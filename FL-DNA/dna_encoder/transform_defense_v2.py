"""DNA Transform v2: lossy random projection plus quantization.

This module is intentionally separate from :mod:`dna_encoder.transform_defense`
(`DNA Transform v1`).  v2 is a stateless, lossy sketch:

    u -> q = Q_delta(R_s u) -> u_hat = R_s^T q

where ``R_s = sqrt(n / k) * P_s * H * D_s``.  ``H`` is a normalized Hadamard
transform over a padded power-of-two dimension ``n``; ``D_s`` is a seeded sign
flip; and ``P_s`` samples ``k < n`` coordinates.  The server must reconstruct
each client update back to full parameter space before FedAvg averaging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DNATransformV2Config:
    """Configuration for the stateless DNA Transform v2 sketch."""

    compression_ratio: float = 0.95
    quantization_eta: float = 0.01
    seed: int | None = None


@dataclass(frozen=True)
class DNATransformV2Metadata:
    """Metadata required by the server to reconstruct the sketched update."""

    original_shape: tuple[int, ...]
    original_size: int
    padded_size: int
    sketch_size: int
    compression_ratio: float
    quantization_eta: float
    quantization_delta: float
    seed: int
    tensor_index: int
    sampled_indices: tuple[int, ...]


@dataclass(frozen=True)
class DNATransformV2Stats:
    """Mechanical diagnostics for one v2 transform/reconstruct operation."""

    original_l2: float
    reconstructed_l2: float
    relative_l2_error: float
    sketch_rank_upper_bound: int
    original_size: int
    padded_size: int
    sketch_size: int
    quantization_delta: float


def transform_update_array_v2(
    array: np.ndarray,
    config: DNATransformV2Config,
    tensor_index: int = 0,
    quantization_seed: int | None = None,
) -> tuple[np.ndarray, DNATransformV2Metadata]:
    """Project and quantize one array.

    Returns the low-dimensional quantized sketch and the metadata required for
    reconstruction.  The returned sketch is one-dimensional float32.
    """

    if config.seed is None:
        raise ValueError("DNATransformV2Config.seed must be provided")
    _validate_config(config)

    original = np.asarray(array, dtype=np.float32)
    flat = original.reshape(-1).astype(np.float64, copy=False)
    original_size = int(flat.size)
    padded_size = _next_power_of_two(max(1, original_size))
    sketch_size = _sketch_size(padded_size, config.compression_ratio)
    seed = _derive_seed(config.seed, tensor_index)

    padded = np.zeros(padded_size, dtype=np.float64)
    padded[:original_size] = flat

    signs = _signs(padded_size, seed)
    sampled = _sampled_indices(padded_size, sketch_size, seed)
    projected_full = _fwht_normalized(padded * signs)
    scale = float(np.sqrt(padded_size / sketch_size))
    sketch = scale * projected_full[sampled]

    delta = _quantization_delta(flat, config.quantization_eta)
    q_seed = _derive_seed(seed, 0 if quantization_seed is None else quantization_seed)
    quantized = _stochastic_quantize(sketch, delta, q_seed)

    metadata = DNATransformV2Metadata(
        original_shape=tuple(original.shape),
        original_size=original_size,
        padded_size=padded_size,
        sketch_size=sketch_size,
        compression_ratio=float(config.compression_ratio),
        quantization_eta=float(config.quantization_eta),
        quantization_delta=float(delta),
        seed=int(seed),
        tensor_index=int(tensor_index),
        sampled_indices=tuple(int(i) for i in sampled.tolist()),
    )
    return quantized.astype(np.float32, copy=False), metadata


def reconstruct_update_array_v2(
    sketch: np.ndarray,
    metadata: DNATransformV2Metadata,
) -> tuple[np.ndarray, DNATransformV2Stats]:
    """Lift a v2 sketch back to the original array shape."""

    q = np.asarray(sketch, dtype=np.float64).reshape(-1)
    if q.size != metadata.sketch_size:
        raise ValueError(
            f"sketch has {q.size} values but metadata expects {metadata.sketch_size}"
        )
    if len(metadata.sampled_indices) != metadata.sketch_size:
        raise ValueError("metadata sampled_indices length does not match sketch_size")

    lifted = np.zeros(metadata.padded_size, dtype=np.float64)
    scale = float(np.sqrt(metadata.padded_size / metadata.sketch_size))
    lifted[np.asarray(metadata.sampled_indices, dtype=np.int64)] = scale * q
    reconstructed_padded = _fwht_normalized(lifted) * _signs(
        metadata.padded_size, metadata.seed
    )
    reconstructed = reconstructed_padded[: metadata.original_size].reshape(
        metadata.original_shape
    )
    stats = _stats(reconstructed, metadata)
    return reconstructed.astype(np.float32, copy=False), stats


def transform_and_reconstruct_array_v2(
    array: np.ndarray,
    config: DNATransformV2Config,
    tensor_index: int = 0,
    quantization_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, DNATransformV2Metadata, DNATransformV2Stats]:
    """Convenience wrapper returning sketch, reconstruction, metadata and stats."""

    sketch, metadata = transform_update_array_v2(
        array, config, tensor_index=tensor_index, quantization_seed=quantization_seed
    )
    reconstructed, stats = reconstruct_update_array_v2(sketch, metadata)
    original = np.asarray(array, dtype=np.float32)
    original_norm = float(np.linalg.norm(original.reshape(-1).astype(np.float64)))
    diff_norm = float(
        np.linalg.norm(
            reconstructed.reshape(-1).astype(np.float64)
            - original.reshape(-1).astype(np.float64)
        )
    )
    stats = DNATransformV2Stats(
        original_l2=original_norm,
        reconstructed_l2=float(np.linalg.norm(reconstructed.reshape(-1).astype(np.float64))),
        relative_l2_error=diff_norm / max(original_norm, np.finfo(float).tiny),
        sketch_rank_upper_bound=metadata.sketch_size,
        original_size=metadata.original_size,
        padded_size=metadata.padded_size,
        sketch_size=metadata.sketch_size,
        quantization_delta=metadata.quantization_delta,
    )
    return sketch, reconstructed, metadata, stats


def materialize_projection_matrix_v2(
    original_size: int,
    config: DNATransformV2Config,
    tensor_index: int = 0,
) -> tuple[np.ndarray, DNATransformV2Metadata]:
    """Materialize ``R_s`` for small mechanical audits and unit tests."""

    if config.seed is None:
        raise ValueError("DNATransformV2Config.seed must be provided")
    _validate_config(config)
    if original_size < 1:
        raise ValueError("original_size must be positive")

    padded_size = _next_power_of_two(original_size)
    sketch_size = _sketch_size(padded_size, config.compression_ratio)
    seed = _derive_seed(config.seed, tensor_index)
    sampled = _sampled_indices(padded_size, sketch_size, seed)
    scale = float(np.sqrt(padded_size / sketch_size))
    signs = _signs(padded_size, seed)

    matrix = np.zeros((sketch_size, original_size), dtype=np.float64)
    for column in range(original_size):
        basis = np.zeros(padded_size, dtype=np.float64)
        basis[column] = signs[column]
        transformed = _fwht_normalized(basis)
        matrix[:, column] = scale * transformed[sampled]

    metadata = DNATransformV2Metadata(
        original_shape=(original_size,),
        original_size=original_size,
        padded_size=padded_size,
        sketch_size=sketch_size,
        compression_ratio=float(config.compression_ratio),
        quantization_eta=float(config.quantization_eta),
        quantization_delta=0.0,
        seed=int(seed),
        tensor_index=int(tensor_index),
        sampled_indices=tuple(int(i) for i in sampled.tolist()),
    )
    return matrix, metadata


def _validate_config(config: DNATransformV2Config) -> None:
    if not 0.0 < config.compression_ratio <= 1.0:
        raise ValueError("compression_ratio must be in (0, 1]")
    if config.quantization_eta < 0.0:
        raise ValueError("quantization_eta must be non-negative")


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _sketch_size(padded_size: int, compression_ratio: float) -> int:
    return max(1, min(padded_size, int(np.ceil(padded_size * compression_ratio))))


def _derive_seed(*values: int) -> int:
    state = 0x9E3779B9
    for value in values:
        state ^= int(value) + 0x9E3779B9 + ((state << 6) & 0xFFFFFFFF) + (state >> 2)
        state &= 0xFFFFFFFF
    return int(state)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) % (2**32))


def _signs(size: int, seed: int) -> np.ndarray:
    return _rng(_derive_seed(seed, 17)).choice(
        np.asarray([-1.0, 1.0], dtype=np.float64), size=size, replace=True
    )


def _sampled_indices(size: int, sketch_size: int, seed: int) -> np.ndarray:
    return np.sort(_rng(_derive_seed(seed, 29)).choice(size, size=sketch_size, replace=False))


def _fwht_normalized(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    n = out.size
    if n < 1 or n & (n - 1):
        raise ValueError("FWHT length must be a positive power of two")
    step = 1
    while step < n:
        pairs = out.reshape(-1, step * 2)
        left = pairs[:, :step].copy()
        right = pairs[:, step : step * 2].copy()
        pairs[:, :step] = left + right
        pairs[:, step : step * 2] = left - right
        step *= 2
    return out / np.sqrt(n)


def _quantization_delta(flat: np.ndarray, eta: float) -> float:
    if eta == 0.0 or flat.size == 0:
        return 0.0
    norm = float(np.linalg.norm(flat.astype(np.float64, copy=False)))
    if norm == 0.0:
        return 0.0
    return float(eta * norm / np.sqrt(flat.size))


def _stochastic_quantize(values: np.ndarray, delta: float, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if delta <= 0.0:
        return values.copy()
    scaled = values / delta
    lower = np.floor(scaled)
    probability_up = scaled - lower
    draws = _rng(_derive_seed(seed, 43)).random(values.shape)
    rounded = lower + (draws < probability_up)
    return rounded * delta


def _stats(reconstructed: np.ndarray, metadata: DNATransformV2Metadata) -> DNATransformV2Stats:
    reconstructed_l2 = float(np.linalg.norm(reconstructed.reshape(-1).astype(np.float64)))
    return DNATransformV2Stats(
        original_l2=float("nan"),
        reconstructed_l2=reconstructed_l2,
        relative_l2_error=float("nan"),
        sketch_rank_upper_bound=metadata.sketch_size,
        original_size=metadata.original_size,
        padded_size=metadata.padded_size,
        sketch_size=metadata.sketch_size,
        quantization_delta=metadata.quantization_delta,
    )
