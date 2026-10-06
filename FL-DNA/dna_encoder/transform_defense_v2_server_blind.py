"""DNA Transform v2-SB: server-blind sketch aggregation.

v2-SB uses the same SRHT-style sketch and scalar quantization idea as
Transform v2, but changes key ownership.  Clients share a 256-bit round key.
The server receives sketches and can aggregate them linearly, but never
receives the key needed to lift a sketch back to update space.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import secrets
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ServerBlindV2Config:
    compression_ratio: float = 0.95
    quantization_eta: float = 0.01


@dataclass(frozen=True)
class ServerBlindV2Metadata:
    original_shape: tuple[int, ...]
    original_size: int
    padded_size: int
    sketch_size: int
    compression_ratio: float
    quantization_eta: float
    quantization_delta: float
    round_number: int
    tensor_index: int


@dataclass(frozen=True)
class ServerBlindV2Sketch:
    sketch: np.ndarray
    metadata: ServerBlindV2Metadata


class ServerBlindAggregator:
    """Server-side aggregation object.

    This object intentionally stores no key material.  It only sums compatible
    sketches with FedAvg weights.
    """

    def __init__(self) -> None:
        self.key = None
        self.round_key = None

    def aggregate(self, sketches: Iterable[ServerBlindV2Sketch], weights: Iterable[float]) -> ServerBlindV2Sketch:
        sketches = list(sketches)
        weights = list(weights)
        if not sketches:
            raise ValueError("at least one sketch is required")
        if len(sketches) != len(weights):
            raise ValueError("sketches and weights must have equal length")
        meta = sketches[0].metadata
        total = np.zeros_like(sketches[0].sketch, dtype=np.float64)
        for item, weight in zip(sketches, weights):
            if not _compatible_for_aggregation(item.metadata, meta):
                raise ValueError("all sketches must share identical metadata")
            total += float(weight) * np.asarray(item.sketch, dtype=np.float64)
        return ServerBlindV2Sketch(total.astype(np.float32), meta)


def generate_round_key() -> bytes:
    return secrets.token_bytes(32)


def validate_round_key(round_key: bytes) -> None:
    if not isinstance(round_key, (bytes, bytearray)) or len(round_key) != 32:
        raise ValueError("v2-SB round key must be exactly 32 bytes (256 bits)")


def sketch_update_array_v2sb(
    array: np.ndarray,
    config: ServerBlindV2Config,
    *,
    round_key: bytes,
    round_number: int,
    tensor_index: int,
    client_id: int,
) -> ServerBlindV2Sketch:
    validate_round_key(round_key)
    _validate_config(config)
    original = np.asarray(array, dtype=np.float32)
    flat = original.reshape(-1).astype(np.float64, copy=False)
    original_size = int(flat.size)
    padded_size = _next_power_of_two(max(1, original_size))
    sketch_size = _sketch_size(padded_size, config.compression_ratio)

    padded = np.zeros(padded_size, dtype=np.float64)
    padded[:original_size] = flat
    signs = _signs(round_key, round_number, tensor_index, padded_size)
    sampled = _sampled_indices(round_key, round_number, tensor_index, padded_size, sketch_size)
    projected = _project_padded(padded, signs, sampled)

    delta = _quantization_delta(flat, config.quantization_eta)
    quantized = _stochastic_quantize(
        projected,
        delta,
        round_key=round_key,
        round_number=round_number,
        tensor_index=tensor_index,
        client_id=client_id,
    )
    return ServerBlindV2Sketch(
        quantized.astype(np.float32, copy=False),
        ServerBlindV2Metadata(
            original_shape=tuple(original.shape),
            original_size=original_size,
            padded_size=padded_size,
            sketch_size=sketch_size,
            compression_ratio=float(config.compression_ratio),
            quantization_eta=float(config.quantization_eta),
            quantization_delta=float(delta),
            round_number=int(round_number),
            tensor_index=int(tensor_index),
        ),
    )


def lift_sketch_array_v2sb(sketch: ServerBlindV2Sketch, *, round_key: bytes) -> np.ndarray:
    validate_round_key(round_key)
    meta = sketch.metadata
    q = np.asarray(sketch.sketch, dtype=np.float64).reshape(-1)
    if q.size != meta.sketch_size:
        raise ValueError("sketch length does not match metadata")
    signs = _signs(round_key, meta.round_number, meta.tensor_index, meta.padded_size)
    sampled = _sampled_indices(round_key, meta.round_number, meta.tensor_index, meta.padded_size, meta.sketch_size)
    recovered = _lift(q, signs, sampled, meta.padded_size, meta.original_size)
    return recovered.reshape(meta.original_shape).astype(np.float32, copy=False)


def materialize_projection_matrix_v2sb(
    original_size: int,
    config: ServerBlindV2Config,
    *,
    round_key: bytes,
    round_number: int,
    tensor_index: int,
) -> np.ndarray:
    validate_round_key(round_key)
    _validate_config(config)
    padded_size = _next_power_of_two(max(1, int(original_size)))
    sketch_size = _sketch_size(padded_size, config.compression_ratio)
    signs = _signs(round_key, round_number, tensor_index, padded_size)
    sampled = _sampled_indices(round_key, round_number, tensor_index, padded_size, sketch_size)
    matrix = np.zeros((sketch_size, int(original_size)), dtype=np.float64)
    for col in range(int(original_size)):
        basis = np.zeros(padded_size, dtype=np.float64)
        basis[col] = 1.0
        matrix[:, col] = _project_padded(basis, signs, sampled)
    return matrix


def lift_raw_sketch_v2sb(
    raw_sketch: np.ndarray,
    *,
    original_shape: tuple[int, ...],
    config: ServerBlindV2Config,
    round_key: bytes,
    round_number: int,
    tensor_index: int,
) -> np.ndarray:
    original_size = int(np.prod(original_shape))
    padded_size = _next_power_of_two(max(1, original_size))
    sketch_size = _sketch_size(padded_size, config.compression_ratio)
    meta = ServerBlindV2Metadata(
        original_shape=tuple(original_shape),
        original_size=original_size,
        padded_size=padded_size,
        sketch_size=sketch_size,
        compression_ratio=float(config.compression_ratio),
        quantization_eta=float(config.quantization_eta),
        quantization_delta=0.0,
        round_number=int(round_number),
        tensor_index=int(tensor_index),
    )
    return lift_sketch_array_v2sb(ServerBlindV2Sketch(np.asarray(raw_sketch, dtype=np.float32), meta), round_key=round_key)


def _validate_config(config: ServerBlindV2Config) -> None:
    if not 0.0 < config.compression_ratio <= 1.0:
        raise ValueError("compression_ratio must be in (0, 1]")
    if config.quantization_eta < 0.0:
        raise ValueError("quantization_eta must be non-negative")


def _compatible_for_aggregation(left: ServerBlindV2Metadata, right: ServerBlindV2Metadata) -> bool:
    """Check sketch-layout compatibility; per-client quantization_delta may differ."""
    return (
        left.original_shape == right.original_shape
        and left.original_size == right.original_size
        and left.padded_size == right.padded_size
        and left.sketch_size == right.sketch_size
        and left.compression_ratio == right.compression_ratio
        and left.quantization_eta == right.quantization_eta
        and left.round_number == right.round_number
        and left.tensor_index == right.tensor_index
    )


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _sketch_size(padded_size: int, compression_ratio: float) -> int:
    return max(1, min(padded_size, int(np.ceil(padded_size * compression_ratio))))


def _shake_bytes(round_key: bytes, *parts: object, nbytes: int) -> bytes:
    validate_round_key(round_key)
    shake = hashlib.shake_256()
    shake.update(b"FL-DNA-v2SB\x00")
    shake.update(round_key)
    for part in parts:
        text = str(part).encode("utf-8")
        shake.update(len(text).to_bytes(4, "little"))
        shake.update(text)
    return shake.digest(nbytes)


def _rng(round_key: bytes, *parts: object) -> np.random.Generator:
    seed_bytes = _shake_bytes(round_key, *parts, nbytes=32)
    seed_int = int.from_bytes(seed_bytes, "little", signed=False)
    return np.random.default_rng(seed_int)


def _signs(round_key: bytes, round_number: int, tensor_index: int, size: int) -> np.ndarray:
    return _rng(round_key, round_number, tensor_index, "signs").choice(
        np.asarray([-1.0, 1.0], dtype=np.float64), size=int(size), replace=True
    )


def _sampled_indices(round_key: bytes, round_number: int, tensor_index: int, size: int, sketch_size: int) -> np.ndarray:
    return np.sort(
        _rng(round_key, round_number, tensor_index, "sampled_indices").choice(
            int(size), size=int(sketch_size), replace=False
        )
    )


def _project_padded(padded: np.ndarray, signs: np.ndarray, sampled: np.ndarray) -> np.ndarray:
    padded = np.asarray(padded, dtype=np.float64)
    scale = float(np.sqrt(padded.size / sampled.size))
    return scale * _fwht_normalized(padded * signs)[sampled]


def _lift(q: np.ndarray, signs: np.ndarray, sampled: np.ndarray, padded_size: int, original_size: int) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    lifted = np.zeros(int(padded_size), dtype=np.float64)
    scale = float(np.sqrt(int(padded_size) / sampled.size))
    lifted[np.asarray(sampled, dtype=np.int64)] = scale * q
    recovered = _fwht_normalized(lifted) * signs
    return recovered[: int(original_size)]


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


def _stochastic_quantize(
    values: np.ndarray,
    delta: float,
    *,
    round_key: bytes,
    round_number: int,
    tensor_index: int,
    client_id: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if delta <= 0.0:
        return values.copy()
    scaled = values / delta
    lower = np.floor(scaled)
    probability_up = scaled - lower
    draws = _rng(round_key, round_number, client_id, tensor_index, "quantization").random(values.shape)
    rounded = lower + (draws < probability_up)
    return rounded * delta
