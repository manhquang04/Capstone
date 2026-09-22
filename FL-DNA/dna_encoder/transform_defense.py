"""DNA-based gradient/update transformation defense."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binary_mapper import float32_array_to_binary
from .dna_mapper import binary_to_dna


@dataclass(frozen=True)
class DNATransformConfig:
    """Configuration for DNA-guided update transformation."""

    block_size: int = 256
    mix_ratio: float = 0.05
    keep_ratio: float = 0.90
    shrink_factor: float = 0.50
    seed: int | None = None


@dataclass(frozen=True)
class DNATransformStats:
    """Per-array diagnostics for transformation strength."""

    transformed_elements: int
    blocks: int
    mean_abs_delta: float
    max_abs_delta: float
    relative_l2_delta: float
    cosine_similarity: float


def transform_update_array(
    array: np.ndarray,
    config: DNATransformConfig,
    tensor_index: int = 0,
) -> tuple[np.ndarray, DNATransformStats]:
    """
    Transform a float32 update using DNA-derived block rules.

    The transform is intentionally not bit-exact. Each block is mapped to a DNA
    sequence to derive a deterministic sequence seed, then a mild permutation,
    selective attenuation, and residual mix are applied. The output remains a
    numeric update that FedAvg can aggregate.
    """
    original = np.asarray(array, dtype=np.float32)
    if config.seed is None:
        raise ValueError("DNATransformConfig.seed must be provided by the caller")
    flat = original.reshape(-1).astype(np.float32, copy=True)
    transformed = flat.copy()
    block_size = max(1, int(config.block_size))
    blocks = 0

    for start in range(0, flat.size, block_size):
        stop = min(start + block_size, flat.size)
        block = flat[start:stop]
        if block.size == 0:
            continue

        block_seed = _dna_block_seed_from_float32_block(block, config.seed, tensor_index, blocks)
        rng = np.random.default_rng(block_seed)
        permutation = rng.permutation(block.size)
        permuted = block[permutation]

        attenuated = permuted.copy()
        if block.size > 1:
            keep_ratio = min(max(config.keep_ratio, 0.0), 1.0)
            threshold = np.quantile(np.abs(permuted), 1.0 - keep_ratio)
            low_energy_mask = np.abs(permuted) < threshold
            attenuated[low_energy_mask] *= config.shrink_factor

        mixed = (1.0 - config.mix_ratio) * block + config.mix_ratio * attenuated
        transformed[start:stop] = mixed.astype(np.float32, copy=False)
        blocks += 1

    delta = transformed - flat
    original_norm = float(np.linalg.norm(flat))
    transformed_norm = float(np.linalg.norm(transformed))
    dot = float(np.dot(flat, transformed))
    denom = max(original_norm * transformed_norm, 1e-12)
    stats = DNATransformStats(
        transformed_elements=int(flat.size),
        blocks=blocks,
        mean_abs_delta=float(np.mean(np.abs(delta))) if delta.size else 0.0,
        max_abs_delta=float(np.max(np.abs(delta))) if delta.size else 0.0,
        relative_l2_delta=float(np.linalg.norm(delta) / max(original_norm, 1e-12)),
        cosine_similarity=float(dot / denom),
    )
    return transformed.reshape(original.shape).astype(np.float32, copy=False), stats


_DNA_SYMBOLS_BY_BYTE = tuple(
    "".join("ATGC"[(byte >> shift) & 0b11] for shift in (6, 4, 2, 0))
    for byte in range(256)
)
_DNA_COUNTS_BY_BYTE = tuple(
    (
        symbols.count("A"),
        symbols.count("T"),
        symbols.count("G"),
        symbols.count("C"),
    )
    for symbols in _DNA_SYMBOLS_BY_BYTE
)


def _dna_block_seed_from_float32_block(
    block: np.ndarray,
    base_seed: int,
    tensor_index: int,
    block_index: int,
) -> int:
    """Compute the same DNA-derived seed without materializing bit/DNA strings."""
    raw = np.ascontiguousarray(np.asarray(block, dtype="<f4")).tobytes()
    count_a = count_t = count_g = count_c = 0
    rolling = 0
    rolling_symbols = 0
    for byte in raw:
        a, t, g, c = _DNA_COUNTS_BY_BYTE[byte]
        count_a += a
        count_t += t
        count_g += g
        count_c += c
        if rolling_symbols < 512:
            for symbol in _DNA_SYMBOLS_BY_BYTE[byte][: 512 - rolling_symbols]:
                rolling += (rolling_symbols + 1) * ord(symbol)
                rolling_symbols += 1
    # Large odd/prime multipliers decorrelate tensor/block indices and DNA base counts.
    seed = (
        base_seed
        + 1_000_003 * tensor_index
        + 97_409 * block_index
        + 17 * count_a
        + 31 * count_t
        + 47 * count_g
        + 61 * count_c
        + rolling
    )
    return int(seed % (2**31))
