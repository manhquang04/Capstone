"""DNA Transform v3: sensitivity-guided blockwise v2 sketching.

v3 is a thin wrapper around :mod:`dna_encoder.transform_defense_v2`.  It keeps
the existing v2 SRHT-plus-quantization primitive unchanged, but applies it
independently to two flattened update blocks:

* the final classifier layer of ``FraudMLP`` (``network.9.weight`` and
  ``network.9.bias``), treated as the pre-registered sensitive block;
* all remaining floating-point update tensors, treated as the rest block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from dna_encoder.transform_defense_v2 import (
    DNATransformV2Config,
    DNATransformV2Metadata,
    DNATransformV2Stats,
    reconstruct_update_array_v2,
    transform_update_array_v2,
)

V3_SENSITIVE_PARAMETER_NAMES: tuple[str, ...] = (
    "network.9.weight",
    "network.9.bias",
)


@dataclass(frozen=True)
class DNATransformV3Config:
    """Configuration for sensitivity-guided blockwise v2 sketching."""

    rest_compression_ratio: float = 0.95
    sensitive_compression_ratio: float = 0.50
    quantization_eta: float = 0.01
    seed: int | None = None
    sensitive_parameter_names: tuple[str, ...] = V3_SENSITIVE_PARAMETER_NAMES


@dataclass(frozen=True)
class DNATransformV3TensorSpec:
    """Placement metadata for one tensor in the flattened update."""

    name: str
    shape: tuple[int, ...]
    size: int
    block: str
    block_start: int
    block_end: int


@dataclass(frozen=True)
class DNATransformV3BlockMetadata:
    """Metadata required to reconstruct one v3 block."""

    block: str
    config: DNATransformV2Config
    metadata: DNATransformV2Metadata


@dataclass(frozen=True)
class DNATransformV3Metadata:
    """Metadata required to reconstruct a full v3-protected update."""

    tensor_specs: tuple[DNATransformV3TensorSpec, ...]
    block_metadata: tuple[DNATransformV3BlockMetadata, ...]
    sensitive_parameter_names: tuple[str, ...]


def split_update_blocks_v3(
    update: Mapping[str, np.ndarray],
    sensitive_parameter_names: tuple[str, ...] = V3_SENSITIVE_PARAMETER_NAMES,
) -> tuple[dict[str, np.ndarray], tuple[DNATransformV3TensorSpec, ...]]:
    """Flatten a state/update mapping into ``rest`` and ``sensitive`` blocks."""

    sensitive_names = set(sensitive_parameter_names)
    block_parts: dict[str, list[np.ndarray]] = {"rest": [], "sensitive": []}
    block_offsets = {"rest": 0, "sensitive": 0}
    specs: list[DNATransformV3TensorSpec] = []
    for name, value in update.items():
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.floating):
            continue
        block = "sensitive" if name in sensitive_names else "rest"
        flat = array.astype(np.float32, copy=False).reshape(-1)
        start = block_offsets[block]
        end = start + int(flat.size)
        block_parts[block].append(flat)
        block_offsets[block] = end
        specs.append(
            DNATransformV3TensorSpec(
                name=name,
                shape=tuple(array.shape),
                size=int(flat.size),
                block=block,
                block_start=start,
                block_end=end,
            )
        )
    blocks = {
        block: np.concatenate(parts).astype(np.float32, copy=False)
        if parts
        else np.zeros(0, dtype=np.float32)
        for block, parts in block_parts.items()
    }
    return blocks, tuple(specs)


def transform_update_state_v3(
    update: Mapping[str, np.ndarray],
    config: DNATransformV3Config,
    quantization_seed: int | None = None,
) -> tuple[dict[str, np.ndarray], DNATransformV3Metadata]:
    """Transform a flattened update with block-specific v2 sketches."""

    if config.seed is None:
        raise ValueError("DNATransformV3Config.seed must be provided")
    blocks, specs = split_update_blocks_v3(update, config.sensitive_parameter_names)
    block_configs = {
        "rest": DNATransformV2Config(
            compression_ratio=config.rest_compression_ratio,
            quantization_eta=config.quantization_eta,
            seed=config.seed,
        ),
        "sensitive": DNATransformV2Config(
            compression_ratio=config.sensitive_compression_ratio,
            quantization_eta=config.quantization_eta,
            seed=config.seed,
        ),
    }
    sketches: dict[str, np.ndarray] = {}
    metadata: list[DNATransformV3BlockMetadata] = []
    for tensor_index, block in enumerate(("rest", "sensitive")):
        sketch, block_meta = transform_update_array_v2(
            blocks[block],
            block_configs[block],
            tensor_index=tensor_index,
            quantization_seed=quantization_seed,
        )
        sketches[block] = sketch
        metadata.append(DNATransformV3BlockMetadata(block, block_configs[block], block_meta))
    return sketches, DNATransformV3Metadata(
        tensor_specs=specs,
        block_metadata=tuple(metadata),
        sensitive_parameter_names=config.sensitive_parameter_names,
    )


def reconstruct_update_state_v3(
    sketches: Mapping[str, np.ndarray],
    metadata: DNATransformV3Metadata,
) -> tuple[dict[str, np.ndarray], dict[str, DNATransformV2Stats]]:
    """Lift v3 sketches and unpack them back into the original tensor mapping."""

    reconstructed_blocks: dict[str, np.ndarray] = {}
    stats: dict[str, DNATransformV2Stats] = {}
    for item in metadata.block_metadata:
        if item.block not in sketches:
            raise ValueError(f"missing sketch block: {item.block}")
        reconstructed, block_stats = reconstruct_update_array_v2(sketches[item.block], item.metadata)
        reconstructed_blocks[item.block] = reconstructed.reshape(-1)
        stats[item.block] = block_stats

    state: dict[str, np.ndarray] = {}
    for spec in metadata.tensor_specs:
        block = reconstructed_blocks[spec.block]
        state[spec.name] = block[spec.block_start : spec.block_end].reshape(spec.shape).astype(
            np.float32,
            copy=False,
        )
    return state, stats
