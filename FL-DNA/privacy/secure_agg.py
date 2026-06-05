"""Practical secure aggregation simulation for FL client updates."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import torch


def secure_aggregate_states(
    local_states: Sequence[OrderedDict[str, torch.Tensor]],
    global_state: OrderedDict[str, torch.Tensor],
    sample_counts: Sequence[int],
    seed: int,
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, float | int | bool]]:
    """
    Aggregate local states with pairwise masks that cancel in the server sum.

    Each client contributes a sample-weighted model update. Pairwise random masks
    are added with opposite signs across client pairs, so the server can recover
    only the aggregate update, not individual raw updates.
    """
    if not local_states or len(local_states) != len(sample_counts):
        raise ValueError("local_states and sample_counts must have equal non-zero length")

    total_samples = sum(sample_counts)
    weights = [sample_count / total_samples for sample_count in sample_counts]
    aggregated_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    max_mask_abs = 0.0
    floating_tensors = 0
    masked_elements = 0

    for name, global_tensor in global_state.items():
        reference = local_states[0][name]
        if not torch.is_floating_point(reference):
            aggregated_state[name] = reference.clone()
            continue

        weighted_updates = [
            (local_state[name] - global_tensor) * weight
            for local_state, weight in zip(local_states, weights)
        ]
        masked_updates = [update.clone() for update in weighted_updates]

        for i in range(len(masked_updates)):
            for j in range(i + 1, len(masked_updates)):
                generator = torch.Generator(device=masked_updates[i].device).manual_seed(
                    _pair_seed(seed, name, i, j)
                )
                mask = torch.randn(
                    masked_updates[i].shape,
                    generator=generator,
                    dtype=masked_updates[i].dtype,
                    device=masked_updates[i].device,
                )
                masked_updates[i] = masked_updates[i] + mask
                masked_updates[j] = masked_updates[j] - mask
                max_mask_abs = max(max_mask_abs, float(mask.abs().max().item()))

        aggregate_update = sum(masked_updates)
        aggregated_state[name] = global_tensor + aggregate_update
        floating_tensors += 1
        masked_elements += reference.numel()

    metadata = {
        "secure_agg_enabled": True,
        "secure_agg_round_seed": seed,
        "server_sees_individual_raw_updates": False,
        "pairwise_masking": True,
        "num_clients_masked": len(local_states),
        "masked_floating_tensors": floating_tensors,
        "masked_elements": masked_elements,
        "max_mask_abs": max_mask_abs,
    }
    return aggregated_state, metadata


def _pair_seed(base_seed: int, tensor_name: str, client_i: int, client_j: int) -> int:
    name_hash = sum((index + 1) * ord(char) for index, char in enumerate(tensor_name))
    return int((base_seed + 1_000_003 * client_i + 9_176 * client_j + name_hash) % (2**31))
