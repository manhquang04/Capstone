"""DP variant helpers for Priority 27 C2."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from math import sqrt

import torch


BN_BUFFER_MARKERS = ("running_mean", "running_var", "num_batches_tracked")


def is_bn_buffer(name: str) -> bool:
    return any(marker in name for marker in BN_BUFFER_MARKERS)


def trainable_names_from_state(local_state: Mapping[str, torch.Tensor], global_state: Mapping[str, torch.Tensor]) -> list[str]:
    return [
        name
        for name, tensor in local_state.items()
        if torch.is_floating_point(tensor) and not is_bn_buffer(name)
    ]


def floating_names(local_state: Mapping[str, torch.Tensor]) -> list[str]:
    return [name for name, tensor in local_state.items() if torch.is_floating_point(tensor)]


def update_norm(local_state: Mapping[str, torch.Tensor], global_state: Mapping[str, torch.Tensor], names: list[str]) -> float:
    return float(torch.sqrt(sum(torch.sum((local_state[name] - global_state[name]).square()) for name in names)).item())


def tensor_update_norms(local_state: Mapping[str, torch.Tensor], global_state: Mapping[str, torch.Tensor], names: list[str]) -> dict[str, float]:
    return {name: float(torch.linalg.vector_norm((local_state[name] - global_state[name]).detach()).item()) for name in names}


def apply_dp_variant_to_local_state(
    local_state: OrderedDict[str, torch.Tensor],
    global_state: OrderedDict[str, torch.Tensor],
    *,
    variant: str,
    clip_spec: float | Mapping[str, float],
    noise_multiplier: float,
    noise_generator: torch.Generator,
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, float]]:
    """Apply a Priority 27 DP variant to one local state."""
    if variant == "full_state_single_clip":
        names = floating_names(local_state)
        clip_norm = float(clip_spec)
        norm = update_norm(local_state, global_state, names)
        clip_factor = min(1.0, clip_norm / max(norm, 1e-12))
        noise_std = noise_multiplier * clip_norm
        out: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, local_tensor in local_state.items():
            global_tensor = global_state[name]
            if torch.is_floating_point(local_tensor):
                noise = torch.randn(local_tensor.shape, generator=noise_generator, dtype=local_tensor.dtype, device="cpu").to(local_tensor.device) * noise_std
                out[name] = global_tensor + (local_tensor - global_tensor) * clip_factor + noise
            else:
                out[name] = local_tensor.clone()
        return out, {"norm_before": norm, "clip_factor": clip_factor, "noise_std": noise_std, "effective_sensitivity": clip_norm}

    if variant == "per_tensor_clip":
        if not isinstance(clip_spec, Mapping):
            raise TypeError("per_tensor_clip requires mapping clip_spec")
        names = floating_names(local_state)
        total_sensitivity = sqrt(sum(float(clip_spec[name]) ** 2 for name in names))
        noise_std = noise_multiplier * total_sensitivity
        out = OrderedDict()
        clip_factors = []
        for name, local_tensor in local_state.items():
            global_tensor = global_state[name]
            if torch.is_floating_point(local_tensor):
                tensor_clip = float(clip_spec[name])
                delta = local_tensor - global_tensor
                norm = float(torch.linalg.vector_norm(delta.detach()).item())
                factor = min(1.0, tensor_clip / max(norm, 1e-12))
                clip_factors.append(factor)
                noise = torch.randn(local_tensor.shape, generator=noise_generator, dtype=local_tensor.dtype, device="cpu").to(local_tensor.device) * noise_std
                out[name] = global_tensor + delta * factor + noise
            else:
                out[name] = local_tensor.clone()
        return out, {
            "mean_clip_factor": float(sum(clip_factors) / max(len(clip_factors), 1)),
            "noise_std": noise_std,
            "effective_sensitivity": total_sensitivity,
        }

    if variant == "fedbn_trainable_only":
        names = trainable_names_from_state(local_state, global_state)
        clip_norm = float(clip_spec)
        norm = update_norm(local_state, global_state, names)
        clip_factor = min(1.0, clip_norm / max(norm, 1e-12))
        noise_std = noise_multiplier * clip_norm
        out = OrderedDict()
        for name, local_tensor in local_state.items():
            global_tensor = global_state[name]
            if torch.is_floating_point(local_tensor) and name in names:
                noise = torch.randn(local_tensor.shape, generator=noise_generator, dtype=local_tensor.dtype, device="cpu").to(local_tensor.device) * noise_std
                out[name] = global_tensor + (local_tensor - global_tensor) * clip_factor + noise
            else:
                # FedBN-style: BN buffers and non-floating buffers are not transmitted.
                out[name] = global_tensor.clone()
        return out, {"norm_before": norm, "clip_factor": clip_factor, "noise_std": noise_std, "effective_sensitivity": clip_norm}

    raise ValueError(f"unknown DP variant: {variant}")

