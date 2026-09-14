"""Client-update Differential Privacy utilities for simulated FL."""

from __future__ import annotations

from collections import OrderedDict

import torch


def apply_dp_to_local_state(
    local_state: OrderedDict[str, torch.Tensor],
    global_state: OrderedDict[str, torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
    noise_generator: torch.Generator | None = None,
) -> tuple[OrderedDict[str, torch.Tensor], float, float, float]:
    """Clip a full client model update and add Gaussian noise to floating tensors."""
    floating_updates = [
        local_state[name] - global_state[name]
        for name in local_state
        if torch.is_floating_point(local_state[name])
    ]
    if not floating_updates:
        raise ValueError("local_state contains no floating-point tensors")

    update_norm = torch.sqrt(
        sum(torch.sum(update.square()) for update in floating_updates)
    )
    update_norm_value = float(update_norm.item())
    clip_factor = min(1.0, clip_norm / max(update_norm_value, 1e-12))
    clipped_norm_value = update_norm_value * clip_factor
    noise_std = noise_multiplier * clip_norm

    dp_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, local_tensor in local_state.items():
        global_tensor = global_state[name]
        if torch.is_floating_point(local_tensor):
            clipped_update = (local_tensor - global_tensor) * clip_factor
            if noise_generator is None:
                gaussian_noise = torch.randn_like(clipped_update) * noise_std
            else:
                # Draw on CPU from a defense-specific generator, then move to
                # the update device. This keeps DP noise from advancing the
                # training/dropout RNG used by paired RQ2 methods and works on
                # both CPU and MPS.
                gaussian_noise = torch.randn(
                    clipped_update.shape,
                    generator=noise_generator,
                    dtype=clipped_update.dtype,
                    device="cpu",
                ).to(clipped_update.device) * noise_std
            dp_state[name] = global_tensor + clipped_update + gaussian_noise
        else:
            dp_state[name] = local_tensor.clone()

    return dp_state, update_norm_value, clipped_norm_value, noise_std
