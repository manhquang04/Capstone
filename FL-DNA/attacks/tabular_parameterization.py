"""PaySim-aware input parameterization and update-matching objectives."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class PaySimManifold:
    """Decode six base numeric fields and derive the two balance differences."""

    center: torch.Tensor
    scale: torch.Tensor
    category_count: int = 5

    @classmethod
    def from_metadata(cls, metadata, *, dtype=torch.float32, device="cpu"):
        return cls(
            center=torch.as_tensor(metadata.numeric_center, dtype=dtype, device=device),
            scale=torch.as_tensor(metadata.numeric_scale, dtype=dtype, device=device),
            category_count=len(metadata.type_categories),
        )

    @property
    def latent_dim(self):
        return 6 + self.category_count

    def decode(self, latent, temperature=1.0):
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected [records, {self.latent_dim}] latent tensor")
        base_scaled = latent[:, :6]
        base_raw = base_scaled * self.scale[:6] + self.center[:6]
        difference_raw = torch.stack(
            (base_raw[:, 2] - base_raw[:, 3], base_raw[:, 5] - base_raw[:, 4]),
            dim=1,
        )
        difference_scaled = (difference_raw - self.center[6:8]) / self.scale[6:8]
        categories = F.softmax(latent[:, 6:] / temperature, dim=1)
        return torch.cat((base_scaled, difference_scaled, categories), dim=1)


def unconstrained_decode(latent, temperature=1.0):
    """Legacy parameterization retained as a development comparator."""
    if latent.ndim != 2 or latent.shape[1] != 13:
        raise ValueError("Expected [records, 13] latent tensor")
    return torch.cat((latent[:, :8], F.softmax(latent[:, 8:] / temperature, dim=1)), dim=1)


def update_matching_objective(candidate, observed, keys, *, mode, bn_weight=1.0):
    """Compare a replayed update with the observed floating model-state delta."""
    if mode == "flat":
        candidate_flat = torch.cat([candidate[key].reshape(-1) for key in keys])
        observed_flat = torch.cat([observed[key].reshape(-1) for key in keys])
        if float(observed_flat.square().sum().detach()) == 0.0:
            return candidate_flat.square().mean()
        cosine = 1 - F.cosine_similarity(candidate_flat, observed_flat, dim=0, eps=1e-12)
        relative_mse = (candidate_flat - observed_flat).square().sum() / observed_flat.square().sum().clamp_min(1e-20)
        return cosine + 0.1 * relative_mse
    if mode != "balanced_bn":
        raise ValueError(f"Unknown objective mode: {mode}")

    parameter_losses = []
    buffer_losses = []
    for key in keys:
        candidate_value = candidate[key].reshape(-1)
        observed_value = observed[key].reshape(-1)
        if float(observed_value.square().sum().detach()) == 0.0:
            loss = candidate_value.square().mean()
        else:
            cosine = 1 - F.cosine_similarity(candidate_value, observed_value, dim=0, eps=1e-12)
            relative_mse = (candidate_value - observed_value).square().sum() / observed_value.square().sum().clamp_min(1e-20)
            loss = cosine + 0.1 * relative_mse
        target = buffer_losses if "running_mean" in key or "running_var" in key else parameter_losses
        target.append(loss)
    parameter_term = torch.stack(parameter_losses).mean()
    if not buffer_losses:
        return parameter_term
    return parameter_term + bn_weight * torch.stack(buffer_losses).mean()
