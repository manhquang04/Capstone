"""Adaptive approximation for the non-differentiable DNA update transform."""
from __future__ import annotations

import numpy as np
import torch

from dna_encoder.transform_defense import DNATransformConfig, transform_update_array


class _DNA_BPDA(torch.autograd.Function):
    """Exact transform in the forward pass and identity BPDA in backward."""

    @staticmethod
    def forward(ctx, value, config, tensor_index):
        transformed, _ = transform_update_array(
            value.detach().cpu().numpy().astype(np.float32, copy=False),
            config,
            tensor_index=tensor_index,
        )
        return torch.from_numpy(transformed.copy()).to(dtype=value.dtype, device=value.device)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None, None


def transform_state_bpda(state, config):
    """Transform floating state deltas while retaining an approximate gradient."""
    return {
        key: _DNA_BPDA.apply(value, config, tensor_index) if value.is_floating_point() else value
        for tensor_index, (key, value) in enumerate(state.items())
    }


def transform_state_exact(state, config):
    """Apply the exact forward transform without building an autograd graph."""
    return {key: value.detach() for key, value in transform_state_bpda(state, config).items()}
