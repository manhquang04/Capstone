"""Priority35A arithmetic primitives; no experiments or artifact writes.

This module does not construct a server knowledge bundle. In particular, a full
client state used in simulator unit tests is not an allowed instrument input.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Mapping

import torch


class _ExactForwardSqrt(torch.autograd.Function):
    """Native forward sqrt, declared zero backward derivative at exactly 0."""
    @staticmethod
    def forward(ctx, value):
        result = value.sqrt()
        ctx.save_for_backward(value, result)
        return result

    @staticmethod
    def backward(ctx, gradient):
        value, result = ctx.saved_tensors
        factor = torch.zeros_like(value)
        positive = value > 0
        # Do not evaluate a hidden 1/0 branch then multiply it by zero.
        factor[positive] = 0.5 / result[positive]
        return gradient * factor


def require_finite_cpu(values: Mapping[str, torch.Tensor], context: str) -> None:
    if not values:
        raise ValueError(f"{context}: missing tensors")
    for name, value in values.items():
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise ValueError(f"{context}: expected CPU tensor {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{context}: non-finite tensor {name}")
        if name.endswith("running_var") and (value < 0).any():
            raise ValueError(f"{context}: negative BN running variance {name}")


def fresh_adam_parameters(
    parameters: Mapping[str, torch.Tensor],
    gradients: Mapping[str, torch.Tensor],
    *,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> OrderedDict[str, torch.Tensor]:
    """One fresh, default Adam step, retaining float operation order.

    This is the non-capturable single-tensor CPU branch: no weight decay,
    AMSGrad, maximization, or prior optimizer state. Returning parameters (not
    only a real-valued update approximation) preserves float32 subtraction and
    parameter-rounding effects. Forward arithmetic is not clamped. A zero
    gradient has a valid forward result; the prospective protocol defines
    sqrt's backward derivative as zero at exactly zero squared gradient.
    """
    require_finite_cpu(parameters, "Adam parameters")
    require_finite_cpu(gradients, "Adam gradients")
    if tuple(parameters) != tuple(gradients):
        raise ValueError("Adam parameter/gradient names or order differ")
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("invalid Adam lr/eps")
    beta1, beta2 = betas
    if not all(math.isfinite(b) and 0 <= b < 1 for b in betas):
        raise ValueError("invalid Adam betas")
    result = OrderedDict()
    for name, parameter in parameters.items():
        gradient = gradients[name]
        if parameter.shape != gradient.shape or parameter.dtype != gradient.dtype:
            raise ValueError(f"Adam tensor metadata mismatch: {name}")
        if not parameter.is_floating_point():
            raise ValueError(f"Adam requires floating parameters: {name}")
        first = torch.zeros_like(parameter).lerp(gradient, 1 - beta1)
        second = torch.zeros_like(parameter).mul(beta2).addcmul(
            gradient, gradient, value=1 - beta2)
        denominator = _ExactForwardSqrt.apply(second).div(math.sqrt(1 - beta2)).add(eps)
        result[name] = parameter.addcdiv(first, denominator, value=-lr / (1 - beta1))
    require_finite_cpu(result, "Adam result")
    return result


def parameter_difference(before, after):
    require_finite_cpu(before, "difference before")
    require_finite_cpu(after, "difference after")
    if tuple(before) != tuple(after):
        raise ValueError("difference tensor names or order differ")
    result = OrderedDict()
    for name, value in before.items():
        if value.shape != after[name].shape or value.dtype != after[name].dtype:
            raise ValueError(f"difference metadata mismatch: {name}")
        result[name] = after[name] - value
    require_finite_cpu(result, "difference result")
    return result
