"""Priority 28-specific harness checks and small numerical helpers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


def least_squares_decode(measurement_matrix: np.ndarray, sketch: np.ndarray) -> np.ndarray:
    """Recover a key-holder estimate with an explicit least-squares solve."""

    matrix = np.asarray(measurement_matrix, dtype=np.float64)
    observed = np.asarray(sketch, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError("measurement_matrix must be two-dimensional")
    if matrix.shape[0] != observed.size:
        raise ValueError("sketch length does not match matrix row count")
    solution, *_ = np.linalg.lstsq(matrix, observed, rcond=None)
    return solution.astype(np.float32, copy=False)


def state_delta_l2(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    """L2 distance between matching floating tensors in two update dictionaries."""

    total = 0.0
    keys = sorted(set(left) & set(right))
    if not keys:
        return 0.0
    for key in keys:
        if not left[key].is_floating_point() or not right[key].is_floating_point():
            continue
        diff = left[key].detach().cpu().double() - right[key].detach().cpu().double()
        total += float(torch.sum(diff * diff).item())
    return float(np.sqrt(total))


def defense_changes_update(
    original: Mapping[str, torch.Tensor],
    defended: Mapping[str, torch.Tensor],
    *,
    tolerance: float = 1e-12,
) -> tuple[bool, float]:
    """Return whether a defense changed an update by more than tolerance."""

    effect = state_delta_l2(original, defended)
    return effect > tolerance, effect

