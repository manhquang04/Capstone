"""Scoring helpers for the Priority 27 harness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import binomtest


@dataclass(frozen=True)
class ScoreRecord:
    target_id: str
    branch: str
    mse: float
    baseline_population_mse: float | None = None
    baseline_constant_mse: float | None = None
    psnr: float | None = None
    ssim: float | None = None


@dataclass(frozen=True)
class DecoyScoreRecord:
    target_id: str
    decoy_source_id: str
    scored_against_target_id: str
    attack_mse: float
    decoy_mse: float


def standardized_mse(estimate: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor, std: np.ndarray | torch.Tensor) -> float:
    est = _to_numpy(estimate).astype(np.float64)
    tgt = _to_numpy(target).astype(np.float64)
    scale = _to_numpy(std).astype(np.float64)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return float(np.mean(((est - tgt) / scale) ** 2))


def image_metrics(estimate: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor, data_range: float = 1.0) -> dict[str, float]:
    est = np.clip(_to_numpy(estimate).astype(np.float64), 0.0, data_range)
    tgt = np.clip(_to_numpy(target).astype(np.float64), 0.0, data_range)
    mse = float(np.mean((est - tgt) ** 2))
    psnr = float("inf") if mse == 0 else float(20 * np.log10(data_range) - 10 * np.log10(mse))
    # Keep a dependency-light global SSIM proxy for harness checks.  Domain-
    # specific image experiments may replace this with skimage windowed SSIM.
    mu_x = float(est.mean())
    mu_y = float(tgt.mean())
    var_x = float(est.var())
    var_y = float(tgt.var())
    cov = float(((est - mu_x) * (tgt - mu_y)).mean())
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return {"mse": mse, "psnr": psnr, "ssim": float(ssim)}


def exact_one_sided_sign_p(wins: int, n: int) -> float:
    return float(binomtest(wins, n, 0.5, alternative="greater").pvalue) if n else 1.0


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)

