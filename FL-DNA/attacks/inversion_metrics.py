"""Metrics for tabular gradient inversion reconstruction quality."""

from __future__ import annotations

import math

import numpy as np

from .pseudo_image import minmax_pair, vector_to_pseudo_image


def reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    """Compute image-style and tabular reconstruction metrics."""
    original = np.asarray(original, dtype=np.float32).reshape(-1)
    reconstructed = np.asarray(reconstructed, dtype=np.float32).reshape(-1)
    if original.shape != reconstructed.shape:
        raise ValueError("original and reconstructed vectors must have the same shape")

    delta = reconstructed - original
    mse = float(np.mean(np.square(delta)))
    cosine = _cosine_similarity(original, reconstructed)
    pearson = _pearson_correlation(original, reconstructed)
    sign_match = float(np.mean(np.signbit(original) == np.signbit(reconstructed)))

    original_image, reconstructed_image = minmax_pair(
        vector_to_pseudo_image(original),
        vector_to_pseudo_image(reconstructed),
    )
    image_mse = float(np.mean(np.square(reconstructed_image - original_image)))

    return {
        "mse": mse,
        "feature_mse": mse,
        "psnr": _psnr(image_mse),
        "ssim": _global_ssim(original_image, reconstructed_image),
        "cosine_similarity": cosine,
        "pearson_correlation": pearson,
        "sign_match_ratio": sign_match,
    }


def _psnr(mse: float, data_range: float = 1.0) -> float:
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))


def _global_ssim(x: np.ndarray, y: np.ndarray, data_range: float = 1.0) -> float:
    """Small-array SSIM approximation using global statistics."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mux = float(np.mean(x))
    muy = float(np.mean(y))
    varx = float(np.var(x))
    vary = float(np.var(y))
    cov = float(np.mean((x - mux) * (y - muy)))
    numerator = (2.0 * mux * muy + c1) * (2.0 * cov + c2)
    denominator = (mux * mux + muy * muy + c1) * (varx + vary + c2)
    return float(numerator / max(denominator, 1e-12))


def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denom = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denom)
