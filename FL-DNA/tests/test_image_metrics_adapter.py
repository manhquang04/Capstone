from __future__ import annotations

import math

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from attacks.image_metrics import compute_image_metrics


def test_identical_rgb_image_has_infinite_psnr_and_unit_ssim() -> None:
    rng = np.random.default_rng(20260917)
    reference_chw = rng.random((3, 32, 32), dtype=np.float64)

    metrics = compute_image_metrics(reference_chw, reference_chw.copy())

    assert metrics.mse == 0.0
    assert math.isinf(metrics.psnr)
    assert metrics.ssim == 1.0


def test_adapter_matches_skimage_reference_for_noisy_image() -> None:
    rng = np.random.default_rng(20260918)
    reference_chw = rng.random((3, 32, 32), dtype=np.float64)
    noisy_chw = np.clip(reference_chw + rng.normal(0.0, 0.20, size=reference_chw.shape), 0.0, 1.0)

    metrics = compute_image_metrics(reference_chw, noisy_chw)

    reference_hwc = np.transpose(reference_chw, (1, 2, 0))
    noisy_hwc = np.transpose(noisy_chw, (1, 2, 0))
    expected_psnr = float(peak_signal_noise_ratio(reference_hwc, noisy_hwc, data_range=1.0))
    expected_ssim = float(
        structural_similarity(reference_hwc, noisy_hwc, data_range=1.0, channel_axis=-1)
    )
    expected_mse = float(np.mean((reference_hwc - noisy_hwc) ** 2))

    assert np.isclose(metrics.mse, expected_mse)
    assert np.isclose(metrics.psnr, expected_psnr)
    assert np.isclose(metrics.ssim, expected_ssim)
    assert metrics.psnr < 25.0
    assert metrics.ssim < 0.90


def test_adapter_clips_out_of_range_values_before_metrics() -> None:
    reference = np.zeros((3, 32, 32), dtype=np.float64)
    reconstruction = np.full((3, 32, 32), 1.5, dtype=np.float64)

    metrics = compute_image_metrics(reference, reconstruction)

    assert metrics.mse == 1.0
    assert metrics.psnr == 0.0
    assert metrics.ssim < 0.01
