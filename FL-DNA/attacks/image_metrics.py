"""Image-domain reconstruction metrics for true image tensors.

This module is intentionally separate from ``attacks/pseudo_image.py`` because
that file builds pseudo-images from tabular PaySim features. CIFAR-10 targets
are real RGB images, so PSNR/SSIM are computed directly on 3x32x32 image
tensors after normalizing layout and clipping to the public [0, 1] range.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isinf
from typing import Any

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_hwc_unit_image(image: Any) -> np.ndarray:
    """Return a float64 HWC image clipped to [0, 1].

    Accepts CHW or HWC arrays/tensors. The function is deliberately small and
    auditable because it is the adapter layer being verified before image-domain
    development gates.
    """

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D image tensor, got shape {arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] not in (1, 3):
        raise ValueError(f"expected final channel dimension 1 or 3, got shape {arr.shape}")
    return np.clip(arr, 0.0, 1.0)


@dataclass(frozen=True)
class ImageReconstructionMetrics:
    mse: float
    psnr: float
    ssim: float


def image_mse(reference: Any, reconstruction: Any) -> float:
    ref = _to_hwc_unit_image(reference)
    rec = _to_hwc_unit_image(reconstruction)
    if ref.shape != rec.shape:
        raise ValueError(f"shape mismatch: reference {ref.shape} vs reconstruction {rec.shape}")
    return float(np.mean((ref - rec) ** 2))


def image_psnr(reference: Any, reconstruction: Any) -> float:
    ref = _to_hwc_unit_image(reference)
    rec = _to_hwc_unit_image(reconstruction)
    if ref.shape != rec.shape:
        raise ValueError(f"shape mismatch: reference {ref.shape} vs reconstruction {rec.shape}")
    value = float(peak_signal_noise_ratio(ref, rec, data_range=1.0))
    return value


def image_ssim(reference: Any, reconstruction: Any) -> float:
    ref = _to_hwc_unit_image(reference)
    rec = _to_hwc_unit_image(reconstruction)
    if ref.shape != rec.shape:
        raise ValueError(f"shape mismatch: reference {ref.shape} vs reconstruction {rec.shape}")
    value = float(structural_similarity(ref, rec, data_range=1.0, channel_axis=-1))
    return value


def compute_image_metrics(reference: Any, reconstruction: Any) -> ImageReconstructionMetrics:
    psnr = image_psnr(reference, reconstruction)
    ssim = image_ssim(reference, reconstruction)
    mse = image_mse(reference, reconstruction)
    if mse == 0.0 and not isinf(psnr):
        raise AssertionError("PSNR should be infinite when MSE is exactly zero")
    return ImageReconstructionMetrics(mse=mse, psnr=psnr, ssim=ssim)

