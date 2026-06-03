"""Pseudo-image conversion for tabular PaySim inversion metrics."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def vector_to_pseudo_image(vector: np.ndarray) -> np.ndarray:
    """
    Convert a tabular feature vector to a fixed 2D pseudo-image.

    PaySim is tabular, so this image is only a consistent visualization and
    enables PSNR/SSIM reporting. Feature-wise metrics remain the primary
    tabular reconstruction metrics.
    """
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    side = int(np.ceil(np.sqrt(flat.size)))
    padded = np.zeros(side * side, dtype=np.float32)
    padded[: flat.size] = flat
    return padded.reshape(side, side)


def minmax_image(image: np.ndarray) -> np.ndarray:
    """Scale an image-like array to [0, 1] for image metrics and PNG output."""
    image = np.asarray(image, dtype=np.float32)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value - min_value < 1e-12:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_value) / (max_value - min_value)


def minmax_pair(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale two arrays with the same reference range for fair image metrics."""
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    min_value = float(np.min(reference))
    max_value = float(np.max(reference))
    if max_value - min_value < 1e-12:
        return np.zeros_like(reference, dtype=np.float32), np.zeros_like(candidate, dtype=np.float32)
    return (
        np.clip((reference - min_value) / (max_value - min_value), 0.0, 1.0),
        np.clip((candidate - min_value) / (max_value - min_value), 0.0, 1.0),
    )


def save_pseudo_image(path: Path, vector: np.ndarray) -> None:
    """Save a tabular vector as a small grayscale pseudo-image."""
    cache_dir = path.parent / ".mplconfig"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    image = minmax_image(vector_to_pseudo_image(vector))
    plt.figure(figsize=(2.2, 2.2))
    plt.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close()
