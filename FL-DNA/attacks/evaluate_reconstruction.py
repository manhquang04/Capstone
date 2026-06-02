"""Evaluate MNIST inversion output quality with MSE, PSNR, and SSIM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "mnist"
RECONSTRUCTION_DIR = RESULTS_DIR / "reconstructions"
ORIGINAL_PATH = RECONSTRUCTION_DIR / "original.png"
METRICS_PATH = RESULTS_DIR / "reconstruction_metrics.json"
METHODS = ("RAW", "DNA", "DP")


def load_grayscale(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Image artifact not found: '{path}'")
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float64) / 255.0


def calculate_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute grayscale SSIM using the standard global-image formula."""
    c1 = 0.01**2
    c2 = 0.03**2
    original_mean = float(np.mean(original))
    reconstructed_mean = float(np.mean(reconstructed))
    original_variance = float(np.var(original))
    reconstructed_variance = float(np.var(reconstructed))
    covariance = float(np.mean(
        (original - original_mean) * (reconstructed - reconstructed_mean)
    ))
    numerator = (
        (2 * original_mean * reconstructed_mean + c1)
        * (2 * covariance + c2)
    )
    denominator = (
        (original_mean**2 + reconstructed_mean**2 + c1)
        * (original_variance + reconstructed_variance + c2)
    )
    return numerator / denominator


def main() -> None:
    original = load_grayscale(ORIGINAL_PATH)
    metrics = {}
    for method in METHODS:
        reconstructed_path = RECONSTRUCTION_DIR / f"reconstructed_{method.lower()}.png"
        reconstructed = load_grayscale(reconstructed_path)
        if reconstructed.shape != original.shape:
            raise ValueError(f"Image shape mismatch for {method}")

        mse = float(np.mean((original - reconstructed) ** 2))
        psnr = float("inf") if mse == 0.0 else float(10 * np.log10(1.0 / mse))
        metrics[method] = {
            "mse": mse,
            "psnr": psnr,
            "ssim": float(calculate_ssim(original, reconstructed)),
        }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {METRICS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
