"""Apply preliminary DP clipping and Gaussian noise to an MNIST gradient."""

from __future__ import annotations

import json
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "mnist"
RAW_GRADIENT_PATH = RESULTS_DIR / "raw_gradient.pt"
DP_GRADIENT_PATH = RESULTS_DIR / "dp_gradient.pt"
METRICS_PATH = RESULTS_DIR / "dp_gradient_metrics.json"
CLIP_NORM = 1.0
NOISE_MULTIPLIER = 0.05
SEED = 42


def main() -> None:
    if not RAW_GRADIENT_PATH.is_file():
        raise FileNotFoundError(
            f"Raw gradient not found at '{RAW_GRADIENT_PATH}'. Run "
            "'python attacks/gradient_extraction.py' first."
        )

    torch.manual_seed(SEED)
    raw_bundle = torch.load(RAW_GRADIENT_PATH, map_location="cpu", weights_only=True)
    gradients = raw_bundle["gradients"]
    gradient_norm = torch.sqrt(sum(torch.sum(gradient.square()) for gradient in gradients))
    gradient_norm_value = float(gradient_norm.item())
    clip_factor = min(1.0, CLIP_NORM / max(gradient_norm_value, 1e-12))
    noise_std = CLIP_NORM * NOISE_MULTIPLIER
    dp_gradients = [
        gradient * clip_factor + torch.randn_like(gradient) * noise_std
        for gradient in gradients
    ]
    metrics = {
        "clip_norm": CLIP_NORM,
        "noise_multiplier": NOISE_MULTIPLIER,
        "noise_std": noise_std,
        "gradient_norm_before_clip": gradient_norm_value,
        "gradient_norm_after_clip": gradient_norm_value * clip_factor,
    }
    dp_bundle = {
        **raw_bundle,
        "gradients": dp_gradients,
        "source": "dp",
        "dp_config": metrics,
    }
    torch.save(dp_bundle, DP_GRADIENT_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved gradient: {DP_GRADIENT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
