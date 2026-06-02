"""Summarize MNIST gradient inversion reconstruction metrics."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "mnist"
METRICS_PATH = RESULTS_DIR / "reconstruction_metrics.json"
SUMMARY_PATH = RESULTS_DIR / "privacy_summary.json"
COMMENTARY = [
    "Lower PSNR and SSIM indicate stronger protection against reconstruction attacks.",
    "DNA maintains model utility while introducing an additional transformation layer.",
    "DP reduces reconstruction quality through clipping and Gaussian noise.",
]


def main() -> None:
    if not METRICS_PATH.is_file():
        raise FileNotFoundError(
            f"Reconstruction metrics not found at '{METRICS_PATH}'. Run "
            "'python attacks/evaluate_reconstruction.py' first."
        )
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    missing = [method for method in ("RAW", "DNA", "DP") if method not in metrics]
    if missing:
        raise ValueError(f"Reconstruction metrics missing methods: {missing}")

    summary = {"metrics": metrics, "commentary": COMMENTARY}
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    header = f"{'Method':<8} {'PSNR':>12} {'SSIM':>12} {'MSE':>12}"
    print(header)
    print("-" * len(header))
    for method in ("RAW", "DNA", "DP"):
        values = metrics[method]
        print(
            f"{method:<8} {values['psnr']:>12.6f} "
            f"{values['ssim']:>12.6f} {values['mse']:>12.6f}"
        )
    print("\n" + " ".join(COMMENTARY))
    print(f"Saved summary: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
