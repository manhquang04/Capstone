"""Compare final-round Credit Card Fraud FL metrics."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "fraud"
BASELINE_PATH = RESULTS_DIR / "baseline_metrics.json"
DNA_PATH = RESULTS_DIR / "dna_metrics.json"
DP_PATH = RESULTS_DIR / "dp_metrics.json"
SUMMARY_PATH = RESULTS_DIR / "comparison_summary.json"
COMMENTARY = (
    "DNA preserves baseline performance because encode/decode is bit-exact. "
    "DP may reduce or change performance due to clipping and Gaussian noise."
)


def load_final_metrics(path: Path) -> dict[str, float | int | None]:
    """Load the final round from an experiment metrics file."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Metrics file not found: '{path}'. Run the corresponding experiment first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rounds = payload.get("rounds", [])
    if not rounds:
        raise ValueError(f"Metrics file has no round results: '{path}'")
    return rounds[-1]


def metric_text(value: float | int | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def comparison_metric(metrics: dict[str, float | int | None], name: str):
    """Read metrics from existing files and the compact DP metric format."""
    aliases = {
        "f1_score": "f1",
        "auc_roc": "auc",
    }
    return metrics.get(name, metrics.get(aliases.get(name, "")))


def main() -> None:
    baseline = load_final_metrics(BASELINE_PATH)
    dna = load_final_metrics(DNA_PATH)
    dp = load_final_metrics(DP_PATH)
    final_metrics = {
        "Baseline": baseline,
        "DNA": dna,
        "DP": dp,
    }
    summary = {
        "final_metrics": final_metrics,
        "commentary": COMMENTARY,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    header = (
        f"{'Method':<12} {'F1':>10} {'AUC':>10} {'Accuracy':>10} "
        f"{'Precision':>10} {'Recall':>10}"
    )
    print(header)
    print("-" * len(header))
    for method, metrics in final_metrics.items():
        print(
            f"{method:<12} {metric_text(comparison_metric(metrics, 'f1_score')):>10} "
            f"{metric_text(comparison_metric(metrics, 'auc_roc')):>10} "
            f"{metric_text(metrics['accuracy']):>10} "
            f"{metric_text(metrics['precision']):>10} "
            f"{metric_text(metrics['recall']):>10}"
        )
    print(f"\nCommentary: {COMMENTARY}")
    print(f"Saved comparison: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
