"""Compare final-round PaySim metrics across FL-DNA-DP configurations."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "fraud"
CENTRALIZED_PATH = RESULTS_DIR / "centralized_metrics.json"
BASELINE_PATH = RESULTS_DIR / "baseline_metrics.json"
DNA_PATH = RESULTS_DIR / "dna_metrics.json"
DP_PATH = RESULTS_DIR / "dp_metrics.json"
DNA_DP_PATH = RESULTS_DIR / "dna_dp_metrics.json"
SUMMARY_PATH = RESULTS_DIR / "comparison_summary.json"
COMMENTARY = (
    "Centralized is the pooled-data upper bound. FL Baseline uses 3 clients, "
    "15 communication rounds, one local epoch, and FedAvg. FL+DNA only adds "
    "DNA encode/decode for local model updates. FL+DP only adds client-update "
    "clipping and Gaussian noise. FL+DNA+DP applies DP first, then DNA transports "
    "the DP-protected update."
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
    centralized = load_final_metrics(CENTRALIZED_PATH)
    baseline = load_final_metrics(BASELINE_PATH)
    dna = load_final_metrics(DNA_PATH)

    final_metrics = {
        "Centralized": centralized,
        "FL_Baseline": baseline,
        "FL_DNA": dna,
    }
    if DP_PATH.is_file():
        final_metrics["FL_DP"] = load_final_metrics(DP_PATH)
    if DNA_DP_PATH.is_file():
        final_metrics["FL_DNA_DP"] = load_final_metrics(DNA_DP_PATH)
    summary = {
        "final_metrics": final_metrics,
        "commentary": COMMENTARY,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    header = (
        f"{'Method':<12} {'Loss':>10} {'F1':>10} {'ROC-AUC':>10} "
        f"{'PR-AUC':>10} {'Precision':>10} {'Recall':>10} {'TP':>8} {'FP':>8} {'FN':>8}"
    )
    print(header)
    print("-" * len(header))
    for method, metrics in final_metrics.items():
        print(
            f"{method:<12} {metric_text(metrics.get('train_loss')):>10} "
            f"{metric_text(comparison_metric(metrics, 'f1_score')):>10} "
            f"{metric_text(comparison_metric(metrics, 'auc_roc')):>10} "
            f"{metric_text(metrics.get('pr_auc')):>10} "
            f"{metric_text(metrics['precision']):>10} "
            f"{metric_text(metrics['recall']):>10} "
            f"{metrics.get('tp', 'N/A'):>8} {metrics.get('fp', 'N/A'):>8} {metrics.get('fn', 'N/A'):>8}"
        )
    print(f"\nCommentary: {COMMENTARY}")
    print(f"Saved comparison: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
