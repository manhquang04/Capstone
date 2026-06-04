"""Compare PaySim metrics across FL-DNA privacy configurations."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "fraud"
CENTRALIZED_PATH = RESULTS_DIR / "centralized_metrics.json"
BASELINE_PATH = RESULTS_DIR / "baseline_metrics.json"
DNA_PATH = RESULTS_DIR / "dna_metrics.json"
DP_PATH = RESULTS_DIR / "dp_metrics.json"
DP_UTILITY_PATH = RESULTS_DIR / "dp_utility_metrics.json"
DP_MILD_PATH = RESULTS_DIR / "dp_mild_metrics.json"
DNA_DP_PATH = RESULTS_DIR / "dna_dp_metrics.json"
SECUREAGG_PATH = RESULTS_DIR / "secureagg_metrics.json"
DNA_SECUREAGG_PATH = RESULTS_DIR / "dna_secureagg_metrics.json"
DNA_TRANSFORM_PATH = RESULTS_DIR / "dna_transform_metrics.json"
DNA_TRANSFORM_CONSERVATIVE_PATH = RESULTS_DIR / "dna_transform_conservative_metrics.json"
DNA_TRANSFORM_MEDIUM_PATH = RESULTS_DIR / "dna_transform_medium_metrics.json"
DNA_TRANSFORM_STRONGER_PATH = RESULTS_DIR / "dna_transform_stronger_metrics.json"
DNA_TRANSFORM_SECUREAGG_PATH = RESULTS_DIR / "dna_transform_secureagg_metrics.json"
SUMMARY_PATH = RESULTS_DIR / "comparison_summary.json"
COMMENTARY = (
    "Centralized is the pooled-data upper bound. FL Baseline uses 3 clients, "
    "configurable communication rounds, one local epoch, and FedAvg. FL+DNA only adds "
    "DNA encode/decode for local model updates. FL+DP only adds client-update "
    "clipping and Gaussian noise; no privacy accountant is implemented, so no formal "
    "epsilon-delta DP claim is made. FL+DNA+DP applies DP first, then DNA transports "
    "the DP-protected update. Secure Aggregation uses pairwise masks so the server "
    "only observes the aggregate update, not individual client updates. Lossless DNA "
    "after SecureAgg has no additional privacy value beyond representation transport. "
    "DNA Transform Defense makes DNA the core update transformation via sequence-seeded "
    "block mixing and selective attenuation."
)
RECOMMENDED_TRANSFORM = "conservative"


def load_payload(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Metrics file not found: '{path}'. Run the corresponding experiment first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_final_metrics(path: Path) -> dict[str, float | int | None]:
    """Load the final round from an experiment metrics file."""
    payload = load_payload(path)
    rounds = payload.get("rounds", [])
    if not rounds:
        raise ValueError(f"Metrics file has no round results: '{path}'")
    return rounds[-1]


def load_entry(method: str, path: Path) -> dict[str, object]:
    payload = load_payload(path)
    rounds = payload.get("rounds", [])
    if not rounds:
        raise ValueError(f"Metrics file has no round results: '{path}'")
    config = payload.get("config", {})
    final = rounds[-1]
    return {
        "method": method,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "metrics": final,
        "configured_rounds": config.get("num_rounds"),
        "observed_rounds": len(rounds),
        "final_round": final.get("round"),
        "config": config,
    }


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
    entries = [
        load_entry("Centralized", CENTRALIZED_PATH),
        load_entry("FL_Baseline", BASELINE_PATH),
        load_entry("FL_DNA", DNA_PATH),
    ]
    optional_paths = [
        ("FL_DP_utility_noise_defense", DP_UTILITY_PATH),
        ("FL_DP_mild_noise_defense", DP_MILD_PATH),
        ("FL_DP_medium_noise_defense", DP_PATH),
        ("FL_DNA_DP_noise_defense", DNA_DP_PATH),
        ("FL_SecureAgg", SECUREAGG_PATH),
        ("FL_DNA_SecureAgg_lossless_transport", DNA_SECUREAGG_PATH),
        ("FL_DNA_TransformDefense_current", DNA_TRANSFORM_PATH),
        ("FL_DNA_TransformDefense_conservative_recommended", DNA_TRANSFORM_CONSERVATIVE_PATH),
        ("FL_DNA_TransformDefense_medium", DNA_TRANSFORM_MEDIUM_PATH),
        ("FL_DNA_TransformDefense_stronger", DNA_TRANSFORM_STRONGER_PATH),
        ("FL_DNA_TransformDefense_SecureAgg", DNA_TRANSFORM_SECUREAGG_PATH),
    ]
    for method, path in optional_paths:
        if path.is_file():
            entries.append(load_entry(method, path))

    target_rounds = _target_rounds(entries)
    comparable_entries = [
        entry for entry in entries if entry["observed_rounds"] == target_rounds
    ]
    excluded_entries = [
        entry for entry in entries if entry["observed_rounds"] != target_rounds
    ]
    final_metrics = {
        str(entry["method"]): entry["metrics"] for entry in comparable_entries
    }
    transform_tradeoff = [
        entry
        for entry in entries
        if str(entry["method"]).startswith("FL_DNA_TransformDefense")
    ]
    secureagg_note = _secureagg_note(entries)
    summary = {
        "target_rounds": target_rounds,
        "main_comparison": final_metrics,
        "excluded_due_round_mismatch": excluded_entries,
        "transform_tradeoff_candidates": transform_tradeoff,
        "recommended_transform": RECOMMENDED_TRANSFORM,
        "secureagg_note": secureagg_note,
        "commentary": COMMENTARY,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    method_width = max(12, max(len(method) for method in final_metrics))
    header = (
        f"{'Method':<{method_width}} {'Loss':>10} {'F1':>10} {'ROC-AUC':>10} "
        f"{'PR-AUC':>10} {'Precision':>10} {'Recall':>10} {'TP':>8} {'FP':>8} {'FN':>8}"
    )
    print(header)
    print("-" * len(header))
    for method, metrics in final_metrics.items():
        print(
            f"{method:<{method_width}} {metric_text(metrics.get('train_loss')):>10} "
            f"{metric_text(comparison_metric(metrics, 'f1_score')):>10} "
            f"{metric_text(comparison_metric(metrics, 'auc_roc')):>10} "
            f"{metric_text(metrics.get('pr_auc')):>10} "
            f"{metric_text(metrics['precision']):>10} "
            f"{metric_text(metrics['recall']):>10} "
            f"{metrics.get('tp', 'N/A'):>8} {metrics.get('fp', 'N/A'):>8} {metrics.get('fn', 'N/A'):>8}"
        )
    print(f"\nCommentary: {COMMENTARY}")
    if excluded_entries:
        print("\nExcluded from main table because round count differs:")
        for entry in excluded_entries:
            print(
                f"- {entry['method']}: observed_rounds={entry['observed_rounds']} "
                f"path={entry['path']}"
            )
    print(f"\nSecureAgg note: {secureagg_note}")
    print(f"Saved comparison: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


def _target_rounds(entries: list[dict[str, object]]) -> int:
    override = None
    # Keep imports local to avoid global env coupling in tests/imports.
    import os

    if os.environ.get("COMPARE_NUM_ROUNDS"):
        override = int(os.environ["COMPARE_NUM_ROUNDS"])
    if override is not None:
        return override
    counts: dict[int, int] = {}
    for entry in entries:
        observed = int(entry["observed_rounds"])
        counts[observed] = counts.get(observed, 0) + 1
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _secureagg_note(entries: list[dict[str, object]]) -> str:
    by_method = {str(entry["method"]): entry for entry in entries}
    secure = by_method.get("FL_SecureAgg")
    dna_secure = by_method.get("FL_DNA_SecureAgg_lossless_transport")
    if not secure or not dna_secure:
        return "SecureAgg/DNA+SecureAgg identity check unavailable because one result is missing."
    core_metrics = [
        "train_loss",
        "f1_score",
        "auc_roc",
        "pr_auc",
        "precision",
        "recall",
        "tn",
        "fp",
        "fn",
        "tp",
        "optimal_threshold",
    ]
    secure_core = {key: secure["metrics"].get(key) for key in core_metrics}
    dna_secure_core = {key: dna_secure["metrics"].get(key) for key in core_metrics}
    if secure_core == dna_secure_core:
        return (
            "FL_DNA_SecureAgg has the same core utility metrics as FL_SecureAgg in "
            "the saved results; treat it as lossless transport/overhead analysis, "
            "not a new privacy defense."
        )
    return (
        "FL_DNA_SecureAgg differs numerically from FL_SecureAgg in saved metrics; "
        "inspect randomness and transport overhead before making privacy claims."
    )


if __name__ == "__main__":
    main()
