"""Analyze DNA Transform v2 Step 4 utility smoke."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V2_METHODS = ["v2_ratio0p95_eta0p01", "v2_ratio0p9_eta0p01"]


def summarize(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/dna_transform_v2/step4_utility_smoke_20260916")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/dna_transform_v2/step4_utility_smoke_20260916")
    args = parser.parse_args()

    registry: dict[int, dict[str, dict[str, float]]] = {}
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(args.run_dir.glob("seed_*/**/metrics.json")):
        seed = int(metrics_path.parts[-3].split("_")[1])
        method = metrics_path.parent.name
        payload = json.loads(metrics_path.read_text())
        final = payload["rounds"][-1]
        config = payload["config"]
        if int(config["seed"]) != seed:
            raise ValueError(f"seed mismatch in {metrics_path}")
        if int(final["round"]) != 10:
            raise ValueError(f"expected 10 rounds in {metrics_path}")
        registry.setdefault(seed, {})[method] = {
            "f1": float(final["f1_score"]),
            "auc": float(final["auc_roc"]),
            "pr_auc": float(final["pr_auc"]),
            "recall": float(final["recall"]),
            "precision": float(final["precision"]),
            "transform_ms": float(final.get("dna_encode_decode_ms", 0.0)),
            "aggregate_relative_l2": float(final.get("dna_transform_v2_aggregate_relative_l2_error", 0.0)),
        }

    expected = {"baseline", *V2_METHODS}
    if not registry or any(set(methods) != expected for methods in registry.values()):
        raise RuntimeError(f"incomplete paired registry; expected {expected}")

    for seed in sorted(registry):
        baseline = registry[seed]["baseline"]
        for method in ["baseline", *V2_METHODS]:
            value = registry[seed][method]
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "f1": value["f1"],
                    "auc": value["auc"],
                    "pr_auc": value["pr_auc"],
                    "precision": value["precision"],
                    "recall": value["recall"],
                    "delta_f1_vs_baseline": value["f1"] - baseline["f1"],
                    "delta_auc_vs_baseline": value["auc"] - baseline["auc"],
                    "delta_pr_auc_vs_baseline": value["pr_auc"] - baseline["pr_auc"],
                    "transform_ms_final_round": value["transform_ms"],
                    "aggregate_relative_l2_final_round": value["aggregate_relative_l2"],
                }
            )

    analyses = {}
    for method in V2_METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        analyses[method] = {
            "f1_delta": summarize([float(row["delta_f1_vs_baseline"]) for row in method_rows]),
            "auc_delta": summarize([float(row["delta_auc_vs_baseline"]) for row in method_rows]),
            "pr_auc_delta": summarize([float(row["delta_pr_auc_vs_baseline"]) for row in method_rows]),
            "final_round_transform_ms": summarize([float(row["transform_ms_final_round"]) for row in method_rows]),
            "final_round_aggregate_relative_l2": summarize([float(row["aggregate_relative_l2_final_round"]) for row in method_rows]),
        }

    # Conservative development rule: proceed to security only if the less lossy
    # v2 config does not show an obvious utility collapse on both primary
    # metrics.  This is a smoke gate, not a confirmatory non-inferiority test.
    primary = analyses["v2_ratio0p95_eta0p01"]
    f1_mean_delta = float(primary["f1_delta"]["mean"])
    auc_mean_delta = float(primary["auc_delta"]["mean"])
    gate_pass = f1_mean_delta >= -0.05 and auc_mean_delta >= -0.01
    conclusion = (
        "UTILITY_SMOKE_PASS_FOR_STEP5_DEVELOPMENT_CALIBRATION"
        if gate_pass
        else "UTILITY_COLLAPSE_OR_MATERIAL_DEGRADATION_STOP_BEFORE_STEP5"
    )

    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "DNA Transform v2 Step 4 development-scale utility smoke; not confirmatory",
        "n_seeds": len(registry),
        "analysis_round": 10,
        "methods": ["baseline", *V2_METHODS],
        "smoke_gate_rule": "primary v2_ratio0p95_eta0p01 proceeds only if mean delta F1 >= -0.05 and mean delta AUC >= -0.01",
        "analyses": analyses,
        "gate_pass": gate_pass,
        "conclusion": conclusion,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "paired_utility_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "utility_smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
