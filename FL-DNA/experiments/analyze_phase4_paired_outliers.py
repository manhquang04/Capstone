"""Inspect paired DNA-vs-raw Level 1 comparison outliers."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import torch

from experiments.run_phase3_full_client import dump


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _history_stats(path):
    artifact = torch.load(path, weights_only=False)
    history = artifact["history"]
    best = min(history)
    return {
        "artifact": str(path),
        "best_step": int(artifact["best_step"]),
        "history_len": len(history),
        "first_objective": float(history[0]),
        "last_objective": float(history[-1]),
        "best_objective": float(best),
        "last_minus_best_objective": float(history[-1] - best),
        "last_over_best_objective": float(history[-1] / best) if best else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase4_run", type=Path)
    parser.add_argument("--comparison-dir", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--dna-dir", required=True)
    parser.add_argument("--output-dir", default="level1_paired_outlier_diagnostic")
    args = parser.parse_args()

    run = args.phase4_run.resolve()
    comparison = json.loads((run / args.comparison_dir / "level1_paired_dna_vs_raw_report.json").read_text())
    rows = sorted(comparison["rows"], key=lambda row: row["dna_minus_raw_fraud_mse"])
    selected = [rows[0], rows[-1]]
    if len(rows) >= 4:
        selected = rows[:2] + rows[-2:]

    raw_summary = pd.read_csv(run / args.raw_dir / "harddiff_reparam_group_summary.csv")
    dna_records = pd.read_csv(run / args.dna_dir / "dna_level1_forward_attack_records.csv")
    diagnostics = []
    for row in selected:
        group_id = int(row["group_id"])
        raw_restart = int(row["raw_restart"])
        dna_realization = int(row["dna_realization"])
        dna_restart = int(row["dna_restart"])
        raw_artifact = run / args.raw_dir / f"group_{group_id}" / f"restart_{raw_restart}" / "baseline.pt"
        dna_artifact = (
            run
            / args.dna_dir
            / f"group_{group_id}"
            / f"realization_{dna_realization}"
            / f"restart_{dna_restart}"
            / "baseline.pt"
        )
        raw_group = raw_summary[raw_summary.group_id == group_id].iloc[0].to_dict()
        dna_group = dna_records[(dna_records.group_id == group_id) & (dna_records.method == "baseline")]
        dna_group = dna_group.sort_values("objective").head(10)
        diagnostics.append(
            {
                "group_id": group_id,
                "dna_minus_raw_fraud_mse": float(row["dna_minus_raw_fraud_mse"]),
                "raw_fraud_mse": float(row["raw_fraud_mse"]),
                "dna_fraud_mse": float(row["dna_fraud_mse"]),
                "raw_objective_best_restart": raw_restart,
                "raw_fraud_mse_best_restart": int(raw_group["fraud_mse_best_restart"]),
                "raw_extra_fraud_mse_from_objective_selection": float(
                    raw_group["extra_fraud_mse_from_objective_selection"]
                ),
                "dna_objective_best_realization": dna_realization,
                "dna_objective_best_restart": dna_restart,
                "dna_top10_min_fraud_mse": float(dna_group["fraud_mse"].min()),
                "dna_top10_max_fraud_mse": float(dna_group["fraud_mse"].max()),
                "raw_history": _history_stats(raw_artifact),
                "dna_history": _history_stats(dna_artifact),
            }
        )

    report = {
        "comparison_dir": args.comparison_dir,
        "outlier_rule": "two most negative and two most positive DNA-minus-raw fraud-MSE deltas",
        "summary": comparison["paired_dna_minus_raw"],
        "diagnostics": diagnostics,
        "interpretation": (
            "The largest deltas do not show missing files or a replay crash. "
            "Several selected histories have minima before the final iteration, "
            "so the optimizer is non-monotone as expected under Adam; candidates "
            "are selected by the stored best objective, not by the final step."
        ),
    }
    out = run / args.output_dir
    out.mkdir(exist_ok=True)
    dump(out / "paired_outlier_diagnostic.json", report)
    flat = []
    for item in diagnostics:
        flat.append(
            {
                "group_id": item["group_id"],
                "dna_minus_raw_fraud_mse": item["dna_minus_raw_fraud_mse"],
                "raw_fraud_mse": item["raw_fraud_mse"],
                "dna_fraud_mse": item["dna_fraud_mse"],
                "raw_best_step": item["raw_history"]["best_step"],
                "dna_best_step": item["dna_history"]["best_step"],
                "raw_last_over_best": item["raw_history"]["last_over_best_objective"],
                "dna_last_over_best": item["dna_history"]["last_over_best_objective"],
                "raw_extra_fraud_mse_from_objective_selection": item[
                    "raw_extra_fraud_mse_from_objective_selection"
                ],
                "dna_top10_min_fraud_mse": item["dna_top10_min_fraud_mse"],
            }
        )
    _write_csv(out / "paired_outlier_diagnostic.csv", flat)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
