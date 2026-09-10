"""Compare paired Level 1 DNA-transformed and raw-update attacks."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from experiments.run_phase3_full_client import dump


def _sign_tail(wins, total):
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / 2**total if total else 1.0


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _by_group(report):
    return {int(row["group_id"]): row for row in report["group_summary"]}


def _summary(values):
    values = np.asarray(values, dtype=float)
    non_ties = values[values != 0]
    wins = int((non_ties > 0).sum())
    return {
        "n": int(len(values)),
        "non_ties": int(len(non_ties)),
        "dna_harder_wins": wins,
        "mean_dna_minus_raw_fraud_mse": float(values.mean()),
        "median_dna_minus_raw_fraud_mse": float(np.median(values)),
        "one_sided_sign_p_dna_harder": _sign_tail(wins, len(non_ties)),
        "dna_harder_gate": bool(values.mean() > 0 and np.median(values) > 0 and _sign_tail(wins, len(non_ties)) < 0.05),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase4_run", type=Path)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--dna-dir", required=True)
    parser.add_argument("--output-dir", default="level1_paired_dna_vs_raw")
    args = parser.parse_args()

    run = args.phase4_run.resolve()
    raw_dir = run / args.raw_dir
    dna_dir = run / args.dna_dir
    raw_report = json.loads((raw_dir / "harddiff_reparam_report.json").read_text())
    dna_report = json.loads((dna_dir / "dna_level1_forward_attack_report.json").read_text())
    raw_groups = _by_group(raw_report)
    dna_groups = _by_group(dna_report)
    group_ids = sorted(set(raw_groups) & set(dna_groups))
    if group_ids != sorted(raw_groups) or group_ids != sorted(dna_groups):
        raise AssertionError("Raw and DNA reports do not contain identical group ids")
    if raw_report.get("target_file") != dna_report.get("target_file"):
        raise AssertionError("Raw and DNA reports do not use the same target file")

    rows = []
    deltas = []
    for group_id in group_ids:
        raw = raw_groups[group_id]
        dna = dna_groups[group_id]
        raw_mse = float(raw["objective_best_fraud_mse"])
        dna_mse = float(dna["objective_best_fraud_mse"])
        delta = dna_mse - raw_mse
        deltas.append(delta)
        rows.append(
            {
                "group_id": group_id,
                "raw_fraud_mse": raw_mse,
                "dna_fraud_mse": dna_mse,
                "dna_minus_raw_fraud_mse": delta,
                "dna_harder": delta > 0,
                "raw_restart": int(raw["objective_best_restart"]),
                "dna_realization": int(dna["objective_best_realization_id"]),
                "dna_restart": int(dna["objective_best_restart"]),
                "raw_prior_fraud_mse_mean": float(raw["prior_fraud_mse_mean"]),
                "dna_prior_fraud_mse_mean": float(dna["prior_fraud_mse_mean"]),
                "raw_zero_fraud_mse": float(raw["zero_objective_best_fraud_mse"]),
                "dna_zero_fraud_mse": float(dna["zero_objective_best_fraud_mse"]),
            }
        )

    report = {
        "target_file": raw_report["target_file"],
        "raw_dir": str(raw_dir),
        "dna_dir": str(dna_dir),
        "comparison": "paired objective-selected fraud reconstruction MSE; positive delta means DNA is harder to invert than raw",
        "raw_gate": raw_report["target_gate"]["objective_selection"],
        "dna_gate": dna_report["target_gate"],
        "paired_dna_minus_raw": _summary(deltas),
        "rows": rows,
    }
    out = run / args.output_dir
    out.mkdir(exist_ok=True)
    dump(out / "level1_paired_dna_vs_raw_report.json", report)
    _write_csv(out / "level1_paired_dna_vs_raw.csv", rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
