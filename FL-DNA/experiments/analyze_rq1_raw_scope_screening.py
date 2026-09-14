"""Analyze raw-only RQ1 scope-boundary coarse screening."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t


def _gate(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    wins = int((x < 0).sum())
    p_value = float(binomtest(wins, len(x), 0.5, alternative="greater").pvalue)
    return {
        "n": int(len(x)),
        "wins": wins,
        "mean_difference": float(x.mean()),
        "median_difference": float(np.median(x)),
        "one_sided_sign_p": p_value,
        "pass": bool(x.mean() < 0 and np.median(x) < 0 and p_value < 0.05),
    }


def _ci(values: list[float]) -> list[float]:
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return [float(x.mean()), float(x.mean())]
    half = float(t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
    mean = float(x.mean())
    return [mean - half, mean + half]


def _load_summary(run_dir: Path, group: int) -> dict:
    root = run_dir / "raw" / f"group_{group}_run"
    reports = list(root.glob("**/harddiff_reparam_report.json"))
    if len(reports) != 1:
        raise RuntimeError(f"expected one raw report for group {group}, found {len(reports)}")
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    if len(report["group_summary"]) != 1:
        raise RuntimeError(f"expected one summary row for group {group}")
    row = dict(report["group_summary"][0])
    row["group_id"] = group
    row["report_path"] = str(reports[0])
    row["restarts"] = int(report["restarts"])
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    groups = int(config["target"]["groups"])
    rows = [_load_summary(args.run_dir, group) for group in range(groups)]
    prior = [float(row["objective_minus_prior_fraud_mse"]) for row in rows]
    zero = [float(row["objective_minus_zero_fraud_mse"]) for row in rows]
    gates = {"prior": _gate(prior), "zero": _gate(zero)}
    branch_pass = bool(gates["prior"]["pass"] and gates["zero"]["pass"])
    summary = {
        "config": str(args.config),
        "run_dir": str(args.run_dir),
        "records_per_group": int(config["screening"]["records_per_group"]),
        "fraud_records_per_group": int(config["screening"]["fraud_records_per_group"]),
        "local_steps": int(config["screening"]["local_steps"]),
        "groups": groups,
        "branch": "raw",
        "control_gates": gates,
        "raw_branch_pass": branch_pass,
        "mean_objective_minus_prior_fraud_mse_ci95": _ci(prior),
        "mean_objective_minus_zero_fraud_mse_ci95": _ci(zero),
        "screening_decision": "PASS_CONTINUE_TO_NEXT_LEVEL" if branch_pass else "FAIL_STOP_SCREENING",
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "raw_group_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
