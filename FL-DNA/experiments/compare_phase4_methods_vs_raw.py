"""Compare multiple Phase 4 defense reports against the paired raw baseline."""
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


def _load_report(path):
    for name in (
        "dna_level1_forward_attack_report.json",
        "simple_defense_attack_report.json",
        "harddiff_reparam_report.json",
    ):
        candidate = path / name
        if candidate.exists():
            return json.loads(candidate.read_text()), name
    raise FileNotFoundError(f"No known report JSON in {path}")


def _raw_groups(report):
    return {int(row["group_id"]): row for row in report["group_summary"]}


def _method_summary(raw_report, method_report):
    raw_groups = _raw_groups(raw_report)
    method_groups = _raw_groups(method_report)
    group_ids = sorted(set(raw_groups) & set(method_groups))
    if group_ids != sorted(raw_groups) or group_ids != sorted(method_groups):
        raise AssertionError("Raw and method reports do not contain identical group ids")
    if raw_report.get("target_file") != method_report.get("target_file"):
        raise AssertionError("Raw and method reports do not use the same target file")
    rows = []
    deltas = []
    for group_id in group_ids:
        raw = raw_groups[group_id]
        method = method_groups[group_id]
        raw_mse = float(raw["objective_best_fraud_mse"])
        method_mse = float(method["objective_best_fraud_mse"])
        delta = method_mse - raw_mse
        deltas.append(delta)
        rows.append(
            {
                "group_id": group_id,
                "raw_fraud_mse": raw_mse,
                "method_fraud_mse": method_mse,
                "method_minus_raw_fraud_mse": delta,
                "method_harder": delta > 0,
            }
        )
    values = np.asarray(deltas, dtype=float)
    non_ties = values[values != 0]
    wins = int((non_ties > 0).sum())
    return {
        "n": int(len(values)),
        "non_ties": int(len(non_ties)),
        "method_harder_wins": wins,
        "mean_method_minus_raw_fraud_mse": float(values.mean()),
        "median_method_minus_raw_fraud_mse": float(np.median(values)),
        "one_sided_sign_p_method_harder": _sign_tail(wins, len(non_ties)),
        "method_harder_gate": bool(values.mean() > 0 and np.median(values) > 0 and _sign_tail(wins, len(non_ties)) < 0.05),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase4_run", type=Path)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--method", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--output-dir", default="level1_methods_vs_raw")
    args = parser.parse_args()

    run = args.phase4_run.resolve()
    raw_report, _ = _load_report(run / args.raw_dir)
    summary_rows = []
    details = {}
    for name, directory in args.method:
        method_report, report_file = _load_report(run / directory)
        result = _method_summary(raw_report, method_report)
        details[name] = {
            "directory": str(run / directory),
            "report_file": report_file,
            "target_file": method_report["target_file"],
            "method_gate": method_report.get("target_gate"),
            "paired_vs_raw": {key: value for key, value in result.items() if key != "rows"},
        }
        summary_rows.append({"method": name, **details[name]["paired_vs_raw"]})
        for row in result["rows"]:
            row["method"] = name
        details[name]["rows"] = result["rows"]

    out = run / args.output_dir
    out.mkdir(exist_ok=True)
    report = {
        "raw_dir": str(run / args.raw_dir),
        "comparison": "paired objective-selected fraud reconstruction MSE; positive delta means method is harder to invert than raw",
        "methods": details,
    }
    dump(out / "level1_methods_vs_raw_report.json", report)
    _write_csv(out / "level1_methods_vs_raw_summary.csv", summary_rows)
    all_rows = [row for detail in details.values() for row in detail["rows"]]
    _write_csv(out / "level1_methods_vs_raw_rows.csv", all_rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
