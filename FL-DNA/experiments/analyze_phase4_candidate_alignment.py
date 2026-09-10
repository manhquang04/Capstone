"""Diagnose whether attack objective selects good reconstructions.

This script reads already generated Phase 4 gate artifacts.  It does not rerun
attacks and does not change any gate.  The goal is to check whether lower
gradient-matching objective correlates with lower reconstruction error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def _load_json(path: Path):
    return json.loads(path.read_text())


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2))


def _write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return None
    return _pearson(_rank(x), _rank(y))


def _sign_tail(wins, total):
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / 2**total if total else 1.0


def _flatten(rows, split):
    flat = []
    for row in rows:
        metrics = row["metrics"]
        classes = metrics["classes"]
        flat.append(
            {
                "split": split,
                "method": row["method"],
                "group_id": row["group_id"],
                "restart": row["restart"],
                "objective": row["objective"],
                "overall_mse": metrics["mean_mse"],
                "fraud_mse": classes["1"]["mean_mse"],
                "non_fraud_mse": classes["0"]["mean_mse"],
                "prior_fraud_mse": row["prior"]["classes"]["1"]["mean_mse"],
                "prior_non_fraud_mse": row["prior"]["classes"]["0"]["mean_mse"],
                "prior_overall_mse": row["prior"]["mean_mse"],
            }
        )
    return flat


def _correlation_summary(flat, split):
    rows = []
    for method in sorted({row["method"] for row in flat}):
        method_rows = [row for row in flat if row["method"] == method]
        objective = [row["objective"] for row in method_rows]
        for metric in ("overall_mse", "fraud_mse", "non_fraud_mse"):
            metric_values = [row[metric] for row in method_rows]
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "metric": metric,
                    "n_candidates": len(method_rows),
                    "pearson_objective_vs_metric": _pearson(objective, metric_values),
                    "spearman_objective_vs_metric": _spearman(objective, metric_values),
                    "objective_direction": "lower objective should align with lower reconstruction MSE",
                }
            )
    return rows


def _best(rows, metric):
    return min(rows, key=lambda row: row[metric])


def _selection_diagnostics(flat, split):
    rows = []
    baseline = [row for row in flat if row["method"] == "baseline"]
    zero = [row for row in flat if row["method"] == "zero_update"]
    for group_id in sorted({row["group_id"] for row in baseline}):
        group_base = [row for row in baseline if row["group_id"] == group_id]
        group_zero = [row for row in zero if row["group_id"] == group_id]
        objective_best = _best(group_base, "objective")
        fraud_best = _best(group_base, "fraud_mse")
        overall_best = _best(group_base, "overall_mse")
        zero_objective_best = _best(group_zero, "objective")
        rows.append(
            {
                "split": split,
                "group_id": group_id,
                "objective_best_restart": objective_best["restart"],
                "fraud_mse_best_restart": fraud_best["restart"],
                "overall_mse_best_restart": overall_best["restart"],
                "zero_objective_best_restart": zero_objective_best["restart"],
                "objective_best_fraud_mse": objective_best["fraud_mse"],
                "fraud_oracle_best_fraud_mse": fraud_best["fraud_mse"],
                "objective_best_overall_mse": objective_best["overall_mse"],
                "overall_oracle_best_overall_mse": overall_best["overall_mse"],
                "prior_fraud_mse_mean": float(np.mean([row["prior_fraud_mse"] for row in group_base])),
                "zero_best_fraud_mse": zero_objective_best["fraud_mse"],
                "objective_minus_fraud_oracle_mse": objective_best["fraud_mse"] - fraud_best["fraud_mse"],
                "objective_minus_prior_fraud_mse": objective_best["fraud_mse"]
                - float(np.mean([row["prior_fraud_mse"] for row in group_base])),
                "objective_minus_zero_fraud_mse": objective_best["fraud_mse"] - zero_objective_best["fraud_mse"],
                "fraud_oracle_minus_prior_fraud_mse": fraud_best["fraud_mse"]
                - float(np.mean([row["prior_fraud_mse"] for row in group_base])),
                "fraud_oracle_minus_zero_fraud_mse": fraud_best["fraud_mse"] - zero_objective_best["fraud_mse"],
            }
        )
    return rows


def _gate(rows, prefix):
    output = {}
    for control in ("prior", "zero"):
        values = np.asarray([row[f"{prefix}_minus_{control}_fraud_mse"] for row in rows], dtype=float)
        non_ties = values[values != 0]
        wins = int((non_ties < 0).sum())
        output[control] = {
            "n": len(rows),
            "wins": wins,
            "mean_difference": float(values.mean()),
            "median_difference": float(np.median(values)),
            "one_sided_sign_p": _sign_tail(wins, len(non_ties)),
            "gate": bool(values.mean() < 0 and np.median(values) < 0 and _sign_tail(wins, len(non_ties)) < 0.05),
        }
    output["gate"] = bool(output["prior"]["gate"] and output["zero"]["gate"])
    return output


def _summarize_selection(rows):
    gaps = np.asarray([row["objective_minus_fraud_oracle_mse"] for row in rows], dtype=float)
    same = sum(row["objective_best_restart"] == row["fraud_mse_best_restart"] for row in rows)
    return {
        "n_groups": len(rows),
        "objective_selected_same_restart_as_fraud_mse_oracle": same,
        "mean_extra_fraud_mse_from_objective_selection": float(gaps.mean()),
        "median_extra_fraud_mse_from_objective_selection": float(np.median(gaps)),
        "objective_selection_gate": _gate(rows, "objective"),
        "fraud_mse_oracle_selection_gate_diagnostic_only": _gate(rows, "fraud_oracle"),
        "note": "Fraud-MSE oracle selection uses ground truth reconstruction error and is diagnostic only.",
    }


def _print_report(report):
    print(json.dumps(report["summary"], indent=2))
    for split, payload in report["by_split"].items():
        print(f"\n{split}")
        for row in payload["correlations"]:
            if row["method"] == "baseline":
                print(
                    f"  baseline objective vs {row['metric']}: "
                    f"pearson={row['pearson_objective_vs_metric']}, "
                    f"spearman={row['spearman_objective_vs_metric']}"
                )
        sel = payload["selection_summary"]
        print(
            "  objective-best same as fraud-MSE-best: "
            f"{sel['objective_selected_same_restart_as_fraud_mse_oracle']}/{sel['n_groups']}"
        )
        print(
            "  mean extra fraud MSE from objective selection: "
            f"{sel['mean_extra_fraud_mse_from_objective_selection']:.6f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    datasets = {
        "development": run / "development_records.json",
        "evaluation": run / "evaluation_records.json",
    }
    report = {
        "run": str(run),
        "purpose": "Check whether lower attack objective aligns with lower reconstruction error using existing candidates.",
        "by_split": {},
    }
    for split, path in datasets.items():
        if not path.exists():
            continue
        records = _load_json(path)
        flat = _flatten(records, split)
        correlations = _correlation_summary(flat, split)
        selection = _selection_diagnostics(flat, split)
        report["by_split"][split] = {
            "n_records": len(records),
            "n_flat_candidates": len(flat),
            "correlations": correlations,
            "selection_rows": selection,
            "selection_summary": _summarize_selection(selection),
        }
        _write_csv(run / f"{split}_candidate_alignment_flat.csv", flat)
        _write_csv(run / f"{split}_candidate_alignment_correlations.csv", correlations)
        _write_csv(run / f"{split}_candidate_alignment_selection.csv", selection)
    report["summary"] = {
        split: payload["selection_summary"] for split, payload in report["by_split"].items()
    }
    _write_json(run / "candidate_alignment_report.json", report)
    _print_report(report)


if __name__ == "__main__":
    main()
