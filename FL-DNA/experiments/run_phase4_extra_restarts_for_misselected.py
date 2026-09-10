"""Run extra restarts only for Phase 4 misselected held-out groups.

This is a diagnostic branch.  It keeps the frozen pre-Phase-4 objective and
attack budget unchanged, then adds restarts for groups where objective-selected
candidates did not match fraud-MSE-best candidates.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from experiments.rerun_adam_ladder_balanced_objective import _run_pair
from experiments.run_phase3_full_client import dump
from privacy.seed_manager import derive_seed


def _load_json(path: Path):
    return json.loads(path.read_text())


def _read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sign_tail(wins, total):
    import math

    return sum(math.comb(total, k) for k in range(wins, total + 1)) / 2**total if total else 1.0


def _flatten(records):
    rows = []
    for row in records:
        metrics = row["metrics"]
        rows.append(
            {
                "method": row["method"],
                "group_id": int(row["group_id"]),
                "restart": int(row["restart"]),
                "objective": float(row["objective"]),
                "overall_mse": float(metrics["mean_mse"]),
                "fraud_mse": float(metrics["classes"]["1"]["mean_mse"]),
                "non_fraud_mse": float(metrics["classes"]["0"]["mean_mse"]),
                "prior_fraud_mse": float(row["prior"]["classes"]["1"]["mean_mse"]),
            }
        )
    return rows


def _best(rows, key):
    return min(rows, key=lambda row: row[key])


def _summarize_group(rows):
    baseline = [row for row in rows if row["method"] == "baseline"]
    zero = [row for row in rows if row["method"] == "zero_update"]
    objective_best = _best(baseline, "objective")
    fraud_best = _best(baseline, "fraud_mse")
    zero_objective_best = _best(zero, "objective")
    prior_mean = float(np.mean([row["prior_fraud_mse"] for row in baseline]))
    return {
        "group_id": objective_best["group_id"],
        "n_baseline_restarts": len(baseline),
        "objective_best_restart": objective_best["restart"],
        "fraud_mse_best_restart": fraud_best["restart"],
        "objective_best_objective": objective_best["objective"],
        "fraud_best_objective": fraud_best["objective"],
        "objective_margin_wrong_minus_fraud_best": objective_best["objective"] - fraud_best["objective"],
        "objective_best_fraud_mse": objective_best["fraud_mse"],
        "fraud_best_fraud_mse": fraud_best["fraud_mse"],
        "extra_fraud_mse_from_objective_selection": objective_best["fraud_mse"] - fraud_best["fraud_mse"],
        "prior_fraud_mse_mean": prior_mean,
        "zero_objective_best_fraud_mse": zero_objective_best["fraud_mse"],
        "objective_minus_prior_fraud_mse": objective_best["fraud_mse"] - prior_mean,
        "objective_minus_zero_fraud_mse": objective_best["fraud_mse"] - zero_objective_best["fraud_mse"],
        "fraud_best_minus_prior_fraud_mse": fraud_best["fraud_mse"] - prior_mean,
        "fraud_best_minus_zero_fraud_mse": fraud_best["fraud_mse"] - zero_objective_best["fraud_mse"],
    }


def _gate(rows, prefix):
    out = {}
    for control in ("prior", "zero"):
        key = f"{prefix}_minus_{control}_fraud_mse"
        values = np.asarray([row[key] for row in rows], dtype=float)
        wins = int((values < 0).sum())
        p_value = _sign_tail(wins, len(values))
        out[control] = {
            "n": len(rows),
            "wins": wins,
            "mean_difference": float(values.mean()),
            "median_difference": float(np.median(values)),
            "one_sided_sign_p": p_value,
            "gate": bool(values.mean() < 0 and np.median(values) < 0 and p_value < 0.05),
        }
    out["gate"] = bool(out["prior"]["gate"] and out["zero"]["gate"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--groups", type=int, nargs="+", default=[0, 2, 6, 9])
    parser.add_argument("--start-restart", type=int, default=3)
    parser.add_argument("--total-restarts", type=int, default=10)
    args = parser.parse_args()

    torch.set_num_threads(1)
    run = args.run.resolve()
    baseline_report = _load_json(run / "baseline_gate.json")
    frozen = _load_json(run / "frozen.json")
    groups = torch.load(run / "evaluation_targets.pt", weights_only=False)
    existing = _load_json(run / "evaluation_records.json")
    out = run / "extra_restarts_misselected"
    out.mkdir(exist_ok=True)

    records = list(existing)
    for group_id in args.groups:
        group = groups[group_id]
        for restart in range(args.start_restart, args.total_restarts):
            folder = out / f"group_{group_id}" / f"restart_{restart}"
            if (folder / "results.json").exists():
                records.extend(_load_json(folder / "results.json"))
                continue
            records.extend(
                _run_pair(
                    folder,
                    group,
                    group_id,
                    restart,
                    baseline_report["protocol"]["run_seed"],
                    frozen["attack_lr"],
                    frozen["iterations"],
                    baseline_report["protocol"]["batch_size"],
                )
            )

    flat = _flatten(records)
    selected_rows = []
    all_group_summaries = []
    for group_id in sorted({row["group_id"] for row in flat}):
        group_rows = [row for row in flat if row["group_id"] == group_id]
        summary = _summarize_group(group_rows)
        all_group_summaries.append(summary)
        if group_id in args.groups:
            selected_rows.append(summary)

    report = {
        "run": str(run),
        "groups_with_extra_restarts": args.groups,
        "start_restart": args.start_restart,
        "total_restarts": args.total_restarts,
        "frozen_attack": frozen,
        "selected_wrong_group_summary": selected_rows,
        "all_group_gate_with_extra_restarts": {
            "objective_selection": _gate(all_group_summaries, "objective"),
            "fraud_mse_oracle_selection_diagnostic_only": _gate(all_group_summaries, "fraud_best"),
        },
        "wrong_groups_only_gate": {
            "objective_selection": _gate(selected_rows, "objective"),
            "fraud_mse_oracle_selection_diagnostic_only": _gate(selected_rows, "fraud_best"),
        },
        "interpretation": "Diagnostic only; extra restarts are run after seeing held-out misselections and cannot be used as the official gate.",
    }
    dump(out / "extra_restart_report.json", report)
    _write_csv(out / "extra_restart_selected_wrong_groups.csv", selected_rows)
    _write_csv(out / "extra_restart_all_group_summary.csv", all_group_summaries)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
