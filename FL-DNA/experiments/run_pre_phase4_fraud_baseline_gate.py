"""Pre-Phase-4 fraud-focused baseline gate.

This runner does not evaluate DNA.  It checks whether a non-class-decomposed
baseline attack can beat controls on fraud records in a bounded Adam scope.
Only scopes that pass this gate can open Phase 4 DNA evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.load_creditcard import BASE_FEATURE_COLUMNS
from experiments import fraud_fl_common as common
from experiments.rerun_adam_ladder_balanced_objective import _run_pair
from experiments.run_phase3_adam_ladder import excluded_rows, make_groups
from experiments.run_phase3_full_client import checksum, dump
from privacy.seed_manager import derive_seed, generate_run_seed


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/phase3/full_20260908T143837588533Z"


def _sign_tail(wins, total):
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / 2**total if total else 1.0


def _select(rows, group_id, method):
    return min(
        (row for row in rows if row["group_id"] == group_id and row["method"] == method),
        key=lambda row: row["objective"],
    )


def _mean_prior(rows, group_id, label):
    values = [
        row["prior"]["classes"][label]["mean_mse"]
        for row in rows
        if row["group_id"] == group_id and row["method"] == "baseline"
    ]
    return float(np.mean(values))


def _evaluate(rows):
    selected = []
    for group_id in sorted({row["group_id"] for row in rows}):
        baseline = _select(rows, group_id, "baseline")
        zero = _select(rows, group_id, "zero_update")
        for label, class_name in (("1", "fraud"), ("0", "non_fraud")):
            baseline_mse = baseline["metrics"]["classes"][label]["mean_mse"]
            prior_mse = _mean_prior(rows, group_id, label)
            zero_mse = zero["metrics"]["classes"][label]["mean_mse"]
            selected.append(
                {
                    "group_id": group_id,
                    "class": class_name,
                    "baseline_restart": baseline["restart"],
                    "zero_restart": zero["restart"],
                    "baseline_mse": baseline_mse,
                    "prior_mse": prior_mse,
                    "zero_mse": zero_mse,
                    "baseline_minus_prior": baseline_mse - prior_mse,
                    "baseline_minus_zero": baseline_mse - zero_mse,
                }
            )
    comparisons = {}
    for class_name in ("fraud", "non_fraud"):
        class_rows = [row for row in selected if row["class"] == class_name]
        comparisons[class_name] = {}
        for control in ("prior", "zero"):
            deltas = np.asarray([row[f"baseline_minus_{control}"] for row in class_rows])
            non_ties = deltas[deltas != 0]
            wins = int((non_ties < 0).sum())
            summary = {
                "n": len(class_rows),
                "wins": wins,
                "non_ties": int(len(non_ties)),
                "mean_difference": float(deltas.mean()),
                "median_difference": float(np.median(deltas)),
                "one_sided_sign_p": _sign_tail(wins, len(non_ties)),
            }
            summary["gate"] = bool(
                summary["mean_difference"] < 0
                and summary["median_difference"] < 0
                and summary["one_sided_sign_p"] < 0.05
            )
            comparisons[class_name][control] = summary
    comparisons["fraud_gate"] = bool(comparisons["fraud"]["prior"]["gate"] and comparisons["fraud"]["zero"]["gate"])
    return comparisons, selected


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _candidate_score(comparisons):
    return max(comparisons["fraud"]["prior"]["mean_difference"], comparisons["fraud"]["zero"]["mean_difference"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-groups", type=int, default=4)
    parser.add_argument("--evaluation-groups", type=int, default=12)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--attack-lrs", type=float, nargs="+", default=[0.05, 0.1])
    parser.add_argument("--iterations", type=int, nargs="+", default=[300, 600])
    args = parser.parse_args()

    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    started = time.perf_counter()
    run_seed = generate_run_seed()
    out = ROOT / "artifacts/phase4" / datetime.now(timezone.utc).strftime("pre_phase4_fraud_gate_%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(ROOT / "datasets/creditcard.csv", usecols=BASE_FEATURE_COLUMNS + ["isFraud"])
    excluded = excluded_rows(frame.isFraud.to_numpy())
    available = np.asarray(sorted(set(range(len(frame))) - excluded))
    protocol = {
        "run_seed": run_seed,
        "scope": "pre-Phase-4 bounded Adam baseline gate; no DNA evaluation",
        "records_per_group": 4,
        "fraud_per_group": 1,
        "batch_size": 4,
        "local_steps": 1,
        "development_groups": args.development_groups,
        "evaluation_groups": args.evaluation_groups,
        "restarts": args.restarts,
        "attack_lrs": args.attack_lrs,
        "iterations": args.iterations,
        "objective": "balanced_tensor over full transmitted update; no class-decomposed oracle signal",
        "attacker_knowledge": "known supervised labels/order/RNG inherited from Phase 3; no oracle class-separated gradients and no class-weighted objective",
        "evaluation": "candidate selected by full-update objective; gate computed on fraud records only and non-fraud reported separately",
        "gate": "fraud class must beat both prior and zero-update controls with negative mean/median and sign-test p<0.05",
        "dataset_sha256": checksum(ROOT / "datasets/creditcard.csv"),
        "checkpoint_sha256": checksum(REFERENCE / "pre_local.pt"),
    }
    dump(out / "protocol_lock.json", protocol)

    development_groups = make_groups(
        frame,
        available,
        args.development_groups,
        protocol["records_per_group"],
        protocol["fraud_per_group"],
        derive_seed(run_seed, "pre-phase4-development-data"),
    )
    development_ids = {source_id for group in development_groups for source_id in group["source_ids"]}
    available = np.asarray(sorted(set(available) - development_ids))
    torch.save(development_groups, out / "development_targets.pt")
    candidates = []
    development_records = []
    for iterations in args.iterations:
        for attack_lr in args.attack_lrs:
            rows = []
            for group_id, group in enumerate(development_groups):
                rows.extend(
                    _run_pair(
                        out / "development" / f"iter_{iterations}" / f"lr_{attack_lr}" / f"group_{group_id}" / "restart_0",
                        group,
                        group_id,
                        0,
                        run_seed,
                        attack_lr,
                        iterations,
                        protocol["batch_size"],
                    )
                )
            comparisons, selected = _evaluate(rows)
            candidates.append(
                {
                    "iterations": iterations,
                    "attack_lr": attack_lr,
                    "selection_score": _candidate_score(comparisons),
                    "comparisons": comparisons,
                    "selected": selected,
                }
            )
            development_records.extend(rows)
    chosen = min(candidates, key=lambda item: item["selection_score"])
    frozen = {
        "attack_lr": chosen["attack_lr"],
        "iterations": chosen["iterations"],
        "restarts": args.restarts,
        "protocol_sha256": checksum(out / "protocol_lock.json"),
        "selected_before_evaluation": True,
    }
    dump(out / "development_records.json", development_records)
    dump(out / "development_summary.json", {"candidates": candidates, "chosen": chosen})
    _write_csv(
        out / "development_summary.csv",
        [
            {
                "iterations": row["iterations"],
                "attack_lr": row["attack_lr"],
                "selection_score": row["selection_score"],
                "fraud_prior_wins": row["comparisons"]["fraud"]["prior"]["wins"],
                "fraud_prior_mean_delta": row["comparisons"]["fraud"]["prior"]["mean_difference"],
                "fraud_zero_wins": row["comparisons"]["fraud"]["zero"]["wins"],
                "fraud_zero_mean_delta": row["comparisons"]["fraud"]["zero"]["mean_difference"],
            }
            for row in candidates
        ],
    )
    dump(out / "frozen.json", frozen)

    evaluation_groups = make_groups(
        frame,
        available,
        args.evaluation_groups,
        protocol["records_per_group"],
        protocol["fraud_per_group"],
        derive_seed(run_seed, "pre-phase4-evaluation-data"),
    )
    evaluation_ids = {source_id for group in evaluation_groups for source_id in group["source_ids"]}
    if development_ids & evaluation_ids:
        raise AssertionError("Development and evaluation source rows overlap")
    torch.save(evaluation_groups, out / "evaluation_targets.pt")
    evaluation_records = []
    for group_id, group in enumerate(evaluation_groups):
        for restart in range(frozen["restarts"]):
            evaluation_records.extend(
                _run_pair(
                    out / "evaluation" / f"group_{group_id}" / f"restart_{restart}",
                    group,
                    group_id,
                    restart,
                    run_seed,
                    frozen["attack_lr"],
                    frozen["iterations"],
                    protocol["batch_size"],
                )
            )
    comparisons, selected = _evaluate(evaluation_records)
    _write_csv(out / "evaluation_selected_by_class.csv", selected)
    dump(out / "evaluation_records.json", evaluation_records)
    report = {
        "status": "FRAUD_SCOPE_GATE_OPEN" if comparisons["fraud_gate"] else "FRAUD_SCOPE_GATE_CLOSED",
        "dna_evaluation_open": comparisons["fraud_gate"],
        "protocol": protocol,
        "frozen_attack": frozen,
        "development": {"chosen": chosen},
        "evaluation": comparisons,
        "selected_evaluation": selected,
        "interpretation": "Pre-Phase-4 baseline gate only. DNA evaluation is not run here.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    dump(out / "baseline_gate.json", report)
    print(json.dumps({"output": str(out), "status": report["status"], "evaluation": comparisons}, indent=2))


if __name__ == "__main__":
    main()
