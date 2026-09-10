"""Develop and confirm a stronger bounded 16-record Adam update attack.

This is the next rung after the validated one-batch diagnostic. Development
chooses one attack configuration on source-disjoint groups; confirmation uses a
fresh target set and opens DNA evaluation only if the baseline beats controls.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.load_creditcard import BASE_FEATURE_COLUMNS
from experiments import fraud_fl_common as common
from experiments.run_attack_redevelopment import VARIANTS, evaluate, run_trial, write_selected_csv
from experiments.run_phase3_adam_ladder import excluded_rows, make_groups
from experiments.run_phase3_full_client import checksum, dump
from privacy.seed_manager import derive_seed, generate_run_seed


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/phase3/full_20260908T143837588533Z"


def _candidate_score(comparisons):
    return max(comparisons["prior"]["mean_difference"], comparisons["zero"]["mean_difference"])


def _write_candidates_csv(path, candidates):
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "variant": candidate["variant"],
                "attack_lr": candidate["attack_lr"],
                "iterations": candidate["iterations"],
                "selection_score": candidate["selection_score"],
                "prior_wins": candidate["comparisons"]["prior"]["wins"],
                "prior_mean_delta": candidate["comparisons"]["prior"]["mean_difference"],
                "prior_median_delta": candidate["comparisons"]["prior"]["median_difference"],
                "prior_p": candidate["comparisons"]["prior"]["one_sided_sign_p"],
                "zero_wins": candidate["comparisons"]["zero"]["wins"],
                "zero_mean_delta": candidate["comparisons"]["zero"]["mean_difference"],
                "zero_median_delta": candidate["comparisons"]["zero"]["median_difference"],
                "zero_p": candidate["comparisons"]["zero"]["one_sided_sign_p"],
            }
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_grid(groups, out, run_seed, variants, attack_lrs, iterations):
    records = []
    candidates = []
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown attack variant: {variant}")
        for attack_lr in attack_lrs:
            rows = []
            for group_id, group in enumerate(groups):
                folder = out / "development" / variant / f"lr_{attack_lr}" / f"group_{group_id}"
                group_rows = run_trial(folder, group, group_id, 0, run_seed, variant, attack_lr, iterations)
                rows.extend(group_rows)
                records.extend(group_rows)
            comparisons, selected = evaluate(rows)
            candidates.append(
                {
                    "variant": variant,
                    "attack_lr": attack_lr,
                    "iterations": iterations,
                    "selection_score": _candidate_score(comparisons),
                    "comparisons": comparisons,
                    "selected": selected,
                }
            )
    return records, candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-groups", type=int, default=3)
    parser.add_argument("--confirmation-groups", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--variants", nargs="+", default=["manifold_bn1", "manifold_bn3"])
    parser.add_argument("--attack-lrs", type=float, nargs="+", default=[0.03, 0.05])
    args = parser.parse_args()

    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    out = ROOT / "artifacts/attack_redevelopment" / datetime.now(timezone.utc).strftime(
        "scale_dev_%Y%m%dT%H%M%S%fZ"
    )
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    run_seed = generate_run_seed()
    frame = pd.read_csv(ROOT / "datasets/creditcard.csv", usecols=BASE_FEATURE_COLUMNS + ["isFraud"])
    excluded = excluded_rows(frame.isFraud.to_numpy())
    available = np.asarray(sorted(set(range(len(frame))) - excluded))
    protocol = {
        "run_seed": run_seed,
        "scope": "source-disjoint 16-record/four-step Adam attack redevelopment",
        "records_per_group": 16,
        "fraud_per_group": 4,
        "batch_size": 4,
        "local_steps": 4,
        "development_groups": args.development_groups,
        "confirmation_groups": args.confirmation_groups,
        "iterations": args.iterations,
        "development_restarts": 1,
        "confirmation_restarts": args.restarts,
        "variants": args.variants,
        "attack_lrs": args.attack_lrs,
        "selection": "minimum max(mean MSE delta vs own prior, mean MSE delta vs paired zero-update)",
        "gate": "for both controls: negative mean and median MSE difference and one-sided exact sign p<0.05",
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
        derive_seed(run_seed, "scale-development-data"),
    )
    development_ids = {source_id for group in development_groups for source_id in group["source_ids"]}
    available = np.asarray(sorted(set(available) - development_ids))
    torch.save(development_groups, out / "development_targets.pt")
    development_records, candidates = _run_grid(
        development_groups, out, run_seed, args.variants, args.attack_lrs, args.iterations
    )
    chosen = min(candidates, key=lambda row: row["selection_score"])
    frozen = {
        "variant": chosen["variant"],
        "attack_lr": chosen["attack_lr"],
        "iterations": args.iterations,
        "restarts": args.restarts,
        "protocol_sha256": checksum(out / "protocol_lock.json"),
        "selected_before_confirmation": True,
    }
    dump(out / "development_records.json", development_records)
    dump(out / "development_summary.json", {"candidates": candidates, "chosen": chosen})
    _write_candidates_csv(out / "development_summary.csv", candidates)
    dump(out / "frozen.json", frozen)

    confirmation_groups = make_groups(
        frame,
        available,
        args.confirmation_groups,
        protocol["records_per_group"],
        protocol["fraud_per_group"],
        derive_seed(run_seed, "scale-confirmation-data"),
    )
    confirmation_ids = {source_id for group in confirmation_groups for source_id in group["source_ids"]}
    if development_ids & confirmation_ids:
        raise AssertionError("Development and confirmation source rows overlap")
    torch.save(confirmation_groups, out / "confirmation_targets.pt")
    confirmation_records = []
    for group_id, group in enumerate(confirmation_groups):
        for restart in range(args.restarts):
            confirmation_records.extend(
                run_trial(
                    out / "confirmation" / f"group_{group_id}" / f"restart_{restart}",
                    group,
                    group_id,
                    restart,
                    run_seed,
                    frozen["variant"],
                    frozen["attack_lr"],
                    frozen["iterations"],
                )
            )
    comparisons, selected = evaluate(confirmation_records)
    gate = comparisons["prior"]["gate"] and comparisons["zero"]["gate"]
    write_selected_csv(out / "confirmation_selected.csv", selected)
    dump(out / "confirmation_records.json", confirmation_records)
    report = {
        "status": "SCALE_ATTACK_VALIDATED" if gate else "SCALE_REDEVELOPMENT_NEGATIVE",
        "baseline_gate_passed": gate,
        "phase4_dna_evaluation_open": gate,
        "protocol": protocol,
        "frozen_attack": frozen,
        "development": {"chosen": chosen},
        "confirmation": comparisons,
        "selected_confirmation": selected,
        "dna_result": None,
        "dna_reason": None if gate else "DNA was not tested because the 16-record baseline failed controls.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    dump(out / "attack_scale_redevelopment_report.json", report)
    print(json.dumps({"output": str(out), "gate": gate, "confirmation": comparisons}, indent=2))


if __name__ == "__main__":
    main()
