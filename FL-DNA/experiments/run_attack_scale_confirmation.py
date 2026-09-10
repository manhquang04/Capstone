"""Test the frozen bounded attack on a source-disjoint four-batch tier."""
from __future__ import annotations

import csv
import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.load_creditcard import BASE_FEATURE_COLUMNS
from experiments import fraud_fl_common as common
from experiments.run_attack_redevelopment import evaluate, run_trial
from experiments.run_phase3_adam_ladder import excluded_rows, make_groups
from experiments.run_phase3_full_client import checksum, dump
from privacy.seed_manager import derive_seed, generate_run_seed


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/phase3/full_20260908T143837588533Z"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-run",
        type=Path,
        default=ROOT / "artifacts/attack_redevelopment/run_20260908T180405538992Z",
        help="Validated one-batch attack run whose frozen config is reused without retuning.",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    source_run = args.source_run.resolve()
    source_frozen = json.loads((source_run / "frozen.json").read_text())
    out = ROOT / "artifacts/attack_redevelopment" / datetime.now(timezone.utc).strftime("scale_%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    run_seed = generate_run_seed()
    protocol = {
        "run_seed": run_seed,
        "scope": "source-disjoint four-batch Adam generalization tier: 16 records, four fraud, batch size four",
        "frozen_from": str(source_run.relative_to(ROOT)),
        "attack_variant": source_frozen["variant"],
        "attack_lr": source_frozen["attack_lr"],
        "iterations": source_frozen["iterations"],
        "restarts": source_frozen["restarts"],
        "groups": 12,
        "selection": "minimum attacker objective within group; no retuning on this tier",
        "gate": "negative mean/median versus prior and zero-update and one-sided exact sign p<0.05",
        "dataset_sha256": checksum(ROOT / "datasets/creditcard.csv"),
        "checkpoint_sha256": checksum(REFERENCE / "pre_local.pt"),
    }
    dump(out / "protocol_lock.json", protocol)
    frame = pd.read_csv(ROOT / "datasets/creditcard.csv", usecols=BASE_FEATURE_COLUMNS + ["isFraud"])
    excluded = excluded_rows(frame.isFraud.to_numpy())
    available = np.asarray(sorted(set(range(len(frame))) - excluded))
    groups = make_groups(frame, available, protocol["groups"], 16, 4, derive_seed(run_seed, "scale-data"))
    torch.save(groups, out / "confirmation_targets.pt")
    rows = []
    for group_id, group in enumerate(groups):
        for restart in range(protocol["restarts"]):
            rows.extend(
                run_trial(
                    out / "confirmation" / f"group_{group_id}" / f"restart_{restart}",
                    group,
                    group_id,
                    restart,
                    run_seed,
                    protocol["attack_variant"],
                    protocol["attack_lr"],
                    protocol["iterations"],
                )
            )
    comparisons, selected = evaluate(rows)
    gate = comparisons["prior"]["gate"] and comparisons["zero"]["gate"]
    with (out / "confirmation_selected.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    dump(out / "confirmation_records.json", rows)
    dump(
        out / "attack_redevelopment_report.json",
        {
            "status": "SCALE_ATTACK_VALIDATED" if gate else "SCALE_CONFIRMATION_NEGATIVE",
            "baseline_gate_passed": gate,
            "phase4_dna_evaluation_open": gate,
            "confirmation": comparisons,
            "selected_confirmation": selected,
            "frozen_attack": source_frozen,
            "dna_result": None,
            "dna_reason": None if gate else "DNA was not tested because the four-batch baseline failed controls.",
            "limitations": [
                "This tier contains 16 records and four Adam steps, not a full utility client.",
                "Labels, order, RNG, architecture, optimizer, and checkpoint are known.",
                "No attack tuning used these confirmation targets.",
            ],
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print(json.dumps({"output": str(out), "gate": gate, "confirmation": comparisons}, indent=2))


if __name__ == "__main__":
    main()
