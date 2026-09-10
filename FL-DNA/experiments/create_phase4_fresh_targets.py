"""Create source-disjoint fresh Phase 4 target groups.

The target file created here is for a single-shot revalidation after the attack
rule has been locked.  It excludes source rows from existing target artifacts so
the final gate is not evaluated on groups that shaped the attacker.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.load_creditcard import BASE_FEATURE_COLUMNS
from experiments.run_phase3_adam_ladder import excluded_rows, make_groups
from experiments.run_phase3_full_client import checksum, dump
from privacy.seed_manager import derive_seed, generate_run_seed


ROOT = Path(__file__).resolve().parents[1]


def _source_ids(groups):
    return {int(source_id) for group in groups for source_id in group["source_ids"]}


def _load_target_ids(path):
    if not path.exists():
        return set()
    payload = torch.load(path, weights_only=False)
    if isinstance(payload, list):
        return _source_ids(payload)
    if isinstance(payload, dict) and "source_ids" in payload:
        return set(map(int, payload["source_ids"]))
    return set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output-name", default="fresh_final_targets.pt")
    parser.add_argument("--groups", type=int, default=12)
    parser.add_argument("--records-per-group", type=int, default=4)
    parser.add_argument("--fraud-per-group", type=int, default=1)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    run = args.run.resolve()
    output = run / args.output_name
    if output.exists():
        raise FileExistsError(output)

    seed = args.seed if args.seed is not None else generate_run_seed()
    frame = pd.read_csv(ROOT / "datasets/creditcard.csv", usecols=BASE_FEATURE_COLUMNS + ["isFraud"])
    excluded = excluded_rows(frame.isFraud.to_numpy())
    available = np.asarray(sorted(set(range(len(frame))) - excluded), dtype=np.int64)
    groups = make_groups(
        frame,
        available,
        args.groups,
        args.records_per_group,
        args.fraud_per_group,
        derive_seed(seed, "phase4-fresh-final-targets"),
    )
    fresh_ids = _source_ids(groups)
    if fresh_ids & excluded:
        raise AssertionError("Fresh final targets overlap existing excluded targets")

    torch.save(groups, output)
    overlaps = {}
    for name in ("development_targets.pt", "evaluation_targets.pt"):
        overlaps[name] = len(fresh_ids & _load_target_ids(run / name))

    provenance = {
        "output": str(output),
        "run_seed": seed,
        "groups": args.groups,
        "records_per_group": args.records_per_group,
        "fraud_per_group": args.fraud_per_group,
        "source_rows": sorted(fresh_ids),
        "fraud_counts": [int(group["y"].sum()) for group in groups],
        "excluded_row_count_before_sampling": len(excluded),
        "available_row_count_before_sampling": int(len(available)),
        "overlaps_with_named_phase4_targets": overlaps,
        "dataset_sha256": checksum(ROOT / "datasets/creditcard.csv"),
        "purpose": "Fresh source-disjoint single-shot Phase 4 baseline revalidation after attack rule lock.",
    }
    dump(run / output.with_suffix(".provenance.json").name, provenance)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
