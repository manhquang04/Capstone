"""Create source-disjoint Phase 4 target groups for development or evaluation.

The sampler excludes the fixed 500k reference split and every existing
``*targets.pt`` artifact before drawing new groups.  This keeps newly created
target sets disjoint from earlier diagnostic, development, evaluation, and
fresh-final sets.
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
    return {
        int(source_id)
        for group in groups
        if isinstance(group, dict) and "source_ids" in group
        for source_id in group["source_ids"]
    }


def _target_source_ids(path):
    try:
        payload = torch.load(path, weights_only=False)
    except (EOFError, KeyError, RuntimeError, TypeError):
        return set()
    if isinstance(payload, list):
        return _source_ids(payload)
    if isinstance(payload, dict) and "source_ids" in payload:
        return set(map(int, payload["source_ids"]))
    return set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--records-per-group", type=int, default=4)
    parser.add_argument("--fraud-per-group", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--purpose", required=True)
    args = parser.parse_args()

    run = args.run.resolve()
    output = run / args.output_name
    if output.exists():
        raise FileExistsError(output)

    seed = args.seed if args.seed is not None else generate_run_seed()
    frame = pd.read_csv(ROOT / "datasets/creditcard.csv", usecols=BASE_FEATURE_COLUMNS + ["isFraud"])
    labels = frame.isFraud.to_numpy()
    excluded = excluded_rows(labels)
    available = np.asarray(sorted(set(range(len(frame))) - excluded), dtype=np.int64)
    groups = make_groups(
        frame,
        available,
        args.groups,
        args.records_per_group,
        args.fraud_per_group,
        derive_seed(seed, "phase4-source-disjoint-targets", args.output_name),
    )
    new_ids = _source_ids(groups)
    if new_ids & excluded:
        raise AssertionError("New targets overlap rows excluded before sampling")

    torch.save(groups, output)
    target_overlaps = {}
    for target_path in sorted((ROOT / "artifacts").rglob("*targets.pt")):
        if target_path.resolve() == output.resolve():
            continue
        target_overlaps[str(target_path.relative_to(ROOT))] = len(new_ids & _target_source_ids(target_path))

    provenance = {
        "output": str(output),
        "run_seed": seed,
        "groups": args.groups,
        "records_per_group": args.records_per_group,
        "fraud_per_group": args.fraud_per_group,
        "source_rows": sorted(new_ids),
        "fraud_counts": [int(group["y"].sum()) for group in groups],
        "excluded_row_count_before_sampling": len(excluded),
        "available_row_count_before_sampling": int(len(available)),
        "overlaps_with_existing_targets": target_overlaps,
        "max_overlap_with_existing_targets": max(target_overlaps.values(), default=0),
        "dataset_sha256": checksum(ROOT / "datasets/creditcard.csv"),
        "purpose": args.purpose,
    }
    dump(run / output.with_suffix(".provenance.json").name, provenance)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
