"""Inspect whether competing reconstructions are plausible feature vectors.

The script compares the objective-best and fraud-MSE-best candidates for one
Phase 4 group.  It is diagnostic-only and uses saved artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.load_creditcard import BASE_FEATURE_COLUMNS, NUMERIC_COLUMNS, _build_features


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/phase3/full_20260908T143837588533Z"


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


def _artifact_path(run: Path, group_id: int, restart: int):
    primary = run / "evaluation" / f"group_{group_id}" / f"restart_{restart}" / "baseline.pt"
    if primary.exists():
        return primary
    extra = run / "extra_restarts_misselected" / f"group_{group_id}" / f"restart_{restart}" / "baseline.pt"
    if extra.exists():
        return extra
    plausible = run / "plausible_init_misselected" / f"group_{group_id}" / f"restart_{restart}" / "baseline.pt"
    if plausible.exists():
        return plausible
    raise FileNotFoundError(f"No baseline artifact for group={group_id} restart={restart}")


def _raw_numeric(scaled, metadata):
    center = np.asarray(metadata["numeric_center"], dtype=float)
    scale = np.asarray(metadata["numeric_scale"], dtype=float)
    return scaled * scale + center


def _numeric_ranges(metadata):
    frame = pd.read_csv(ROOT / "datasets/creditcard.csv", usecols=BASE_FEATURE_COLUMNS + ["isFraud"])
    features = _build_features(frame)
    rows = {}
    for name in NUMERIC_COLUMNS:
        values = features[name].to_numpy(dtype=float)
        rows[name] = {
            "min": float(values.min()),
            "p01": float(np.quantile(values, 0.01)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(values.max()),
        }
    return rows


def _record_rows(name, vector, feature_names, metadata, ranges):
    numeric_names = feature_names[: len(NUMERIC_COLUMNS)]
    numeric_raw = _raw_numeric(np.asarray(vector[: len(NUMERIC_COLUMNS)], dtype=float), metadata)
    rows = []
    for idx, feature in enumerate(numeric_names):
        raw = float(numeric_raw[idx])
        bounds = ranges[feature]
        rows.append(
            {
                "candidate": name,
                "feature": feature,
                "scaled_value": float(vector[idx]),
                "raw_value": raw,
                "dataset_min": bounds["min"],
                "dataset_p01": bounds["p01"],
                "dataset_p99": bounds["p99"],
                "dataset_max": bounds["max"],
                "outside_dataset_range": bool(raw < bounds["min"] or raw > bounds["max"]),
                "outside_1_99pct_range": bool(raw < bounds["p01"] or raw > bounds["p99"]),
            }
        )
    type_values = np.asarray(vector[len(NUMERIC_COLUMNS) :], dtype=float)
    for offset, feature in enumerate(feature_names[len(NUMERIC_COLUMNS) :]):
        rows.append(
            {
                "candidate": name,
                "feature": feature,
                "scaled_value": float(type_values[offset]),
                "raw_value": float(type_values[offset]),
                "dataset_min": 0.0,
                "dataset_p01": 0.0,
                "dataset_p99": 1.0,
                "dataset_max": 1.0,
                "outside_dataset_range": bool(type_values[offset] < 0.0 or type_values[offset] > 1.0),
                "outside_1_99pct_range": False,
            }
        )
    return rows


def _constraints(vector, metadata):
    raw = dict(zip(NUMERIC_COLUMNS, _raw_numeric(np.asarray(vector[: len(NUMERIC_COLUMNS)], dtype=float), metadata)))
    type_values = np.asarray(vector[len(NUMERIC_COLUMNS) :], dtype=float)
    return {
        "amount_negative": bool(raw["amount"] < 0),
        "amount": float(raw["amount"]),
        "balance_diff_orig_residual": float(
            raw["balance_diff_orig"] - (raw["oldbalanceOrg"] - raw["newbalanceOrig"])
        ),
        "balance_diff_dest_residual": float(
            raw["balance_diff_dest"] - (raw["newbalanceDest"] - raw["oldbalanceDest"])
        ),
        "type_sum": float(type_values.sum()),
        "type_max": float(type_values.max()),
        "type_argmax": int(type_values.argmax()),
        "type_entropy": float(-(type_values * np.log(np.clip(type_values, 1e-12, 1.0))).sum()),
    }


def _fraud_vector(payload):
    labels = payload["labels"].reshape(-1).detach().cpu().numpy()
    ids = np.where(labels == 1)[0]
    if len(ids) != 1:
        raise ValueError(f"Expected one fraud record, found {len(ids)}")
    idx = int(ids[0])
    return idx, payload["original"][idx].detach().cpu().numpy(), payload["aligned"][idx].detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--group-id", type=int, default=2)
    parser.add_argument("--objective-restart", type=int, default=0)
    parser.add_argument("--fraud-best-restart", type=int, default=6)
    parser.add_argument("--tag", default="", help="Optional suffix for output files, e.g. standard or plausible_init.")
    args = parser.parse_args()

    run = args.run.resolve()
    metadata = _load_json(REFERENCE / "preprocessing.json")
    ranges = _numeric_ranges(metadata)
    feature_names = metadata["feature_names"]
    objective_payload = torch.load(_artifact_path(run, args.group_id, args.objective_restart), weights_only=False)
    fraud_payload = torch.load(_artifact_path(run, args.group_id, args.fraud_best_restart), weights_only=False)
    fraud_idx, original, objective_candidate = _fraud_vector(objective_payload)
    _, _, fraud_best_candidate = _fraud_vector(fraud_payload)

    candidates = {
        "original": original,
        "objective_best": objective_candidate,
        "fraud_mse_best": fraud_best_candidate,
    }
    feature_rows = []
    constraint_rows = []
    for name, vector in candidates.items():
        feature_rows.extend(_record_rows(name, vector, feature_names, metadata, ranges))
        constraint_rows.append({"candidate": name, **_constraints(vector, metadata)})
    diff_rows = []
    for idx, feature in enumerate(feature_names):
        diff_rows.append(
            {
                "feature": feature,
                "objective_minus_original_abs": float(abs(objective_candidate[idx] - original[idx])),
                "fraud_best_minus_original_abs": float(abs(fraud_best_candidate[idx] - original[idx])),
                "objective_minus_fraud_best_abs": float(abs(objective_candidate[idx] - fraud_best_candidate[idx])),
            }
        )
    diff_rows.sort(key=lambda row: row["objective_minus_fraud_best_abs"], reverse=True)
    report = {
        "run": str(run),
        "group_id": args.group_id,
        "fraud_ordered_record_index": fraud_idx,
        "objective_restart": args.objective_restart,
        "fraud_best_restart": args.fraud_best_restart,
        "constraints": constraint_rows,
        "largest_candidate_differences": diff_rows[:8],
        "interpretation": "Diagnostic only; compares saved aligned reconstructions for the fraud record.",
    }
    suffix = f"_{args.tag}" if args.tag else ""
    stem = f"group_{args.group_id}_candidate_plausibility{suffix}"
    _write_json(run / f"{stem}.json", report)
    _write_csv(run / f"{stem}_features.csv", feature_rows)
    _write_csv(run / f"{stem}_constraints.csv", constraint_rows)
    _write_csv(run / f"{stem}_diffs.csv", diff_rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
