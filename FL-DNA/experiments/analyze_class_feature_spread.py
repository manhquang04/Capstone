"""Measure class-wise feature spread for Phase 3 attack interpretation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from data.load_creditcard import BASE_FEATURE_COLUMNS, NUMERIC_COLUMNS, _build_features
from experiments.run_phase3_full_client import dump


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/phase3/full_20260908T143837588533Z"


def _load_metadata():
    return SimpleNamespace(**json.loads((REFERENCE / "preprocessing.json").read_text()))


def _scaled_features(frame, metadata):
    features = _build_features(frame.copy())
    scaled = (features[NUMERIC_COLUMNS].to_numpy() - np.asarray(metadata.numeric_center)) / np.asarray(metadata.numeric_scale)
    onehot = np.column_stack([(features.type.to_numpy() == value).astype(float) for value in metadata.type_categories])
    return np.hstack((scaled, onehot)).astype("float64")


def _pairwise_sample(values, seed, limit):
    if len(values) < 2:
        return np.asarray([], dtype="float64")
    rng = np.random.default_rng(seed)
    count = min(len(values), limit)
    chosen = values[rng.choice(len(values), count, replace=False)]
    distances = []
    for i in range(len(chosen)):
        diff = chosen[i + 1 :] - chosen[i]
        if len(diff):
            distances.extend(np.square(diff).mean(axis=1).tolist())
    return np.asarray(distances, dtype="float64")


def _summarize_class(values, feature_names, seed):
    centroid = values.mean(axis=0)
    centered_mse = np.square(values - centroid).mean(axis=1)
    feature_variance = np.var(values, axis=0)
    feature_iqr = np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0)
    pairwise = _pairwise_sample(values, seed, 512)
    return {
        "n": int(len(values)),
        "mean_centroid_mse": float(centered_mse.mean()),
        "median_centroid_mse": float(np.median(centered_mse)),
        "mean_feature_variance": float(feature_variance.mean()),
        "median_feature_variance": float(np.median(feature_variance)),
        "mean_feature_iqr": float(feature_iqr.mean()),
        "median_feature_iqr": float(np.median(feature_iqr)),
        "mean_pairwise_mse": float(pairwise.mean()) if len(pairwise) else None,
        "median_pairwise_mse": float(np.median(pairwise)) if len(pairwise) else None,
        "top_variance_features": [
            {"feature": feature_names[index], "variance": float(feature_variance[index]), "iqr": float(feature_iqr[index])}
            for index in np.argsort(feature_variance)[::-1][:5]
        ],
    }


def _summarize_scope(name, frame, metadata, seed):
    labels = frame.isFraud.to_numpy()
    x = _scaled_features(frame, metadata)
    feature_names = metadata.feature_names
    classes = {}
    for label, class_name in ((0, "non_fraud"), (1, "fraud")):
        values = x[labels == label]
        classes[class_name] = _summarize_class(values, feature_names, seed + label)
    fraud = classes["fraud"]
    non = classes["non_fraud"]
    ratios = {}
    for key in (
        "mean_centroid_mse",
        "median_centroid_mse",
        "mean_feature_variance",
        "median_feature_variance",
        "mean_feature_iqr",
        "median_feature_iqr",
        "mean_pairwise_mse",
        "median_pairwise_mse",
    ):
        if fraud[key] is None or non[key] in (None, 0.0):
            ratios[f"fraud_to_non_{key}"] = None
        else:
            ratios[f"fraud_to_non_{key}"] = float(fraud[key] / non[key])
    return {"scope": name, "classes": classes, "ratios": ratios}


def _target_frame(dataset, targets_path):
    targets = torch.load(targets_path, weights_only=False)
    source_ids = sorted({source_id for group in targets for source_id in group["source_ids"]})
    return dataset.iloc[source_ids].copy()


def _train_reference_frame(dataset, seed, max_rows):
    if max_rows and max_rows < len(dataset):
        _, sampled = train_test_split(dataset, test_size=max_rows, random_state=seed, stratify=dataset["isFraud"])
        dataset = sampled.reset_index(drop=True)
    train, _ = train_test_split(dataset, test_size=0.35, random_state=seed, stratify=dataset["isFraud"])
    return train.copy()


def _write_summary_csv(path, report):
    rows = []
    for scope in report["scopes"]:
        for class_name, payload in scope["classes"].items():
            row = {"scope": scope["scope"], "class": class_name}
            for key, value in payload.items():
                if key != "top_variance_features":
                    row[key] = value
            rows.append(row)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path, report):
    lines = [
        "# Class Feature Spread Diagnostic",
        "",
        "This diagnostic measures whether non-fraud records are statistically tighter than fraud records under the same scaled feature representation used by the attack.",
        "",
        "| Scope | Class | n | Mean centroid MSE | Median centroid MSE | Mean pairwise MSE | Mean feature variance |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scope in report["scopes"]:
        for class_name, payload in scope["classes"].items():
            lines.append(
                f"| {scope['scope']} | {class_name} | {payload['n']} | "
                f"{payload['mean_centroid_mse']:.6f} | {payload['median_centroid_mse']:.6f} | "
                f"{payload['mean_pairwise_mse']:.6f} | {payload['mean_feature_variance']:.6f} |"
            )
    lines.extend(["", "Fraud/non-fraud ratios:", "", "| Scope | Mean centroid ratio | Median centroid ratio | Mean pairwise ratio | Mean variance ratio |", "|---|---:|---:|---:|---:|"])
    for scope in report["scopes"]:
        ratios = scope["ratios"]
        lines.append(
            f"| {scope['scope']} | {ratios['fraud_to_non_mean_centroid_mse']:.6f} | "
            f"{ratios['fraud_to_non_median_centroid_mse']:.6f} | "
            f"{ratios['fraud_to_non_mean_pairwise_mse']:.6f} | "
            f"{ratios['fraud_to_non_mean_feature_variance']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: lower centroid and pairwise MSE means records in that class are closer to their class-level typical point. A low non-fraud spread makes an unoptimized prior more competitive.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/attack_redevelopment/class_feature_spread")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--max-rows", type=int, default=500000)
    parser.add_argument(
        "--target-runs",
        type=Path,
        nargs="+",
        default=[
            ROOT / "artifacts/attack_redevelopment/run_20260908T180405538992Z/confirmation_targets.pt",
            ROOT / "artifacts/attack_redevelopment/scale_dev_20260908T191737994986Z/confirmation_targets.pt",
            ROOT / "artifacts/attack_redevelopment/class_decomp_20260908T200353687099Z/confirmation_targets.pt",
        ],
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = _load_metadata()
    dataset = pd.read_csv(ROOT / "datasets/creditcard.csv", usecols=BASE_FEATURE_COLUMNS + ["isFraud"])
    scopes = [_summarize_scope("train_reference_500k", _train_reference_frame(dataset, args.seed, args.max_rows), metadata, args.seed)]
    for target_path in args.target_runs:
        if not target_path.exists():
            continue
        name = target_path.parent.name
        scopes.append(_summarize_scope(name, _target_frame(dataset, target_path), metadata, args.seed))
    report = {
        "seed": args.seed,
        "max_rows": args.max_rows,
        "feature_representation": "RobustScaler numeric fields plus one-hot transaction type, using stored Phase 3 metadata.",
        "note": "Diagnostic only; it explains reconstruction controls and does not evaluate DNA protection.",
        "scopes": scopes,
    }
    dump(args.output / "class_feature_spread.json", report)
    _write_summary_csv(args.output / "class_feature_spread_summary.csv", report)
    _write_markdown(args.output / "class_feature_spread.md", report)
    print(json.dumps({"output": str(args.output), "scopes": [scope["scope"] for scope in scopes]}, indent=2))


if __name__ == "__main__":
    main()
