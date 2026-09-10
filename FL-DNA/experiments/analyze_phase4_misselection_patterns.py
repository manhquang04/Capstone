"""Analyze held-out groups where objective selection misses fraud reconstruction.

The script is diagnostic-only.  It reads Phase 4 pre-gate artifacts and asks
whether misselected groups share feature spread, gradient contribution, or
restart-margin patterns.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


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


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2))


def _as_float(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _group_candidates(flat_rows, group_id):
    return [
        row
        for row in flat_rows
        if row["split"] == "evaluation" and row["method"] == "baseline" and int(row["group_id"]) == group_id
    ]


def _best(rows, key):
    return min(rows, key=lambda row: float(row[key]))


def _feature_stats(group):
    x = np.asarray(group["x"], dtype=float)
    y = np.asarray(group["y"], dtype=int).reshape(-1)
    fraud = x[y == 1]
    non = x[y == 0]
    return {
        "fraud_feature_l2_mean": float(np.linalg.norm(fraud, axis=1).mean()) if len(fraud) else None,
        "non_fraud_feature_l2_mean": float(np.linalg.norm(non, axis=1).mean()) if len(non) else None,
        "non_fraud_feature_std_mean": float(non.std(axis=0).mean()) if len(non) else None,
        "non_fraud_pairwise_distance_mean": _pairwise_mean(non),
        "fraud_to_non_centroid_distance": _centroid_distance(fraud, non),
    }


def _pairwise_mean(x):
    if len(x) < 2:
        return None
    distances = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            distances.append(float(np.linalg.norm(x[i] - x[j])))
    return float(np.mean(distances))


def _centroid_distance(a, b):
    if len(a) == 0 or len(b) == 0:
        return None
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))


def _summary(rows):
    output = {}
    for bucket in ("selected_correct", "selected_wrong"):
        bucket_rows = [row for row in rows if row["selection_bucket"] == bucket]
        payload = {"n_groups": len(bucket_rows)}
        for key in (
            "train_shared_bn_weighted_ratio",
            "train_shared_bn_fraud_cosine_to_full",
            "eval_weighted_ratio",
            "non_fraud_feature_std_mean",
            "non_fraud_pairwise_distance_mean",
            "fraud_to_non_centroid_distance",
            "objective_margin_wrong_minus_fraud_best",
            "fraud_mse_gap_objective_minus_best",
        ):
            values = [row[key] for row in bucket_rows if row[key] is not None]
            payload[f"{key}_mean"] = float(np.mean(values)) if values else None
            payload[f"{key}_median"] = float(np.median(values)) if values else None
        output[bucket] = payload
    restarts = {}
    for row in rows:
        key = f"objective_restart_{row['objective_best_restart']}_fraud_best_{row['fraud_mse_best_restart']}"
        restarts[key] = restarts.get(key, 0) + 1
    output["restart_pattern_counts"] = restarts
    output["wrong_group_ids"] = [row["group_id"] for row in rows if row["selection_bucket"] == "selected_wrong"]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    selection = _read_csv(run / "evaluation_candidate_alignment_selection.csv")
    flat = _read_csv(run / "evaluation_candidate_alignment_flat.csv")
    gradient_rows = _read_csv(run / "class_gradient_contribution.csv")
    targets = torch.load(run / "evaluation_targets.pt", weights_only=False)

    gradients = {}
    for row in gradient_rows:
        gradients[(int(row["group_id"]), row["mode"])] = row

    rows = []
    for row in selection:
        group_id = int(row["group_id"])
        candidates = _group_candidates(flat, group_id)
        objective_best = _best(candidates, "objective")
        fraud_best = _best(candidates, "fraud_mse")
        bucket = (
            "selected_correct"
            if int(row["objective_best_restart"]) == int(row["fraud_mse_best_restart"])
            else "selected_wrong"
        )
        eval_grad = gradients.get((group_id, "eval"), {})
        train_grad = gradients.get((group_id, "train_shared_bn"), {})
        stats = _feature_stats(targets[group_id])
        rows.append(
            {
                "group_id": group_id,
                "selection_bucket": bucket,
                "objective_best_restart": int(row["objective_best_restart"]),
                "fraud_mse_best_restart": int(row["fraud_mse_best_restart"]),
                "objective_best_objective": float(objective_best["objective"]),
                "fraud_best_objective": float(fraud_best["objective"]),
                "objective_margin_wrong_minus_fraud_best": float(objective_best["objective"])
                - float(fraud_best["objective"]),
                "objective_best_fraud_mse": float(row["objective_best_fraud_mse"]),
                "fraud_oracle_best_fraud_mse": float(row["fraud_oracle_best_fraud_mse"]),
                "fraud_mse_gap_objective_minus_best": float(row["objective_minus_fraud_oracle_mse"]),
                "eval_weighted_ratio": _as_float(eval_grad, "fraud_to_non_weighted_norm_ratio"),
                "eval_fraud_cosine_to_full": _as_float(eval_grad, "fraud_cosine_to_full"),
                "train_shared_bn_weighted_ratio": _as_float(train_grad, "fraud_to_non_weighted_norm_ratio"),
                "train_shared_bn_fraud_cosine_to_full": _as_float(train_grad, "fraud_cosine_to_full"),
                **stats,
            }
        )

    report = {
        "run": str(run),
        "purpose": "Compare held-out groups where objective-best restart does or does not match fraud-MSE-best restart.",
        "summary": _summary(rows),
        "rows": rows,
        "interpretation_hint": "If wrong groups have small objective margins but large fraud-MSE gaps, selection is unstable even when good fraud candidates exist.",
    }
    _write_csv(run / "misselection_pattern_rows.csv", rows)
    _write_json(run / "misselection_pattern_report.json", report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
