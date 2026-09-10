"""Measure real PaySim balance-difference residuals by class.

This diagnostic informs whether balance-difference penalties are safe priors or
could become hidden class hints for fraud records.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.load_creditcard import BASE_FEATURE_COLUMNS, TARGET_COLUMN, _build_features


ROOT = Path(__file__).resolve().parents[1]


def _summarize(values):
    values = np.asarray(values, dtype=float)
    abs_values = np.abs(values)
    return {
        "n": int(len(values)),
        "mean_abs": float(abs_values.mean()),
        "median_abs": float(np.median(abs_values)),
        "p90_abs": float(np.quantile(abs_values, 0.90)),
        "p99_abs": float(np.quantile(abs_values, 0.99)),
        "max_abs": float(abs_values.max()),
        "near_zero_rate_1e-6": float((abs_values <= 1e-6).mean()),
        "near_zero_rate_1": float((abs_values <= 1.0).mean()),
        "near_zero_rate_100": float((abs_values <= 100.0).mean()),
    }


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    frame = pd.read_csv(ROOT / "datasets/creditcard.csv", usecols=BASE_FEATURE_COLUMNS + [TARGET_COLUMN])
    if args.max_rows:
        frame = frame.iloc[: args.max_rows].copy()
    features = _build_features(frame)
    labels = frame[TARGET_COLUMN].to_numpy()
    residuals = {
        "balance_diff_orig": features["balance_diff_orig"].to_numpy()
        - (features["oldbalanceOrg"].to_numpy() - features["newbalanceOrig"].to_numpy()),
        "balance_diff_dest": features["balance_diff_dest"].to_numpy()
        - (features["newbalanceDest"].to_numpy() - features["oldbalanceDest"].to_numpy()),
    }
    rows = []
    for label, class_name in ((0, "non_fraud"), (1, "fraud")):
        mask = labels == label
        for residual_name, values in residuals.items():
            rows.append({"class": class_name, "residual": residual_name, **_summarize(values[mask])})
    report = {
        "dataset": str(ROOT / "datasets/creditcard.csv"),
        "max_rows": args.max_rows,
        "rows": rows,
        "interpretation": "If fraud residuals are naturally large, strong balance-diff penalties may distort fraud reconstructions.",
    }
    out = ROOT / "artifacts/phase4/paysim_balance_residuals.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    _write_csv(ROOT / "artifacts/phase4/paysim_balance_residuals.csv", rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
