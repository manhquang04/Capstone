"""Verify RQ2 Stage-1 replay metrics while excluding timing-only fields."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_FIELDS = [
    "round", "train_loss", "f1_score", "auc_roc", "pr_auc", "accuracy",
    "precision", "recall", "optimal_threshold", "tn", "fp", "fn", "tp",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, default=ROOT / "artifacts/rq2/stage1_development_20260912")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--methods", nargs="+", default=["baseline", "dna_lossless", "dna_transform", "dp_0.0001"])
    args = parser.parse_args()
    batch = args.batch_dir.resolve()
    records = []
    for method in args.methods:
        original = json.loads((batch / f"seed_{args.seed}" / method / "metrics.json").read_text())
        replay = json.loads((batch / "replay_validation" / f"seed_{args.seed}_{method}" / "metrics.json").read_text())
        differences = {}
        for field in CORE_FIELDS:
            values = []
            for first, second in zip(original["rounds"], replay["rounds"]):
                if first[field] is not None and second[field] is not None:
                    values.append(abs(float(first[field]) - float(second[field])))
            differences[field] = max(values, default=0.0)
        records.append(
            {
                "method": method,
                "seed": args.seed,
                "config_equal": original["config"] == replay["config"],
                "round_count_equal": len(original["rounds"]) == len(replay["rounds"]),
                "core_metrics_bit_equal": all(value == 0.0 for value in differences.values()),
                "max_absolute_differences": differences,
                "timing_fields_excluded": True,
            }
        )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "RQ2 Stage-1 seed-101 deterministic replay",
        "records": records,
        "gate_pass": all(row["config_equal"] and row["round_count_equal"] and row["core_metrics_bit_equal"] for row in records),
    }
    output = batch / "replay_validation" / "replay_validation_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["gate_pass"] else 1)


if __name__ == "__main__":
    main()
