"""Analyze Priority 14 DP comparator utility against reused RQ2-v2 baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_stats(values: list[float], margin: float) -> dict:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    half = float(t.ppf(0.975, len(arr) - 1) * sd / math.sqrt(len(arr)))
    lower = mean - half
    upper = mean + half
    return {
        "n": int(len(arr)),
        "mean_delta": mean,
        "sd": sd,
        "median_delta": float(np.median(arr)),
        "ci95": [lower, upper],
        "noninferiority_margin": margin,
        "noninferiority_pass": bool(lower >= -margin),
    }


def load_metric(path: Path, seed: int, rounds: int) -> dict[str, float]:
    payload = json.loads(path.read_text())
    final = payload["rounds"][-1]
    if int(payload["config"]["seed"]) != int(seed):
        raise ValueError(f"seed mismatch: {path}")
    if int(final["round"]) != int(rounds):
        raise ValueError(f"round mismatch: {path}")
    return {"f1": float(final["f1_score"]), "auc": float(final["auc_roc"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    seeds = [int(seed) for seed in config["confirmatory_seeds"]["values"]]
    methods = [method["method_id"] for method in config["dp_methods"]]
    baseline_source = config["baseline_source"]
    baseline_dir = ROOT / baseline_source["run_dir"]
    baseline_summary = baseline_dir / "execution_summary.json"
    if sha256(baseline_summary) != baseline_source["execution_summary_sha256"]:
        raise RuntimeError("Baseline execution summary checksum mismatch")
    rounds = int(config["training"]["num_rounds"])

    registry: dict[int, dict[str, dict[str, float]]] = {}
    for seed in seeds:
        baseline_path = baseline_dir / f"seed_{seed}" / "baseline" / "metrics.json"
        registry.setdefault(seed, {})["baseline"] = load_metric(baseline_path, seed, rounds)
        for method_id in methods:
            dp_path = args.run_dir / f"seed_{seed}" / method_id / "metrics.json"
            registry[seed][method_id] = load_metric(dp_path, seed, rounds)

    rows = []
    analyses = {}
    for method_id in methods:
        delta_f1 = []
        delta_auc = []
        for seed in seeds:
            baseline = registry[seed]["baseline"]
            dp = registry[seed][method_id]
            row = {
                "seed": seed,
                "method": method_id,
                "baseline_f1": baseline["f1"],
                "dp_f1": dp["f1"],
                "delta_f1": dp["f1"] - baseline["f1"],
                "baseline_auc": baseline["auc"],
                "dp_auc": dp["auc"],
                "delta_auc": dp["auc"] - baseline["auc"],
            }
            rows.append(row)
            delta_f1.append(row["delta_f1"])
            delta_auc.append(row["delta_auc"])
        analyses[method_id] = {
            "f1": paired_stats(delta_f1, float(config["statistics"]["f1_noninferiority_margin"])),
            "auc": paired_stats(delta_auc, float(config["statistics"]["auc_noninferiority_margin"])),
        }
        analyses[method_id]["primary_noninferiority_pass"] = bool(
            analyses[method_id]["f1"]["noninferiority_pass"]
            and analyses[method_id]["auc"]["noninferiority_pass"]
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "run_dir": str(args.run_dir),
        "baseline_source": baseline_source,
        "n_seeds": len(seeds),
        "contrast": "FL_DP_COMPARATOR_MINUS_FL_BASELINE",
        "endpoint_rule": config["statistics"]["endpoint_rule"],
        "alpha": float(config["statistics"]["alpha"]),
        "analyses": analyses,
    }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
