"""Analyze the frozen RQ2 DNA Transform v2-SB confirmatory execution."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    seeds = [int(seed) for seed in config["confirmatory_seeds"]["values"]]
    registry: dict[int, dict[str, dict[str, float]]] = {}
    for path in sorted(args.run_dir.glob("seed_*/**/metrics.json")):
        seed = int(path.parts[-3].split("_")[1])
        method = path.parent.name
        payload = json.loads(path.read_text())
        final = payload["rounds"][-1]
        if int(payload["config"]["seed"]) != seed:
            raise ValueError(f"seed mismatch: {path}")
        if int(final["round"]) != int(config["training"]["num_rounds"]):
            raise ValueError(f"round mismatch: {path}")
        registry.setdefault(seed, {})[method] = {
            "f1": float(final["f1_score"]),
            "auc": float(final["auc_roc"]),
            "pr_auc": float(final["pr_auc"]),
        }
    expected_methods = {"baseline", "dna_transform_v2sb"}
    if set(registry) != set(seeds):
        raise RuntimeError("confirmatory seed registry is incomplete")
    if any(set(methods) != expected_methods for methods in registry.values()):
        raise RuntimeError("confirmatory method registry is incomplete")

    rows = []
    delta_f1 = []
    delta_auc = []
    delta_pr_auc = []
    for seed in seeds:
        baseline = registry[seed]["baseline"]
        v2sb = registry[seed]["dna_transform_v2sb"]
        row = {
            "seed": seed,
            "baseline_f1": baseline["f1"],
            "v2sb_f1": v2sb["f1"],
            "delta_f1": v2sb["f1"] - baseline["f1"],
            "baseline_auc": baseline["auc"],
            "v2sb_auc": v2sb["auc"],
            "delta_auc": v2sb["auc"] - baseline["auc"],
            "baseline_pr_auc": baseline["pr_auc"],
            "v2sb_pr_auc": v2sb["pr_auc"],
            "delta_pr_auc": v2sb["pr_auc"] - baseline["pr_auc"],
        }
        rows.append(row)
        delta_f1.append(row["delta_f1"])
        delta_auc.append(row["delta_auc"])
        delta_pr_auc.append(row["delta_pr_auc"])

    analyses = {
        "f1": paired_stats(delta_f1, float(config["statistics"]["f1_noninferiority_margin"])),
        "auc": paired_stats(delta_auc, float(config["statistics"]["auc_noninferiority_margin"])),
        "pr_auc_descriptive": paired_stats(delta_pr_auc, 0.0),
    }
    primary_pass = bool(analyses["f1"]["noninferiority_pass"] and analyses["auc"]["noninferiority_pass"])
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_id": config["config_id"],
        "n_seeds": len(seeds),
        "contrast": "DNA_TRANSFORM_V2SB_MINUS_BASELINE",
        "endpoint_rule": "both_f1_and_auc_must_pass",
        "alpha": float(config["statistics"]["alpha"]),
        "analyses": analyses,
        "primary_noninferiority_pass": primary_pass,
        "primary_conclusion": (
            "DNA Transform v2-SB establishes non-inferiority for both RQ2 endpoints"
            if primary_pass
            else "DNA Transform v2-SB does not establish non-inferiority for both RQ2 endpoints"
        ),
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
