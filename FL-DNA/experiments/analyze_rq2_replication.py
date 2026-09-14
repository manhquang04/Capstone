"""Analyze one frozen 21-seed RQ2 replication."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import t

METHODS = ["dna_transform", "dna_lossless", "dp_0.00025"]


def stats(values: list[float], margin: Optional[float] = None) -> dict:
    values_array = np.asarray(values, float)
    mean = float(values_array.mean())
    half = float(t.ppf(.975, len(values_array) - 1) * values_array.std(ddof=1) /
                 math.sqrt(len(values_array)))
    result = {"n": len(values_array), "mean": mean,
              "sd": float(values_array.std(ddof=1)),
              "median": float(np.median(values_array)), "ci95": [mean-half, mean+half]}
    if margin is not None:
        result.update({"noninferiority_margin": margin,
                       "noninferiority_pass": mean-half >= -margin})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()); seeds = config["seeds"]
    values = {}; rows = []
    for path in sorted(args.run_dir.glob("seed_*/**/metrics.json")):
        seed = int(path.parts[-3].split("_")[1]); method = path.parent.name
        payload = json.loads(path.read_text()); final = payload["rounds"][-1]
        if int(payload["config"]["seed"]) != seed or int(final["round"]) != 50:
            raise ValueError(f"seed/round mismatch: {path}")
        values.setdefault(seed, {})[method] = {"f1": float(final["f1_score"]),
                                               "auc": float(final["auc_roc"])}
    expected = {"baseline", *METHODS}
    if set(values) != set(seeds) or any(set(methods) != expected for methods in values.values()):
        raise RuntimeError("incomplete paired registry")
    for seed in seeds:
        baseline = values[seed]["baseline"]
        for method, observed in values[seed].items():
            rows.append({"seed": seed, "method": method, **observed,
                         "delta_f1": observed["f1"] - baseline["f1"],
                         "delta_auc": observed["auc"] - baseline["auc"]})
    analyses = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        analyses[method] = {
            "f1": stats([row["delta_f1"] for row in selected], .02 if method == "dna_transform" else None),
            "auc": stats([row["delta_auc"] for row in selected], .005 if method == "dna_transform" else None),
        }
    primary = analyses["dna_transform"]
    passed = primary["f1"]["noninferiority_pass"] and primary["auc"]["noninferiority_pass"]
    summary = {"replication_id": config["replication_id"], "n_seeds": len(seeds),
               "analyses": analyses, "primary_noninferiority_pass": passed,
               "primary_conclusion": "DNA Transform establishes non-inferiority for both endpoints" if passed else "DNA Transform does not establish non-inferiority for both endpoints"}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
