"""Primary paired RQ2 analysis; every one of the frozen189 results required."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import t
from experiments.priority32_multidataset import OUT, DATASETS, METHODS, SEEDS, write, sha


def main():
    rows, paired = [], []
    for dataset in DATASETS:
        values = {}
        for method in METHODS:
            values[method] = []
            for seed in SEEDS:
                path = OUT/"rq2_jobs"/dataset/method/f"seed_{seed}.json"
                doc = json.loads(path.read_text())
                assert doc["status"] == "COMPLETED" and doc["seed"] == seed and doc["dataset"] == dataset
                values[method].append(doc["test"])
        for method in METHODS[1:]:
            for endpoint, margin in (("f1", .02), ("auc_roc", .005), ("pr_auc", None)):
                baseline = np.array([v[endpoint] for v in values["baseline"]], dtype=np.float64)
                defended = np.array([v[endpoint] for v in values[method]], dtype=np.float64)
                delta = defended-baseline
                sd = float(np.std(delta, ddof=1))
                half = float(t.ppf(.975, 20)*sd/np.sqrt(21))
                mean = float(np.mean(delta))
                rows.append({"dataset": dataset, "method": method, "endpoint": endpoint, "n": 21,
                             "baseline_mean": float(baseline.mean()), "defended_mean": float(defended.mean()),
                             "mean_delta": mean, "sd_delta": sd, "ci95_lower": mean-half, "ci95_upper": mean+half,
                             "margin": margin, "gate": "DESCRIPTIVE" if margin is None else ("PASS" if mean-half > -margin else "FAIL")})
                for seed, b, d, v in zip(SEEDS, baseline, defended, delta):
                    paired.append({"dataset": dataset, "method": method, "endpoint": endpoint, "seed": seed,
                                   "baseline": float(b), "defended": float(d), "delta": float(v)})
    destination = OUT/"rq2_analysis"
    destination.mkdir(parents=True, exist_ok=True)
    for name, records in (("combined_summary.csv", rows), ("paired_differences.csv", paired)):
        with (destination/name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    write(destination/"summary.json", {"rows": rows, "jobs_required": 189,
          "overall": {f"{d}/{m}": "PASS" if all(r["gate"] == "PASS" for r in rows if r["dataset"] == d and r["method"] == m and r["endpoint"] != "pr_auc") else "FAIL" for d in DATASETS for m in METHODS[1:]}})
    print(json.dumps({"stage": 5, "summary_sha256": sha(destination/"summary.json")}), flush=True)


if __name__ == "__main__":
    main()
