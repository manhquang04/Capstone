"""Analyze RQ2 Stage-1 utility matching and conservative paired power."""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import chi2, nct, t


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ["dna_lossless", "dna_transform"]
ENDPOINTS = {"f1": "f1_score", "auc": "auc_roc"}
MARGINS = {"f1": 0.02, "auc": 0.005}
UTILITY_DP_METHODS = {"dp_0.0001", "dp_0.0005", "dp_0.001", "dp_0.005", "dp_0.01"}
CONTEXTUAL_DP_METHOD = "dp_0.00025"


def upper_variance(values: np.ndarray, confidence: float = 0.95) -> float:
    if values.size < 2:
        raise ValueError("at least two paired values are required")
    variance = float(np.var(values, ddof=1))
    if variance == 0.0:
        return 0.0
    df = values.size - 1
    return float(df * variance / chi2.ppf(1.0 - confidence, df))


def required_n(sigma: float, margin: float, alpha: float, power: float, maximum_search: int = 10_000) -> int:
    if sigma <= 0.0:
        return 2
    for n in range(2, maximum_search + 1):
        df = n - 1
        critical = t.ppf(1.0 - alpha, df)
        noncentrality = margin * math.sqrt(n) / sigma
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            achieved = nct.sf(critical, df, noncentrality)
        if achieved >= power:
            return n
    raise RuntimeError("required n exceeds search range")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, default=ROOT / "artifacts/rq2/stage1_development_20260912")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--variance-confidence", type=float, default=0.95)
    parser.add_argument("--maximum-feasible-seeds", type=int, default=30)
    args = parser.parse_args()
    batch = args.batch_dir.resolve()

    metrics: dict[int, dict[str, dict[str, float]]] = {}
    for path in sorted(batch.glob("seed_*/**/metrics.json")):
        seed = int(path.parts[-3].split("_")[1])
        method = path.parent.name
        payload = json.loads(path.read_text())
        if int(payload["config"]["seed"]) != seed:
            raise ValueError(f"seed mismatch in {path}")
        final = payload["rounds"][-1]
        if int(final["round"]) != 50:
            raise ValueError(f"incomplete 50-round run: {path}")
        metrics.setdefault(seed, {})[method] = {name: float(final[key]) for name, key in ENDPOINTS.items()}

    expected_methods = {"baseline", "dna_lossless", "dna_transform", CONTEXTUAL_DP_METHOD, *UTILITY_DP_METHODS}
    for seed, values in metrics.items():
        if set(values) != expected_methods:
            raise ValueError(f"seed {seed} has methods {sorted(values)}, expected {sorted(expected_methods)}")

    paired_rows = []
    for seed in sorted(metrics):
        baseline = metrics[seed]["baseline"]
        for method, values in metrics[seed].items():
            paired_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "f1": values["f1"],
                    "auc": values["auc"],
                    "delta_f1_vs_baseline": values["f1"] - baseline["f1"],
                    "delta_auc_vs_baseline": values["auc"] - baseline["auc"],
                }
            )

    power_rows = []
    for method in PRIMARY:
        for endpoint in ENDPOINTS:
            differences = np.asarray(
                [row[f"delta_{endpoint}_vs_baseline"] for row in paired_rows if row["method"] == method],
                dtype=float,
            )
            sample_variance = float(np.var(differences, ddof=1))
            variance_upper = upper_variance(differences, args.variance_confidence)
            need = required_n(math.sqrt(variance_upper), MARGINS[endpoint], args.alpha, args.power)
            mean = float(np.mean(differences))
            se = float(np.std(differences, ddof=1) / math.sqrt(len(differences)))
            half_width = float(t.ppf(0.975, len(differences) - 1) * se)
            power_rows.append(
                {
                    "method": method,
                    "endpoint": endpoint,
                    "n_pilot": len(differences),
                    "mean_paired_difference": mean,
                    "mean_paired_difference_ci95_low": mean - half_width,
                    "mean_paired_difference_ci95_high": mean + half_width,
                    "sd_paired_difference": float(np.std(differences, ddof=1)),
                    "sample_variance": sample_variance,
                    "upper_95pct_variance": variance_upper,
                    "margin": MARGINS[endpoint],
                    "required_n": need,
                }
            )

    dna_degradation = {
        endpoint: float(np.mean([row[f"delta_{endpoint}_vs_baseline"] for row in paired_rows if row["method"] == "dna_transform"]))
        for endpoint in ENDPOINTS
    }
    utility_rows = []
    for method in sorted(expected_methods):
        if method not in UTILITY_DP_METHODS:
            continue
        mean_delta = {
            endpoint: float(np.mean([row[f"delta_{endpoint}_vs_baseline"] for row in paired_rows if row["method"] == method]))
            for endpoint in ENDPOINTS
        }
        f1_deviation = abs(mean_delta["f1"] - dna_degradation["f1"])
        auc_deviation = abs(mean_delta["auc"] - dna_degradation["auc"])
        utility_rows.append(
            {
                "method": method,
                "noise_multiplier": float(method.removeprefix("dp_")),
                "mean_delta_f1": mean_delta["f1"],
                "mean_delta_auc": mean_delta["auc"],
                "dna_mean_delta_f1": dna_degradation["f1"],
                "dna_mean_delta_auc": dna_degradation["auc"],
                "absolute_f1_deviation": f1_deviation,
                "absolute_auc_deviation": auc_deviation,
                "distance": max(f1_deviation / 0.01, auc_deviation / 0.0025),
                "within_f1_tolerance": f1_deviation <= 0.01,
                "within_auc_tolerance": auc_deviation <= 0.0025,
                "eligible": f1_deviation <= 0.01 and auc_deviation <= 0.0025,
            }
        )
    eligible = [row for row in utility_rows if row["eligible"]]
    selected = min(eligible, key=lambda row: (float(row["distance"]), float(row["noise_multiplier"]))) if eligible else None
    overall_required = max(int(row["required_n"]) for row in power_rows)

    for name, rows in (("paired_metrics.csv", paired_rows), ("power_analysis.csv", power_rows), ("utility_grid.csv", utility_rows)):
        with (batch / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "RQ2 Stage-1 development only",
        "seeds": sorted(metrics),
        "utility_matching": {
            "f1_tolerance": 0.01,
            "auc_tolerance": 0.0025,
            "distance": "max normalized absolute F1/AUC deviation",
            "selected_candidate": selected,
            "status": "MATCHED" if selected else "NOT_FOUND",
        },
        "distortion_matched_context": {
            "method": CONTEXTUAL_DP_METHOD,
            "selection_basis": "RQ1 relative-L2 calibration only; excluded from frozen utility candidate grid",
            "mean_delta_f1": float(np.mean([row["delta_f1_vs_baseline"] for row in paired_rows if row["method"] == CONTEXTUAL_DP_METHOD])),
            "mean_delta_auc": float(np.mean([row["delta_auc_vs_baseline"] for row in paired_rows if row["method"] == CONTEXTUAL_DP_METHOD])),
        },
        "power": {
            "method": "one-sided paired noninferiority noncentral-t approximation",
            "variance": "one-sided 95% chi-square upper confidence bound",
            "alpha": args.alpha,
            "target_power": args.power,
            "required_confirmatory_seed_count": overall_required,
            "maximum_feasible_seed_count": args.maximum_feasible_seeds,
            "within_resource_ceiling": overall_required <= args.maximum_feasible_seeds,
            "rows": power_rows,
        },
    }
    (batch / "stage1_analysis_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
