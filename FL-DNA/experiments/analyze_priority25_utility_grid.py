"""Analyze Priority 25 utility-matched DP development grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.dp_update_accounting import alpha_grid, epsilon_from_rdp


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method_id(sigma: float) -> str:
    return f"dp_sigma_{sigma:.0e}".replace("-", "m").replace("+", "")


def load_final(path: Path, seed: int, rounds: int) -> dict[str, float]:
    payload = json.loads(path.read_text())
    final = payload["rounds"][-1]
    if int(payload["config"]["seed"]) != int(seed):
        raise ValueError(f"seed mismatch: {path}")
    if int(final["round"]) != int(rounds):
        raise ValueError(f"round mismatch: {path}")
    return {
        "f1": float(final["f1_score"]),
        "auc": float(final["auc_roc"]),
        "pr_auc": float(final["pr_auc"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    seeds = [int(seed) for seed in config["seeds"]]
    sigmas = [float(x) for x in config["dp"]["noise_multiplier_grid"]]
    rounds = int(config["training"]["num_rounds"])
    run_dir = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    orders = alpha_grid()

    rows = []
    summaries = []
    for sigma in sigmas:
        mid = method_id(sigma)
        deltas = {"f1": [], "auc": [], "pr_auc": []}
        for seed in seeds:
            base = load_final(run_dir / f"seed_{seed}" / "baseline" / "metrics.json", seed, rounds)
            dp = load_final(run_dir / f"seed_{seed}" / mid / "metrics.json", seed, rounds)
            row = {
                "seed": seed,
                "method": mid,
                "noise_multiplier": sigma,
                "baseline_f1": base["f1"],
                "dp_f1": dp["f1"],
                "delta_f1": dp["f1"] - base["f1"],
                "baseline_auc": base["auc"],
                "dp_auc": dp["auc"],
                "delta_auc": dp["auc"] - base["auc"],
                "baseline_pr_auc": base["pr_auc"],
                "dp_pr_auc": dp["pr_auc"],
                "delta_pr_auc": dp["pr_auc"] - base["pr_auc"],
            }
            rows.append(row)
            for key in deltas:
                deltas[key].append(row[f"delta_{key}"])
        eps_one = epsilon_from_rdp(
            noise_multiplier=sigma,
            sensitivity_ratio=1.0,
            delta=1e-5,
            compositions=1,
            orders=orders,
        )
        eps_50 = epsilon_from_rdp(
            noise_multiplier=sigma,
            sensitivity_ratio=1.0,
            delta=1e-5,
            compositions=50,
            orders=orders,
        )
        summaries.append(
            {
                "method": mid,
                "noise_multiplier": sigma,
                "clip_norm": float(config["dp"]["clip_norm"]),
                "mean_delta_f1": float(np.mean(deltas["f1"])),
                "sd_delta_f1": float(np.std(deltas["f1"], ddof=1)),
                "mean_delta_auc": float(np.mean(deltas["auc"])),
                "sd_delta_auc": float(np.std(deltas["auc"], ddof=1)),
                "mean_delta_pr_auc": float(np.mean(deltas["pr_auc"])),
                "sd_delta_pr_auc": float(np.std(deltas["pr_auc"], ddof=1)),
                "epsilon_one_release_delta_1e_minus_5": float(eps_one["epsilon"]),
                "epsilon_50_releases_delta_1e_minus_5": float(eps_50["epsilon"]),
            }
        )

    selections = {}
    for transform, info in config["matching_targets"].items():
        threshold = float(info["rq2_mean_delta_f1"]) - float(config["matching_rule"]["f1_slack"])
        eligible = [row for row in summaries if float(row["mean_delta_f1"]) >= threshold]
        if eligible:
            selected = max(eligible, key=lambda row: float(row["noise_multiplier"]))
            status = "MATCH_FOUND"
        else:
            selected = min(summaries, key=lambda row: float(row["noise_multiplier"]))
            status = "NO_GRID_POINT_WITHIN_RULE_USED_SMALLEST_SIGMA"
        selections[transform] = {
            "status": status,
            "threshold_delta_f1": threshold,
            "selected_method": selected["method"],
            "selected_noise_multiplier": selected["noise_multiplier"],
            "selected_clip_norm": selected["clip_norm"],
            "selected_mean_delta_f1": selected["mean_delta_f1"],
            "selected_mean_delta_auc": selected["mean_delta_auc"],
            "selected_mean_delta_pr_auc": selected["mean_delta_pr_auc"],
            "epsilon_one_release_delta_1e_minus_5": selected["epsilon_one_release_delta_1e_minus_5"],
            "epsilon_50_releases_delta_1e_minus_5": selected["epsilon_50_releases_delta_1e_minus_5"],
        }

    with (output / "per_seed_grid.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "grid_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "run_execution_summary_sha256": sha256(run_dir / "execution_summary.json"),
        "grid": summaries,
        "matching_rule": config["matching_rule"],
        "matching_targets": config["matching_targets"],
        "selections": selections,
    }
    (output / "utility_grid_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
