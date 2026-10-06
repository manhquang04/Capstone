"""Analyze Priority 27 C2 DP-variant utility grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.dp_update_accounting import alpha_grid, epsilon_from_rdp
from experiments.run_priority27_utility_grid import method_id


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def metric_path(seed: int, method: str, run_dirs: list[Path]) -> Path:
    for run_dir in run_dirs:
        path = run_dir / f"seed_{seed}" / method / "metrics.json"
        if path.exists():
            return path
    raise FileNotFoundError(f"missing metrics for seed={seed} method={method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extension-run-dir", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    run_dirs = [args.run_dir.resolve()]
    sigmas = [float(x) for x in config["initial_sigma_grid"]]
    if args.extension_run_dir:
        run_dirs.insert(0, args.extension_run_dir.resolve())
        sigmas += [float(x) for x in config["extension_sigma_grid"]]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    seeds = [int(x) for x in config["seeds"]]
    rounds = int(config["training"]["num_rounds"])
    orders = alpha_grid()
    rows = []
    summaries = []
    for variant, clip_spec in config["clip_specs"].items():
        effective_sensitivity = float(config["effective_sensitivities"][variant])
        for sigma in sigmas:
            mid = method_id(variant, sigma)
            deltas = {"f1": [], "auc": [], "pr_auc": []}
            for seed in seeds:
                base = load_final(metric_path(seed, "baseline", run_dirs), seed, rounds)
                dp = load_final(metric_path(seed, mid, run_dirs), seed, rounds)
                row = {
                    "seed": seed,
                    "variant": variant,
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
            eps_one = epsilon_from_rdp(sigma, 1.0, 1e-5, 1, orders)
            eps_50 = epsilon_from_rdp(sigma, 1.0, 1e-5, 50, orders)
            summaries.append(
                {
                    "variant": variant,
                    "method": mid,
                    "noise_multiplier": sigma,
                    "effective_sensitivity": effective_sensitivity,
                    "n_replicates": len(seeds),
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
    for variant in config["clip_specs"]:
        ordered = sorted([row for row in summaries if row["variant"] == variant], key=lambda row: row["noise_multiplier"])
        selections[variant] = {}
        for transform, info in config["matching_targets"].items():
            threshold = float(info["threshold_delta_f1"])
            eligible = [i for i, row in enumerate(ordered) if float(row["mean_delta_f1"]) >= threshold]
            idx = max(eligible) if eligible else 0
            selected = ordered[idx]
            next_row = ordered[idx + 1] if idx + 1 < len(ordered) else None
            bracketed = bool(eligible and next_row is not None and float(next_row["mean_delta_f1"]) < threshold)
            selections[variant][transform] = {
                "selected_noise_multiplier": selected["noise_multiplier"],
                "selected_method": selected["method"],
                "selected_mean_delta_f1": selected["mean_delta_f1"],
                "selected_sd_delta_f1": selected["sd_delta_f1"],
                "selected_mean_delta_auc": selected["mean_delta_auc"],
                "selected_mean_delta_pr_auc": selected["mean_delta_pr_auc"],
                "threshold_delta_f1": threshold,
                "bracketed": bracketed,
                "bracket_status": "BRACKETED" if bracketed else ("NOT_BRACKETED_SELECTED_LARGEST_GRID_POINT" if next_row is None else "NOT_BRACKETED_NEXT_GRID_POINT_STILL_ELIGIBLE"),
                "next_larger_noise_multiplier": None if next_row is None else next_row["noise_multiplier"],
                "next_larger_mean_delta_f1": None if next_row is None else next_row["mean_delta_f1"],
                "epsilon_one_release_delta_1e_minus_5": selected["epsilon_one_release_delta_1e_minus_5"],
                "epsilon_50_releases_delta_1e_minus_5": selected["epsilon_50_releases_delta_1e_minus_5"],
            }
    with (output / "per_seed_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "grid_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "run_dirs": [str(path.relative_to(ROOT)) for path in run_dirs],
        "grid": summaries,
        "matching_targets": config["matching_targets"],
        "selections": selections,
        "all_bracketed": all(sel["bracketed"] for by_transform in selections.values() for sel in by_transform.values()),
        "extension_required": any(not sel["bracketed"] for by_transform in selections.values() for sel in by_transform.values()),
    }
    (output / "utility_grid_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

