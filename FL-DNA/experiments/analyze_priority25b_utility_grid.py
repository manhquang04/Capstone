"""Analyze Priority 25b bracketed utility-matched DP grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


def metric_path(seed: int, method: str, old_run: Path, new_run: Path) -> Path:
    new_path = new_run / f"seed_{seed}" / method / "metrics.json"
    if new_path.exists():
        return new_path
    old_path = old_run / f"seed_{seed}" / method / "metrics.json"
    if old_path.exists():
        return old_path
    raise FileNotFoundError(f"missing metrics for seed={seed} method={method}")


def summarize(config: dict, old_run: Path, new_run: Path, sigmas: list[float]) -> tuple[list[dict], list[dict]]:
    seeds = [int(x) for x in config["all_seeds"]]
    rounds = int(config["training"]["num_rounds"])
    orders = alpha_grid()
    rows = []
    summaries = []
    for sigma in sigmas:
        mid = method_id(sigma)
        deltas = {"f1": [], "auc": [], "pr_auc": []}
        for seed in seeds:
            base = load_final(metric_path(seed, "baseline", old_run, new_run), seed, rounds)
            dp = load_final(metric_path(seed, mid, old_run, new_run), seed, rounds)
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
        eps_one = epsilon_from_rdp(sigma, 1.0, 1e-5, 1, orders)
        eps_50 = epsilon_from_rdp(sigma, 1.0, 1e-5, 50, orders)
        summaries.append(
            {
                "method": mid,
                "noise_multiplier": sigma,
                "clip_norm": float(config["dp"]["clip_norm"]),
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
    return rows, summaries


def select(config: dict, summaries: list[dict]) -> dict:
    ordered = sorted(summaries, key=lambda row: float(row["noise_multiplier"]))
    selections = {}
    for transform, info in config["matching_targets"].items():
        threshold = float(info["threshold_delta_f1"])
        eligible_indices = [i for i, row in enumerate(ordered) if float(row["mean_delta_f1"]) >= threshold]
        if not eligible_indices:
            idx = 0
            status = "NO_ELIGIBLE_GRID_POINT_USED_SMALLEST_SIGMA"
        else:
            idx = max(eligible_indices)
            status = "MATCH_FOUND"
        selected = ordered[idx]
        next_row = ordered[idx + 1] if idx + 1 < len(ordered) else None
        bracketed = bool(
            status == "MATCH_FOUND"
            and next_row is not None
            and float(next_row["mean_delta_f1"]) < threshold
        )
        if status == "MATCH_FOUND" and next_row is None:
            bracket_status = "NOT_BRACKETED_SELECTED_LARGEST_GRID_POINT"
        elif status == "MATCH_FOUND" and not bracketed:
            bracket_status = "NOT_BRACKETED_NEXT_GRID_POINT_STILL_ELIGIBLE"
        elif bracketed:
            bracket_status = "BRACKETED"
        else:
            bracket_status = "NOT_BRACKETED_NO_ELIGIBLE_GRID_POINT"
        selections[transform] = {
            "status": status,
            "bracket_status": bracket_status,
            "bracketed": bracketed,
            "threshold_delta_f1": threshold,
            "selected_method": selected["method"],
            "selected_noise_multiplier": selected["noise_multiplier"],
            "selected_clip_norm": selected["clip_norm"],
            "selected_mean_delta_f1": selected["mean_delta_f1"],
            "selected_sd_delta_f1": selected["sd_delta_f1"],
            "selected_mean_delta_auc": selected["mean_delta_auc"],
            "selected_mean_delta_pr_auc": selected["mean_delta_pr_auc"],
            "epsilon_one_release_delta_1e_minus_5": selected["epsilon_one_release_delta_1e_minus_5"],
            "epsilon_50_releases_delta_1e_minus_5": selected["epsilon_50_releases_delta_1e_minus_5"],
            "next_larger_method": None if next_row is None else next_row["method"],
            "next_larger_noise_multiplier": None if next_row is None else next_row["noise_multiplier"],
            "next_larger_mean_delta_f1": None if next_row is None else next_row["mean_delta_f1"],
        }
    return selections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extension-run-dir", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    old_run = (ROOT / config["prior_priority25_run_dir"]).resolve()
    run_dir = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sigmas = [float(x) for x in config["dp"]["full_initial_noise_multiplier_grid"]]
    if args.extension_run_dir is not None:
        sigmas += [float(x) for x in config["dp"]["extension_noise_multiplier_grid"]]
        # Overlay extension metrics by copying lookup priority into a combined path list:
        # extension metrics are passed as the primary new_run after initial analysis rerun.
        # To keep the code simple, require caller to merge/symlink extension run into run-dir
        # or run the analyzer with a run-dir that contains all missing metrics.
        run_dir = args.extension_run_dir.resolve()
    rows, summaries = summarize(config, old_run, run_dir, sigmas)
    selections = select(config, summaries)
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
        "prior_priority25_run_dir": str(old_run.relative_to(ROOT)),
        "priority25b_run_dir": str(run_dir.relative_to(ROOT)),
        "priority25b_run_execution_summary_sha256": sha256(run_dir / "execution_summary.json"),
        "grid": summaries,
        "matching_rule": config["matching_rule"],
        "matching_targets": config["matching_targets"],
        "selections": selections,
        "all_bracketed": bool(all(item["bracketed"] for item in selections.values())),
        "extension_required": bool(any(not item["bracketed"] for item in selections.values())),
    }
    (output / "utility_grid_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
