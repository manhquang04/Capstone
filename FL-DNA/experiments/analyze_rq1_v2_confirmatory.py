"""Analyze the frozen RQ1-v2 confirmatory family."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta, binomtest, t


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.inversion_metrics import reconstruction_metrics


BRANCHES = ("raw", "dna_v2", "dp_v2")


def _best(paths: list[Path], method: str):
    rows = []
    for path in paths:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("method") == method:
            rows.append((float(artifact["best_objective"]), str(path), path, artifact))
    if not rows:
        raise RuntimeError(f"no {method} artifact")
    return min(rows, key=lambda row: (row[0], row[1]))[2:]


def _metrics(artifact: dict) -> dict:
    return reconstruction_metrics(
        artifact["original"].detach().cpu().numpy(),
        artifact.get("aligned", artifact["reconstruction"]).detach().cpu().numpy(),
    )


def _ci(values: list[float]) -> list[float]:
    data = np.asarray(values, dtype=float)
    mean = float(data.mean())
    half = float(t.ppf(0.975, len(data) - 1) * data.std(ddof=1) / math.sqrt(len(data))) if len(data) > 1 else 0.0
    return [mean - half, mean + half]


def _gate(values: list[float]) -> dict:
    data = np.asarray(values, dtype=float)
    wins = int((data < 0).sum())
    p_value = float(binomtest(wins, len(data), 0.5, alternative="greater").pvalue)
    return {
        "n": len(data),
        "wins": wins,
        "mean_difference": float(data.mean()),
        "median_difference": float(np.median(data)),
        "one_sided_sign_p": p_value,
        "pass": bool(data.mean() < 0 and np.median(data) < 0 and p_value < 0.05),
    }


def _report_path(root: Path, branch: str) -> Path:
    patterns = {
        "raw": "**/harddiff_reparam_report.json",
        "dna_v2": "**/dna_v2_iht_attack_report.json",
        "dp_v2": "**/simple_defense_attack_report.json",
    }
    matches = list(root.glob(patterns[branch]))
    if not matches:
        matches = list(root.glob("**/*_report.json"))
    if not matches:
        raise RuntimeError(f"no report json under {root}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    groups = int(config["target"]["groups"])
    threshold = float(config["statistics"]["tie_threshold_mse"])
    rows = []
    control = {branch: {"prior": [], "zero": []} for branch in BRANCHES}
    for group in range(groups):
        selected = {}
        for branch in BRANCHES:
            root = args.run_dir / branch / f"group_{group}_run"
            artifact_path, artifact = _best(list(root.glob("**/baseline.pt")), "baseline")
            _, zero = _best(list(root.glob("**/zero_update.pt")), "zero_update")
            report = json.loads(_report_path(root, branch).read_text(encoding="utf-8"))
            summary = report["group_summary"][0]
            control[branch]["prior"].append(float(summary["objective_minus_prior_fraud_mse"]))
            control[branch]["zero"].append(float(summary["objective_minus_zero_fraud_mse"]))
            metric = _metrics(artifact)
            selected[branch] = (artifact, metric)
            rows.append(
                {
                    "group_id": group,
                    "branch": branch,
                    "selected_artifact": str(artifact_path),
                    "selection_objective": float(artifact["best_objective"]),
                    **metric,
                }
            )
        ids = [list(map(int, selected[branch][0]["source_ids"])) for branch in BRANCHES]
        if not ids[0] == ids[1] == ids[2]:
            raise AssertionError(f"source mismatch {group}")

    gates = {branch: {name: _gate(control[branch][name]) for name in ("prior", "zero")} for branch in BRANCHES}
    for branch in gates:
        gates[branch]["pass"] = gates[branch]["prior"]["pass"] and gates[branch]["zero"]["pass"]

    lookup = {(row["group_id"], row["branch"]): row for row in rows}
    paired = []
    for group in range(groups):
        raw = lookup[(group, "raw")]
        dna = lookup[(group, "dna_v2")]
        dp = lookup[(group, "dp_v2")]
        difference = dna["feature_mse"] - dp["feature_mse"]
        paired.append(
            {
                "group_id": group,
                "mse_raw": raw["feature_mse"],
                "mse_dna_v2": dna["feature_mse"],
                "mse_dp_v2": dp["feature_mse"],
                "delta_dna_v2_raw": dna["feature_mse"] - raw["feature_mse"],
                "delta_dp_v2_raw": dp["feature_mse"] - raw["feature_mse"],
                "D_dna_v2_minus_dp_v2": difference,
                "tie": abs(difference) <= threshold,
                "psnr_raw": raw["psnr"],
                "psnr_dna_v2": dna["psnr"],
                "psnr_dp_v2": dp["psnr"],
                "ssim_raw": raw["ssim"],
                "ssim_dna_v2": dna["ssim"],
                "ssim_dp_v2": dp["ssim"],
            }
        )
    non_tied = [row for row in paired if not row["tie"]]
    wins = sum(row["D_dna_v2_minus_dp_v2"] > threshold for row in non_tied)
    losses = len(non_tied) - wins
    p_value = float(binomtest(wins, len(non_tied), 0.5, alternative="greater").pvalue) if non_tied else 1.0
    win_ci = [
        0.0 if wins == 0 else float(beta.ppf(0.025, wins, losses + 1)),
        1.0 if losses == 0 else float(beta.ppf(0.975, wins + 1, losses)),
    ]
    primary_valid = gates["dna_v2"]["pass"] and gates["dp_v2"]["pass"]
    all_branch_gates_pass = primary_valid and gates["raw"]["pass"]
    differences = [row["D_dna_v2_minus_dp_v2"] for row in paired]
    summary = {
        "variant": config["variant"],
        "scope": "single frozen RQ1-v2 confirmatory execution",
        "n_targets": groups,
        "tie_threshold_mse": threshold,
        "branch_gates": gates,
        "primary": {
            "contrast": "DNA_TRANSFORM_V2_MINUS_DP_DISTORTION_MATCHED_V2",
            "valid": primary_valid,
            "all_branch_gates_pass": all_branch_gates_pass,
            "wins": wins,
            "losses": losses,
            "ties": groups - len(non_tied),
            "non_tied_n": len(non_tied),
            "win_probability": wins / len(non_tied) if non_tied else None,
            "win_probability_exact_ci95": win_ci,
            "one_sided_exact_sign_p": p_value,
            "reject_h0": bool(primary_valid and p_value < 0.05),
            "mean_mse_difference": float(np.mean(differences)),
            "mean_mse_difference_ci95": _ci(differences),
            "conclusion": (
                "DNA v2 shows significantly greater reconstruction resistance than distortion-matched clipping/noise v2"
                if primary_valid and p_value < 0.05
                else (
                    "No significant DNA v2 advantage over distortion-matched clipping/noise v2 under the frozen test"
                    if primary_valid
                    else "Primary contrast withheld because DNA v2 or DP v2 branch gate failed"
                )
            ),
        },
        "secondary": {
            "dna_v2_minus_raw_mse_ci95": _ci([row["delta_dna_v2_raw"] for row in paired]),
            "dp_v2_minus_raw_mse_ci95": _ci([row["delta_dp_v2_raw"] for row in paired]),
            "dna_v2_minus_dp_v2_psnr_ci95": _ci([row["psnr_dna_v2"] - row["psnr_dp_v2"] for row in paired]),
            "dna_v2_minus_dp_v2_ssim_ci95": _ci([row["ssim_dna_v2"] - row["ssim_dp_v2"] for row in paired]),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in (("selected_metrics.csv", rows), ("paired_metrics.csv", paired)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
