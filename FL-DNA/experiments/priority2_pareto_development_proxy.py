"""Priority 2 development-only Pareto/proxy sweep for RQ1.

This script does not create confirmatory targets and does not read old
post-hoc target outcomes for point selection.  It uses the frozen development
gate target/artifact pool to compute update-distortion proxies for a grid of
DNA Transform and DP-style clipping/noise settings, adds epsilon diagnostics
for DP-style settings, and attaches already-existing utility or development
attack measurements only when the source is explicitly known.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dna_encoder.transform_defense import DNATransformConfig, transform_update_array
from experiments.calibrate_rq1_dp_distortion import (
    DEFAULT_PHASE4,
    dp_relative_l2,
    floating_keys,
    relative_l2,
)
from experiments.dp_update_accounting import alpha_grid, epsilon_from_rdp
from privacy.seed_manager import derive_seed


RAW_DEV_DIR = "harddiff_reparam_development_standard_nonneg_0p001"
DNA_DEV_DIR = "dna_level1_forward_attack_development_gate_c4_r8_i600_lr0p1_nonneg0p001"
DP_LEGACY_DEV_DIR = (
    "simple_defense_attack_clipping_noise_mc_development_gate_"
    "r8_i600_lr0p1_nonneg0p001_rel0p1_mc100"
)
OBSERVED_UPDATE_DIR = DNA_DEV_DIR

DNA_GRID: list[dict[str, Any]] = [
    {"point_id": "dna_very_light", "mix_ratio": 0.03, "keep_ratio": 0.95, "shrink_factor": 0.70, "base_seed": 91827364},
    {"point_id": "dna_current", "mix_ratio": 0.05, "keep_ratio": 0.90, "shrink_factor": 0.50, "base_seed": 477885591},
    {"point_id": "dna_conservative", "mix_ratio": 0.08, "keep_ratio": 0.88, "shrink_factor": 0.45, "base_seed": 477885591},
    {"point_id": "dna_medium", "mix_ratio": 0.10, "keep_ratio": 0.85, "shrink_factor": 0.40, "base_seed": 2036071981},
    {"point_id": "dna_stronger", "mix_ratio": 0.12, "keep_ratio": 0.82, "shrink_factor": 0.35, "base_seed": 1211543167},
    {"point_id": "dna_extra_strong", "mix_ratio": 0.16, "keep_ratio": 0.78, "shrink_factor": 0.30, "base_seed": 64291357},
]

DP_GRID = [0.00001, 0.00005, 0.0001, 0.0002, 0.00025, 0.0003, 0.0004, 0.0005, 0.001, 0.005, 0.01]
CALIBRATION_SEEDS = [11, 22, 33, 44, 55]
CLIP_NORM = 100.0
DELTA = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_development_updates(phase4_root: Path) -> list[tuple[int, dict[str, torch.Tensor], list[str]]]:
    targets_path = phase4_root / "development_gate_targets.pt"
    groups = torch.load(targets_path, map_location="cpu", weights_only=False)
    updates: list[tuple[int, dict[str, torch.Tensor], list[str]]] = []
    for group_id in range(len(groups)):
        artifact_path = (
            phase4_root
            / OBSERVED_UPDATE_DIR
            / f"group_{group_id}/realization_0/restart_0/baseline.pt"
        )
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        observed = artifact["observed_raw_update"]
        updates.append((group_id, observed, floating_keys(observed)))
    return updates


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }


def final_round_metrics(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    payload = load_json(path)
    rounds = payload.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        return None
    last = rounds[-1]
    return {
        "f1": float(last["f1_score"]),
        "auc": float(last["auc_roc"]),
        "relative_l2": float(last["dna_transform_relative_l2_delta"])
        if "dna_transform_relative_l2_delta" in last
        else math.nan,
    }


def utility_lookup() -> dict[str, dict[str, Any]]:
    baseline = final_round_metrics(ROOT / "results/fraud/baseline_metrics.json")
    out: dict[str, dict[str, Any]] = {}
    if baseline is None:
        return out
    sources = {
        "dna_current": ROOT / "results/fraud/dna_transform_metrics.json",
        "dna_conservative": ROOT / "results/fraud/dna_transform_conservative_metrics.json",
        "dna_medium": ROOT / "results/fraud/dna_transform_medium_metrics.json",
        "dna_stronger": ROOT / "results/fraud/dna_transform_stronger_metrics.json",
        "dp_0.0001": ROOT / "results/fraud/dp_utility_metrics.json",
        "dp_0.001": ROOT / "results/fraud/dp_mild_metrics.json",
        "dp_0.005": ROOT / "results/fraud/dp_metrics.json",
    }
    for point_id, path in sources.items():
        metrics = final_round_metrics(path)
        if metrics is None:
            continue
        out[point_id] = {
            "utility_source": str(path.relative_to(ROOT)),
            "f1_delta_vs_baseline": metrics["f1"] - baseline["f1"],
            "auc_delta_vs_baseline": metrics["auc"] - baseline["auc"],
            "historical_single_seed_f1": metrics["f1"],
            "historical_single_seed_auc": metrics["auc"],
        }
    rq2_summary = ROOT / "results/rq2/confirmatory_20260913/rq2_summary.json"
    if rq2_summary.is_file():
        summary = load_json(rq2_summary)
        dp = summary.get("analyses", {}).get("dp_0.00025", {})
        if dp:
            out["dp_0.00025"] = {
                "utility_source": str(rq2_summary.relative_to(ROOT)),
                "f1_delta_vs_baseline": float(dp["f1"]["mean"]),
                "auc_delta_vs_baseline": float(dp["auc"]["mean"]),
                "historical_single_seed_f1": math.nan,
                "historical_single_seed_auc": math.nan,
            }
    return out


def mean_fraud_mse_from_group_summary(path: Path) -> float | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    key = "objective_best_fraud_mse"
    if not rows or key not in rows[0]:
        return None
    return float(np.mean([float(row[key]) for row in rows]))


def measured_reconstruction_lookup(phase4_root: Path) -> dict[str, dict[str, Any]]:
    raw = mean_fraud_mse_from_group_summary(phase4_root / RAW_DEV_DIR / "harddiff_reparam_group_summary.csv")
    if raw is None:
        return {}
    out: dict[str, dict[str, Any]] = {
        "raw_development": {
            "reconstruction_source": str((phase4_root / RAW_DEV_DIR / "harddiff_reparam_group_summary.csv").relative_to(ROOT)),
            "mean_feature_mse": raw,
            "feature_mse_delta_vs_raw": 0.0,
        }
    }
    dna = mean_fraud_mse_from_group_summary(phase4_root / DNA_DEV_DIR / "dna_level1_forward_attack_group_summary.csv")
    if dna is not None:
        out["dna_conservative"] = {
            "reconstruction_source": str((phase4_root / DNA_DEV_DIR / "dna_level1_forward_attack_group_summary.csv").relative_to(ROOT)),
            "mean_feature_mse": dna,
            "feature_mse_delta_vs_raw": dna - raw,
        }
    dp_legacy = mean_fraud_mse_from_group_summary(phase4_root / DP_LEGACY_DEV_DIR / "simple_defense_attack_group_summary.csv")
    if dp_legacy is not None:
        out["dp_legacy_target_rel_l2_0.1_mc100"] = {
            "reconstruction_source": str((phase4_root / DP_LEGACY_DEV_DIR / "simple_defense_attack_group_summary.csv").relative_to(ROOT)),
            "mean_feature_mse": dp_legacy,
            "feature_mse_delta_vs_raw": dp_legacy - raw,
        }
    return out


def dp_epsilon(noise_multiplier: float) -> float:
    result = epsilon_from_rdp(
        noise_multiplier=noise_multiplier,
        sensitivity_ratio=1.0,
        compositions=1,
        delta=DELTA,
        orders=alpha_grid(),
    )
    return float(result["epsilon"])


def build_rows(phase4_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    updates = load_development_updates(phase4_root)
    utilities = utility_lookup()
    reconstruction = measured_reconstruction_lookup(phase4_root)

    rows: list[dict[str, Any]] = []
    dna_detail_rows: list[dict[str, Any]] = []
    dp_detail_rows: list[dict[str, Any]] = []

    for cfg in DNA_GRID:
        values: list[float] = []
        for group_id, observed, keys in updates:
            transformed: OrderedDict[str, torch.Tensor] = OrderedDict()
            seed = derive_seed(int(cfg["base_seed"]), "priority2-pareto-dna-transform", "development_gate_targets.pt", group_id)
            transform_config = DNATransformConfig(
                block_size=256,
                mix_ratio=float(cfg["mix_ratio"]),
                keep_ratio=float(cfg["keep_ratio"]),
                shrink_factor=float(cfg["shrink_factor"]),
                seed=seed,
            )
            for tensor_index, (name, value) in enumerate(observed.items()):
                if value.is_floating_point():
                    array, _ = transform_update_array(value.detach().cpu().numpy(), replace(transform_config, seed=seed), tensor_index)
                    transformed[name] = torch.from_numpy(array)
                else:
                    transformed[name] = value.clone()
            value = relative_l2(observed, transformed, keys)
            values.append(value)
            dna_detail_rows.append({"point_id": cfg["point_id"], "group_id": group_id, "relative_l2": value, "seed": seed})
        stats = summarize(values)
        point_id = str(cfg["point_id"])
        row = {
            "family": "DNA",
            "point_id": point_id,
            "mix_ratio": cfg["mix_ratio"],
            "keep_ratio": cfg["keep_ratio"],
            "shrink_factor": cfg["shrink_factor"],
            "noise_multiplier": "",
            "median_relative_l2": stats["median"],
            "mean_relative_l2": stats["mean"],
            "epsilon_delta_1e_5_single_release": "",
            "epsilon_interpretation": "",
            "selection_eligible_after_supervisor_review": True,
            "measurement_status": "DISTORTION_PROXY_MEASURED",
        }
        row.update(utilities.get(point_id, {"utility_source": "MISSING_UTILITY_MEASUREMENT", "f1_delta_vs_baseline": "", "auc_delta_vs_baseline": ""}))
        row.update(reconstruction.get(point_id, {"reconstruction_source": "MISSING_ATTACK_MEASUREMENT", "mean_feature_mse": "", "feature_mse_delta_vs_raw": ""}))
        rows.append(row)

    for multiplier in DP_GRID:
        values = []
        for calibration_seed in CALIBRATION_SEEDS:
            for group_id, observed, keys in updates:
                value, clip_factor, noise_std = dp_relative_l2(
                    observed,
                    keys,
                    CLIP_NORM,
                    multiplier,
                    derive_seed(calibration_seed, "priority2-pareto-dp", group_id),
                )
                values.append(value)
                dp_detail_rows.append(
                    {
                        "point_id": f"dp_{multiplier:g}",
                        "noise_multiplier": multiplier,
                        "calibration_seed": calibration_seed,
                        "group_id": group_id,
                        "relative_l2": value,
                        "clip_factor": clip_factor,
                        "noise_std": noise_std,
                    }
                )
        stats = summarize(values)
        point_id = f"dp_{multiplier:g}"
        row = {
            "family": "DP_STYLE_CLIPPING_NOISE",
            "point_id": point_id,
            "mix_ratio": "",
            "keep_ratio": "",
            "shrink_factor": "",
            "noise_multiplier": multiplier,
            "median_relative_l2": stats["median"],
            "mean_relative_l2": stats["mean"],
            "epsilon_delta_1e_5_single_release": dp_epsilon(multiplier),
            "epsilon_interpretation": "weak_formal_accounting_large_epsilon",
            "selection_eligible_after_supervisor_review": True,
            "measurement_status": "DISTORTION_PROXY_AND_EPSILON_MEASURED",
        }
        row.update(utilities.get(point_id, {"utility_source": "MISSING_UTILITY_MEASUREMENT", "f1_delta_vs_baseline": "", "auc_delta_vs_baseline": ""}))
        row.update(reconstruction.get(point_id, {"reconstruction_source": "MISSING_ATTACK_MEASUREMENT", "mean_feature_mse": "", "feature_mse_delta_vs_raw": ""}))
        rows.append(row)

    legacy = reconstruction.get("dp_legacy_target_rel_l2_0.1_mc100")
    if legacy is not None:
        rows.append(
            {
                "family": "DP_STYLE_CLIPPING_NOISE",
                "point_id": "dp_legacy_target_rel_l2_0.1_mc100",
                "mix_ratio": "",
                "keep_ratio": "",
                "shrink_factor": "",
                "noise_multiplier": "",
                "median_relative_l2": 0.1,
                "mean_relative_l2": 0.1,
                "epsilon_delta_1e_5_single_release": "",
                "epsilon_interpretation": "legacy_target_relative_l2_not_a_noise_multiplier",
                "selection_eligible_after_supervisor_review": False,
                "measurement_status": "DEVELOPMENT_ATTACK_MEASURED_BUT_LEGACY_PARAMETERIZATION",
                "utility_source": "MISSING_UTILITY_MEASUREMENT",
                "f1_delta_vs_baseline": "",
                "auc_delta_vs_baseline": "",
                **legacy,
            }
        )

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Priority 2 development-only Pareto/proxy sweep; not confirmatory",
        "phase4_root": str(phase4_root),
        "development_target": str((phase4_root / "development_gate_targets.pt").relative_to(ROOT)),
        "development_target_sha256": sha256(phase4_root / "development_gate_targets.pt"),
        "observed_update_source": OBSERVED_UPDATE_DIR,
        "dna_grid": DNA_GRID,
        "dp_grid": DP_GRID,
        "dp_calibration_seeds": CALIBRATION_SEEDS,
        "clip_norm": CLIP_NORM,
        "epsilon_delta": DELTA,
        "post_hoc_policy": "No post-hoc target outcomes are used for selection; old target sets may only be used for source-disjointness checks.",
        "limitation": "Most new grid points have distortion proxy and epsilon only; reconstruction attack measurements require a separate development attack sweep before scientific selection by reconstruction resistance.",
    }
    metadata["detail_rows"] = {
        "dna": dna_detail_rows,
        "dp": dp_detail_rows,
    }
    return rows, metadata


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        path.with_suffix(".txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for family, marker in [("DNA", "o"), ("DP_STYLE_CLIPPING_NOISE", "s")]:
        subset = [row for row in rows if row["family"] == family and row["selection_eligible_after_supervisor_review"]]
        xs = [float(row["median_relative_l2"]) for row in subset]
        ys = []
        for row in subset:
            value = row.get("feature_mse_delta_vs_raw")
            ys.append(float(value) if value not in ("", None) else math.nan)
        ax.scatter(xs, ys, label=family, marker=marker)
        for row, x, y in zip(subset, xs, ys):
            label = str(row["point_id"]).replace("dna_", "D:").replace("dp_", "P:")
            ax.annotate(label, (x, y if not math.isnan(y) else 0.0), fontsize=7, alpha=0.8)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Median relative-L2 update distortion (development)")
    ax.set_ylabel("Feature-MSE delta vs raw where measured; 0 placeholder if missing")
    ax.set_title("Priority 2 RQ1 development Pareto/proxy sweep")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase4-root", type=Path, default=DEFAULT_PHASE4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/rq1/priority2_pareto_development_20260916")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, metadata = build_rows(args.phase4_root.resolve())
    write_csv(output / "pareto_points.csv", rows)
    write_csv(output / "dna_relative_l2_detail.csv", metadata["detail_rows"]["dna"])
    write_csv(output / "dp_relative_l2_detail.csv", metadata["detail_rows"]["dp"])
    serializable_metadata = dict(metadata)
    del serializable_metadata["detail_rows"]
    (output / "pareto_report.json").write_text(json.dumps({"metadata": serializable_metadata, "points": rows}, indent=2) + "\n", encoding="utf-8")
    write_plot(output / "pareto_proxy.png", rows)
    print(json.dumps({"output_dir": str(output), "points": len(rows), "metadata": serializable_metadata}, indent=2))


if __name__ == "__main__":
    main()
