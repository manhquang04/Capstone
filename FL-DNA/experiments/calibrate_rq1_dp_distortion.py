"""Calibrate the Stage-1 DP-style comparator to DNA relative-L2 distortion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy.seed_manager import derive_seed
from experiments.run_phase4_simple_defense_attack import _dp_clipping_noise_plan


DEFAULT_PHASE4 = ROOT / "artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z"
DEFAULT_DNA_DIR = "dna_level1_forward_attack_development_gate_c4_r8_i600_lr0p1_nonneg0p001"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def floating_keys(update: dict[str, torch.Tensor]) -> list[str]:
    return [key for key, value in update.items() if value.is_floating_point()]


def norm(update: dict[str, torch.Tensor], keys: list[str]) -> float:
    return float(torch.stack([update[key].detach().double().square().sum() for key in keys]).sum().sqrt())


def relative_l2(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor], keys: list[str]) -> float:
    difference = {key: second[key] - first[key] for key in keys}
    return norm(difference, keys) / max(norm(first, keys), np.finfo(float).tiny)


def dp_relative_l2(
    update: dict[str, torch.Tensor],
    keys: list[str],
    clip_norm: float,
    multiplier: float,
    seed: int,
) -> tuple[float, float, float]:
    plan = _dp_clipping_noise_plan(update, keys, clip_norm, multiplier, seed)
    clip_factor = float(plan["clip_factor"])
    noise_std = float(plan["noise_std"])
    defended = {key: update[key] * clip_factor + plan["noise"][key] for key in keys}
    return relative_l2(update, defended, keys), clip_factor, noise_std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase4-root", type=Path, default=DEFAULT_PHASE4)
    parser.add_argument("--dna-dir", default=DEFAULT_DNA_DIR)
    parser.add_argument("--clip-norm", type=float, default=100.0)
    parser.add_argument("--multipliers", nargs="+", type=float, default=[0.0001, 0.0005, 0.001, 0.005, 0.01])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/rq1/stage1_distortion_calibration_20260912")
    args = parser.parse_args()

    phase4 = args.phase4_root.resolve()
    dna_dir = phase4 / args.dna_dir
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets_path = phase4 / "development_gate_targets.pt"
    targets = torch.load(targets_path, map_location="cpu", weights_only=False)
    dna_rows: list[dict[str, object]] = []
    updates: list[tuple[int, dict[str, torch.Tensor], list[str]]] = []
    for group_id in range(len(targets)):
        artifact_path = dna_dir / f"group_{group_id}/realization_0/restart_0/baseline.pt"
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        observed = artifact["observed_raw_update"]
        transmitted = artifact["transmitted_update"]
        keys = floating_keys(observed)
        distortion = relative_l2(observed, transmitted, keys)
        dna_rows.append({"group_id": group_id, "relative_l2": distortion, "artifact": str(artifact_path)})
        updates.append((group_id, observed, keys))

    dna_target = float(np.median([float(row["relative_l2"]) for row in dna_rows]))
    rows: list[dict[str, object]] = []
    for multiplier in args.multipliers:
        for calibration_seed in args.seeds:
            for group_id, observed, keys in updates:
                value, clip_factor, noise_std = dp_relative_l2(
                    observed, keys, args.clip_norm, multiplier, derive_seed(calibration_seed, group_id)
                )
                rows.append(
                    {
                        "noise_multiplier": multiplier,
                        "calibration_seed": calibration_seed,
                        "group_id": group_id,
                        "relative_l2": value,
                        "clip_factor": clip_factor,
                        "noise_std": noise_std,
                    }
                )

    summaries = []
    for multiplier in args.multipliers:
        values = [float(row["relative_l2"]) for row in rows if row["noise_multiplier"] == multiplier]
        median = float(np.median(values))
        relative_distance = abs(median - dna_target) / max(dna_target, np.finfo(float).tiny)
        summaries.append(
            {
                "noise_multiplier": multiplier,
                "median_relative_l2": median,
                "dna_median_relative_l2": dna_target,
                "absolute_distance": abs(median - dna_target),
                "relative_distance": relative_distance,
                "within_tolerance": relative_distance <= args.tolerance,
                "observations": len(values),
            }
        )
    selected = min(summaries, key=lambda row: (float(row["absolute_distance"]), float(row["noise_multiplier"])))
    matched = bool(selected["within_tolerance"])

    for name, payload in (("full_grid.csv", rows), ("grid_summary.csv", summaries), ("dna_target.csv", dna_rows)):
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload[0]))
            writer.writeheader()
            writer.writerows(payload)
    report = {
        "schema_version": 1,
        "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "development-only RQ1 distortion calibration",
        "target_file": str(targets_path),
        "target_sha256": sha256(targets_path),
        "dna_artifact_dir": str(dna_dir),
        "dna_target_median_relative_l2": dna_target,
        "clip_norm": args.clip_norm,
        "noise_multiplier_grid": args.multipliers,
        "calibration_seed_list": args.seeds,
        "matching_tolerance": args.tolerance,
        "target_statistic": "median_relative_l2",
        "selection_rule": "minimum absolute distance; smallest multiplier tie-break",
        "selected_candidate": selected,
        "match_succeeded": matched,
        "full_grid_file": str(output / "full_grid.csv"),
        "amendment_required_before_confirmatory_freeze": not matched,
    }
    (output / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
