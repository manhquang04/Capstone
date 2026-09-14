"""Calibrate DP distortion separately for frozen medium/stronger transforms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dna_encoder.transform_defense import DNATransformConfig, transform_update_array
from experiments.calibrate_rq1_dp_distortion import dp_relative_l2, floating_keys, relative_l2
from privacy.seed_manager import derive_seed

PHASE4 = ROOT / "artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z"
RAW_DEV = PHASE4 / "dna_level1_forward_attack_development_gate_c4_r8_i600_lr0p1_nonneg0p001"
GRID = [0.0001, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005, 0.00075, 0.001, 0.005, 0.01]
SEEDS = [11, 22, 33, 44, 55]
VARIANTS = {
    "medium": DNATransformConfig(256, 0.10, 0.85, 0.40),
    "stronger": DNATransformConfig(256, 0.12, 0.82, 0.35),
}
BASE_SEEDS = {"medium": 2036071981, "stronger": 1211543167}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, default=GRID)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    target = PHASE4 / "development_gate_targets.pt"
    groups = torch.load(target, map_location="cpu", weights_only=False)
    config = VARIANTS[args.variant]
    updates = []
    dna_rows = []
    for group_id in range(len(groups)):
        artifact = torch.load(RAW_DEV / f"group_{group_id}/realization_0/restart_0/baseline.pt", map_location="cpu", weights_only=False)
        observed = artifact["observed_raw_update"]
        keys = floating_keys(observed)
        transformed: OrderedDict[str, torch.Tensor] = OrderedDict()
        local_seed = derive_seed(BASE_SEEDS[args.variant], "phase4-dna-transform", target.name, group_id)
        for tensor_index, (name, value) in enumerate(observed.items()):
            if value.is_floating_point():
                array, _ = transform_update_array(value.detach().cpu().numpy(), replace(config, seed=local_seed), tensor_index)
                transformed[name] = torch.from_numpy(array)
            else:
                transformed[name] = value.clone()
        distortion = relative_l2(observed, transformed, keys)
        dna_rows.append({"variant": args.variant, "group_id": group_id, "relative_l2": distortion, "transform_seed": local_seed})
        updates.append((group_id, observed, keys))
    dna_target = float(np.median([row["relative_l2"] for row in dna_rows]))
    full = []
    for multiplier in args.multipliers:
        for calibration_seed in SEEDS:
            for group_id, observed, keys in updates:
                value, clip_factor, noise_std = dp_relative_l2(observed, keys, 100.0, multiplier, derive_seed(calibration_seed, group_id))
                full.append({"noise_multiplier": multiplier, "calibration_seed": calibration_seed, "group_id": group_id, "relative_l2": value, "clip_factor": clip_factor, "noise_std": noise_std})
    summaries = []
    for multiplier in args.multipliers:
        values = [row["relative_l2"] for row in full if row["noise_multiplier"] == multiplier]
        median = float(np.median(values)); relative_distance = abs(median - dna_target) / dna_target
        summaries.append({"noise_multiplier": multiplier, "median_relative_l2": median, "dna_median_relative_l2": dna_target, "absolute_distance": abs(median-dna_target), "relative_distance": relative_distance, "within_tolerance": relative_distance <= 0.05, "observations": len(values)})
    selected = min(summaries, key=lambda row: (row["absolute_distance"], row["noise_multiplier"]))
    _write_csv(output / "dna_target.csv", dna_rows); _write_csv(output / "full_grid.csv", full); _write_csv(output / "grid_summary.csv", summaries)
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "scope": "development-only variant distortion calibration", "variant": args.variant, "target": str(target), "target_sha256": _sha(target), "dna_config": config.__dict__, "dna_base_seed": BASE_SEEDS[args.variant], "dna_median_relative_l2": dna_target, "clip_norm": 100.0, "noise_multiplier_grid": args.multipliers, "calibration_seed_list": SEEDS, "matching_tolerance": 0.05, "selection_rule": "minimum absolute distance; smallest multiplier tie-break", "selected_candidate": selected, "match_succeeded": bool(selected["within_tolerance"]), "amendment": "protocols/amendments/2026-09-13_rq1_variant_and_utility_extension.md"}
    (output / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
