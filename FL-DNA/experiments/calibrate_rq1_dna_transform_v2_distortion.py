"""Calibrate DP distortion for DNA Transform v2 on development data only."""

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

from dna_encoder.transform_defense_v2 import DNATransformV2Config, transform_and_reconstruct_array_v2
from experiments.calibrate_rq1_dp_distortion import dp_relative_l2, floating_keys, relative_l2
from privacy.seed_manager import derive_seed


PHASE4 = ROOT / "artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z"
RAW_DEV = PHASE4 / "dna_level1_forward_attack_development_gate_c4_r8_i600_lr0p1_nonneg0p001"
TARGET = PHASE4 / "development_gate_targets.pt"
GRID = [0.00025, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
SEEDS = [11, 22, 33, 44, 55]
BASE_SEED = 20260916


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def v2_defended_update(
    observed: dict[str, torch.Tensor],
    config: DNATransformV2Config,
    group_id: int,
) -> OrderedDict[str, torch.Tensor]:
    defended: OrderedDict[str, torch.Tensor] = OrderedDict()
    for tensor_index, (name, value) in enumerate(observed.items()):
        if not value.is_floating_point():
            defended[name] = value.clone()
            continue
        _, reconstructed, _, _ = transform_and_reconstruct_array_v2(
            value.detach().cpu().numpy().astype(np.float32, copy=False),
            config,
            tensor_index=tensor_index,
            quantization_seed=group_id,
        )
        defended[name] = torch.from_numpy(reconstructed.copy()).to(dtype=value.dtype)
    return defended


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/dna_transform_v2/step5_distortion_calibration_20260916")
    parser.add_argument("--compression-ratio", type=float, default=0.95)
    parser.add_argument("--quantization-eta", type=float, default=0.01)
    parser.add_argument("--clip-norm", type=float, default=100.0)
    parser.add_argument("--multipliers", nargs="+", type=float, default=GRID)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    groups = torch.load(TARGET, map_location="cpu", weights_only=False)
    config = DNATransformV2Config(
        compression_ratio=args.compression_ratio,
        quantization_eta=args.quantization_eta,
        seed=BASE_SEED,
    )

    dna_rows: list[dict[str, object]] = []
    updates: list[tuple[int, dict[str, torch.Tensor], list[str]]] = []
    for group_id in range(len(groups)):
        artifact_path = RAW_DEV / f"group_{group_id}/realization_0/restart_0/baseline.pt"
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        observed = artifact["observed_raw_update"]
        keys = floating_keys(observed)
        defended = v2_defended_update(observed, config, group_id)
        distortion = relative_l2(observed, defended, keys)
        dna_rows.append(
            {
                "group_id": group_id,
                "relative_l2": distortion,
                "compression_ratio": args.compression_ratio,
                "quantization_eta": args.quantization_eta,
                "artifact": str(artifact_path),
            }
        )
        updates.append((group_id, observed, keys))

    dna_target = float(np.median([float(row["relative_l2"]) for row in dna_rows]))
    full: list[dict[str, object]] = []
    for multiplier in args.multipliers:
        for calibration_seed in args.seeds:
            for group_id, observed, keys in updates:
                value, clip_factor, noise_std = dp_relative_l2(
                    observed,
                    keys,
                    args.clip_norm,
                    multiplier,
                    derive_seed(calibration_seed, group_id),
                )
                full.append(
                    {
                        "noise_multiplier": multiplier,
                        "calibration_seed": calibration_seed,
                        "group_id": group_id,
                        "relative_l2": value,
                        "clip_factor": clip_factor,
                        "noise_std": noise_std,
                    }
                )

    summaries: list[dict[str, object]] = []
    for multiplier in args.multipliers:
        values = [float(row["relative_l2"]) for row in full if row["noise_multiplier"] == multiplier]
        median = float(np.median(values))
        absolute_distance = abs(median - dna_target)
        relative_distance = absolute_distance / max(dna_target, np.finfo(float).tiny)
        summaries.append(
            {
                "noise_multiplier": multiplier,
                "median_relative_l2": median,
                "dna_median_relative_l2": dna_target,
                "absolute_distance": absolute_distance,
                "relative_distance": relative_distance,
                "within_tolerance": relative_distance <= args.tolerance,
                "observations": len(values),
            }
        )
    selected = min(summaries, key=lambda row: (float(row["absolute_distance"]), float(row["noise_multiplier"])))

    write_csv(output / "dna_target.csv", dna_rows)
    write_csv(output / "full_grid.csv", full)
    write_csv(output / "grid_summary.csv", summaries)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "DNA Transform v2 Step 5 development-only DP distortion calibration",
        "amendment": "protocols/amendments/2026-09-16_dna_transform_v2_step5_development_calibration.md",
        "target": str(TARGET),
        "target_sha256": sha256(TARGET),
        "raw_development_artifact_dir": str(RAW_DEV),
        "v2_config": config.__dict__,
        "v2_base_seed": BASE_SEED,
        "dna_median_relative_l2": dna_target,
        "clip_norm": args.clip_norm,
        "noise_multiplier_grid": args.multipliers,
        "calibration_seed_list": args.seeds,
        "matching_tolerance": args.tolerance,
        "target_statistic": "median_relative_l2",
        "selection_rule": "minimum absolute distance; smallest multiplier tie-break",
        "selected_candidate": selected,
        "match_succeeded": bool(selected["within_tolerance"]),
    }
    (output / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
