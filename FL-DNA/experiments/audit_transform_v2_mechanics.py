"""Mechanical audit for DNA Transform v2.

Step 3 only: rank/condition diagnostics and known-seed reconstruction error.
This script does not run FL utility, gradient inversion attacks, calibration,
or confirmatory experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dna_encoder.transform_defense_v2 import (
    DNATransformV2Config,
    materialize_projection_matrix_v2,
    transform_and_reconstruct_array_v2,
)


PHASE4 = ROOT / "artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z"
OBSERVED_DIR = PHASE4 / "dna_level1_forward_attack_development_gate_c4_r8_i600_lr0p1_nonneg0p001"
TARGET = PHASE4 / "development_gate_targets.pt"
BASE_SEED = 20260916

CONFIGS = [
    ("v2_ratio0p95_eta0", 0.95, 0.0),
    ("v2_ratio0p95_eta0p01", 0.95, 0.01),
    ("v2_ratio0p95_eta0p02", 0.95, 0.02),
    ("v2_ratio0p9_eta0", 0.90, 0.0),
    ("v2_ratio0p9_eta0p01", 0.90, 0.01),
    ("v2_ratio0p9_eta0p02", 0.90, 0.02),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def observed_update(group_id: int) -> dict[str, torch.Tensor]:
    path = OBSERVED_DIR / f"group_{group_id}/realization_0/restart_0/baseline.pt"
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    return artifact["observed_raw_update"]


def aggregate_group_error(update: dict[str, torch.Tensor], config: DNATransformV2Config, group_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total_original_sq = 0.0
    total_diff_sq = 0.0
    total_original_size = 0
    total_rank_upper = 0
    rank_loss_tensors = 0
    tensor_rows: list[dict[str, Any]] = []

    for tensor_index, (name, value) in enumerate(update.items()):
        if not value.is_floating_point():
            continue
        array = value.detach().cpu().numpy().astype("float32", copy=False)
        _, reconstructed, metadata, stats = transform_and_reconstruct_array_v2(
            array,
            config,
            tensor_index=tensor_index,
            quantization_seed=group_id,
        )
        original = array.reshape(-1).astype(np.float64)
        recovered = reconstructed.reshape(-1).astype(np.float64)
        diff = recovered - original
        original_sq = float(np.dot(original, original))
        diff_sq = float(np.dot(diff, diff))
        rank_upper = min(metadata.sketch_size, metadata.original_size)
        has_rank_loss = rank_upper < metadata.original_size

        total_original_sq += original_sq
        total_diff_sq += diff_sq
        total_original_size += metadata.original_size
        total_rank_upper += rank_upper
        rank_loss_tensors += int(has_rank_loss)
        tensor_rows.append(
            {
                "group_id": group_id,
                "config_id": config_id(config),
                "tensor_index": tensor_index,
                "tensor_name": name,
                "original_size": metadata.original_size,
                "padded_size": metadata.padded_size,
                "sketch_size": metadata.sketch_size,
                "rank_upper_bound": rank_upper,
                "rank_deficient": has_rank_loss,
                "relative_l2_error": math.sqrt(diff_sq) / max(math.sqrt(original_sq), np.finfo(float).tiny),
                "quantization_delta": metadata.quantization_delta,
            }
        )

    return (
        {
            "group_id": group_id,
            "config_id": config_id(config),
            "compression_ratio": config.compression_ratio,
            "quantization_eta": config.quantization_eta,
            "floating_tensors": len(tensor_rows),
            "rank_loss_tensors": rank_loss_tensors,
            "total_original_size": total_original_size,
            "total_rank_upper_bound": total_rank_upper,
            "rank_retained_upper_fraction": total_rank_upper / max(total_original_size, 1),
            "relative_l2_reconstruction_error": math.sqrt(total_diff_sq) / max(math.sqrt(total_original_sq), np.finfo(float).tiny),
        },
        tensor_rows,
    )


def config_id(config: DNATransformV2Config) -> str:
    ratio = f"{config.compression_ratio:g}".replace(".", "p")
    eta = f"{config.quantization_eta:g}".replace(".", "p")
    return f"v2_ratio{ratio}_eta{eta}"


def rank_matrix_checks() -> list[dict[str, Any]]:
    rows = []
    for original_size in (8, 13, 32, 64):
        for _, ratio, eta in CONFIGS:
            config = DNATransformV2Config(compression_ratio=ratio, quantization_eta=eta, seed=BASE_SEED)
            matrix, metadata = materialize_projection_matrix_v2(original_size, config)
            singular = np.linalg.svd(matrix, compute_uv=False)
            nonzero = singular[singular > 1e-10]
            rank = int(np.linalg.matrix_rank(matrix, tol=1e-10))
            rows.append(
                {
                    "original_size": original_size,
                    "config_id": config_id(config),
                    "padded_size": metadata.padded_size,
                    "sketch_size": metadata.sketch_size,
                    "rank": rank,
                    "rank_deficient": rank < original_size,
                    "nonzero_singular_min": float(nonzero.min()) if nonzero.size else 0.0,
                    "nonzero_singular_max": float(nonzero.max()) if nonzero.size else 0.0,
                    "nonzero_condition_number": float(nonzero.max() / nonzero.min()) if nonzero.size else math.inf,
                    "full_condition_number": math.inf if rank < original_size else float(singular.max() / singular.min()),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for cid in sorted({row["config_id"] for row in rows}):
        values = np.asarray([row["relative_l2_reconstruction_error"] for row in rows if row["config_id"] == cid], dtype=float)
        rank_values = np.asarray([row["rank_retained_upper_fraction"] for row in rows if row["config_id"] == cid], dtype=float)
        cfg_rows = [row for row in rows if row["config_id"] == cid]
        out.append(
            {
                "config_id": cid,
                "groups": len(values),
                "compression_ratio": cfg_rows[0]["compression_ratio"],
                "quantization_eta": cfg_rows[0]["quantization_eta"],
                "mean_relative_l2_reconstruction_error": float(values.mean()),
                "median_relative_l2_reconstruction_error": float(np.median(values)),
                "min_relative_l2_reconstruction_error": float(values.min()),
                "max_relative_l2_reconstruction_error": float(values.max()),
                "mean_rank_retained_upper_fraction": float(rank_values.mean()),
                "all_groups_rank_loss": bool(all(row["rank_loss_tensors"] > 0 for row in cfg_rows)),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/dna_transform_v2/step3_mechanics_20260916")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target_groups = torch.load(TARGET, map_location="cpu", weights_only=False)
    group_count = len(target_groups)

    group_rows: list[dict[str, Any]] = []
    tensor_rows: list[dict[str, Any]] = []
    for label, ratio, eta in CONFIGS:
        config = DNATransformV2Config(compression_ratio=ratio, quantization_eta=eta, seed=BASE_SEED)
        if config_id(config) != label:
            raise AssertionError((label, config_id(config)))
        for group_id in range(group_count):
            group_row, tensor_detail = aggregate_group_error(observed_update(group_id), config, group_id)
            group_rows.append(group_row)
            tensor_rows.extend(tensor_detail)

    matrix_rows = rank_matrix_checks()
    summary_rows = summarize(group_rows)
    write_csv(output / "group_reconstruction_errors.csv", group_rows)
    write_csv(output / "tensor_rank_and_error_detail.csv", tensor_rows)
    write_csv(output / "small_matrix_rank_checks.csv", matrix_rows)
    write_csv(output / "summary.csv", summary_rows)

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "DNA Transform v2 Step 3 mechanical audit only",
        "development_target": str(TARGET.relative_to(ROOT)),
        "development_target_sha256": sha256(TARGET),
        "observed_update_source": str(OBSERVED_DIR.relative_to(ROOT)),
        "groups": group_count,
        "base_seed": BASE_SEED,
        "configs": [
            {"config_id": label, "compression_ratio": ratio, "quantization_eta": eta}
            for label, ratio, eta in CONFIGS
        ],
        "rank_interpretation": (
            "R has rank at most k.  The nonzero singular values of the ideal row "
            "sketch are well-conditioned, but the lifted operator R^T R is rank "
            "deficient and therefore has infinite full condition number whenever k < d."
        ),
        "v1_level2_reference_relative_error": 3.41e-08,
        "summary": summary_rows,
        "gate": {
            "rank_loss_observed": all(row["all_groups_rank_loss"] for row in summary_rows),
            "known_seed_reconstruction_error_exceeds_v1": all(
                row["median_relative_l2_reconstruction_error"] > 1e-4 for row in summary_rows
            ),
        },
    }
    report["gate"]["pass"] = bool(
        report["gate"]["rank_loss_observed"]
        and report["gate"]["known_seed_reconstruction_error_exceeds_v1"]
    )
    (output / "mechanical_audit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
