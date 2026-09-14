"""Coarse-screening sanity check for Phase 4 Level-B.

This development-only check runs the locked 96-cell coarse grid on one group and
verifies whether the true bounded architecture (B3) remains in the top three
architectures after each architecture is represented by its own best
configuration.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None

from attacks.local_update import simulate
from experiments import fraud_fl_common as common
from experiments.analyze_phase4_level_b_projection_sanity import (
    ARCHITECTURES,
    PROJECTION_INPUT_DIM,
    PROJECTION_OUTPUT_DIM,
    PROJECTION_SEED,
    SURROGATE_INIT_SEED,
    signed_hadamard_jl,
    safe_l2_normalize,
)
from experiments.run_phase3_adam_ladder import capture
from experiments.run_phase3_full_client import checksum, dump
from experiments.run_phase4_harddiff_reparam_for_misselected import (
    _decode_harddiff,
    _feature_distribution,
    _initial,
    _nonnegative_penalty,
)
from experiments.run_phase4_level_b_runtime_probe import (
    flatten_delta_for_objective,
    make_surrogate,
)
from privacy.seed_manager import derive_seed


ROOT = Path(__file__).resolve().parents[1]
PHASE4_ROOT = ROOT / "artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z"
LOCAL_LR_GRID = [0.0005, 0.001, 0.002, 0.005]
LOSS_GRID = [
    {"loss_id": "focal_low", "alpha": 0.90, "gamma": 2.0},
    {"loss_id": "focal_main", "alpha": 0.95, "gamma": 2.0},
    {"loss_id": "focal_steep", "alpha": 0.95, "gamma": 1.5},
    {"loss_id": "weighted_bce", "alpha": None, "gamma": None},
]


def _rss_mb() -> float | None:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(usage / (1024 * 1024) if usage > 10_000_000 else usage / 1024)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _criterion(loss_config: dict[str, object], y: torch.Tensor) -> nn.Module:
    if loss_config["loss_id"] == "weighted_bce":
        positives = y.sum().clamp_min(1.0)
        negatives = (y.numel() - y.sum()).clamp_min(1.0)
        return nn.BCEWithLogitsLoss(pos_weight=(negatives / positives).reshape(()))
    return common.BinaryFocalLoss(alpha=float(loss_config["alpha"]), gamma=float(loss_config["gamma"]))


def _parameter_count(arch_id: str, input_dim: int) -> int:
    return sum(parameter.numel() for parameter in make_surrogate(arch_id, input_dim).parameters())


def _complexity_factor(parameter_count: int, reference_parameter_count: float) -> float:
    return 1.0 + 0.05 * max(0.0, math.log(parameter_count / reference_parameter_count))


def _projected_loss(delta: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    vector = flatten_delta_for_objective(delta)
    projected = safe_l2_normalize(signed_hadamard_jl(vector))
    return (projected - target).square().mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase4-root", type=Path, default=PHASE4_ROOT)
    parser.add_argument("--target-file", default="development_gate_targets.pt")
    parser.add_argument("--group-index", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--attack-lr", type=float, default=0.1)
    parser.add_argument("--nonnegative-lambda", type=float, default=0.001)
    parser.add_argument("--init-mode", choices=["standard", "plausible"], default="plausible")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    start = time.perf_counter()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.phase4_root / datetime.now(timezone.utc).strftime("level_b_coarse_sanity_%Y%m%dT%H%M%S%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)

    groups = torch.load(args.phase4_root / args.target_file, weights_only=False)
    group = groups[args.group_index]
    _, _, x, y, batches, rng_state, observed = capture(
        group,
        derive_seed(PROJECTION_SEED, "level_b_coarse_sanity", args.group_index),
        batch_size=4,
    )
    target = safe_l2_normalize(signed_hadamard_jl(flatten_delta_for_objective(observed))).detach()
    distribution = _feature_distribution()
    meta = distribution[0]
    parameter_counts = {arch_id: _parameter_count(arch_id, x.shape[1]) for arch_id in ARCHITECTURES}
    sorted_counts = sorted(parameter_counts.values())
    reference_parameter_count = 0.5 * (sorted_counts[2] + sorted_counts[3])
    complexity = {
        arch_id: _complexity_factor(count, reference_parameter_count)
        for arch_id, count in parameter_counts.items()
    }

    rows: list[dict[str, object]] = []
    for arch_id in sorted(ARCHITECTURES):
        for local_lr in LOCAL_LR_GRID:
            for loss_config in LOSS_GRID:
                cell_start = time.perf_counter()
                best_raw = float("inf")
                best_step = 0
                best_restart = 0
                for restart in range(args.restarts):
                    model = make_surrogate(arch_id, input_dim=x.shape[1])
                    criterion = _criterion(loss_config, y)
                    initial = _initial(
                        x.shape,
                        derive_seed(
                            PROJECTION_SEED,
                            "level_b_coarse_initial",
                            arch_id,
                            local_lr,
                            loss_config["loss_id"],
                            restart,
                            args.init_mode,
                        ),
                        args.init_mode,
                        distribution,
                    )
                    latent = initial.clone().requires_grad_(True)
                    optimizer = torch.optim.Adam([latent], lr=args.attack_lr)
                    for step in range(args.iterations + 1):
                        reconstruction = _decode_harddiff(latent, meta)
                        delta = simulate(model, criterion, reconstruction, y, batches, rng_state, lr=local_lr)
                        gradient_loss = _projected_loss(delta, target)
                        penalty = _nonnegative_penalty(latent, meta)
                        raw_objective = gradient_loss + args.nonnegative_lambda * penalty
                        value = float(raw_objective.detach())
                        if value < best_raw:
                            best_raw = value
                            best_step = step
                            best_restart = restart
                        if step < args.iterations:
                            (gradient,) = torch.autograd.grad(raw_objective, latent)
                            optimizer.zero_grad()
                            latent.grad = gradient
                            optimizer.step()
                rows.append(
                    {
                        "architecture_id": arch_id,
                        "parameter_count": parameter_counts[arch_id],
                        "local_lr": local_lr,
                        "loss_id": loss_config["loss_id"],
                        "raw_objective": best_raw,
                        "complexity_factor": complexity[arch_id],
                        "complexity_adjusted_objective": best_raw * complexity[arch_id],
                        "best_restart": best_restart,
                        "best_step": best_step,
                        "seconds": time.perf_counter() - cell_start,
                    }
                )

    best_by_architecture = []
    for arch_id in sorted(ARCHITECTURES):
        best = min(
            (row for row in rows if row["architecture_id"] == arch_id),
            key=lambda row: row["complexity_adjusted_objective"],
        )
        best_by_architecture.append(dict(best))
    ranked_architectures = sorted(best_by_architecture, key=lambda row: row["complexity_adjusted_objective"])
    for rank, row in enumerate(ranked_architectures, start=1):
        row["architecture_rank"] = rank
    b3_rank = next(row["architecture_rank"] for row in ranked_architectures if row["architecture_id"] == "B3")
    report = {
        "protocol": {
            "purpose": "coarse_screening_sanity_only",
            "observation_mode": "public_projection_matrix",
            "projection_input_dim": PROJECTION_INPUT_DIM,
            "projection_output_dim": PROJECTION_OUTPUT_DIM,
            "projection_seed": PROJECTION_SEED,
            "projection_distribution": "signed_hadamard_jl",
            "projection_objective_normalization": "l2_normalize_projected_update",
            "surrogate_init_seed": SURROGATE_INIT_SEED,
            "target_file": args.target_file,
            "target_file_sha256": checksum(args.phase4_root / args.target_file),
            "group_index": args.group_index,
            "source_ids": group["source_ids"],
            "iterations": args.iterations,
            "restarts": args.restarts,
            "attack_lr": args.attack_lr,
            "nonnegative_lambda": args.nonnegative_lambda,
            "init_mode": args.init_mode,
            "local_lr_grid": LOCAL_LR_GRID,
            "loss_grid": LOSS_GRID,
            "reference_parameter_count": reference_parameter_count,
            "top3_rule": "best configuration per architecture by complexity_adjusted_objective",
        },
        "all_cells": rows,
        "ranked_architectures": ranked_architectures,
        "summary": {
            "b3_rank": b3_rank,
            "b3_in_top3": b3_rank <= 3,
            "gate": "PASS" if b3_rank <= 3 else "FAIL",
            "seconds": time.perf_counter() - start,
            "max_rss_mb": _rss_mb(),
        },
    }
    _write_csv(output_dir / "level_b_coarse_sanity_cells.csv", rows)
    _write_csv(output_dir / "level_b_coarse_sanity_ranked_architectures.csv", ranked_architectures)
    dump(output_dir / "level_b_coarse_sanity_report.json", report)
    print(json.dumps(report["summary"], indent=2, allow_nan=False))
    print(json.dumps(ranked_architectures, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
