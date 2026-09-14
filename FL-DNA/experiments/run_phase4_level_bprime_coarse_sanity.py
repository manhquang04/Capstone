"""Level-B' narrow architecture uncertainty coarse-screening sanity check.

Level-B' fixes the architecture pattern and training hyperparameters to the
Mức-A values, then searches only over a narrow 3x3x3 hidden-width grid around
the deployed bounded architecture.
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
    CandidateMLP,
    PROJECTION_INPUT_DIM,
    PROJECTION_OUTPUT_DIM,
    PROJECTION_SEED,
    SURROGATE_INIT_SEED,
    safe_l2_normalize,
    signed_hadamard_jl,
)
from experiments.run_phase3_adam_ladder import capture
from experiments.run_phase3_full_client import checksum, dump
from experiments.run_phase4_harddiff_reparam_for_misselected import (
    _decode_harddiff,
    _feature_distribution,
    _initial,
    _nonnegative_penalty,
)
from experiments.run_phase4_level_b_runtime_probe import flatten_delta_for_objective
from privacy.seed_manager import derive_seed


ROOT = Path(__file__).resolve().parents[1]
PHASE4_ROOT = ROOT / "artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z"
LAYER_1 = [102, 128, 154]
LAYER_2 = [51, 64, 77]
LAYER_3 = [26, 32, 38]
TRUE_WIDTHS = (128, 64, 32)


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


def _width_grid() -> list[tuple[int, int, int]]:
    return [(a, b, c) for a in LAYER_1 for b in LAYER_2 for c in LAYER_3]


def _make_model(widths: tuple[int, int, int], input_dim: int) -> CandidateMLP:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(SURROGATE_INIT_SEED, "level-bprime-widths", *widths))
        model = CandidateMLP(
            input_dim=input_dim,
            hidden_widths=list(widths),
            batch_norm=True,
            dropout=[0.3, 0.2, 0.1],
        )
    return model.train()


def _parameter_count(widths: tuple[int, int, int], input_dim: int) -> int:
    return sum(parameter.numel() for parameter in _make_model(widths, input_dim).parameters())


def _complexity_factor(parameter_count: int, reference_parameter_count: float) -> float:
    return 1.0 + 0.05 * max(0.0, math.log(parameter_count / reference_parameter_count))


def _projected_loss(delta: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    vector = flatten_delta_for_objective(delta)
    projected = safe_l2_normalize(signed_hadamard_jl(vector))
    return (projected - target).square().mean()


def _run_width_tuple(
    widths: tuple[int, int, int],
    x: torch.Tensor,
    y: torch.Tensor,
    batches: list[slice],
    rng_state: torch.Tensor,
    target: torch.Tensor,
    distribution,
    args: argparse.Namespace,
) -> dict[str, object]:
    meta = distribution[0]
    criterion = common.BinaryFocalLoss(alpha=0.95, gamma=2.0)
    best_raw = float("inf")
    best_step = 0
    best_restart = 0
    started = time.perf_counter()
    for restart in range(args.restarts):
        model = _make_model(widths, x.shape[1])
        initial = _initial(
            x.shape,
            derive_seed(PROJECTION_SEED, "level-bprime-initial", *widths, restart, args.init_mode),
            args.init_mode,
            distribution,
        )
        latent = initial.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=args.attack_lr)
        for step in range(args.iterations + 1):
            reconstruction = _decode_harddiff(latent, meta)
            delta = simulate(model, criterion, reconstruction, y, batches, rng_state, lr=0.001)
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
    return {
        "width_1": widths[0],
        "width_2": widths[1],
        "width_3": widths[2],
        "width_tuple": f"{widths[0]},{widths[1]},{widths[2]}",
        "is_true_width_tuple": widths == TRUE_WIDTHS,
        "raw_objective": best_raw,
        "best_restart": best_restart,
        "best_step": best_step,
        "seconds": time.perf_counter() - started,
    }


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
        output_dir = args.phase4_root / datetime.now(timezone.utc).strftime("level_bprime_coarse_sanity_%Y%m%dT%H%M%S%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)

    groups = torch.load(args.phase4_root / args.target_file, weights_only=False)
    group = groups[args.group_index]
    _, _, x, y, batches, rng_state, observed = capture(
        group,
        derive_seed(PROJECTION_SEED, "level_bprime_coarse_sanity", args.group_index),
        batch_size=4,
    )
    target = safe_l2_normalize(signed_hadamard_jl(flatten_delta_for_objective(observed))).detach()
    distribution = _feature_distribution()
    widths = _width_grid()
    parameter_counts = {width: _parameter_count(width, x.shape[1]) for width in widths}
    sorted_counts = sorted(parameter_counts.values())
    reference_parameter_count = float(sorted_counts[len(sorted_counts) // 2])

    rows: list[dict[str, object]] = []
    for width in widths:
        row = _run_width_tuple(width, x, y, batches, rng_state, target, distribution, args)
        count = parameter_counts[width]
        factor = _complexity_factor(count, reference_parameter_count)
        row.update(
            {
                "parameter_count": count,
                "complexity_factor": factor,
                "complexity_adjusted_objective": float(row["raw_objective"]) * factor,
            }
        )
        rows.append(row)

    ranked = sorted(rows, key=lambda row: row["complexity_adjusted_objective"])
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    true_row = next(row for row in ranked if row["is_true_width_tuple"])
    report = {
        "protocol": {
            "purpose": "level_bprime_coarse_screening_sanity_only",
            "attacker_model": "Level-B' narrow architecture uncertainty",
            "known_architecture_pattern": "3 hidden layers + BatchNorm + Dropout",
            "width_grid": {"layer_1": LAYER_1, "layer_2": LAYER_2, "layer_3": LAYER_3},
            "true_width_tuple": TRUE_WIDTHS,
            "fixed_local_lr": 0.001,
            "fixed_loss": {"loss_id": "focal_main", "alpha": 0.95, "gamma": 2.0},
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
            "reference_parameter_count": reference_parameter_count,
        },
        "ranked_width_tuples": ranked,
        "summary": {
            "true_width_rank": int(true_row["rank"]),
            "true_width_in_top3": int(true_row["rank"]) <= 3,
            "gate": "PASS" if int(true_row["rank"]) <= 3 else "FAIL",
            "seconds": time.perf_counter() - start,
            "max_rss_mb": _rss_mb(),
        },
    }
    _write_csv(output_dir / "level_bprime_coarse_sanity_ranked_widths.csv", ranked)
    dump(output_dir / "level_bprime_coarse_sanity_report.json", report)
    print(json.dumps(report["summary"], indent=2, allow_nan=False))
    print(json.dumps(ranked[:10], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
