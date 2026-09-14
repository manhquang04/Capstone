"""Runtime probe for one Phase 4 Level-B projected-observation grid cell.

This is not a final Level-B attack. It measures whether the locked projected
observation path is computationally feasible before running the 96-cell grid.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None

from attacks.local_update import simulate
from experiments import fraud_fl_common as common
from experiments.analyze_phase4_level_b_projection_sanity import (
    ARCHITECTURES,
    CandidateMLP,
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
from privacy.seed_manager import derive_seed


ROOT = Path(__file__).resolve().parents[1]
PHASE4_ROOT = ROOT / "artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z"


def _rss_mb() -> float | None:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KiB.
    return float(usage / (1024 * 1024) if usage > 10_000_000 else usage / 1024)


def make_surrogate(arch_id: str, input_dim: int) -> CandidateMLP:
    config = ARCHITECTURES[arch_id]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(SURROGATE_INIT_SEED, "architecture", arch_id))
        model = CandidateMLP(
            input_dim=input_dim,
            hidden_widths=config["hidden_widths"],
            batch_norm=config["batch_norm"],
            dropout=config["dropout"],
        )
    return model.train()


def flatten_delta_for_objective(delta: dict[str, torch.Tensor]) -> torch.Tensor:
    parts = [value.reshape(-1).float() for key, value in sorted(delta.items()) if torch.is_floating_point(value)]
    if not parts:
        raise ValueError("delta has no floating tensors")
    return torch.cat(parts)


def projected_objective(delta: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    vector = flatten_delta_for_objective(delta)
    projected = safe_l2_normalize(signed_hadamard_jl(vector))
    return (projected - target).square().mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase4-root", type=Path, default=PHASE4_ROOT)
    parser.add_argument("--target-file", default="development_gate_targets.pt")
    parser.add_argument("--group-index", type=int, default=0)
    parser.add_argument("--architecture-id", default="B3", choices=sorted(ARCHITECTURES))
    parser.add_argument("--local-lr", type=float, default=0.001)
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
        output_dir = args.phase4_root / datetime.now(timezone.utc).strftime("level_b_runtime_probe_%Y%m%dT%H%M%S%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)

    groups = torch.load(args.phase4_root / args.target_file, weights_only=False)
    group = groups[args.group_index]
    _, _, x, y, batches, rng_state, observed = capture(
        group,
        derive_seed(PROJECTION_SEED, "level_b_runtime_probe", args.group_index),
        batch_size=4,
    )
    target = safe_l2_normalize(signed_hadamard_jl(flatten_delta_for_objective(observed))).detach()
    distribution = _feature_distribution()
    meta = distribution[0]

    rows = []
    for restart in range(args.restarts):
        model = make_surrogate(args.architecture_id, input_dim=x.shape[1])
        criterion = common.BinaryFocalLoss()
        initial = _initial(
            x.shape,
            derive_seed(PROJECTION_SEED, "level_b_probe_initial", args.architecture_id, restart, args.init_mode),
            args.init_mode,
            distribution,
        )
        latent = initial.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=args.attack_lr)
        best = float("inf")
        best_step = 0
        history = []
        restart_start = time.perf_counter()
        for step in range(args.iterations + 1):
            reconstruction = _decode_harddiff(latent, meta)
            delta = simulate(model, criterion, reconstruction, y, batches, rng_state, lr=args.local_lr)
            gradient_loss = projected_objective(delta, target)
            penalty = _nonnegative_penalty(latent, meta)
            loss = gradient_loss + args.nonnegative_lambda * penalty
            value = float(loss.detach())
            history.append(value)
            if value < best:
                best = value
                best_step = step
            if step < args.iterations:
                (gradient,) = torch.autograd.grad(loss, latent)
                optimizer.zero_grad()
                latent.grad = gradient
                optimizer.step()
        rows.append(
            {
                "restart": restart,
                "best_objective": best,
                "best_step": best_step,
                "initial_objective": history[0],
                "final_objective": history[-1],
                "seconds": time.perf_counter() - restart_start,
            }
        )

    total_seconds = time.perf_counter() - start
    best_row = min(rows, key=lambda row: row["best_objective"])
    report = {
        "protocol": {
            "purpose": "runtime_probe_only",
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
            "architecture_id": args.architecture_id,
            "local_lr": args.local_lr,
            "iterations": args.iterations,
            "restarts": args.restarts,
            "attack_lr": args.attack_lr,
            "nonnegative_lambda": args.nonnegative_lambda,
            "init_mode": args.init_mode,
        },
        "rows": rows,
        "summary": {
            "best_restart": best_row["restart"],
            "best_objective": best_row["best_objective"],
            "total_seconds": total_seconds,
            "seconds_per_restart": total_seconds / args.restarts,
            "estimated_96_cell_coarse_seconds": total_seconds * 96.0,
            "estimated_96_cell_coarse_hours": total_seconds * 96.0 / 3600.0,
            "max_rss_mb": _rss_mb(),
            "status": "RUNTIME_PROBE_ONLY",
        },
    }
    dump(output_dir / "level_b_runtime_probe_report.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
