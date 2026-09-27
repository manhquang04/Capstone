"""Priority 16 v1-medium clean-vector DNA-vs-DP probe.

This is a cheap n=8 development probe.  It optimizes candidates directly on the
trainable-parameter-only defended signal for DNA and DP branches, then performs a
head-to-head clean-vector MSE comparison.  It does not run controls, n=24, or
confirmatory experiments.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.local_update import simulate
from dna_encoder.transform_defense import DNATransformConfig
from experiments import fraud_fl_common as common
from experiments.phase3_bounded_validation import _align_for_evaluation, update_objective
from experiments.run_phase4_dna_level1_forward_attack import (
    _apply_surrogate_realization_torch,
    _surrogate_plan_from_state,
    _transmit_observed,
)
from experiments.run_phase4_dna_v2_sketch_space_attack import (
    _candidate_sketches,
    _observed_sketches,
)
from experiments.run_phase4_harddiff_reparam_for_misselected import (
    _decode_harddiff,
    _feature_distribution,
    _initial,
    _max_balance_residual,
    _nonnegative_penalty,
)
from experiments.run_phase4_simple_defense_attack import (
    _apply_candidate_defense,
    _apply_observed_defense,
    _plan as simple_defense_plan,
)
from experiments.run_phase4_dna_v2_iht_attack import _capture_for_iht as capture_paysim
from experiments.run_phase3_full_client import dump, score
from privacy.seed_manager import derive_seed


AMENDMENT = "protocols/amendments/2026-09-27_v1_medium_priority14_style_dp_head_to_head.md"

CELLS = {
    "v1_medium_vs_dp_0p000315": {
        "dna_kind": "v1",
        "dna_config": dict(block_size=256, mix_ratio=0.10, keep_ratio=0.85, shrink_factor=0.40, seed=681958327),
        "dp": dict(clip_norm=100.0, noise_multiplier=0.000315, mc_noise_samples=100, defense_seed=314159265),
    },
}


def _trainable_names(model: torch.nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def _filter_trainable(state: dict[str, torch.Tensor], names: list[str]) -> dict[str, torch.Tensor]:
    missing = [name for name in names if name not in state]
    if missing:
        raise KeyError(f"missing trainable keys: {missing[:5]}")
    return {name: state[name] for name in names}


def _mse(candidate: dict[str, torch.Tensor], signal: dict[str, torch.Tensor]) -> float:
    keys = [key for key, value in signal.items() if value.is_floating_point()]
    parts = [(candidate[key].detach().double() - signal[key].detach().double()).reshape(-1).square() for key in keys]
    return float(torch.cat(parts).mean())


def _dna_payload(observed: dict[str, torch.Tensor], cell: dict, group: int, restart: int):
    if cell["dna_kind"] == "v1":
        config = DNATransformConfig(**cell["dna_config"])
        transmitted = _transmit_observed(observed, config)
        plan = _surrogate_plan_from_state(observed, config, group, restart)
        return transmitted, ("v1", plan)
    if cell["dna_kind"] == "v2":
        config = SimpleNamespace(**cell["dna_config"])
        observed_sketch_payload = _observed_sketches(observed, config, group)
        sketch = {key: item["sketch"].to(dtype=observed[key].dtype) for key, item in observed_sketch_payload.items()}
        return sketch, ("v2", observed_sketch_payload, config)
    raise ValueError(cell["dna_kind"])


def _candidate_dna(delta: dict[str, torch.Tensor], payload):
    kind = payload[0]
    if kind == "v1":
        return _apply_surrogate_realization_torch(delta, payload[1])
    if kind == "v2":
        _, plan, config = payload
        args = argparse.Namespace(
            compression_ratio=float(config.compression_ratio),
            quantization_eta=float(config.quantization_eta),
            v2_base_seed=int(config.seed),
            ste_quantization=True,
        )
        return _candidate_sketches(delta, plan, args)
    raise ValueError(kind)


def _dp_payload(observed: dict[str, torch.Tensor], dp: dict, group: int):
    keys = [key for key, value in observed.items() if value.is_floating_point()]
    args = SimpleNamespace(
        target_rel_l2=0.1,
        clip_factor=0.95,
        clip_norm=float(dp["clip_norm"]),
        noise_multiplier=float(dp["noise_multiplier"]),
        defense_seed=int(dp["defense_seed"]),
        mc_noise_samples=int(dp["mc_noise_samples"]),
        topk_temperature=1e-3,
    )
    plan = simple_defense_plan("clipping_noise_mc", observed, keys, group, args)
    signal = _apply_observed_defense(observed, keys, "clipping_noise_mc", plan)
    return signal, (keys, plan)


def _candidate_dp(delta: dict[str, torch.Tensor], payload):
    keys, plan = payload
    return _apply_candidate_defense(delta, keys, "clipping_noise_mc", plan)


def _optimize_branch(job: dict, branch: str, signal: dict[str, torch.Tensor], payload, model, criterion, original, labels, batches, rng, trainable, distribution, folder: Path) -> dict:
    meta = distribution[0]
    initial = _initial(
        original.shape,
        derive_seed(job["seed"], "priority14-init", job["cell"], branch, job["group"], job["restart"]),
        "standard",
        distribution,
    )
    latent = initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=0.1)
    best = float("inf")
    best_step = 0
    best_latent = latent.detach().clone()
    history = []
    for step in range(601):
        reconstruction = _decode_harddiff(latent, meta)
        delta_full = simulate(model, criterion, reconstruction, labels, batches, rng)
        delta = _filter_trainable(delta_full, trainable)
        candidate = _candidate_dna(delta, payload) if branch == "dna" else _candidate_dp(delta, payload)
        penalty = 0.001 * _nonnegative_penalty(latent, meta)
        loss = update_objective(candidate, signal, list(signal), reference=signal, mode="balanced_tensor") + penalty
        value = float(loss.detach())
        history.append(value)
        if value < best:
            best = value
            best_step = step
            best_latent = latent.detach().clone()
        if step < 600:
            (gradient,) = torch.autograd.grad(loss, latent)
            optimizer.zero_grad()
            latent.grad = gradient
            optimizer.step()
    reconstruction = _decode_harddiff(best_latent, meta).detach()
    aligned = _align_for_evaluation(original, reconstruction, labels)
    delta_full = simulate(model, criterion, reconstruction, labels, batches, rng)
    delta = _filter_trainable(delta_full, trainable)
    candidate = _candidate_dna(delta, payload) if branch == "dna" else _candidate_dp(delta, payload)
    mse = _mse(candidate, signal)
    metrics = score(original, aligned, labels, meta, folder / f"{branch}.csv")
    artifact = {
        "method": branch,
        "cell": job["cell"],
        "group_id": job["group"],
        "restart": job["restart"],
        "source_ids": job["source_ids"],
        "original": original,
        "labels": labels,
        "initial": initial,
        "reconstruction": reconstruction,
        "aligned": aligned,
        "best_objective": best,
        "best_step": best_step,
        "history": history,
        "clean_vector_mse": mse,
        "vector_scope": "trainable_parameters_only",
        "excluded_buffers": "all model.named_buffers() including BatchNorm running stats",
        "trainable_keys": trainable,
        "max_balance_residual": _max_balance_residual(reconstruction, meta),
        "feature_metrics": metrics,
    }
    torch.save(artifact, folder / f"{branch}.pt")
    return artifact


def _job(job: dict) -> dict:
    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    target = Path(job["target"])
    groups = torch.load(target, map_location="cpu", weights_only=False)
    group = groups[job["group"]]
    distribution = _feature_distribution()
    local_seed = derive_seed(job["seed"], "priority14-local", job["cell"], job["group"])
    model, criterion, original, labels, batches, rng, observed_full = capture_paysim(group, local_seed, 4)
    trainable = _trainable_names(model)
    observed = _filter_trainable(observed_full, trainable)
    cell = CELLS[job["cell"]]
    dna_signal, dna_payload = _dna_payload(observed, cell, job["group"], job["restart"])
    dp_signal, dp_payload = _dp_payload(observed, cell["dp"], job["group"])
    folder = Path(job["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    job_context = {
        "seed": job["seed"],
        "cell": job["cell"],
        "group": job["group"],
        "restart": job["restart"],
        "source_ids": group["source_ids"],
    }
    dna = _optimize_branch(job_context, "dna", dna_signal, dna_payload, model, criterion, original, labels, batches, rng, trainable, distribution, folder)
    dp = _optimize_branch(job_context, "dp", dp_signal, dp_payload, model, criterion, original, labels, batches, rng, trainable, distribution, folder)
    result = {
        "cell": job["cell"],
        "group": job["group"],
        "restart": job["restart"],
        "source_ids": group["source_ids"],
        "dna_mse": dna["clean_vector_mse"],
        "dp_mse": dp["clean_vector_mse"],
        "D_dna_minus_dp": dna["clean_vector_mse"] - dp["clean_vector_mse"],
        "status": "SUCCESS",
    }
    dump(folder / "result.json", result)
    return result


def _summarize(records: list[dict]) -> dict:
    selected = []
    for group in sorted({row["group"] for row in records}):
        rows = [row for row in records if row["group"] == group]
        # Select independently best DNA and best DP across restarts by their own MSE.
        dna_best = min(rows, key=lambda row: row["dna_mse"])
        dp_best = min(rows, key=lambda row: row["dp_mse"])
        D = dna_best["dna_mse"] - dp_best["dp_mse"]
        selected.append(
            {
                "group_id": group,
                "dna_restart": dna_best["restart"],
                "dp_restart": dp_best["restart"],
                "dna_mse": dna_best["dna_mse"],
                "dp_mse": dp_best["dp_mse"],
                "D_dna_minus_dp": D,
                "winner": "DNA" if D > 0 else ("DP" if D < 0 else "TIE"),
            }
        )
    wins = sum(row["D_dna_minus_dp"] > 0 for row in selected)
    losses = sum(row["D_dna_minus_dp"] < 0 for row in selected)
    ties = sum(row["D_dna_minus_dp"] == 0 for row in selected)
    n = wins + losses
    return {
        "selected": selected,
        "zero_threshold": {
            "dna_wins": wins,
            "dna_losses_dp_wins": losses,
            "exact_ties": ties,
            "non_tied_n": n,
            "dna_advantage_p": float(binomtest(wins, n, 0.5, alternative="greater").pvalue) if n else 1.0,
            "dp_advantage_p": float(binomtest(losses, n, 0.5, alternative="greater").pvalue) if n else 1.0,
            "mean_D_dna_minus_dp": float(np.mean([row["D_dna_minus_dp"] for row in selected])),
            "median_D_dna_minus_dp": float(np.median([row["D_dna_minus_dp"] for row in selected])),
        },
    }


def execute(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": args.amendment,
        "target": str(args.target),
        "cell": args.cell,
        "groups": args.groups,
        "restarts": args.restarts,
        "jobs": args.groups * args.restarts,
        "workers": args.workers,
        "torch_num_threads": 1,
        "vector_scope": "trainable_parameters_only",
        "attacker": "simple_balanced_tensor_raw_lift_clean_vector",
        "no_escalation_authorized": False,
        "protocol_scope": "v1_medium_priority14_style_escalation_chain",
    }
    dump(output / "execution_manifest.json", manifest)
    jobs = []
    for group in range(args.groups):
        for restart in range(args.restarts):
            jobs.append(
                {
                    "cell": args.cell,
                    "group": group,
                    "restart": restart,
                    "seed": args.seed,
                    "target": str(args.target.resolve()),
                    "folder": str(output / args.cell / f"group_{group}" / f"restart_{restart}"),
                }
            )
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_job, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            print(json.dumps({key: row[key] for key in ("cell", "group", "restart", "status")}), flush=True)
    summary = {**manifest, "completed_jobs": len(records), "records": sorted(records, key=lambda r: (r["group"], r["restart"])), "summary": _summarize(records)}
    dump(output / "priority14_probe_report.json", summary)
    print(json.dumps(summary["summary"]["zero_threshold"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=tuple(CELLS), required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amendment", default=AMENDMENT)
    args = parser.parse_args()
    if args.workers > 9:
        raise ValueError("workers must be <= 9")
    execute(args)


if __name__ == "__main__":
    main()
