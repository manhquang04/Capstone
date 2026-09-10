"""Fresh Phase 4 Level-1 surrogate-forward DNA attack.

This runner optimizes dummy data directly against the transmitted DNA update:

    L_balanced(M_r delta(candidate), transmitted_update)

where M_r is a fixed surrogate realization chosen by candidate id.  It does not
reuse the baseline candidate pool for optimization or selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from attacks.local_update import simulate
from dna_encoder.transform_defense import DNATransformConfig, transform_update_array
from experiments import fraud_fl_common as common
from experiments.analyze_phase4_dna_level1_surrogate_inversion import (
    _candidate_realization_matrix,
    _surrogate_seed,
)
from experiments.phase3_bounded_validation import _align_for_evaluation, update_objective
from experiments.run_phase3_adam_ladder import capture
from experiments.run_phase3_full_client import checksum, dump, score
from experiments.run_phase4_harddiff_reparam_for_misselected import (
    _decode_harddiff,
    _feature_distribution,
    _initial,
    _max_balance_residual,
    _nonnegative_penalty,
)
from privacy.seed_manager import derive_seed, generate_run_seed


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sign_tail(wins, total):
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / 2**total if total else 1.0


def _surrogate_plan_from_state(state_delta, config, group_id, realization_id):
    plan = {}
    for tensor_index, (name, value) in enumerate(state_delta.items()):
        if not value.is_floating_point():
            continue
        flat_size = value.numel()
        rows = []
        for block_index, start in enumerate(range(0, flat_size, config.block_size)):
            stop = min(start + config.block_size, flat_size)
            seed = _surrogate_seed(config, group_id, realization_id, tensor_index, block_index)
            matrix = torch.as_tensor(
                _candidate_realization_matrix(stop - start, config, seed),
                dtype=value.dtype,
                device=value.device,
            )
            rows.append((start, stop, matrix))
        plan[name] = rows
    return plan


def _apply_surrogate_realization_torch(state_delta, plan):
    transformed = {}
    for name, value in state_delta.items():
        if not value.is_floating_point():
            transformed[name] = value.clone()
            continue
        flat = value.reshape(-1)
        out = []
        for start, stop, matrix in plan[name]:
            out.append(matrix @ flat[start:stop])
        transformed[name] = torch.cat(out).reshape_as(value)
    return transformed


def _transmit_observed(observed, config):
    transmitted = {}
    for tensor_index, (name, delta) in enumerate(observed.items()):
        if delta.is_floating_point():
            array, _ = transform_update_array(delta.detach().cpu().numpy(), config, tensor_index=tensor_index)
            transmitted[name] = torch.from_numpy(array).to(dtype=delta.dtype)
        else:
            transmitted[name] = delta.detach().cpu()
    return transmitted


def _run_one(folder, method, group, group_id, restart, realization_id, protocol, frozen, distribution, dna_run_seed, args):
    folder.mkdir(parents=True, exist_ok=True)
    meta = distribution[0]
    local_seed = derive_seed(protocol["run_seed"], "local", group_id, protocol["batch_size"])
    model, criterion, x, y, batches, rng, observed = capture(group, local_seed, protocol["batch_size"])
    keys = [key for key, value in observed.items() if value.is_floating_point()]
    transform_config = DNATransformConfig(
        block_size=args.block_size,
        mix_ratio=args.mix_ratio,
        keep_ratio=args.keep_ratio,
        shrink_factor=args.shrink_factor,
        seed=derive_seed(dna_run_seed, "phase4-dna-transform", args.target_file, group_id),
    )
    transmitted = _transmit_observed(observed, transform_config)
    surrogate_plan = _surrogate_plan_from_state(observed, transform_config, group_id, realization_id)
    signal = transmitted if method == "baseline" else {key: torch.zeros_like(value) for key, value in transmitted.items()}
    initial_seed = derive_seed(
        protocol["run_seed"],
        "level1-forward-initial",
        args.init_mode,
        group_id,
        realization_id,
        restart,
        protocol["batch_size"],
    )
    initial = _initial(x.shape, initial_seed, args.init_mode, distribution)
    prior = _align_for_evaluation(x, _decode_harddiff(initial, meta), y)
    prior_metrics = score(x, prior, y, meta, folder / "prior.csv")
    latent = initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=args.attack_lr if args.attack_lr is not None else frozen["attack_lr"])
    iterations = args.iterations if args.iterations is not None else frozen["iterations"]
    best = float("inf")
    best_step = 0
    best_latent = latent.detach().clone()
    history = []
    for step in range(iterations + 1):
        reconstruction = _decode_harddiff(latent, meta)
        delta = simulate(model, criterion, reconstruction, y, batches, rng)
        candidate_transmitted = _apply_surrogate_realization_torch(delta, surrogate_plan)
        gradient_loss = update_objective(candidate_transmitted, signal, keys, reference=transmitted, mode="balanced_tensor")
        penalty = torch.zeros((), dtype=latent.dtype, device=latent.device)
        loss = gradient_loss
        if args.nonnegative_lambda:
            penalty = _nonnegative_penalty(latent, meta)
            loss = loss + args.nonnegative_lambda * penalty
        value = float(loss.detach())
        history.append(value)
        if value < best:
            best = value
            best_step = step
            best_latent = latent.detach().clone()
        if step < iterations:
            (gradient,) = torch.autograd.grad(loss, latent)
            optimizer.zero_grad()
            latent.grad = gradient
            optimizer.step()
    reconstruction = _decode_harddiff(best_latent, meta).detach()
    aligned = _align_for_evaluation(x, reconstruction, y)
    metrics = score(x, aligned, y, meta, folder / f"{method}.csv")
    artifact = {
        "method": method,
        "group_id": group_id,
        "restart": restart,
        "realization_id": realization_id,
        "source_ids": group["source_ids"],
        "original": x,
        "labels": y,
        "initial": initial,
        "latent": best_latent,
        "reconstruction": reconstruction,
        "aligned": aligned,
        "best_objective": best,
        "best_step": best_step,
        "history": history,
        "objective_mode": "balanced_tensor",
        "selection_objective": "level1_forward_transmitted_balanced_tensor_plus_nonnegative_penalty",
        "nonnegative_lambda": args.nonnegative_lambda,
        "parameterization": "hard_balance_diff_reparameterization",
        "initialization": args.init_mode,
        "local_seed": local_seed,
        "initial_seed": initial_seed,
        "dna_run_seed": dna_run_seed,
        "dna_transform_seed": transform_config.seed,
        "dna_transform_config": transform_config.__dict__,
        "surrogate_realization_id": realization_id,
        "observed_raw_update": observed,
        "transmitted_update": transmitted,
        "keys": keys,
    }
    torch.save(artifact, folder / f"{method}.pt")
    return {
        "method": method,
        "group_id": group_id,
        "restart": restart,
        "realization_id": realization_id,
        "objective": best,
        "best_step": best_step,
        "metrics": metrics,
        "prior": prior_metrics,
        "max_balance_residual_raw": _max_balance_residual(reconstruction, meta),
    }


def _flatten(row):
    return {
        "method": row["method"],
        "group_id": int(row["group_id"]),
        "restart": int(row["restart"]),
        "realization_id": int(row["realization_id"]),
        "objective": float(row["objective"]),
        "fraud_mse": float(row["metrics"]["classes"]["1"]["mean_mse"]),
        "non_fraud_mse": float(row["metrics"]["classes"]["0"]["mean_mse"]),
        "prior_fraud_mse": float(row["prior"]["classes"]["1"]["mean_mse"]),
        "max_balance_residual_raw": float(row["max_balance_residual_raw"]),
    }


def _summarize_group(rows):
    baseline = [row for row in rows if row["method"] == "baseline"]
    zero = [row for row in rows if row["method"] == "zero_update"]
    best = min(baseline, key=lambda row: row["objective"])
    zero_best = min(zero, key=lambda row: row["objective"])
    prior_mean = float(np.mean([row["prior_fraud_mse"] for row in baseline]))
    return {
        "group_id": int(best["group_id"]),
        "n_baseline_candidates": len(baseline),
        "objective_best_restart": int(best["restart"]),
        "objective_best_realization_id": int(best["realization_id"]),
        "objective_best_objective": float(best["objective"]),
        "objective_best_fraud_mse": float(best["fraud_mse"]),
        "objective_best_non_fraud_mse": float(best["non_fraud_mse"]),
        "prior_fraud_mse_mean": prior_mean,
        "zero_objective_best_fraud_mse": float(zero_best["fraud_mse"]),
        "objective_minus_prior_fraud_mse": float(best["fraud_mse"] - prior_mean),
        "objective_minus_zero_fraud_mse": float(best["fraud_mse"] - zero_best["fraud_mse"]),
        "max_balance_residual_raw": float(max(row["max_balance_residual_raw"] for row in baseline)),
    }


def _gate(summaries):
    out = {}
    for control in ("prior", "zero"):
        values = np.asarray([row[f"objective_minus_{control}_fraud_mse"] for row in summaries], dtype=float)
        wins = int((values < 0).sum())
        p_value = _sign_tail(wins, len(values))
        out[control] = {
            "n": len(values),
            "wins": wins,
            "mean_difference": float(values.mean()),
            "median_difference": float(np.median(values)),
            "one_sided_sign_p": p_value,
            "gate": bool(values.mean() < 0 and np.median(values) < 0 and p_value < 0.05),
        }
    out["gate"] = bool(out["prior"]["gate"] and out["zero"]["gate"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase4_run", type=Path)
    parser.add_argument("--target-file", default="development_targets.pt")
    parser.add_argument("--groups", type=int, nargs="+")
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--attack-lr", type=float)
    parser.add_argument("--init-mode", choices=("standard", "plausible"), default="standard")
    parser.add_argument("--nonnegative-lambda", type=float, default=0.001)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--mix-ratio", type=float, default=0.08)
    parser.add_argument("--keep-ratio", type=float, default=0.88)
    parser.add_argument("--shrink-factor", type=float, default=0.45)
    parser.add_argument("--dna-run-seed", type=int)
    args = parser.parse_args()

    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    run = args.phase4_run.resolve()
    gate = json.loads((run / "baseline_gate.json").read_text())
    frozen = json.loads((run / "frozen.json").read_text())
    target_groups = torch.load(run / args.target_file, weights_only=False)
    selected_groups = args.groups if args.groups is not None else list(range(len(target_groups)))
    dna_run_seed = args.dna_run_seed if args.dna_run_seed is not None else generate_run_seed()
    distribution = _feature_distribution()
    target_tag = (run / args.target_file).stem.replace("_targets", "")
    iterations = args.iterations if args.iterations is not None else frozen["iterations"]
    attack_lr = args.attack_lr if args.attack_lr is not None else frozen["attack_lr"]
    lr_tag = f"{attack_lr:g}".replace(".", "p")
    lambda_tag = f"{args.nonnegative_lambda:g}".replace(".", "p")
    config_tag = ""
    if (args.mix_ratio, args.keep_ratio, args.shrink_factor) != (0.08, 0.88, 0.45):
        mix_tag = f"{args.mix_ratio:g}".replace(".", "p")
        keep_tag = f"{args.keep_ratio:g}".replace(".", "p")
        shrink_tag = f"{args.shrink_factor:g}".replace(".", "p")
        config_tag = f"_mix{mix_tag}_keep{keep_tag}_shrink{shrink_tag}"
    out = run / (
        f"dna_level1_forward_attack_{target_tag}_c{args.candidates}_r{args.restarts}"
        f"_i{iterations}_lr{lr_tag}_nonneg{lambda_tag}{config_tag}"
    )
    out.mkdir(exist_ok=True)
    records = []
    for group_id in selected_groups:
        for realization_id in range(args.candidates):
            for restart in range(args.restarts):
                folder = out / f"group_{group_id}" / f"realization_{realization_id}" / f"restart_{restart}"
                if (folder / "results.json").exists():
                    records.extend(json.loads((folder / "results.json").read_text()))
                    continue
                rows = []
                for method in ("baseline", "zero_update"):
                    rows.append(
                        _run_one(
                            folder,
                            method,
                            target_groups[group_id],
                            group_id,
                            restart,
                            realization_id,
                            gate["protocol"],
                            frozen,
                            distribution,
                            dna_run_seed,
                            args,
                        )
                    )
                dump(folder / "results.json", rows)
                records.extend(rows)
    flat = [_flatten(row) for row in records]
    summaries = [_summarize_group([row for row in flat if row["group_id"] == group_id]) for group_id in selected_groups]
    report = {
        "target_file": args.target_file,
        "groups": selected_groups,
        "diagnostic": "fresh optimized Level 1 surrogate-forward DNA attack",
        "dna_run_seed": dna_run_seed,
        "surrogate_candidates_per_group": args.candidates,
        "restarts_per_candidate": args.restarts,
        "iterations": iterations,
        "attack_lr": attack_lr,
        "nonnegative_lambda": args.nonnegative_lambda,
        "selection_formula": "argmin_r,c L_balanced(M_r delta(c), transmitted_update; reference=transmitted_update)",
        "caveat": "Surrogate realizations do not equal the raw-block-derived realization; this is a Level-1 diagnostic attack, not Level 2.",
        "group_summary": summaries,
        "target_gate": _gate(summaries),
        "dataset_sha256": checksum(ROOT / "datasets/creditcard.csv"),
    }
    dump(out / "dna_level1_forward_attack_report.json", report)
    _write_csv(out / "dna_level1_forward_attack_records.csv", flat)
    _write_csv(out / "dna_level1_forward_attack_group_summary.csv", summaries)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
