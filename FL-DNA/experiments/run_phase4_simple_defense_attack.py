"""Run bounded Level-1-style attacks against simple update defenses.

This runner evaluates simple, non-DNA defenses in the same 4-record/1-step
scope used by the Phase 4 paired DNA comparison.  Retention masks, top-k
thresholds, and clipping factors are attacker-visible.  Gaussian noise
realizations are not attacker-visible, so candidate updates are not given the
same noise sample during objective matching.  The Monte Carlo clipping/noise
variant uses independent surrogate noise samples and optimizes the equivalent
mean-noise objective.
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
from experiments import fraud_fl_common as common
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
from privacy.seed_manager import derive_seed


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


def _floating_keys(update):
    return [key for key, value in update.items() if value.is_floating_point()]


def _global_l2(update, keys):
    terms = [update[key].detach().double().square().sum() for key in keys]
    return torch.stack(terms).sum().sqrt()


def _scale_like(update, factor):
    return {key: value * factor if value.is_floating_point() else value.clone() for key, value in update.items()}


def _random_retention_plan(update, keys, keep_ratio, seed):
    generator = torch.Generator().manual_seed(seed)
    return {
        key: (torch.rand(update[key].shape, generator=generator, dtype=update[key].dtype) < keep_ratio).to(update[key].dtype)
        for key in keys
    }


def _topk_plan(update, keys, target_rel_l2):
    flat = torch.cat([update[key].detach().reshape(-1).abs().double() for key in keys])
    if flat.numel() == 0:
        return {key: torch.ones_like(update[key]) for key in keys}
    total_energy = float(flat.square().sum())
    if total_energy == 0.0:
        return {key: torch.ones_like(update[key]) for key in keys}
    sorted_values = torch.sort(flat.square(), descending=True).values
    retained = torch.cumsum(sorted_values, dim=0)
    target_energy = (1.0 - target_rel_l2**2) * total_energy
    keep_count = int(torch.searchsorted(retained, torch.tensor(target_energy, dtype=retained.dtype)).item()) + 1
    keep_count = min(max(1, keep_count), flat.numel())
    threshold = float(torch.sort(flat, descending=True).values[keep_count - 1])
    return {
        "mask": {key: (update[key].detach().abs() >= threshold).to(update[key].dtype) for key in keys},
        "threshold": threshold,
        "keep_count": keep_count,
        "total_count": int(flat.numel()),
    }


def _raw_noise(update, keys, seed):
    generator = torch.Generator().manual_seed(seed)
    return {
        key: torch.randn(update[key].shape, generator=generator, dtype=update[key].dtype)
        for key in keys
    }


def _noise_plan(update, keys, target_rel_l2, seed):
    raw = _raw_noise(update, keys, seed)
    update_norm = _global_l2(update, keys).clamp_min(torch.finfo(torch.float64).tiny)
    noise_norm = _global_l2(raw, keys).clamp_min(torch.finfo(torch.float64).tiny)
    scale = float(target_rel_l2 * update_norm / noise_norm)
    return {key: raw[key] * scale for key in keys}


def _noise_mean(update, keys, scale, samples, seed):
    if samples <= 0:
        return {key: torch.zeros_like(update[key]) for key in keys}
    acc = {key: torch.zeros_like(update[key]) for key in keys}
    for sample_id in range(samples):
        raw = _raw_noise(update, keys, derive_seed(seed, "mc-noise", sample_id))
        for key in keys:
            acc[key] = acc[key] + raw[key] * scale
    return {key: value / float(samples) for key, value in acc.items()}


def _clipping_noise_plan(update, keys, target_rel_l2, clip_factor, seed):
    raw = _raw_noise(update, keys, seed)
    update_norm_sq = _global_l2(update, keys).square()
    target_sq = (target_rel_l2**2) * update_norm_sq
    clip_delta = {key: (clip_factor - 1.0) * update[key] for key in keys}
    clip_delta_sq = _global_l2(clip_delta, keys).square()
    raw_sq = _global_l2(raw, keys).square().clamp_min(torch.finfo(torch.float64).tiny)
    cross = torch.stack([(clip_delta[key].double() * raw[key].double()).sum() for key in keys]).sum()
    discriminant = (2.0 * cross).square() - 4.0 * raw_sq * (clip_delta_sq - target_sq)
    if float(discriminant) >= 0.0:
        root = (-2.0 * cross + discriminant.sqrt()) / (2.0 * raw_sq)
        scale = max(0.0, float(root))
    else:
        residual = (target_sq - clip_delta_sq).clamp_min(0.0)
        scale = float((residual / raw_sq).sqrt())
    return {
        "defense": "clipping_noise",
        "noise": {key: raw[key] * scale for key in keys},
        "noise_scale": scale,
        "clip_factor": clip_factor,
        "keep_ratio": 1.0,
    }


def _apply_observed_defense(update, keys, defense, plan):
    defended = {}
    for key, value in update.items():
        if not value.is_floating_point():
            defended[key] = value.clone()
            continue
        if defense in ("random_retention", "topk_retention", "soft_topk_retention"):
            defended[key] = value * plan["mask"][key]
        elif defense == "gaussian_noise":
            defended[key] = value + plan["noise"][key]
        elif defense in ("clipping_noise", "clipping_noise_mc"):
            defended[key] = plan["clip_factor"] * value + plan["noise"][key]
        else:
            raise ValueError(f"Unknown defense: {defense}")
    return defended


def _apply_candidate_defense(update, keys, defense, plan):
    defended = {}
    for key, value in update.items():
        if not value.is_floating_point():
            defended[key] = value.clone()
            continue
        if defense in ("random_retention", "topk_retention"):
            defended[key] = value * plan["mask"][key]
        elif defense == "soft_topk_retention":
            threshold = plan["threshold"]
            temperature = max(plan["temperature"], torch.finfo(value.dtype).eps)
            soft_mask = torch.sigmoid((value.abs() - threshold) / temperature)
            defended[key] = value * plan["mask"][key] * soft_mask
        elif defense == "gaussian_noise":
            defended[key] = value
        elif defense == "clipping_noise":
            defended[key] = plan["clip_factor"] * value
        elif defense == "clipping_noise_mc":
            defended[key] = plan["clip_factor"] * value + plan["mc_noise_mean"][key]
        else:
            raise ValueError(f"Unknown defense: {defense}")
    return defended


def _plan(defense, observed, keys, group_id, args):
    if defense == "random_retention":
        keep_ratio = 1.0 - args.target_rel_l2**2
        mask = _random_retention_plan(
            observed,
            keys,
            keep_ratio,
            derive_seed(args.defense_seed, "random-retention", group_id),
        )
        return {"defense": defense, "mask": mask, "keep_ratio": keep_ratio}
    if defense == "topk_retention":
        topk = _topk_plan(observed, keys, args.target_rel_l2)
        mask = topk["mask"]
        kept = sum(int(mask[key].sum().item()) for key in keys)
        total = sum(mask[key].numel() for key in keys)
        return {
            "defense": defense,
            "mask": mask,
            "threshold": topk["threshold"],
            "keep_ratio": kept / total,
        }
    if defense == "soft_topk_retention":
        topk = _topk_plan(observed, keys, args.target_rel_l2)
        mask = topk["mask"]
        kept = sum(int(mask[key].sum().item()) for key in keys)
        total = sum(mask[key].numel() for key in keys)
        threshold = topk["threshold"]
        return {
            "defense": defense,
            "mask": mask,
            "threshold": threshold,
            "temperature": max(args.topk_temperature * max(abs(threshold), 1e-12), 1e-12),
            "temperature_multiplier": args.topk_temperature,
            "keep_ratio": kept / total,
        }
    if defense == "gaussian_noise":
        noise = _noise_plan(
            observed,
            keys,
            args.target_rel_l2,
            derive_seed(args.defense_seed, "gaussian-noise", group_id),
        )
        return {"defense": defense, "noise": noise, "keep_ratio": 1.0}
    if defense in ("clipping_noise", "clipping_noise_mc"):
        plan = _clipping_noise_plan(
            observed,
            keys,
            args.target_rel_l2,
            args.clip_factor,
            derive_seed(args.defense_seed, "clipping-noise", group_id),
        )
        plan["defense"] = defense
        if defense == "clipping_noise_mc":
            plan["mc_noise_samples"] = args.mc_noise_samples
            plan["mc_noise_mean"] = _noise_mean(
                observed,
                keys,
                plan["noise_scale"],
                args.mc_noise_samples,
                derive_seed(args.defense_seed, "clipping-noise-mc", group_id),
            )
        return plan
    raise ValueError(f"Unknown defense: {defense}")


def _defense_stats(observed, defended, keys):
    diff = {key: defended[key] - observed[key] for key in keys}
    observed_norm = _global_l2(observed, keys)
    diff_norm = _global_l2(diff, keys)
    return {
        "observed_l2": float(observed_norm),
        "defense_delta_l2": float(diff_norm),
        "relative_l2_delta": float(diff_norm / observed_norm.clamp_min(torch.finfo(torch.float64).tiny)),
    }


def _run_one(folder, method, group, group_id, restart, protocol, frozen, distribution, args):
    folder.mkdir(parents=True, exist_ok=True)
    meta = distribution[0]
    local_seed = derive_seed(protocol["run_seed"], "local", group_id, protocol["batch_size"])
    model, criterion, x, y, batches, rng, observed = capture(group, local_seed, protocol["batch_size"])
    keys = _floating_keys(observed)
    plan = _plan(args.defense, observed, keys, group_id, args)
    defended_observed = _apply_observed_defense(observed, keys, args.defense, plan)
    signal = defended_observed if method == "baseline" else {key: torch.zeros_like(value) for key, value in defended_observed.items()}
    initial_seed = derive_seed(
        protocol["run_seed"],
        "harddiff-initial",
        args.init_mode,
        group_id,
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
        candidate_defended = _apply_candidate_defense(delta, keys, args.defense, plan)
        gradient_loss = update_objective(candidate_defended, signal, keys, reference=defended_observed, mode="balanced_tensor")
        loss = gradient_loss
        if args.nonnegative_lambda:
            loss = loss + args.nonnegative_lambda * _nonnegative_penalty(latent, meta)
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
    stats = _defense_stats(observed, defended_observed, keys)
    torch.save(
        {
            "method": method,
            "defense": args.defense,
            "group_id": group_id,
            "restart": restart,
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
            "selection_objective": f"{args.defense}_balanced_tensor_plus_nonnegative_penalty",
            "nonnegative_lambda": args.nonnegative_lambda,
            "parameterization": "hard_balance_diff_reparameterization",
            "initialization": args.init_mode,
            "local_seed": local_seed,
            "initial_seed": initial_seed,
            "observed_raw_update": observed,
            "defended_update": defended_observed,
            "defense_stats": stats,
            "defense_plan_public": {
                "defense": args.defense,
                "clip_factor": float(plan["clip_factor"]) if "clip_factor" in plan else None,
                "keep_ratio": float(plan.get("keep_ratio", 1.0)),
                "threshold": float(plan["threshold"]) if "threshold" in plan else None,
                "temperature": float(plan["temperature"]) if "temperature" in plan else None,
                "mc_noise_samples": int(plan["mc_noise_samples"]) if "mc_noise_samples" in plan else 0,
            },
            "keys": keys,
        },
        folder / f"{method}.pt",
    )
    return {
        "method": method,
        "group_id": group_id,
        "restart": restart,
        "objective": best,
        "best_step": best_step,
        "metrics": metrics,
        "prior": prior_metrics,
        "max_balance_residual_raw": _max_balance_residual(reconstruction, meta),
        "defense_relative_l2_delta": stats["relative_l2_delta"],
        "defense_keep_ratio": float(plan.get("keep_ratio", 1.0)),
    }


def _flatten(row):
    return {
        "method": row["method"],
        "group_id": int(row["group_id"]),
        "restart": int(row["restart"]),
        "objective": float(row["objective"]),
        "fraud_mse": float(row["metrics"]["classes"]["1"]["mean_mse"]),
        "non_fraud_mse": float(row["metrics"]["classes"]["0"]["mean_mse"]),
        "prior_fraud_mse": float(row["prior"]["classes"]["1"]["mean_mse"]),
        "max_balance_residual_raw": float(row["max_balance_residual_raw"]),
        "defense_relative_l2_delta": float(row["defense_relative_l2_delta"]),
        "defense_keep_ratio": float(row["defense_keep_ratio"]),
    }


def _summarize_group(rows):
    baseline = [row for row in rows if row["method"] == "baseline"]
    zero = [row for row in rows if row["method"] == "zero_update"]
    best = min(baseline, key=lambda row: row["objective"])
    zero_best = min(zero, key=lambda row: row["objective"])
    prior_mean = float(np.mean([row["prior_fraud_mse"] for row in baseline]))
    return {
        "group_id": int(best["group_id"]),
        "n_baseline_restarts": len(baseline),
        "objective_best_restart": int(best["restart"]),
        "objective_best_objective": float(best["objective"]),
        "objective_best_fraud_mse": float(best["fraud_mse"]),
        "objective_best_non_fraud_mse": float(best["non_fraud_mse"]),
        "prior_fraud_mse_mean": prior_mean,
        "zero_objective_best_fraud_mse": float(zero_best["fraud_mse"]),
        "objective_minus_prior_fraud_mse": float(best["fraud_mse"] - prior_mean),
        "objective_minus_zero_fraud_mse": float(best["fraud_mse"] - zero_best["fraud_mse"]),
        "max_balance_residual_raw": float(max(row["max_balance_residual_raw"] for row in baseline)),
        "mean_defense_relative_l2_delta": float(np.mean([row["defense_relative_l2_delta"] for row in baseline])),
        "mean_defense_keep_ratio": float(np.mean([row["defense_keep_ratio"] for row in baseline])),
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
    parser.add_argument("--target-file", default="level1_fresh_final_targets.pt")
    parser.add_argument(
        "--defense",
        choices=(
            "random_retention",
            "topk_retention",
            "soft_topk_retention",
            "gaussian_noise",
            "clipping_noise",
            "clipping_noise_mc",
        ),
        required=True,
    )
    parser.add_argument("--groups", type=int, nargs="+")
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--attack-lr", type=float)
    parser.add_argument("--init-mode", choices=("standard", "plausible"), default="standard")
    parser.add_argument("--nonnegative-lambda", type=float, default=0.001)
    parser.add_argument("--target-rel-l2", type=float, default=0.10)
    parser.add_argument("--clip-factor", type=float, default=0.95)
    parser.add_argument("--mc-noise-samples", type=int, default=0)
    parser.add_argument("--topk-temperature", type=float, default=0.05)
    parser.add_argument("--defense-seed", type=int, default=314159265)
    args = parser.parse_args()

    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    run = args.phase4_run.resolve()
    gate = json.loads((run / "baseline_gate.json").read_text())
    frozen = json.loads((run / "frozen.json").read_text())
    target_groups = torch.load(run / args.target_file, weights_only=False)
    selected_groups = args.groups if args.groups is not None else list(range(len(target_groups)))
    distribution = _feature_distribution()
    iterations = args.iterations if args.iterations is not None else frozen["iterations"]
    attack_lr = args.attack_lr if args.attack_lr is not None else frozen["attack_lr"]
    lr_tag = f"{attack_lr:g}".replace(".", "p")
    lambda_tag = f"{args.nonnegative_lambda:g}".replace(".", "p")
    rel_tag = f"{args.target_rel_l2:g}".replace(".", "p")
    extra_tags = []
    if args.defense == "clipping_noise_mc":
        extra_tags.append(f"mc{args.mc_noise_samples}")
    if args.defense == "soft_topk_retention":
        extra_tags.append(f"temp{args.topk_temperature:g}".replace(".", "p"))
    extra_tag = "_" + "_".join(extra_tags) if extra_tags else ""
    target_tag = (run / args.target_file).stem.replace("_targets", "")
    out = run / (
        f"simple_defense_attack_{args.defense}_{target_tag}_r{args.restarts}"
        f"_i{iterations}_lr{lr_tag}_nonneg{lambda_tag}_rel{rel_tag}{extra_tag}"
    )
    out.mkdir(exist_ok=True)
    records = []
    for group_id in selected_groups:
        for restart in range(args.restarts):
            folder = out / f"group_{group_id}" / f"restart_{restart}"
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
                        gate["protocol"],
                        frozen,
                        distribution,
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
        "defense": args.defense,
        "diagnostic": (
            "bounded simple-defense attack; attacker-visible masks/thresholds/factors for retention and clipping; "
            "unknown realization for Gaussian noise"
        ),
        "target_relative_l2_delta": args.target_rel_l2,
        "defense_seed": args.defense_seed,
        "restarts": args.restarts,
        "iterations": iterations,
        "attack_lr": attack_lr,
        "nonnegative_lambda": args.nonnegative_lambda,
        "clip_factor": args.clip_factor if args.defense in ("clipping_noise", "clipping_noise_mc") else None,
        "mc_noise_samples": args.mc_noise_samples if args.defense == "clipping_noise_mc" else 0,
        "topk_temperature": args.topk_temperature if args.defense == "soft_topk_retention" else None,
        "selection_formula": (
            f"argmin_c L_balanced(T_visible_{args.defense}(delta(c)), defended_update; "
            "reference=defended_update)"
        ),
        "group_summary": summaries,
        "target_gate": _gate(summaries),
        "mean_defense_relative_l2_delta": float(np.mean([row["mean_defense_relative_l2_delta"] for row in summaries])),
        "mean_defense_keep_ratio": float(np.mean([row["mean_defense_keep_ratio"] for row in summaries])),
        "dataset_sha256": checksum(ROOT / "datasets/creditcard.csv"),
    }
    dump(out / "simple_defense_attack_report.json", report)
    _write_csv(out / "simple_defense_attack_records.csv", flat)
    _write_csv(out / "simple_defense_attack_group_summary.csv", summaries)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
