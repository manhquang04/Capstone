"""Development gate attack for DNA Transform v2 lifted updates.

This is Step 5 development work only.  The attacker observes the v2 defended
full update after server-side lifting and optimizes the existing hard-diff
parameterization directly against that signal.
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
from dna_encoder.transform_defense_v2 import DNATransformV2Config, transform_and_reconstruct_array_v2
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


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sign_tail(wins: int, total: int) -> float:
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / 2**total if total else 1.0


def _floating_keys(update: dict[str, torch.Tensor]) -> list[str]:
    return [key for key, value in update.items() if value.is_floating_point()]


def _global_l2(update: dict[str, torch.Tensor], keys: list[str]) -> torch.Tensor:
    return torch.stack([update[key].detach().double().square().sum() for key in keys]).sum().sqrt()


def _v2_lifted_update(
    observed: dict[str, torch.Tensor],
    config: DNATransformV2Config,
    group_id: int,
) -> dict[str, torch.Tensor]:
    defended = {}
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


def _defense_stats(observed: dict[str, torch.Tensor], defended: dict[str, torch.Tensor], keys: list[str]) -> dict[str, float]:
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
    v2_config = DNATransformV2Config(
        compression_ratio=args.compression_ratio,
        quantization_eta=args.quantization_eta,
        seed=args.v2_base_seed,
    )
    defended_observed = _v2_lifted_update(observed, v2_config, group_id)
    signal = defended_observed if method == "baseline" else {key: torch.zeros_like(value) for key, value in defended_observed.items()}
    initial_seed = derive_seed(
        protocol["run_seed"],
        "dna-v2-harddiff-initial",
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
        gradient_loss = update_objective(delta, signal, keys, reference=defended_observed, mode="balanced_tensor")
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
            "defense": "dna_transform_v2_lifted",
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
            "selection_objective": "dna_transform_v2_lifted_raw_style_balanced_tensor_plus_nonnegative_penalty",
            "nonnegative_lambda": args.nonnegative_lambda,
            "parameterization": "hard_balance_diff_reparameterization",
            "initialization": args.init_mode,
            "local_seed": local_seed,
            "initial_seed": initial_seed,
            "observed_raw_update": observed,
            "defended_update": defended_observed,
            "defense_stats": stats,
            "v2_config": v2_config.__dict__,
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
    parser.add_argument("--target-file", default="development_gate_targets.pt")
    parser.add_argument("--groups", type=int, nargs="+")
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--attack-lr", type=float, default=0.1)
    parser.add_argument("--init-mode", choices=("standard", "plausible"), default="standard")
    parser.add_argument("--nonnegative-lambda", type=float, default=0.001)
    parser.add_argument("--compression-ratio", type=float, default=0.95)
    parser.add_argument("--quantization-eta", type=float, default=0.01)
    parser.add_argument("--v2-base-seed", type=int, default=20260916)
    args = parser.parse_args()

    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    run = args.phase4_run.resolve()
    gate = json.loads((run / "baseline_gate.json").read_text())
    frozen = json.loads((run / "frozen.json").read_text())
    target_groups = torch.load(run / args.target_file, weights_only=False)
    selected_groups = args.groups if args.groups is not None else list(range(len(target_groups)))
    distribution = _feature_distribution()
    ratio_tag = f"{args.compression_ratio:g}".replace(".", "p")
    eta_tag = f"{args.quantization_eta:g}".replace(".", "p")
    lr_tag = f"{args.attack_lr:g}".replace(".", "p")
    lambda_tag = f"{args.nonnegative_lambda:g}".replace(".", "p")
    target_tag = (run / args.target_file).stem.replace("_targets", "")
    out = run / (
        f"dna_v2_development_gate_{target_tag}_ratio{ratio_tag}_eta{eta_tag}"
        f"_r{args.restarts}_i{args.iterations}_lr{lr_tag}_nonneg{lambda_tag}"
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
        "defense": "dna_transform_v2_lifted",
        "diagnostic": "development-only raw-style attacker against v2 lifted full update",
        "amendment": "protocols/amendments/2026-09-16_dna_transform_v2_step5_development_calibration.md",
        "v2_config": {
            "compression_ratio": args.compression_ratio,
            "quantization_eta": args.quantization_eta,
            "seed": args.v2_base_seed,
        },
        "restarts": args.restarts,
        "iterations": args.iterations,
        "attack_lr": args.attack_lr,
        "nonnegative_lambda": args.nonnegative_lambda,
        "selection_formula": "argmin_x L_balanced(delta(x), lifted_v2_update; reference=lifted_v2_update)",
        "group_summary": summaries,
        "target_gate": _gate(summaries),
        "mean_defense_relative_l2_delta": float(np.mean([row["mean_defense_relative_l2_delta"] for row in summaries])),
        "dataset_sha256": checksum(ROOT / "datasets/creditcard.csv"),
    }
    dump(out / "dna_v2_development_gate_report.json", report)
    _write_csv(out / "dna_v2_development_gate_records.csv", flat)
    _write_csv(out / "dna_v2_development_gate_group_summary.csv", summaries)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
