"""Development-only sketch-space attacker for DNA Transform v2.

This runner attacks the observed v2 sketch q = Q(R_s u) directly and uses a
straight-through estimator for quantization.  It is not confirmatory.
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
from dna_encoder.transform_defense_v2 import (
    DNATransformV2Config,
    transform_update_array_v2,
    _derive_seed,
    _sampled_indices,
    _signs,
)
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


def _fwht_normalized_torch(values: torch.Tensor) -> torch.Tensor:
    out = values
    n = out.numel()
    if n < 1 or n & (n - 1):
        raise ValueError("FWHT length must be a positive power of two")
    step = 1
    while step < n:
        reshaped = out.reshape(-1, step * 2)
        left = reshaped[:, :step].clone()
        right = reshaped[:, step:].clone()
        reshaped[:, :step] = left + right
        reshaped[:, step:] = left - right
        out = reshaped.reshape(n)
        step *= 2
    return out / math.sqrt(n)


def _candidate_sketch_torch(
    value: torch.Tensor,
    *,
    compression_ratio: float,
    quantization_eta: float,
    base_seed: int,
    tensor_index: int,
    quantization_delta: float,
    ste_quantization: bool,
) -> torch.Tensor:
    flat = value.reshape(-1)
    original_size = flat.numel()
    padded_size = 1 << (max(1, original_size) - 1).bit_length()
    sketch_size = max(1, min(padded_size, int(math.ceil(padded_size * compression_ratio))))
    seed = _derive_seed(base_seed, tensor_index)
    signs_np = _signs(padded_size, seed)
    sampled_np = _sampled_indices(padded_size, sketch_size, seed)
    signs = torch.as_tensor(signs_np, dtype=flat.dtype, device=flat.device)
    sampled = torch.as_tensor(sampled_np, dtype=torch.long, device=flat.device)
    padded = torch.zeros(padded_size, dtype=flat.dtype, device=flat.device)
    padded[:original_size] = flat
    projected_full = _fwht_normalized_torch(padded * signs)
    sketch = math.sqrt(padded_size / sketch_size) * projected_full[sampled]
    if ste_quantization and quantization_delta > 0.0:
        delta = torch.as_tensor(quantization_delta, dtype=sketch.dtype, device=sketch.device)
        quantized = torch.round(sketch / delta) * delta
        sketch = sketch + (quantized - sketch).detach()
    return sketch


def _observed_sketches(
    observed: dict[str, torch.Tensor],
    config: DNATransformV2Config,
    group_id: int,
) -> dict[str, dict[str, torch.Tensor | float]]:
    sketches: dict[str, dict[str, torch.Tensor | float]] = {}
    for tensor_index, (name, value) in enumerate(observed.items()):
        if not value.is_floating_point():
            continue
        sketch, metadata = transform_update_array_v2(
            value.detach().cpu().numpy().astype(np.float32, copy=False),
            config,
            tensor_index=tensor_index,
            quantization_seed=group_id,
        )
        sketches[name] = {
            "sketch": torch.from_numpy(sketch.copy()).to(dtype=value.dtype),
            "quantization_delta": float(metadata.quantization_delta),
            "tensor_index": tensor_index,
        }
    return sketches


def _candidate_sketches(
    delta: dict[str, torch.Tensor],
    observed_sketches: dict[str, dict[str, torch.Tensor | float]],
    args,
) -> dict[str, torch.Tensor]:
    out = {}
    for name, payload in observed_sketches.items():
        sketch = _candidate_sketch_torch(
            delta[name],
            compression_ratio=args.compression_ratio,
            quantization_eta=args.quantization_eta,
            base_seed=args.v2_base_seed,
            tensor_index=int(payload["tensor_index"]),
            quantization_delta=float(payload["quantization_delta"]),
            ste_quantization=args.ste_quantization,
        )
        out[name] = sketch
    return out


def _zero_like_signal(signal: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: torch.zeros_like(value) for key, value in signal.items()}


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
    observed_sketch_payload = _observed_sketches(observed, v2_config, group_id)
    observed_sketch = {key: value["sketch"].to(dtype=observed[key].dtype) for key, value in observed_sketch_payload.items()}
    signal = observed_sketch if method == "baseline" else _zero_like_signal(observed_sketch)
    initial_seed = derive_seed(
        protocol["run_seed"],
        "dna-v2-sketch-initial",
        args.init_mode,
        group_id,
        restart,
        protocol["batch_size"],
        args.attack_lr,
        args.l1_update_lambda,
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
        candidate_sketch = _candidate_sketches(delta, observed_sketch_payload, args)
        sketch_loss = update_objective(candidate_sketch, signal, list(signal), reference=observed_sketch, mode="balanced_tensor")
        loss = args.sketch_loss_weight * sketch_loss
        if args.nonnegative_lambda:
            loss = loss + args.nonnegative_lambda * _nonnegative_penalty(latent, meta)
        if args.l1_update_lambda:
            l1_terms = [delta[key].abs().mean() for key in keys]
            loss = loss + args.l1_update_lambda * torch.stack(l1_terms).mean()
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
    torch.save(
        {
            "method": method,
            "defense": "dna_transform_v2_sketch_space",
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
            "objective_mode": "sketch_space_balanced_tensor",
            "selection_objective": "STE_Q(R_s_delta(candidate))_vs_observed_sketch",
            "nonnegative_lambda": args.nonnegative_lambda,
            "l1_update_lambda": args.l1_update_lambda,
            "parameterization": "hard_balance_diff_reparameterization",
            "initialization": args.init_mode,
            "local_seed": local_seed,
            "initial_seed": initial_seed,
            "observed_raw_update": observed,
            "observed_sketch": observed_sketch,
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
    parser.add_argument("--l1-update-lambda", type=float, default=0.0)
    parser.add_argument("--sketch-loss-weight", type=float, default=1.0)
    parser.add_argument("--compression-ratio", type=float, default=0.95)
    parser.add_argument("--quantization-eta", type=float, default=0.01)
    parser.add_argument("--v2-base-seed", type=int, default=20260916)
    parser.add_argument("--no-ste-quantization", action="store_true")
    args = parser.parse_args()
    args.ste_quantization = not args.no_ste_quantization

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
    l1_tag = f"{args.l1_update_lambda:g}".replace(".", "p")
    lambda_tag = f"{args.nonnegative_lambda:g}".replace(".", "p")
    ste_tag = "ste" if args.ste_quantization else "noste"
    target_tag = (run / args.target_file).stem.replace("_targets", "")
    out = run / (
        f"dna_v2_sketch_attack_{target_tag}_ratio{ratio_tag}_eta{eta_tag}"
        f"_r{args.restarts}_i{args.iterations}_lr{lr_tag}_nonneg{lambda_tag}_l1{l1_tag}_{ste_tag}"
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
        "defense": "dna_transform_v2_sketch_space",
        "diagnostic": "development-only sketch-space STE attacker for DNA Transform v2",
        "amendment": "protocols/amendments/2026-09-16_dna_transform_v2_structured_attacker_development.md",
        "v2_config": {
            "compression_ratio": args.compression_ratio,
            "quantization_eta": args.quantization_eta,
            "seed": args.v2_base_seed,
        },
        "restarts": args.restarts,
        "iterations": args.iterations,
        "attack_lr": args.attack_lr,
        "nonnegative_lambda": args.nonnegative_lambda,
        "l1_update_lambda": args.l1_update_lambda,
        "ste_quantization": args.ste_quantization,
        "selection_formula": "argmin_x L_balanced(STE_Q(R_s delta(x)), q_obs)",
        "group_summary": summaries,
        "target_gate": _gate(summaries),
        "dataset_sha256": checksum(ROOT / "datasets/creditcard.csv"),
    }
    dump(out / "dna_v2_sketch_attack_report.json", report)
    _write_csv(out / "dna_v2_sketch_attack_records.csv", flat)
    _write_csv(out / "dna_v2_sketch_attack_group_summary.csv", summaries)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
