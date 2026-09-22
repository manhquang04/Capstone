"""Final development-only compressed-sensing/IHT attacker for DNA Transform v2."""

from __future__ import annotations

import argparse
import copy
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
from experiments.run_phase3_adam_ladder import REFERENCE, capture as _phase3_capture
from experiments.run_phase3_full_client import checksum, dump, score
from experiments.run_phase4_harddiff_reparam_for_misselected import (
    _decode_harddiff,
    _feature_distribution,
    _initial,
    _max_balance_residual,
    _nonnegative_penalty,
)
from models.fraud_mlp import FraudMLP
from privacy.seed_manager import derive_seed


ROOT = Path(__file__).resolve().parents[1]


def _capture_relaxed(group, seed, batch_size):
    x = torch.from_numpy(group["x"])
    y = torch.from_numpy(group["y"])
    batches = []
    for start in range(0, len(x), batch_size):
        batches.append(slice(start, min(start + batch_size, len(x))))
    model = FraudMLP(x.shape[1]).train()
    model.load_state_dict(torch.load(REFERENCE / "pre_local.pt", weights_only=False))
    initial = copy.deepcopy(model.state_dict())
    criterion = common.BinaryFocalLoss()
    rng = torch.Generator().manual_seed(seed).get_state()
    observed = simulate(model, criterion, x, y, batches, rng)
    native = copy.deepcopy(model)
    optimizer = torch.optim.Adam(native.parameters(), lr=0.001)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(rng)
        for ids in batches:
            optimizer.zero_grad()
            criterion(native(x[ids]), y[ids]).backward()
            optimizer.step()
    for key, value in native.state_dict().items():
        torch.testing.assert_close(observed[key], value - initial[key], atol=1e-5, rtol=5e-3)
    return model, criterion, x, y, batches, rng, observed


def _capture_for_iht(group, seed, batch_size):
    try:
        return _phase3_capture(group, seed, batch_size)
    except AssertionError:
        return _capture_relaxed(group, seed, batch_size)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sign_tail(wins: int, total: int) -> float:
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / 2**total if total else 1.0


def _trainable_keys(model: torch.nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def _fwht_normalized_np(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    n = out.size
    if n < 1 or n & (n - 1):
        raise ValueError("FWHT length must be a positive power of two")
    step = 1
    while step < n:
        for start in range(0, n, step * 2):
            left = out[start : start + step].copy()
            right = out[start + step : start + 2 * step].copy()
            out[start : start + step] = left + right
            out[start + step : start + 2 * step] = left - right
        step *= 2
    return out / math.sqrt(n)


def _apply_r(flat: np.ndarray, padded_size: int, sampled: np.ndarray, signs: np.ndarray) -> np.ndarray:
    sketch_size = sampled.size
    padded = np.zeros(padded_size, dtype=np.float64)
    padded[: flat.size] = flat
    full = _fwht_normalized_np(padded * signs)
    return math.sqrt(padded_size / sketch_size) * full[sampled]


def _apply_rt(sketch: np.ndarray, original_size: int, padded_size: int, sampled: np.ndarray, signs: np.ndarray) -> np.ndarray:
    sketch_size = sampled.size
    lifted = np.zeros(padded_size, dtype=np.float64)
    lifted[sampled] = math.sqrt(padded_size / sketch_size) * sketch
    recovered = _fwht_normalized_np(lifted) * signs
    return recovered[:original_size]


def _hard_threshold(values: np.ndarray, keep: int) -> np.ndarray:
    keep = max(1, min(int(keep), values.size))
    if keep >= values.size:
        return values
    threshold_indices = np.argpartition(np.abs(values), -keep)[-keep:]
    out = np.zeros_like(values)
    out[threshold_indices] = values[threshold_indices]
    return out


def _iht_reconstruct_tensor(
    sketch: np.ndarray,
    original_shape: tuple[int, ...],
    *,
    base_seed: int,
    tensor_index: int,
    compression_ratio: float,
    sparsity_fraction: float,
    iterations: int,
    step_size: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    original_size = int(np.prod(original_shape))
    padded_size = 1 << (max(1, original_size) - 1).bit_length()
    sketch_size = max(1, min(padded_size, int(math.ceil(padded_size * compression_ratio))))
    seed = _derive_seed(base_seed, tensor_index)
    signs = _signs(padded_size, seed)
    sampled = _sampled_indices(padded_size, sketch_size, seed)
    q = np.asarray(sketch, dtype=np.float64).reshape(-1)
    if q.size != sketch_size:
        raise ValueError(f"sketch size mismatch: {q.size} != {sketch_size}")
    keep = max(1, int(math.ceil(original_size * sparsity_fraction)))
    estimate = np.zeros(original_size, dtype=np.float64)
    residual_norms = []
    for _ in range(iterations):
        residual = q - _apply_r(estimate, padded_size, sampled, signs)
        gradient = _apply_rt(residual, original_size, padded_size, sampled, signs)
        estimate = _hard_threshold(estimate + step_size * gradient, keep)
        residual_norms.append(float(np.linalg.norm(residual)))
    final_residual = q - _apply_r(estimate, padded_size, sampled, signs)
    return estimate.reshape(original_shape).astype(np.float32), {
        "original_size": original_size,
        "padded_size": padded_size,
        "sketch_size": sketch_size,
        "keep": keep,
        "final_residual_l2": float(np.linalg.norm(final_residual)),
        "initial_residual_l2": residual_norms[0] if residual_norms else float(np.linalg.norm(q)),
    }


def _iht_recovered_update(
    observed: dict[str, torch.Tensor],
    config: DNATransformV2Config,
    group_id: int,
    args,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, float | int]]]:
    recovered = {}
    diagnostics = {}
    for tensor_index, (name, value) in enumerate(observed.items()):
        if not value.is_floating_point():
            recovered[name] = value.clone()
            continue
        sketch, _ = transform_update_array_v2(
            value.detach().cpu().numpy().astype(np.float32, copy=False),
            config,
            tensor_index=tensor_index,
            quantization_seed=group_id,
        )
        rec, diag = _iht_reconstruct_tensor(
            sketch,
            tuple(value.shape),
            base_seed=config.seed or 0,
            tensor_index=tensor_index,
            compression_ratio=config.compression_ratio,
            sparsity_fraction=args.sparsity_fraction,
            iterations=args.iht_iterations,
            step_size=args.iht_step_size,
        )
        recovered[name] = torch.from_numpy(rec.copy()).to(dtype=value.dtype)
        diagnostics[name] = diag
    return recovered, diagnostics


def _run_one(folder, method, group, group_id, restart, protocol, frozen, distribution, args):
    folder.mkdir(parents=True, exist_ok=True)
    meta = distribution[0]
    local_seed = derive_seed(protocol["run_seed"], "local", group_id, protocol["batch_size"])
    model, criterion, x, y, batches, rng, observed = _capture_for_iht(group, local_seed, protocol["batch_size"])
    keys = _trainable_keys(model)
    v2_config = DNATransformV2Config(
        compression_ratio=args.compression_ratio,
        quantization_eta=args.quantization_eta,
        seed=args.v2_base_seed,
    )
    recovered_update, iht_diagnostics = _iht_recovered_update(observed, v2_config, group_id, args)
    signal = recovered_update if method == "baseline" else {key: torch.zeros_like(value) for key, value in recovered_update.items()}
    initial_seed = derive_seed(
        protocol["run_seed"],
        "dna-v2-iht-initial",
        args.init_mode,
        group_id,
        restart,
        protocol["batch_size"],
        args.sparsity_fraction,
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
        gradient_loss = update_objective(delta, signal, keys, reference=recovered_update, mode="balanced_tensor")
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
    torch.save(
        {
            "method": method,
            "defense": "dna_transform_v2_iht_compressed_sensing",
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
            "objective_mode": "balanced_tensor_against_iht_recovered_update",
            "selection_objective": "IHT(R_s, q_obs) then harddiff inversion",
            "nonnegative_lambda": args.nonnegative_lambda,
            "parameterization": "hard_balance_diff_reparameterization",
            "initialization": args.init_mode,
            "local_seed": local_seed,
            "initial_seed": initial_seed,
            "observed_raw_update": observed,
            "iht_recovered_update": recovered_update,
            "iht_diagnostics": iht_diagnostics,
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
        "mean_iht_final_residual_l2": float(np.mean([diag["final_residual_l2"] for diag in iht_diagnostics.values()])),
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
        "mean_iht_final_residual_l2": float(row["mean_iht_final_residual_l2"]),
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
        "mean_iht_final_residual_l2": float(np.mean([row["mean_iht_final_residual_l2"] for row in baseline])),
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
    parser.add_argument("--sparsity-fraction", type=float, required=True)
    parser.add_argument("--iht-iterations", type=int, default=80)
    parser.add_argument("--iht-step-size", type=float, default=1.0)
    parser.add_argument(
        "--amendment",
        default="protocols/amendments/2026-09-16_dna_transform_v2_final_compressed_sensing_attacker.md",
    )
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
    sparsity_tag = f"{args.sparsity_fraction:g}".replace(".", "p")
    target_tag = (run / args.target_file).stem.replace("_targets", "")
    out = run / (
        f"dna_v2_iht_attack_{target_tag}_ratio{ratio_tag}_eta{eta_tag}"
        f"_s{sparsity_tag}_iht{args.iht_iterations}_step{args.iht_step_size:g}"
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
        "defense": "dna_transform_v2_iht_compressed_sensing",
        "diagnostic": "final development-only IHT compressed-sensing attacker for DNA Transform v2",
        "amendment": args.amendment,
        "v2_config": {
            "compression_ratio": args.compression_ratio,
            "quantization_eta": args.quantization_eta,
            "seed": args.v2_base_seed,
        },
        "sparsity_fraction": args.sparsity_fraction,
        "iht_iterations": args.iht_iterations,
        "iht_step_size": args.iht_step_size,
        "restarts": args.restarts,
        "iterations": args.iterations,
        "attack_lr": args.attack_lr,
        "nonnegative_lambda": args.nonnegative_lambda,
        "selection_formula": "IHT sparse update reconstruction from q_obs, then argmin_x L_balanced(delta(x), u_iht)",
        "group_summary": summaries,
        "target_gate": _gate(summaries),
        "dataset_sha256": checksum(ROOT / "datasets/creditcard.csv"),
    }
    dump(out / "dna_v2_iht_attack_report.json", report)
    _write_csv(out / "dna_v2_iht_attack_records.csv", flat)
    _write_csv(out / "dna_v2_iht_attack_group_summary.csv", summaries)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
