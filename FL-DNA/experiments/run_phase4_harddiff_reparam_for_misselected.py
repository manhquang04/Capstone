"""Hard-reparameterize PaySim balance-diff features on Phase 4 hard groups.

This diagnostic keeps the locked pre-Phase-4 attack objective and budget
unchanged.  The only change is the reconstruction parameterization:
``balance_diff_orig`` and ``balance_diff_dest`` are computed from the balance
columns instead of optimized as independent dummy features.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from attacks.local_update import simulate
from data.load_creditcard import BASE_FEATURE_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN, _build_features
from experiments import fraud_fl_common as common
from experiments.phase3_bounded_validation import _align_for_evaluation, update_objective
from experiments.run_phase3_adam_ladder import capture, metadata
from experiments.run_phase3_full_client import checksum, dump, score
from privacy.seed_manager import derive_seed


ROOT = Path(__file__).resolve().parents[1]
BASE_NUMERIC = (
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)
DERIVED_NUMERIC = ("balance_diff_orig", "balance_diff_dest")


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sign_tail(wins, total):
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / 2**total if total else 1.0


def _feature_distribution():
    meta = metadata()
    frame = pd.read_csv(ROOT / "datasets/creditcard.csv", usecols=BASE_FEATURE_COLUMNS + [TARGET_COLUMN])
    features = _build_features(frame)
    scaled = (features[NUMERIC_COLUMNS].to_numpy() - np.asarray(meta.numeric_center)) / np.asarray(meta.numeric_scale)
    base_indices = [NUMERIC_COLUMNS.index(name) for name in BASE_NUMERIC]
    numeric_low = np.quantile(scaled[:, base_indices], 0.01, axis=0)
    numeric_high = np.quantile(scaled[:, base_indices], 0.99, axis=0)
    type_probs = np.asarray([(features["type"].to_numpy() == value).mean() for value in meta.type_categories])
    type_probs = type_probs / type_probs.sum()
    return meta, numeric_low.astype("float32"), numeric_high.astype("float32"), type_probs.astype("float64")


def _initial(shape, seed, init_mode, distribution):
    rows = shape[0]
    type_count = shape[1] - 8
    if init_mode == "standard":
        generator = torch.Generator().manual_seed(seed)
        return torch.randn((rows, len(BASE_NUMERIC) + type_count), generator=generator)
    if init_mode != "plausible":
        raise ValueError(f"Unknown init mode: {init_mode}")

    _, numeric_low, numeric_high, type_probs = distribution
    rng = np.random.default_rng(seed)
    numeric = rng.uniform(numeric_low, numeric_high, size=(rows, len(BASE_NUMERIC))).astype("float32")
    types = rng.choice(len(type_probs), size=rows, p=type_probs)
    probs = np.full((rows, len(type_probs)), 0.01, dtype="float32")
    probs[np.arange(rows), types] = 0.96
    probs = probs / probs.sum(axis=1, keepdims=True)
    logits = np.log(probs).astype("float32")
    return torch.from_numpy(np.hstack((numeric, logits)))


def _decode_harddiff(latent, meta):
    """Decode latent features while enforcing PaySim balance-diff identities."""
    center = torch.as_tensor(meta.numeric_center, dtype=latent.dtype, device=latent.device)
    scale = torch.as_tensor(meta.numeric_scale, dtype=latent.dtype, device=latent.device)
    base = latent[:, : len(BASE_NUMERIC)]
    raw = base * scale[: len(BASE_NUMERIC)] + center[: len(BASE_NUMERIC)]
    diff_orig_raw = raw[:, 2] - raw[:, 3]
    diff_dest_raw = raw[:, 5] - raw[:, 4]
    diff_orig = (diff_orig_raw - center[6]) / scale[6]
    diff_dest = (diff_dest_raw - center[7]) / scale[7]
    numeric = torch.cat((base, diff_orig[:, None], diff_dest[:, None]), dim=1)
    return torch.cat((numeric, latent[:, len(BASE_NUMERIC):].softmax(-1)), dim=1)


def _max_balance_residual(candidate, meta):
    numeric = candidate[:, :8].detach().cpu().numpy().astype("float64")
    raw = numeric * np.asarray(meta.numeric_scale, dtype="float64") + np.asarray(meta.numeric_center, dtype="float64")
    orig_residual = raw[:, 6] - (raw[:, 2] - raw[:, 3])
    dest_residual = raw[:, 7] - (raw[:, 5] - raw[:, 4])
    return float(max(np.abs(orig_residual).max(), np.abs(dest_residual).max()))


def _nonnegative_penalty(latent, meta):
    """Softly penalize impossible negative amount/balance raw values."""
    center = torch.as_tensor(meta.numeric_center, dtype=latent.dtype, device=latent.device)
    scale = torch.as_tensor(meta.numeric_scale, dtype=latent.dtype, device=latent.device)
    raw = latent[:, : len(BASE_NUMERIC)] * scale[: len(BASE_NUMERIC)] + center[: len(BASE_NUMERIC)]
    constrained = raw[:, [1, 2, 3, 4, 5]]
    constrained_scale = scale[[1, 2, 3, 4, 5]].clamp_min(torch.finfo(latent.dtype).tiny)
    return torch.relu(-constrained / constrained_scale).square().mean()


def _run_pair_harddiff(
    folder,
    group,
    group_id,
    restart,
    run_seed,
    lr,
    iterations,
    batch_size,
    distribution,
    init_mode,
    nonnegative_lambda,
):
    folder.mkdir(parents=True, exist_ok=False)
    meta = distribution[0]
    model, criterion, x, y, batches, rng, observed = capture(
        group, derive_seed(run_seed, "local", group_id, batch_size), batch_size
    )
    keys = [key for key, value in observed.items() if value.is_floating_point()]
    initial = _initial(
        x.shape,
        derive_seed(run_seed, "harddiff-initial", init_mode, group_id, restart, batch_size),
        init_mode,
        distribution,
    )
    prior = _align_for_evaluation(x, _decode_harddiff(initial, meta), y)
    prior_metrics = score(x, prior, y, meta, folder / "prior.csv")
    rows = []
    for method in ("baseline", "zero_update"):
        signal = observed if method == "baseline" else {key: torch.zeros_like(value) for key, value in observed.items()}
        latent = initial.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=lr)
        best = float("inf")
        best_step = 0
        best_latent = latent.detach().clone()
        history = []
        for step in range(iterations + 1):
            reconstruction = _decode_harddiff(latent, meta)
            delta = simulate(model, criterion, reconstruction, y, batches, rng)
            gradient_loss = update_objective(delta, signal, keys, reference=observed, mode="balanced_tensor")
            loss = gradient_loss
            penalty = torch.zeros((), dtype=latent.dtype, device=latent.device)
            if nonnegative_lambda:
                penalty = _nonnegative_penalty(latent, meta)
                loss = loss + nonnegative_lambda * penalty
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
                "selection_objective": "balanced_tensor_plus_nonnegative_penalty",
                "nonnegative_lambda": nonnegative_lambda,
                "parameterization": "hard_balance_diff_reparameterization",
                "initialization": init_mode,
                "observed": observed,
                "keys": keys,
            },
            folder / f"{method}.pt",
        )
        rows.append(
            {
                "method": method,
                "group_id": group_id,
                "restart": restart,
                "objective": best,
                "best_step": best_step,
                "metrics": metrics,
                "prior": prior_metrics,
                "max_balance_residual_raw": _max_balance_residual(reconstruction, meta),
            }
        )
    dump(folder / "results.json", rows)
    return rows


def _best(rows, key):
    return min(rows, key=lambda row: row[key])


def _flatten(records):
    flat = []
    for row in records:
        flat.append(
            {
                "method": row["method"],
                "group_id": int(row["group_id"]),
                "restart": int(row["restart"]),
                "objective": float(row["objective"]),
                "overall_mse": float(row["metrics"]["mean_mse"]),
                "fraud_mse": float(row["metrics"]["classes"]["1"]["mean_mse"]),
                "non_fraud_mse": float(row["metrics"]["classes"]["0"]["mean_mse"]),
                "prior_fraud_mse": float(row["prior"]["classes"]["1"]["mean_mse"]),
                "max_balance_residual_raw": float(row["max_balance_residual_raw"]),
            }
        )
    return flat


def _summarize_group(rows):
    baseline = [row for row in rows if row["method"] == "baseline"]
    zero = [row for row in rows if row["method"] == "zero_update"]
    objective_best = _best(baseline, "objective")
    fraud_best = _best(baseline, "fraud_mse")
    zero_best = _best(zero, "objective")
    prior_mean = float(np.mean([row["prior_fraud_mse"] for row in baseline]))
    return {
        "group_id": objective_best["group_id"],
        "n_baseline_restarts": len(baseline),
        "objective_best_restart": objective_best["restart"],
        "fraud_mse_best_restart": fraud_best["restart"],
        "objective_best_fraud_mse": objective_best["fraud_mse"],
        "fraud_best_fraud_mse": fraud_best["fraud_mse"],
        "extra_fraud_mse_from_objective_selection": objective_best["fraud_mse"] - fraud_best["fraud_mse"],
        "objective_best_objective": objective_best["objective"],
        "fraud_best_objective": fraud_best["objective"],
        "objective_margin_wrong_minus_fraud_best": objective_best["objective"] - fraud_best["objective"],
        "prior_fraud_mse_mean": prior_mean,
        "zero_objective_best_fraud_mse": zero_best["fraud_mse"],
        "objective_minus_prior_fraud_mse": objective_best["fraud_mse"] - prior_mean,
        "objective_minus_zero_fraud_mse": objective_best["fraud_mse"] - zero_best["fraud_mse"],
        "fraud_best_minus_prior_fraud_mse": fraud_best["fraud_mse"] - prior_mean,
        "fraud_best_minus_zero_fraud_mse": fraud_best["fraud_mse"] - zero_best["fraud_mse"],
        "max_balance_residual_raw": max(row["max_balance_residual_raw"] for row in baseline),
    }


def _gate(rows, prefix):
    out = {}
    for control in ("prior", "zero"):
        values = np.asarray([row[f"{prefix}_minus_{control}_fraud_mse"] for row in rows], dtype=float)
        wins = int((values < 0).sum())
        p_value = _sign_tail(wins, len(values))
        out[control] = {
            "n": len(rows),
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
    parser.add_argument("run", type=Path)
    parser.add_argument("--target-file", default="evaluation_targets.pt")
    parser.add_argument("--groups", type=int, nargs="+")
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--init-mode", choices=("standard", "plausible"), default="standard")
    parser.add_argument("--nonnegative-lambda", type=float, default=0.0)
    args = parser.parse_args()

    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    run = args.run.resolve()
    gate = json.loads((run / "baseline_gate.json").read_text())
    frozen = json.loads((run / "frozen.json").read_text())
    target_path = run / args.target_file
    groups = torch.load(target_path, weights_only=False)
    selected_groups = args.groups if args.groups is not None else list(range(len(groups)))
    distribution = _feature_distribution()
    lambda_tag = f"nonneg_{args.nonnegative_lambda:g}".replace(".", "p")
    target_tag = target_path.stem.replace("_targets", "")
    out = run / f"harddiff_reparam_{target_tag}_{args.init_mode}_{lambda_tag}"
    out.mkdir(exist_ok=True)
    records = []
    for group_id in selected_groups:
        for restart in range(args.restarts):
            folder = out / f"group_{group_id}" / f"restart_{restart}"
            if (folder / "results.json").exists():
                records.extend(json.loads((folder / "results.json").read_text()))
                continue
            records.extend(
                _run_pair_harddiff(
                    folder,
                    groups[group_id],
                    group_id,
                    restart,
                    gate["protocol"]["run_seed"],
                    frozen["attack_lr"],
                    frozen["iterations"],
                    gate["protocol"]["batch_size"],
                    distribution,
                    args.init_mode,
                    args.nonnegative_lambda,
                )
            )
    flat = _flatten(records)
    summaries = [_summarize_group([row for row in flat if row["group_id"] == group_id]) for group_id in selected_groups]
    report = {
        "run": str(run),
        "target_file": args.target_file,
        "groups": selected_groups,
        "restarts": args.restarts,
        "initialization": args.init_mode,
        "nonnegative_lambda": args.nonnegative_lambda,
        "parameterization": (
            "balance_diff_orig = oldbalanceOrg - newbalanceOrig and "
            "balance_diff_dest = newbalanceDest - oldbalanceDest are enforced during optimization"
        ),
        "frozen_attack": frozen,
        "group_summary": summaries,
        "target_gate": {
            "objective_selection": _gate(summaries, "objective"),
            "fraud_mse_oracle_selection_diagnostic_only": _gate(summaries, "fraud_best"),
        },
        "wrong_groups_only_gate": {
            "objective_selection": _gate(summaries, "objective"),
            "fraud_mse_oracle_selection_diagnostic_only": _gate(summaries, "fraud_best"),
        },
        "dataset_sha256": checksum(ROOT / "datasets/creditcard.csv"),
        "interpretation": (
            "Fresh single-shot target evaluation after attack-rule lock."
            if "fresh" in args.target_file
            else "Diagnostic/development target run; do not use as an independent final gate."
        ),
    }
    dump(out / "harddiff_reparam_report.json", report)
    _write_csv(out / "harddiff_reparam_group_summary.csv", summaries)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
