"""Test plausible initialization on Phase 4 misselected groups.

This diagnostic keeps the locked gradient-matching objective unchanged.  It
changes only the dummy-data initialization so optimization starts inside a more
plausible feature domain.
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
from experiments.phase3_bounded_validation import _align_for_evaluation, _decode, update_objective
from experiments.run_phase3_adam_ladder import capture, metadata
from experiments.run_phase3_full_client import checksum, dump, score
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


def _feature_distribution():
    meta = metadata()
    frame = pd.read_csv(ROOT / "datasets/creditcard.csv", usecols=BASE_FEATURE_COLUMNS + [TARGET_COLUMN])
    features = _build_features(frame)
    scaled = (features[NUMERIC_COLUMNS].to_numpy() - np.asarray(meta.numeric_center)) / np.asarray(meta.numeric_scale)
    numeric_low = np.quantile(scaled, 0.01, axis=0)
    numeric_high = np.quantile(scaled, 0.99, axis=0)
    type_probs = np.asarray([(features["type"].to_numpy() == value).mean() for value in meta.type_categories])
    type_probs = type_probs / type_probs.sum()
    return meta, numeric_low.astype("float32"), numeric_high.astype("float32"), type_probs.astype("float64")


def _plausible_initial(shape, seed, numeric_low, numeric_high, type_probs):
    rng = np.random.default_rng(seed)
    numeric = rng.uniform(numeric_low, numeric_high, size=(shape[0], len(numeric_low))).astype("float32")
    types = rng.choice(len(type_probs), size=shape[0], p=type_probs)
    probs = np.full((shape[0], len(type_probs)), 0.01, dtype="float32")
    probs[np.arange(shape[0]), types] = 0.96
    probs = probs / probs.sum(axis=1, keepdims=True)
    logits = np.log(probs).astype("float32")
    return torch.from_numpy(np.hstack((numeric, logits)))


def _run_pair_plausible(folder, group, group_id, restart, run_seed, lr, iterations, batch_size, distribution):
    folder.mkdir(parents=True, exist_ok=False)
    meta, numeric_low, numeric_high, type_probs = distribution
    model, criterion, x, y, batches, rng, observed = capture(
        group, derive_seed(run_seed, "local", group_id, batch_size), batch_size
    )
    keys = [key for key, value in observed.items() if value.is_floating_point()]
    initial = _plausible_initial(
        x.shape,
        derive_seed(run_seed, "plausible-initial", group_id, restart, batch_size),
        numeric_low,
        numeric_high,
        type_probs,
    )
    prior = _align_for_evaluation(x, _decode(initial), y)
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
            delta = simulate(model, criterion, _decode(latent), y, batches, rng)
            loss = update_objective(delta, signal, keys, reference=observed, mode="balanced_tensor")
            value = float(loss.detach())
            history.append(value)
            if value < best:
                best = value
                best_step = step
                best_latent = latent.detach().clone()
            if step < iterations:
                gradient, = torch.autograd.grad(loss, latent)
                optimizer.zero_grad()
                latent.grad = gradient
                optimizer.step()
        reconstruction = _decode(best_latent).detach()
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
                "reconstruction": reconstruction,
                "aligned": aligned,
                "best_objective": best,
                "best_step": best_step,
                "history": history,
                "objective_mode": "balanced_tensor",
                "initialization": "plausible_global_1_99pct_numeric_type_prior",
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
    parser.add_argument("--groups", type=int, nargs="+", default=[0, 2, 6, 9])
    parser.add_argument("--restarts", type=int, default=10)
    args = parser.parse_args()

    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    run = args.run.resolve()
    gate = json.loads((run / "baseline_gate.json").read_text())
    frozen = json.loads((run / "frozen.json").read_text())
    groups = torch.load(run / "evaluation_targets.pt", weights_only=False)
    distribution = _feature_distribution()
    out = run / "plausible_init_misselected"
    out.mkdir(exist_ok=True)
    records = []
    for group_id in args.groups:
        for restart in range(args.restarts):
            folder = out / f"group_{group_id}" / f"restart_{restart}"
            if (folder / "results.json").exists():
                records.extend(json.loads((folder / "results.json").read_text()))
                continue
            records.extend(
                _run_pair_plausible(
                    folder,
                    groups[group_id],
                    group_id,
                    restart,
                    gate["protocol"]["run_seed"],
                    frozen["attack_lr"],
                    frozen["iterations"],
                    gate["protocol"]["batch_size"],
                    distribution,
                )
            )
    flat = _flatten(records)
    summaries = []
    for group_id in args.groups:
        summaries.append(_summarize_group([row for row in flat if row["group_id"] == group_id]))
    report = {
        "run": str(run),
        "groups": args.groups,
        "restarts": args.restarts,
        "initialization": "numeric uniform between global 1st and 99th scaled percentiles; type logits sampled from global type prior",
        "frozen_attack": frozen,
        "group_summary": summaries,
        "wrong_groups_only_gate": {
            "objective_selection": _gate(summaries, "objective"),
            "fraud_mse_oracle_selection_diagnostic_only": _gate(summaries, "fraud_best"),
        },
        "dataset_sha256": checksum(ROOT / "datasets/creditcard.csv"),
        "interpretation": "Diagnostic only; held-out groups were chosen after misselection analysis.",
    }
    dump(out / "plausible_init_report.json", report)
    _write_csv(out / "plausible_init_group_summary.csv", summaries)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
