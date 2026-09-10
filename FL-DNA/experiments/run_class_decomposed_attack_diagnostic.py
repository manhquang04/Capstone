"""Oracle class-decomposed gradient diagnostic for Phase 3.

This runner intentionally does not represent the normal server-side FL threat
model.  It exposes separate fraud and non-fraud gradient contributions and
matches them with equal class weight.  The goal is narrow: test whether the
16-record attack failed because the minority fraud signal overwhelms the
non-fraud signal in the aggregate objective.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.func import functional_call
from torch.nn import functional as F

from attacks.tabular_parameterization import PaySimManifold, update_matching_objective
from data.load_creditcard import BASE_FEATURE_COLUMNS
from experiments import fraud_fl_common as common
from experiments.phase3_bounded_validation import _align_for_evaluation
from experiments.run_attack_redevelopment import _initial_latent, decoder_for, sign_tail
from experiments.run_phase3_adam_ladder import excluded_rows, make_groups, metadata
from experiments.run_phase3_full_client import checksum, dump, score
from models.fraud_mlp import FraudMLP
from privacy.seed_manager import derive_seed, generate_run_seed


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/phase3/full_20260908T143837588533Z"


def _focal_per_sample(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_t = common.FOCAL_ALPHA * targets + (1.0 - common.FOCAL_ALPHA) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(common.FOCAL_GAMMA) * bce).reshape(-1)


def _flat(values):
    return torch.cat([value.reshape(-1) for value in values])


def _class_contribution_gradients(model, x, y, *, seed, create_graph):
    """Return train-mode full-batch class contribution gradients.

    The forward pass is shared by both classes, so BatchNorm and Dropout match
    the same full-batch graph instead of recomputing separate subset statistics.
    """
    params = {name: value for name, value in model.named_parameters()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    names = list(params)
    labels = y.reshape(-1)
    masks = {"fraud": labels == 1, "non_fraud": labels == 0}
    model.train(True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        logits = functional_call(model, (params, buffers), (x,))
        per_sample = _focal_per_sample(logits, y)
    total = len(x)
    payload = {}
    for class_name, mask in masks.items():
        if int(mask.sum()) == 0:
            raise ValueError(f"Missing class in target group: {class_name}")
        contribution_loss = per_sample[mask].sum() / total
        gradients = torch.autograd.grad(
            contribution_loss,
            tuple(params.values()),
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=False,
        )
        payload[class_name] = {name: grad for name, grad in zip(names, gradients)}
    return payload, names


def _zero_like(signal):
    return {class_name: {key: torch.zeros_like(value) for key, value in values.items()} for class_name, values in signal.items()}


def _objective(candidate_signal, observed_signal, keys, *, class_weight="equal"):
    losses = []
    for class_name in ("fraud", "non_fraud"):
        loss = update_matching_objective(
            candidate_signal[class_name],
            observed_signal[class_name],
            keys,
            mode="balanced_bn",
            bn_weight=0.0,
        )
        losses.append(loss)
    if class_weight != "equal":
        raise ValueError(f"Unknown class weighting: {class_weight}")
    return torch.stack(losses).mean()


def _class_mse(original, candidate, labels):
    values = (original.double() - candidate.double()).square().mean(dim=1).detach().cpu().numpy()
    flat_labels = labels.reshape(-1).detach().cpu().numpy()
    out = {}
    for label, name in ((0, "non_fraud"), (1, "fraud")):
        selected = values[flat_labels == label]
        out[name] = {
            "n": int(len(selected)),
            "mean_mse": float(selected.mean()),
            "median_mse": float(np.median(selected)),
        }
    return out


def _run_trial(folder, group, group_id, restart, run_seed, variant, attack_lr, iterations):
    folder.mkdir(parents=True, exist_ok=False)
    x = torch.from_numpy(group["x"])
    y = torch.from_numpy(group["y"])
    model = FraudMLP(x.shape[1])
    model.load_state_dict(torch.load(REFERENCE / "pre_local.pt", weights_only=False))
    observed_seed = derive_seed(run_seed, "class-gradient", group_id)
    observed, keys = _class_contribution_gradients(model, x, y, seed=observed_seed, create_graph=False)
    decode, latent_dim = decoder_for(variant, dtype=x.dtype)
    initial = _initial_latent(run_seed, group_id, restart, latent_dim, len(x))
    initial_decoded = decode(initial)
    prior_aligned = _align_for_evaluation(x, initial_decoded, y)
    meta = metadata()
    prior_metrics = score(x, prior_aligned, y, meta, folder / "prior.csv")
    rows = []
    for method in ("baseline", "zero_update"):
        signal = observed if method == "baseline" else _zero_like(observed)
        latent = initial.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=attack_lr)
        best_objective = float("inf")
        best_step = 0
        best_latent = latent.detach().clone()
        history = []
        started = time.perf_counter()
        for step in range(iterations + 1):
            reconstruction = decode(latent)
            candidate_signal, _ = _class_contribution_gradients(model, reconstruction, y, seed=observed_seed, create_graph=True)
            loss = _objective(candidate_signal, signal, keys)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite class-decomposed objective at step {step}")
            value = float(loss.detach())
            history.append(value)
            if value < best_objective:
                best_objective = value
                best_step = step
                best_latent = latent.detach().clone()
            if step < iterations:
                gradient, = torch.autograd.grad(loss, latent)
                if not torch.isfinite(gradient).all():
                    raise RuntimeError(f"Non-finite latent gradient at step {step}")
                optimizer.zero_grad()
                latent.grad = gradient
                optimizer.step()

        reconstruction = decode(best_latent).detach()
        aligned = _align_for_evaluation(x, reconstruction, y)
        metrics = score(x, aligned, y, meta, folder / f"{method}.csv")
        artifact_path = folder / f"{method}.pt"
        torch.save(
            {
                "method": method,
                "variant": variant,
                "group_id": group_id,
                "restart": restart,
                "source_ids": group["source_ids"],
                "labels": y,
                "original": x,
                "initial_latent": initial,
                "initial_reconstruction": initial_decoded,
                "best_latent": best_latent,
                "best_reconstruction": reconstruction,
                "aligned_reconstruction": aligned,
                "best_objective": best_objective,
                "best_step": best_step,
                "history": history,
                "observed_seed": observed_seed,
                "initial_seed": derive_seed(run_seed, "initial", group_id, restart),
            },
            artifact_path,
        )
        loaded = torch.load(artifact_path, weights_only=False)
        loaded_signal, _ = _class_contribution_gradients(
            model,
            loaded["best_reconstruction"],
            y,
            seed=observed_seed,
            create_graph=False,
        )
        reloaded_objective = float(_objective(loaded_signal, signal, keys).detach())
        if not math.isclose(reloaded_objective, best_objective, rel_tol=1e-5, abs_tol=1e-10):
            raise AssertionError("Reloaded candidate does not match best class-decomposed objective")
        rows.append(
            {
                "method": method,
                "variant": variant,
                "group_id": group_id,
                "restart": restart,
                "attack_lr": attack_lr,
                "iterations": iterations,
                "best_step": best_step,
                "objective": best_objective,
                "reloaded_objective": reloaded_objective,
                "metrics": metrics,
                "class_metrics": _class_mse(x, aligned, y),
                "prior": prior_metrics,
                "prior_class_metrics": _class_mse(x, prior_aligned, y),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    dump(folder / "results.json", rows)
    return rows


def _select(rows, group_id, method):
    return min(
        (row for row in rows if row["group_id"] == group_id and row["method"] == method),
        key=lambda row: row["objective"],
    )


def _evaluate(rows):
    selected = []
    for group_id in sorted({row["group_id"] for row in rows}):
        baseline = _select(rows, group_id, "baseline")
        zero = _select(rows, group_id, "zero_update")
        prior = float(np.mean([row["prior"]["mean_mse"] for row in rows if row["group_id"] == group_id and row["method"] == "baseline"]))
        selected.append(
            {
                "group_id": group_id,
                "baseline_restart": baseline["restart"],
                "zero_restart": zero["restart"],
                "baseline_mse": baseline["metrics"]["mean_mse"],
                "prior_mse": prior,
                "zero_mse": zero["metrics"]["mean_mse"],
                "baseline_minus_prior": baseline["metrics"]["mean_mse"] - prior,
                "baseline_minus_zero": baseline["metrics"]["mean_mse"] - zero["metrics"]["mean_mse"],
                "baseline_fraud_mse": baseline["class_metrics"]["fraud"]["mean_mse"],
                "baseline_non_fraud_mse": baseline["class_metrics"]["non_fraud"]["mean_mse"],
                "prior_fraud_mse": float(
                    np.mean(
                        [
                            row["prior_class_metrics"]["fraud"]["mean_mse"]
                            for row in rows
                            if row["group_id"] == group_id and row["method"] == "baseline"
                        ]
                    )
                ),
                "prior_non_fraud_mse": float(
                    np.mean(
                        [
                            row["prior_class_metrics"]["non_fraud"]["mean_mse"]
                            for row in rows
                            if row["group_id"] == group_id and row["method"] == "baseline"
                        ]
                    )
                ),
            }
        )
    comparisons = {}
    for control in ("prior", "zero"):
        differences = np.asarray([row[f"baseline_minus_{control}"] for row in selected])
        non_ties = differences[differences != 0]
        wins = int((non_ties < 0).sum())
        comparisons[control] = {
            "n": len(selected),
            "wins": wins,
            "non_ties": int(len(non_ties)),
            "mean_difference": float(differences.mean()),
            "median_difference": float(np.median(differences)),
            "one_sided_sign_p": sign_tail(wins, len(non_ties)),
        }
        comparisons[control]["gate"] = bool(
            comparisons[control]["mean_difference"] < 0
            and comparisons[control]["median_difference"] < 0
            and comparisons[control]["one_sided_sign_p"] < 0.05
        )
    for class_name in ("fraud", "non_fraud"):
        differences = np.asarray([row[f"baseline_{class_name}_mse"] - row[f"prior_{class_name}_mse"] for row in selected])
        non_ties = differences[differences != 0]
        wins = int((non_ties < 0).sum())
        comparisons[f"{class_name}_vs_prior"] = {
            "n": len(selected),
            "wins": wins,
            "non_ties": int(len(non_ties)),
            "mean_difference": float(differences.mean()),
            "median_difference": float(np.median(differences)),
            "one_sided_sign_p": sign_tail(wins, len(non_ties)),
        }
    return comparisons, selected


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _candidate_score(comparisons):
    return max(comparisons["prior"]["mean_difference"], comparisons["zero"]["mean_difference"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-groups", type=int, default=3)
    parser.add_argument("--confirmation-groups", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--attack-lrs", type=float, nargs="+", default=[0.03, 0.05])
    parser.add_argument("--variant", default="manifold_bn1")
    args = parser.parse_args()

    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    started = time.perf_counter()
    run_seed = generate_run_seed()
    out = ROOT / "artifacts/attack_redevelopment" / datetime.now(timezone.utc).strftime("class_decomp_%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(ROOT / "datasets/creditcard.csv", usecols=BASE_FEATURE_COLUMNS + ["isFraud"])
    excluded = excluded_rows(frame.isFraud.to_numpy())
    available = np.asarray(sorted(set(range(len(frame))) - excluded))
    protocol = {
        "run_seed": run_seed,
        "scope": "oracle class-decomposed 16-record train-mode gradient diagnostic",
        "records_per_group": 16,
        "fraud_per_group": 4,
        "development_groups": args.development_groups,
        "confirmation_groups": args.confirmation_groups,
        "iterations": args.iterations,
        "development_restarts": 1,
        "confirmation_restarts": args.restarts,
        "attack_lrs": args.attack_lrs,
        "variant": args.variant,
        "threat_model_status": "diagnostic_oracle_not_server_visible",
        "observation": "separate fraud and non-fraud train-mode gradient contributions from one shared full-batch graph",
        "objective": "equal-weight match of fraud and non-fraud contribution gradients; no ground truth in candidate selection",
        "purpose": "test whether non-fraud reconstruction is hidden by class-skewed aggregate gradients",
        "dataset_sha256": checksum(ROOT / "datasets/creditcard.csv"),
        "checkpoint_sha256": checksum(REFERENCE / "pre_local.pt"),
    }
    dump(out / "protocol_lock.json", protocol)
    development_groups = make_groups(
        frame,
        available,
        args.development_groups,
        protocol["records_per_group"],
        protocol["fraud_per_group"],
        derive_seed(run_seed, "class-decomp-development-data"),
    )
    development_ids = {source_id for group in development_groups for source_id in group["source_ids"]}
    available = np.asarray(sorted(set(available) - development_ids))
    torch.save(development_groups, out / "development_targets.pt")
    candidates = []
    development_records = []
    for lr in args.attack_lrs:
        rows = []
        for group_id, group in enumerate(development_groups):
            rows.extend(_run_trial(out / "development" / f"lr_{lr}" / f"group_{group_id}", group, group_id, 0, run_seed, args.variant, lr, args.iterations))
        comparisons, selected = _evaluate(rows)
        candidates.append({"attack_lr": lr, "selection_score": _candidate_score(comparisons), "comparisons": comparisons, "selected": selected})
        development_records.extend(rows)
    chosen = min(candidates, key=lambda item: item["selection_score"])
    frozen = {
        "variant": args.variant,
        "attack_lr": chosen["attack_lr"],
        "iterations": args.iterations,
        "restarts": args.restarts,
        "protocol_sha256": checksum(out / "protocol_lock.json"),
        "selected_before_confirmation": True,
    }
    dump(out / "development_records.json", development_records)
    dump(out / "development_summary.json", {"candidates": candidates, "chosen": chosen})
    _write_csv(
        out / "development_summary.csv",
        [
            {
                "attack_lr": candidate["attack_lr"],
                "selection_score": candidate["selection_score"],
                "prior_wins": candidate["comparisons"]["prior"]["wins"],
                "prior_mean_delta": candidate["comparisons"]["prior"]["mean_difference"],
                "zero_wins": candidate["comparisons"]["zero"]["wins"],
                "zero_mean_delta": candidate["comparisons"]["zero"]["mean_difference"],
            }
            for candidate in candidates
        ],
    )
    dump(out / "frozen.json", frozen)

    confirmation_groups = make_groups(
        frame,
        available,
        args.confirmation_groups,
        protocol["records_per_group"],
        protocol["fraud_per_group"],
        derive_seed(run_seed, "class-decomp-confirmation-data"),
    )
    confirmation_ids = {source_id for group in confirmation_groups for source_id in group["source_ids"]}
    if development_ids & confirmation_ids:
        raise AssertionError("Development and confirmation source rows overlap")
    torch.save(confirmation_groups, out / "confirmation_targets.pt")
    confirmation_records = []
    for group_id, group in enumerate(confirmation_groups):
        for restart in range(args.restarts):
            confirmation_records.extend(
                _run_trial(
                    out / "confirmation" / f"group_{group_id}" / f"restart_{restart}",
                    group,
                    group_id,
                    restart,
                    run_seed,
                    frozen["variant"],
                    frozen["attack_lr"],
                    frozen["iterations"],
                )
            )
    comparisons, selected = _evaluate(confirmation_records)
    gate = comparisons["prior"]["gate"] and comparisons["zero"]["gate"]
    _write_csv(out / "confirmation_selected.csv", selected)
    dump(out / "confirmation_records.json", confirmation_records)
    report = {
        "status": "CLASS_DECOMPOSED_GATE_PASSED" if gate else "CLASS_DECOMPOSED_GATE_FAILED",
        "baseline_gate_passed": gate,
        "protocol": protocol,
        "frozen_attack": frozen,
        "development": {"chosen": chosen},
        "confirmation": comparisons,
        "selected_confirmation": selected,
        "interpretation": "Diagnostic-only oracle check; separate class gradients are not available to a normal FL server.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    dump(out / "class_decomposed_attack_report.json", report)
    print(json.dumps({"output": str(out), "gate": gate, "confirmation": comparisons}, indent=2))


if __name__ == "__main__":
    main()
