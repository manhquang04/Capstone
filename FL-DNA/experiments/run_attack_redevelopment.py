"""Prospective redevelopment gate for bounded Adam client-update inversion.

Development data select one fixed attack configuration. A source-disjoint
confirmation set is evaluated once. DNA evaluation is not opened unless the
baseline attack beats both no-update controls on the locked confirmation set.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from attacks.local_update import simulate
from attacks.tabular_parameterization import (
    PaySimManifold,
    unconstrained_decode,
    update_matching_objective,
)
from data.load_creditcard import BASE_FEATURE_COLUMNS
from experiments import fraud_fl_common as common
from experiments.phase3_bounded_validation import _align_for_evaluation
from experiments.run_phase3_adam_ladder import capture, excluded_rows, make_groups, metadata
from experiments.run_phase3_full_client import checksum, dump, score
from privacy.seed_manager import derive_seed, generate_run_seed


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/phase3/full_20260908T143837588533Z"


VARIANTS = {
    "legacy_flat": {"parameterization": "legacy", "objective": "flat", "bn_weight": 1.0},
    "manifold_flat": {"parameterization": "manifold", "objective": "flat", "bn_weight": 1.0},
    "manifold_bn1": {"parameterization": "manifold", "objective": "balanced_bn", "bn_weight": 1.0},
    "manifold_bn3": {"parameterization": "manifold", "objective": "balanced_bn", "bn_weight": 3.0},
}


def sign_tail(wins, total):
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / 2**total if total else 1.0


def decoder_for(variant, *, dtype=torch.float32):
    config = VARIANTS[variant]
    if config["parameterization"] == "legacy":
        return unconstrained_decode, 13
    manifold = PaySimManifold.from_metadata(metadata(), dtype=dtype)
    return manifold.decode, manifold.latent_dim


def _initial_latent(run_seed, group_id, restart, latent_dim, record_count=4):
    # The six base fields and five category logits share identical draws across
    # parameterizations. Legacy-only engineered fields are appended separately.
    generator = torch.Generator().manual_seed(derive_seed(run_seed, "initial", group_id, restart))
    common_draw = torch.randn((record_count, 11), generator=generator)
    if latent_dim == 11:
        return common_draw
    engineered = torch.randn((record_count, 2), generator=generator)
    return torch.cat((common_draw[:, :6], engineered, common_draw[:, 6:]), dim=1)


def run_trial(folder, group, group_id, restart, run_seed, variant, attack_lr, iterations):
    folder.mkdir(parents=True, exist_ok=False)
    local_seed = derive_seed(run_seed, "local", group_id)
    model, criterion, original, labels, batches, rng, observed = capture(group, local_seed, 4)
    keys = [key for key, value in observed.items() if value.is_floating_point()]
    decode, latent_dim = decoder_for(variant, dtype=original.dtype)
    initial = _initial_latent(run_seed, group_id, restart, latent_dim, len(original))
    initial_decoded = decode(initial)
    prior_aligned = _align_for_evaluation(original, initial_decoded, labels)
    prior_metrics = score(original, prior_aligned, labels, metadata(), folder / "prior.csv")
    config = VARIANTS[variant]
    rows = []

    for method in ("baseline", "zero_update"):
        signal = observed if method == "baseline" else {key: torch.zeros_like(value) for key, value in observed.items()}
        latent = initial.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=attack_lr)
        best_objective = float("inf")
        best_step = 0
        best_latent = latent.detach().clone()
        history = []
        started = time.perf_counter()
        for step in range(iterations + 1):
            reconstruction = decode(latent)
            candidate_update = simulate(model, criterion, reconstruction, labels, batches, rng)
            loss = update_matching_objective(
                candidate_update,
                signal,
                keys,
                mode=config["objective"],
                bn_weight=config["bn_weight"],
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite attack objective at step {step}")
            objective = float(loss.detach())
            history.append(objective)
            if objective < best_objective:
                best_objective = objective
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
        aligned = _align_for_evaluation(original, reconstruction, labels)
        metrics = score(original, aligned, labels, metadata(), folder / f"{method}.csv")
        artifact = {
            "method": method,
            "variant": variant,
            "group_id": group_id,
            "source_ids": group["source_ids"],
            "labels": labels,
            "original": original,
            "initial_latent": initial,
            "initial_reconstruction": initial_decoded,
            "best_latent": best_latent,
            "best_reconstruction": reconstruction,
            "aligned_reconstruction": aligned,
            "best_objective": best_objective,
            "best_step": best_step,
            "history": history,
            "local_seed": local_seed,
            "initial_seed": derive_seed(run_seed, "initial", group_id, restart),
        }
        artifact_path = folder / f"{method}.pt"
        torch.save(artifact, artifact_path)

        # The persisted candidate, not only the in-memory tensor, must reproduce
        # the recorded best objective.
        loaded = torch.load(artifact_path, weights_only=False)
        loaded_update = simulate(model, criterion, loaded["best_reconstruction"], labels, batches, rng)
        reloaded_objective = float(
            update_matching_objective(
                loaded_update,
                signal,
                keys,
                mode=config["objective"],
                bn_weight=config["bn_weight"],
            ).detach()
        )
        if not math.isclose(reloaded_objective, best_objective, rel_tol=1e-5, abs_tol=1e-10):
            raise AssertionError("Reloaded candidate does not match best attack objective")
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
                "prior": prior_metrics,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    dump(folder / "results.json", rows)
    return rows


def select_by_attacker_objective(rows, group_id, method):
    return min(
        (row for row in rows if row["group_id"] == group_id and row["method"] == method),
        key=lambda row: row["objective"],
    )


def evaluate(rows):
    selected = []
    for group_id in sorted({row["group_id"] for row in rows}):
        baseline = select_by_attacker_objective(rows, group_id, "baseline")
        zero = select_by_attacker_objective(rows, group_id, "zero_update")
        priors = [
            row["prior"]["mean_mse"]
            for row in rows
            if row["group_id"] == group_id and row["method"] == "baseline"
        ]
        prior = float(np.mean(priors))
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
            "non_ties": len(non_ties),
            "mean_difference": float(differences.mean()),
            "median_difference": float(np.median(differences)),
            "one_sided_sign_p": sign_tail(wins, len(non_ties)),
        }
        comparisons[control]["gate"] = bool(
            comparisons[control]["mean_difference"] < 0
            and comparisons[control]["median_difference"] < 0
            and comparisons[control]["one_sided_sign_p"] < 0.05
        )
    return comparisons, selected


def write_selected_csv(path, selected):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)


def main():
    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    out = ROOT / "artifacts/attack_redevelopment" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    run_seed = generate_run_seed()
    frame = pd.read_csv(ROOT / "datasets/creditcard.csv", usecols=BASE_FEATURE_COLUMNS + ["isFraud"])
    excluded = excluded_rows(frame.isFraud.to_numpy())
    available = np.asarray(sorted(set(range(len(frame))) - excluded))
    protocol = {
        "run_seed": run_seed,
        "scope": "known-label/order/RNG, one Adam step on one four-record batch (one fraud, three non-fraud)",
        "observation": "individual full floating model-state delta, including BatchNorm buffers",
        "local_optimizer": "Adam(lr=0.001, betas=(0.9,0.999), eps=1e-8)",
        "model_mode": "train with replayed dropout RNG",
        "development_groups": 6,
        "confirmation_groups": 12,
        "attack_iterations": 300,
        "development_restarts": 1,
        "confirmation_restarts": 3,
        "attack_learning_rates": [0.01, 0.05],
        "variants": VARIANTS,
        "development_selection": "minimum max(mean MSE delta vs own prior, mean MSE delta vs paired zero-update); ground truth development-only",
        "candidate_selection": "minimum observed-signal objective; ground truth never used",
        "confirmation_gate": "for both controls: negative mean and median MSE difference and one-sided exact sign p<0.05",
        "decision_rule": "open DNA comparison only if the locked baseline confirmation gate passes",
        "dataset_sha256": checksum(ROOT / "datasets/creditcard.csv"),
        "checkpoint_sha256": checksum(REFERENCE / "pre_local.pt"),
    }
    dump(out / "protocol_lock.json", protocol)

    development_groups = make_groups(
        frame, available, protocol["development_groups"], 4, 1, derive_seed(run_seed, "development-data")
    )
    development_ids = {source_id for group in development_groups for source_id in group["source_ids"]}
    available = np.asarray(sorted(set(available) - development_ids))
    torch.save(development_groups, out / "development_targets.pt")
    development_records = []
    candidates = []
    for variant in VARIANTS:
        for attack_lr in protocol["attack_learning_rates"]:
            variant_rows = []
            for group_id, group in enumerate(development_groups):
                folder = out / "development" / variant / f"lr_{attack_lr}" / f"group_{group_id}"
                rows = run_trial(folder, group, group_id, 0, run_seed, variant, attack_lr, 300)
                variant_rows.extend(rows)
                development_records.extend(rows)
            comparisons, selected = evaluate(variant_rows)
            score_value = max(
                comparisons["prior"]["mean_difference"], comparisons["zero"]["mean_difference"]
            )
            candidates.append(
                {
                    "variant": variant,
                    "attack_lr": attack_lr,
                    "selection_score": score_value,
                    "comparisons": comparisons,
                    "selected": selected,
                }
            )
    chosen = min(candidates, key=lambda row: row["selection_score"])
    frozen = {
        "variant": chosen["variant"],
        "attack_lr": chosen["attack_lr"],
        "iterations": protocol["attack_iterations"],
        "restarts": protocol["confirmation_restarts"],
        "protocol_sha256": checksum(out / "protocol_lock.json"),
        "selected_before_confirmation": True,
    }
    dump(out / "development_summary.json", {"candidates": candidates, "chosen": chosen})
    dump(out / "frozen.json", frozen)

    confirmation_groups = make_groups(
        frame, available, protocol["confirmation_groups"], 4, 1, derive_seed(run_seed, "confirmation-data")
    )
    confirmation_ids = {source_id for group in confirmation_groups for source_id in group["source_ids"]}
    if development_ids & confirmation_ids:
        raise AssertionError("Development and confirmation source rows overlap")
    torch.save(confirmation_groups, out / "confirmation_targets.pt")
    confirmation_records = []
    for group_id, group in enumerate(confirmation_groups):
        for restart in range(frozen["restarts"]):
            folder = out / "confirmation" / f"group_{group_id}" / f"restart_{restart}"
            confirmation_records.extend(
                run_trial(
                    folder,
                    group,
                    group_id,
                    restart,
                    run_seed,
                    frozen["variant"],
                    frozen["attack_lr"],
                    frozen["iterations"],
                )
            )
    comparisons, selected = evaluate(confirmation_records)
    gate = comparisons["prior"]["gate"] and comparisons["zero"]["gate"]
    write_selected_csv(out / "confirmation_selected.csv", selected)
    report = {
        "status": "BASELINE_ATTACK_VALIDATED" if gate else "REDEVELOPMENT_COMPLETE_NEGATIVE",
        "baseline_gate_passed": gate,
        "phase4_dna_evaluation_open": gate,
        "frozen_attack": frozen,
        "confirmation": comparisons,
        "selected_confirmation": selected,
        "dna_result": None,
        "dna_reason": None if gate else "DNA comparison was not run because the baseline attack failed its preregistered controls.",
        "limitations": [
            "This is a bounded one-batch Adam diagnostic, not inversion of a full utility-client FedAvg update.",
            "Known labels, record order, local RNG, architecture, optimizer, and checkpoint favor the attacker.",
            "If this gate fails, this attack should not be used as a defense evaluator at this scope.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    dump(out / "attack_redevelopment_report.json", report)
    dump(out / "development_records.json", development_records)
    dump(out / "confirmation_records.json", confirmation_records)
    print(json.dumps({"output": str(out), **report}, indent=2))


if __name__ == "__main__":
    main()
