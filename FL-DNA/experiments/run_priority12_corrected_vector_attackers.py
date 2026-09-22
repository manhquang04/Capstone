"""Priority 12 corrected-vector tabular attacker gates.

This runner reuses the Priority 6 SOTA-style attacker logic but applies DNA
transforms and attacker matching losses only to trainable parameters.  BatchNorm
buffers and all other model buffers are excluded before defense construction and
candidate matching.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.local_update import simulate
from experiments import fraud_fl_common as common
from experiments.phase3_bounded_validation import _align_for_evaluation
from experiments.run_phase4_dna_v2_iht_attack import _capture_for_iht as capture_paysim
from experiments.run_phase4_harddiff_reparam_for_misselected import (
    _decode_harddiff,
    _feature_distribution,
    _initial as init_paysim,
    _max_balance_residual,
    _nonnegative_penalty,
)
from experiments.run_phase3_full_client import dump, score as score_paysim
from experiments.run_priority6_sota_style_attackers import (
    DEFENSES,
    GENERATIONS,
    _canonical_ieee_labels,
    _defense_payload,
    _candidate_defended,
    _evaluate,
    _loss,
    init_ieee,
    decode_ieee,
    score_ieee,
    capture_ieee,
    prepare_ieee,
)
from privacy.seed_manager import derive_seed


AMENDMENT = "protocols/amendments/2026-09-21_priority12_corrected_vector_reconfirmation.md"


def _trainable_names(model: torch.nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def _filter_trainable(state: dict[str, torch.Tensor], names: list[str]) -> dict[str, torch.Tensor]:
    missing = [name for name in names if name not in state]
    if missing:
        raise KeyError(f"missing trainable keys in update state: {missing[:5]}")
    return {name: state[name] for name in names}


def _paysim_job(job: dict) -> dict:
    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    target_path = Path(job["target"])
    groups = torch.load(target_path, weights_only=False)
    group = groups[job["group"]]
    distribution = _feature_distribution()
    meta = distribution[0]
    local_seed = derive_seed(job["seed"], "priority12-paysim-local", job["group"])
    model, criterion, original, true_labels, batches, rng, observed_full = capture_paysim(group, local_seed, 4)
    trainable = _trainable_names(model)
    observed = _filter_trainable(observed_full, trainable)
    initial = init_paysim(
        original.shape,
        derive_seed(job["seed"], "priority12-paysim-init", job["generation"], job["defense"], job["group"], job["restart"]),
        "standard",
        distribution,
    )
    prior = _align_for_evaluation(original, _decode_harddiff(initial, meta), true_labels)
    observed_defended, payload = _defense_payload(observed, job["defense"], job["group"], job["restart"])
    rows = []
    folder = Path(job["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    prior_metrics = score_paysim(original, prior, true_labels, meta, folder / "prior.csv")
    for method in ("baseline", "zero_update"):
        signal = observed_defended if method == "baseline" else {k: torch.zeros_like(v) for k, v in observed_defended.items()}
        latent = initial.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=0.1)
        best = float("inf")
        best_step = 0
        best_latent = latent.detach().clone()
        for step in range(601):
            reconstruction = _decode_harddiff(latent, meta)
            delta_full = simulate(model, criterion, reconstruction, true_labels, batches, rng)
            delta = _filter_trainable(delta_full, trainable)
            defended = _candidate_defended(delta, payload, job["defense"])
            penalty = 0.001 * _nonnegative_penalty(latent, meta)
            loss = _loss(defended, signal, observed_defended, job["generation"], penalty)
            value = float(loss.detach())
            if value < best:
                best, best_step, best_latent = value, step, latent.detach().clone()
            if step < 600:
                (gradient,) = torch.autograd.grad(loss, latent)
                optimizer.zero_grad()
                latent.grad = gradient
                optimizer.step()
        reconstruction = _decode_harddiff(best_latent, meta).detach()
        aligned = _align_for_evaluation(original, reconstruction, true_labels)
        metrics = score_paysim(original, aligned, true_labels, meta, folder / f"{method}.csv")
        artifact = {
            "dataset": "PaySim",
            "generation": job["generation"],
            "defense": job["defense"],
            "method": method,
            "group_id": job["group"],
            "restart": job["restart"],
            "source_ids": group["source_ids"],
            "original": original,
            "labels": true_labels,
            "initial": initial,
            "reconstruction": reconstruction,
            "aligned": aligned,
            "best_objective": best,
            "best_step": best_step,
            "objective": "corrected_trainable_vector_global_cosine" if job["generation"] == "GEN_COSINE_TV" else "corrected_trainable_vector_fixed_label_balanced_tensor",
            "lambda_tabular_tv": 0.0,
            "vector_scope": "trainable_parameters_only",
            "excluded_buffers": "all model.named_buffers() including BatchNorm running stats",
            "trainable_keys": trainable,
            "max_balance_residual": _max_balance_residual(reconstruction, meta),
        }
        torch.save(artifact, folder / f"{method}.pt")
        rows.append({"method": method, "objective": best, "best_step": best_step, "metrics": metrics, "prior": prior_metrics})
    dump(folder / "results.json", rows)
    return {**{k: job[k] for k in ("dataset", "defense", "generation", "group", "restart")}, "status": "SUCCESS", "rows": rows}


def _ieee_job(job: dict) -> dict:
    torch.set_num_threads(1)
    bundle = torch.load(job["bundle"], weights_only=False)
    group = bundle["targets"][job["group"]]
    contract = bundle["contract"]
    local_seed = derive_seed(job["seed"], "priority12-ieee-local", job["group"])
    model, criterion, original, true_labels, rng, observed_full = capture_ieee(
        group,
        bundle["state_dict"],
        bundle["pos_weight"],
        local_seed,
        bundle["local_lr"],
    )
    trainable = _trainable_names(model)
    observed = _filter_trainable(observed_full, trainable)
    init_numeric, init_cats, init_label_logits = init_ieee(
        len(original),
        contract,
        derive_seed(job["seed"], "priority12-ieee-init", job["generation"], job["defense"], job["group"], job["restart"]),
    )
    with torch.no_grad():
        prior_x, _ = decode_ieee(init_numeric, init_cats, init_label_logits, contract)
    folder = Path(job["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    prior_metrics = score_ieee(original, prior_x, true_labels, contract, folder / "prior.csv")
    observed_defended, payload = _defense_payload(observed, job["defense"], job["group"], job["restart"])
    fixed_labels, label_diagnostic = _canonical_ieee_labels(len(original), observed)
    rows = []
    for method in ("baseline", "zero_update"):
        signal = observed_defended if method == "baseline" else {k: torch.zeros_like(v) for k, v in observed_defended.items()}
        numeric = init_numeric.detach().clone().requires_grad_(True)
        cats = [value.detach().clone().requires_grad_(True) for value in init_cats]
        if job["generation"] == "GEN_COSINE_TV":
            label_logits = init_label_logits.detach().clone().requires_grad_(True)
            params = [numeric, *cats, label_logits]
        else:
            label_logits = None
            params = [numeric, *cats]
        optimizer = torch.optim.Adam(params, lr=0.05)
        best = float("inf")
        best_step = 0
        best_state = None
        for step in range(301):
            if label_logits is None:
                dummy_logits = torch.zeros((len(original), 1), dtype=numeric.dtype)
                candidate_x, _ = decode_ieee(numeric, cats, dummy_logits, contract)
                candidate_y = fixed_labels
            else:
                candidate_x, candidate_y = decode_ieee(numeric, cats, label_logits, contract)
            batches = [torch.arange(len(original))]
            delta_full = simulate(model, criterion, candidate_x, candidate_y, batches, rng, lr=bundle["local_lr"])
            delta = _filter_trainable(delta_full, trainable)
            defended = _candidate_defended(delta, payload, job["defense"])
            loss = _loss(defended, signal, observed_defended, job["generation"], torch.zeros((), dtype=numeric.dtype))
            value = float(loss.detach())
            if value < best:
                best, best_step = value, step
                best_state = (numeric.detach().clone(), [v.detach().clone() for v in cats], None if label_logits is None else label_logits.detach().clone())
            if step < 300:
                gradients = torch.autograd.grad(loss, params)
                optimizer.zero_grad()
                for parameter, gradient in zip(params, gradients):
                    parameter.grad = gradient
                optimizer.step()
        assert best_state is not None
        if best_state[2] is None:
            dummy_logits = torch.zeros((len(original), 1), dtype=best_state[0].dtype)
            reconstruction, _ = decode_ieee(best_state[0], best_state[1], dummy_logits, contract)
            reconstructed_labels = fixed_labels
        else:
            reconstruction, reconstructed_labels = decode_ieee(best_state[0], best_state[1], best_state[2], contract)
        metrics = score_ieee(original, reconstruction.detach(), true_labels, contract, folder / f"{method}.csv")
        artifact = {
            "dataset": "IEEE-CIS",
            "generation": job["generation"],
            "defense": job["defense"],
            "method": method,
            "group_id": job["group"],
            "restart": job["restart"],
            "source_rows": group["source_rows"],
            "transaction_ids": group["transaction_ids"],
            "original": original,
            "true_labels_evaluation_only": true_labels,
            "attacker_labels": reconstructed_labels.detach(),
            "label_diagnostic": label_diagnostic,
            "reconstruction": reconstruction.detach(),
            "best_objective": best,
            "best_step": best_step,
            "objective": "corrected_trainable_vector_global_cosine" if job["generation"] == "GEN_COSINE_TV" else "corrected_trainable_vector_fixed_label_balanced_tensor",
            "lambda_tabular_tv": 0.0,
            "vector_scope": "trainable_parameters_only",
            "excluded_buffers": "all model.named_buffers() including BatchNorm running stats",
            "trainable_keys": trainable,
        }
        torch.save(artifact, folder / f"{method}.pt")
        rows.append({"method": method, "objective": best, "best_step": best_step, "metrics": metrics, "prior": prior_metrics})
    dump(folder / "results.json", rows)
    return {**{k: job[k] for k in ("dataset", "defense", "generation", "group", "restart")}, "status": "SUCCESS", "rows": rows}


def execute(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    defenses = tuple(args.defenses)
    generations = tuple(args.generations)
    invalid_defenses = sorted(set(defenses) - set(DEFENSES))
    invalid_generations = sorted(set(generations) - set(GENERATIONS))
    if invalid_defenses or invalid_generations:
        raise ValueError(f"invalid defenses={invalid_defenses} generations={invalid_generations}")
    jobs = []
    for defense in defenses:
        for generation in generations:
            for group in range(args.groups):
                for restart in range(4):
                    jobs.append({
                        "dataset": args.dataset,
                        "defense": defense,
                        "generation": generation,
                        "group": group,
                        "restart": restart,
                        "seed": args.seed,
                        "folder": str(output / args.dataset / defense / generation / f"group_{group}" / f"restart_{restart}"),
                        "target": str(args.target.resolve()) if args.target else None,
                        "bundle": str(args.bundle.resolve()) if args.bundle else None,
                    })
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": args.amendment,
        "dataset": args.dataset,
        "jobs": len(jobs),
        "workers": args.workers,
        "groups": args.groups,
        "generations": generations,
        "defenses": defenses,
        "lambda_tabular_tv": 0.0,
        "torch_num_threads": 1,
        "vector_scope": "trainable_parameters_only",
        "excluded_buffers": "all model.named_buffers() including BatchNorm running stats",
    }
    dump(output / "execution_manifest.json", manifest)
    worker = _paysim_job if args.dataset == "paysim" else _ieee_job
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            print(json.dumps({key: row[key] for key in ("dataset", "defense", "generation", "group", "restart", "status")}), flush=True)
    summaries = {}
    for defense in defenses:
        for generation in generations:
            cell = [row for row in records if row["defense"] == defense and row["generation"] == generation]
            summaries[f"{defense}__{generation}"] = _evaluate(cell, args.dataset)
    report = {**manifest, "completed_jobs": len(records), "failed_jobs": 0, "cells": summaries}
    dump(output / "priority12_gate_report.json", report)
    print(json.dumps({"completed_jobs": len(records), "cells": {key: value["gate"] for key, value in summaries.items()}}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-ieee")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--seed", type=int, required=True)
    prep.add_argument("--groups", type=int, default=8)
    prep.add_argument("--max-rows", type=int, default=50000)
    prep.add_argument("--amendment", default=AMENDMENT)
    exe = sub.add_parser("execute")
    exe.add_argument("--dataset", choices=("paysim", "ieee"), required=True)
    exe.add_argument("--target", type=Path)
    exe.add_argument("--bundle", type=Path)
    exe.add_argument("--output", type=Path, required=True)
    exe.add_argument("--seed", type=int, required=True)
    exe.add_argument("--workers", type=int, default=8)
    exe.add_argument("--groups", type=int, required=True)
    exe.add_argument("--defenses", nargs="+", required=True)
    exe.add_argument("--generations", nargs="+", required=True)
    exe.add_argument("--amendment", default=AMENDMENT)
    args = parser.parse_args()
    if args.command == "prepare-ieee":
        prepare_ieee(args.output, args.seed, args.groups, args.max_rows, args.amendment)
    else:
        if args.dataset == "paysim" and args.target is None:
            raise ValueError("--target is required for paysim")
        if args.dataset == "ieee" and args.bundle is None:
            raise ValueError("--bundle is required for ieee")
        execute(args)


if __name__ == "__main__":
    main()

