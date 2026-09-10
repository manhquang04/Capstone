"""Paired adaptive DNA evaluation after a validated baseline attack gate."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from attacks.adaptive_dna import transform_state_bpda, transform_state_exact
from attacks.local_update import simulate
from attacks.tabular_parameterization import update_matching_objective
from dna_encoder.encoder import DNAEncoder
from dna_encoder.transform_defense import DNATransformConfig
from experiments.phase3_bounded_validation import _align_for_evaluation
from experiments.run_attack_redevelopment import (
    decoder_for,
    _initial_latent,
    select_by_attacker_objective,
    sign_tail,
)
from experiments.run_phase3_adam_ladder import capture, metadata
from experiments.run_phase3_full_client import checksum, dump, score
from privacy.seed_manager import derive_seed


def lossless_round_trip(state):
    encoder = DNAEncoder(key=b"phase4-lossless-comparator-key!!")
    restored = {}
    for key, value in state.items():
        if not value.is_floating_point():
            restored[key] = value.detach().clone()
            continue
        array = value.detach().cpu().numpy().astype("float32", copy=False)
        decoded = encoder.decode_array(encoder.encode_array(array), array.shape)
        restored[key] = torch.from_numpy(decoded.copy()).to(dtype=value.dtype)
    return restored


def run_dna_trial(folder, group, group_id, restart, protocol, frozen):
    result_path = folder / "results.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved["group_id"] != group_id or saved["restart"] != restart:
            raise RuntimeError(f"Resume artifact identity mismatch: {result_path}")
        return saved
    # A terminated process may leave a partial folder. Completed trials are
    # immutable; only a folder without results.json is recomputed.
    folder.mkdir(parents=True, exist_ok=True)
    run_seed = protocol["run_seed"]
    model, criterion, original, labels, batches, rng, raw_observed = capture(
        group, derive_seed(run_seed, "local", group_id), 4
    )
    transform_seed = derive_seed(run_seed, "adaptive-dna", group_id)
    transform_config = DNATransformConfig(
        block_size=256,
        mix_ratio=0.08,
        keep_ratio=0.88,
        shrink_factor=0.45,
        seed=transform_seed,
    )
    observed = transform_state_exact(raw_observed, transform_config)
    keys = [key for key, value in observed.items() if value.is_floating_point()]
    decode, latent_dim = decoder_for(frozen["variant"], dtype=original.dtype)
    initial = _initial_latent(run_seed, group_id, restart, latent_dim, len(original))
    latent = initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=frozen["attack_lr"])
    best_objective = float("inf")
    best_step = 0
    best_latent = latent.detach().clone()
    history = []
    started = time.perf_counter()
    for step in range(frozen["iterations"] + 1):
        reconstruction = decode(latent)
        raw_candidate = simulate(model, criterion, reconstruction, labels, batches, rng)
        transformed_candidate = transform_state_bpda(raw_candidate, transform_config)
        loss = update_matching_objective(
            transformed_candidate,
            observed,
            keys,
            mode="balanced_bn",
            bn_weight=1.0,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite DNA objective at step {step}")
        objective = float(loss.detach())
        history.append(objective)
        if objective < best_objective:
            best_objective = objective
            best_step = step
            best_latent = latent.detach().clone()
        if step < frozen["iterations"]:
            gradient, = torch.autograd.grad(loss, latent)
            if not torch.isfinite(gradient).all():
                raise RuntimeError(f"Non-finite DNA latent gradient at step {step}")
            optimizer.zero_grad()
            latent.grad = gradient
            optimizer.step()

    reconstruction = decode(best_latent).detach()
    aligned = _align_for_evaluation(original, reconstruction, labels)
    metrics = score(original, aligned, labels, metadata(), folder / "dna_transform.csv")
    artifact = {
        "method": "dna_transform_bpda",
        "group_id": group_id,
        "restart": restart,
        "source_ids": group["source_ids"],
        "original": original,
        "labels": labels,
        "initial_latent": initial,
        "best_latent": best_latent,
        "best_reconstruction": reconstruction,
        "aligned_reconstruction": aligned,
        "best_objective": best_objective,
        "best_step": best_step,
        "history": history,
        "transform_seed": transform_seed,
        "transform_config": transform_config.__dict__,
        "adaptive_model": "exact DNA forward transform with identity BPDA backward",
    }
    path = folder / "dna_transform.pt"
    torch.save(artifact, path)
    loaded = torch.load(path, weights_only=False)
    loaded_raw = simulate(model, criterion, loaded["best_reconstruction"], labels, batches, rng)
    loaded_transformed = transform_state_exact(loaded_raw, transform_config)
    reloaded_objective = float(
        update_matching_objective(
            loaded_transformed, observed, keys, mode="balanced_bn", bn_weight=1.0
        ).detach()
    )
    if not math.isclose(reloaded_objective, best_objective, rel_tol=1e-5, abs_tol=1e-10):
        raise AssertionError("Reloaded adaptive DNA candidate does not match best objective")
    row = {
        "method": "dna_transform_bpda",
        "group_id": group_id,
        "restart": restart,
        "objective": best_objective,
        "reloaded_objective": reloaded_objective,
        "best_step": best_step,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "transform_seed": transform_seed,
    }
    dump(folder / "results.json", row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    report = json.loads((run / "attack_redevelopment_report.json").read_text())
    if not report["baseline_gate_passed"]:
        raise RuntimeError("Baseline confirmation gate did not pass; DNA evaluation is not valid")
    protocol = json.loads((run / "protocol_lock.json").read_text())
    frozen = json.loads((run / "frozen.json").read_text())
    if checksum(run / "protocol_lock.json") != frozen["protocol_sha256"]:
        raise RuntimeError("Protocol changed after attack configuration was frozen")
    groups = torch.load(run / "confirmation_targets.pt", weights_only=False)

    # Lossless encoding must not change the attack signal. This is checked on
    # every confirmation group before reusing baseline results.
    lossless_checks = []
    for group_id, group in enumerate(groups):
        _, _, _, _, _, _, observed = capture(group, derive_seed(protocol["run_seed"], "local", group_id), 4)
        restored = lossless_round_trip(observed)
        max_error = max(float((restored[key] - observed[key]).abs().max()) for key in observed)
        lossless_checks.append({"group_id": group_id, "max_abs_error": max_error})
        if max_error != 0.0:
            raise AssertionError("Lossless DNA update path changed the observed signal")

    dna_rows = []
    for group_id, group in enumerate(groups):
        for restart in range(frozen["restarts"]):
            dna_rows.append(
                run_dna_trial(
                    run / "dna_confirmation" / f"group_{group_id}" / f"restart_{restart}",
                    group,
                    group_id,
                    restart,
                    protocol,
                    frozen,
                )
            )
    baseline_rows = json.loads((run / "confirmation_records.json").read_text())
    selected = []
    for group_id in range(len(groups)):
        baseline = select_by_attacker_objective(baseline_rows, group_id, "baseline")
        dna = min(
            (row for row in dna_rows if row["group_id"] == group_id),
            key=lambda row: row["objective"],
        )
        difference = dna["metrics"]["mean_mse"] - baseline["metrics"]["mean_mse"]
        selected.append(
            {
                "group_id": group_id,
                "baseline_mse": baseline["metrics"]["mean_mse"],
                "dna_mse": dna["metrics"]["mean_mse"],
                "dna_minus_baseline": difference,
                "baseline_restart": baseline["restart"],
                "dna_restart": dna["restart"],
                "transform_seed": dna["transform_seed"],
            }
        )
    differences = np.asarray([row["dna_minus_baseline"] for row in selected])
    non_ties = differences[differences != 0]
    privacy_wins = int((non_ties > 0).sum())
    result = {
        "scope": "paired bounded one-batch Adam confirmation; same targets, initialization seeds, budget, labels, order, RNG, and checkpoint",
        "lossless_dna": {
            "max_abs_error": max(row["max_abs_error"] for row in lossless_checks),
            "interpretation": "bit-exact transport comparator; reconstruction equals the validated raw baseline",
        },
        "dna_transform": {
            "adaptive_attack": "exact forward DNA transform, identity BPDA backward; transform config and seed disclosed",
            "n": len(selected),
            "higher_mse_wins": privacy_wins,
            "mean_mse_difference": float(differences.mean()),
            "median_mse_difference": float(np.median(differences)),
            "one_sided_sign_p": sign_tail(privacy_wins, len(non_ties)),
            "mean_baseline_mse": float(np.mean([row["baseline_mse"] for row in selected])),
            "mean_dna_mse": float(np.mean([row["dna_mse"] for row in selected])),
        },
        "limitations": [
            "BPDA is an adaptive approximation, not an exact derivative of the discrete DNA seed/permutation rule.",
            "The result applies to a four-record, one-step Adam update under attacker-favorable knowledge.",
            "It does not establish formal privacy or inversion resistance for full-client FedAvg updates.",
        ],
        "selected": selected,
    }
    with (run / "dna_confirmation_selected.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    dump(run / "dna_confirmation_records.json", dna_rows)
    dump(run / "dna_confirmation_report.json", result)
    report["dna_result"] = result
    report["dna_reason"] = None
    dump(run / "attack_redevelopment_report.json", report)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
