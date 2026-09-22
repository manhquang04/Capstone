"""Priority 6 literature-style attacker development gates.

The runner implements two pre-registered generations:

* GEN_COSINE_TV: global cosine matching plus existing range constraints only.
* GEN_IDLG_STYLE: fixed labels plus the existing balanced-tensor objective.

Despite the historical generation name, GEN_COSINE_TV has no tabular-TV term;
the supervisor froze ``lambda_tabular_tv = 0`` before these gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.local_update import simulate
from dna_encoder.transform_defense import DNATransformConfig
from dna_encoder.transform_defense_v2 import DNATransformV2Config
from dna_encoder.transform_defense_v3 import (
    DNATransformV3Config,
    V3_SENSITIVE_PARAMETER_NAMES,
    transform_update_state_v3,
)
from experiments import fraud_fl_common as common
from experiments.phase3_bounded_validation import _align_for_evaluation, update_objective
from experiments.run_ieee_cis_rq1_development_gate import (
    FeatureContract,
    align_all_records,
    capture as capture_ieee,
    decode as decode_ieee,
    fit_contract,
    init_latent as init_ieee,
    make_groups as make_ieee_groups,
    read_ieee_frame,
    score_reconstruction as score_ieee,
    transform_frame,
    warmup_model,
)
from experiments.run_phase4_dna_v2_iht_attack import _capture_for_iht as capture_paysim
from experiments.run_phase3_full_client import checksum, dump, score as score_paysim
from experiments.run_phase4_dna_level1_forward_attack import (
    _apply_surrogate_realization_torch,
    _surrogate_plan_from_state,
    _transmit_observed,
)
from experiments.run_phase4_dna_v2_sketch_space_attack import (
    _candidate_sketches,
    _observed_sketches,
)
from experiments.run_phase4_harddiff_reparam_for_misselected import (
    _decode_harddiff,
    _feature_distribution,
    _initial as init_paysim,
    _max_balance_residual,
    _nonnegative_penalty,
)
from models.fraud_mlp import FraudMLP
from privacy.seed_manager import derive_seed


AMENDMENT = "protocols/amendments/2026-09-16_priority6_sota_style_attackers_design.md"
GENERATIONS = ("GEN_COSINE_TV", "GEN_IDLG_STYLE")
DEFENSES = (
    "v1_conservative",
    "v1_medium",
    "v1_stronger",
    "v2_ratio0p95_eta0p01",
    "v2_ratio0p9_eta0p01",
    "v3_sensitive_final_layer_ratio0p50_rest0p95_eta0p01",
)
V1_CONFIGS = {
    "v1_conservative": dict(block_size=256, mix_ratio=0.08, keep_ratio=0.88, shrink_factor=0.45, seed=681958327),
    "v1_medium": dict(block_size=256, mix_ratio=0.10, keep_ratio=0.85, shrink_factor=0.40, seed=681958327),
    "v1_stronger": dict(block_size=256, mix_ratio=0.12, keep_ratio=0.82, shrink_factor=0.35, seed=681958327),
}
V2_CONFIGS = {
    "v2_ratio0p95_eta0p01": dict(compression_ratio=0.95, quantization_eta=0.01, seed=20260916),
    "v2_ratio0p9_eta0p01": dict(compression_ratio=0.90, quantization_eta=0.01, seed=20260916),
}
V3_CONFIGS = {
    "v3_sensitive_final_layer_ratio0p50_rest0p95_eta0p01": dict(
        rest_compression_ratio=0.95,
        sensitive_compression_ratio=0.50,
        quantization_eta=0.01,
        seed=20260916,
        sensitive_parameter_names=V3_SENSITIVE_PARAMETER_NAMES,
    ),
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _source_rows_from_ieee_target(path: Path) -> set[int]:
    try:
        payload = torch.load(path, weights_only=False)
    except (EOFError, KeyError, RuntimeError, TypeError, ValueError):
        return set()
    rows: set[int] = set()
    if isinstance(payload, list):
        for group in payload:
            if isinstance(group, dict):
                rows.update(int(v) for v in group.get("source_rows", []))
    elif isinstance(payload, dict):
        rows.update(int(v) for v in payload.get("source_rows", []))
        for group in payload.get("groups", []):
            if isinstance(group, dict):
                rows.update(int(v) for v in group.get("source_rows", []))
    return rows


def prepare_ieee(
    output: Path, seed: int, groups_count: int, max_rows: int, amendment: str
) -> None:
    """Create a fresh IEEE bundle and prove target-source disjointness."""
    torch.set_num_threads(1)
    _seed_everything(seed)
    output.mkdir(parents=True, exist_ok=False)

    historical: dict[str, set[int]] = {}
    excluded: set[int] = set()
    candidate_paths = set((ROOT / "artifacts/priority5_ieee_cis").rglob("*targets.pt"))
    candidate_paths.update((ROOT / "artifacts").rglob("ieee_priority6_bundle.pt"))
    for path in sorted(candidate_paths):
        rows = _source_rows_from_ieee_target(path)
        historical[str(path.relative_to(ROOT))] = rows
        excluded.update(rows)

    frame = read_ieee_frame(0, derive_seed(seed, "ieee-full-frame"))
    available = frame.loc[~frame["_source_row"].isin(excluded)].reset_index(drop=True)
    sampled, _ = train_test_split(
        available,
        train_size=min(max_rows, len(available)),
        random_state=derive_seed(seed, "priority6-ieee-sample"),
        stratify=available["isFraud"],
    )
    sampled = sampled.reset_index(drop=True)
    warmup, target_pool = train_test_split(
        sampled,
        test_size=0.20,
        random_state=derive_seed(seed, "priority6-ieee-target-pool"),
        stratify=sampled["isFraud"],
    )
    warmup = warmup.reset_index(drop=True)
    target_pool = target_pool.reset_index(drop=True)
    contract, scaler, encoder, medians = fit_contract(warmup)
    warmup_x = transform_frame(warmup, contract, scaler, encoder, medians)
    warmup_y = warmup["isFraud"].astype(np.float32).to_numpy()
    target_x = transform_frame(target_pool, contract, scaler, encoder, medians)
    state, pos_weight = warmup_model(
        warmup_x,
        warmup_y,
        contract.input_dim,
        derive_seed(seed, "priority6-ieee-warmup"),
        clients=3,
        rounds=1,
        batch_size=1024,
    )
    targets = make_ieee_groups(
        target_pool,
        target_x,
        groups_count,
        derive_seed(seed, "priority6-ieee-targets"),
    )
    new_rows = {int(v) for group in targets for v in group["source_rows"]}
    overlap = {path: len(new_rows & rows) for path, rows in historical.items()}
    warmup_rows = set(warmup["_source_row"].astype(int).tolist())
    if max(overlap.values(), default=0) != 0 or new_rows & warmup_rows:
        raise RuntimeError("IEEE Priority 6 source-disjointness gate failed")

    bundle = {
        "targets": targets,
        "contract": contract,
        "state_dict": state,
        "pos_weight": pos_weight,
        "seed": seed,
        "local_lr": 0.001,
    }
    torch.save(bundle, output / "ieee_priority6_bundle.pt")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": amendment,
        "dataset": "IEEE-CIS",
        "seed": seed,
        "groups": groups_count,
        "records_per_group": 4,
        "fraud_per_group": 1,
        "new_source_rows": sorted(new_rows),
        "historical_target_files_checked": len(historical),
        "overlaps": overlap,
        "max_overlap": max(overlap.values(), default=0),
        "warmup_target_overlap": len(new_rows & warmup_rows),
        "disjointness_gate": "PASS",
        "dataset_sha256": checksum(ROOT / "datasets/ieee-fraud-detection/train_transaction.csv"),
        "bundle_sha256": _sha(output / "ieee_priority6_bundle.pt"),
    }
    dump(output / "ieee_target_firewall.json", manifest)
    print(json.dumps(manifest, indent=2))


def _global_cosine(candidate: dict[str, torch.Tensor], signal: dict[str, torch.Tensor]) -> torch.Tensor:
    keys = [key for key in signal if key in candidate and signal[key].is_floating_point()]
    if not keys:
        raise ValueError("No floating tensors for cosine objective")
    left = torch.cat([candidate[key].reshape(-1) for key in keys])
    right = torch.cat([signal[key].reshape(-1).to(dtype=left.dtype) for key in keys])
    return 1.0 - F.cosine_similarity(left, right, dim=0, eps=1e-12)


def _defense_payload(observed: dict[str, torch.Tensor], defense: str, group_id: int, realization: int):
    if defense in V1_CONFIGS:
        config = DNATransformConfig(**V1_CONFIGS[defense])
        transmitted = _transmit_observed(observed, config)
        plan = _surrogate_plan_from_state(observed, config, group_id, realization)
        return transmitted, ("v1", plan)
    if defense in V2_CONFIGS:
        config = DNATransformV2Config(**V2_CONFIGS[defense])
        payload = _observed_sketches(observed, config, group_id)
        sketch = {key: item["sketch"].to(dtype=observed[key].dtype) for key, item in payload.items()}
        return sketch, ("v2", payload)
    if defense in V3_CONFIGS:
        config = DNATransformV3Config(**V3_CONFIGS[defense])
        arrays = {
            key: value.detach().cpu().numpy().astype(np.float32, copy=False)
            for key, value in observed.items()
            if value.is_floating_point()
        }
        sketches, metadata = transform_update_state_v3(arrays, config, quantization_seed=group_id)
        reference = next(value for value in observed.values() if value.is_floating_point())
        signal = {
            block: torch.from_numpy(sketch.copy()).to(dtype=reference.dtype)
            for block, sketch in sketches.items()
        }
        payload = {
            "metadata": metadata,
            "sketches": signal,
            "config": config,
        }
        return signal, ("v3", payload)
    raise ValueError(defense)


def _candidate_defended(delta: dict[str, torch.Tensor], payload, defense: str) -> dict[str, torch.Tensor]:
    kind, plan = payload
    if kind == "v1":
        return _apply_surrogate_realization_torch(delta, plan)
    if kind == "v2":
        v2 = V2_CONFIGS[defense]
        args = argparse.Namespace(
            compression_ratio=v2["compression_ratio"],
            quantization_eta=v2["quantization_eta"],
            v2_base_seed=v2["seed"],
            ste_quantization=True,
        )
        return _candidate_sketches(delta, plan, args)
    if kind == "v3":
        return _candidate_v3_sketches(delta, plan, defense)
    raise ValueError(kind)


def _flatten_v3_blocks_torch(delta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    parts: dict[str, list[torch.Tensor]] = {"rest": [], "sensitive": []}
    sensitive = set(V3_SENSITIVE_PARAMETER_NAMES)
    for name, value in delta.items():
        if not value.is_floating_point():
            continue
        block = "sensitive" if name in sensitive else "rest"
        parts[block].append(value.reshape(-1))
    out = {}
    reference = next(value for value in delta.values() if value.is_floating_point())
    for block, values in parts.items():
        out[block] = torch.cat(values) if values else torch.zeros(0, dtype=reference.dtype, device=reference.device)
    return out


def _candidate_v3_sketches(delta: dict[str, torch.Tensor], payload: dict, defense: str) -> dict[str, torch.Tensor]:
    from experiments.run_phase4_dna_v2_sketch_space_attack import _candidate_sketch_torch

    config = V3_CONFIGS[defense]
    blocks = _flatten_v3_blocks_torch(delta)
    metadata_by_block = {item.block: item.metadata for item in payload["metadata"].block_metadata}
    compression = {
        "rest": float(config["rest_compression_ratio"]),
        "sensitive": float(config["sensitive_compression_ratio"]),
    }
    out = {}
    for tensor_index, block in enumerate(("rest", "sensitive")):
        block_meta = metadata_by_block[block]
        out[block] = _candidate_sketch_torch(
            blocks[block],
            compression_ratio=compression[block],
            quantization_eta=float(config["quantization_eta"]),
            base_seed=int(config["seed"]),
            tensor_index=tensor_index,
            quantization_delta=float(block_meta.quantization_delta),
            ste_quantization=True,
        )
    return out


def _loss(candidate, signal, reference, generation, range_penalty):
    if generation == "GEN_COSINE_TV":
        return _global_cosine(candidate, signal) + range_penalty
    if generation == "GEN_IDLG_STYLE":
        keys = [key for key, value in signal.items() if value.is_floating_point()]
        return update_objective(candidate, signal, keys, reference=reference, mode="balanced_tensor") + range_penalty
    raise ValueError(generation)


def _canonical_ieee_labels(records: int, observed: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    labels = torch.zeros((records, 1), dtype=torch.float32)
    labels[0, 0] = 1.0
    bias_candidates = [(key, value) for key, value in observed.items() if value.is_floating_point() and value.numel() == 1]
    key, value = bias_candidates[-1] if bias_candidates else (None, torch.tensor(float("nan")))
    diagnostic = {
        "rule": "public one-fraud-of-four label multiset in canonical permutation; no target-row label membership",
        "last_scalar_update_key": key,
        "last_scalar_update_value": float(value.reshape(-1)[0]),
    }
    return labels, diagnostic


def _paysim_job(job: dict) -> dict:
    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    target_path = Path(job["target"])
    groups = torch.load(target_path, weights_only=False)
    group = groups[job["group"]]
    distribution = _feature_distribution()
    meta = distribution[0]
    local_seed = derive_seed(job["seed"], "priority6-paysim-local", job["group"])
    model, criterion, original, true_labels, batches, rng, observed = capture_paysim(group, local_seed, 4)
    initial = init_paysim(
        original.shape,
        derive_seed(job["seed"], "priority6-paysim-init", job["generation"], job["defense"], job["group"], job["restart"]),
        "standard",
        distribution,
    )
    prior = _align_for_evaluation(original, _decode_harddiff(initial, meta), true_labels)
    observed_defended, payload = _defense_payload(observed, job["defense"], job["group"], job["restart"])
    rows = []
    Path(job["folder"]).mkdir(parents=True, exist_ok=True)
    # Score() writes after the folder exists; write the prior now.
    prior_metrics = score_paysim(original, prior, true_labels, meta, Path(job["folder"]) / "prior.csv")
    for method in ("baseline", "zero_update"):
        signal = observed_defended if method == "baseline" else {k: torch.zeros_like(v) for k, v in observed_defended.items()}
        latent = initial.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=0.1)
        best = float("inf")
        best_step = 0
        best_latent = latent.detach().clone()
        for step in range(601):
            reconstruction = _decode_harddiff(latent, meta)
            delta = simulate(model, criterion, reconstruction, true_labels, batches, rng)
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
        metrics = score_paysim(original, aligned, true_labels, meta, Path(job["folder"]) / f"{method}.csv")
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
            "objective": "global_cosine_plus_range_only" if job["generation"] == "GEN_COSINE_TV" else "fixed_label_balanced_tensor",
            "lambda_tabular_tv": 0.0,
            "max_balance_residual": _max_balance_residual(reconstruction, meta),
        }
        torch.save(artifact, Path(job["folder"]) / f"{method}.pt")
        rows.append({"method": method, "objective": best, "best_step": best_step, "metrics": metrics, "prior": prior_metrics})
    dump(Path(job["folder"]) / "results.json", rows)
    return {**{k: job[k] for k in ("dataset", "defense", "generation", "group", "restart")}, "status": "SUCCESS", "rows": rows}


def _ieee_job(job: dict) -> dict:
    torch.set_num_threads(1)
    bundle = torch.load(job["bundle"], weights_only=False)
    group = bundle["targets"][job["group"]]
    contract: FeatureContract = bundle["contract"]
    model, criterion, original, true_labels, rng, observed = capture_ieee(
        group,
        bundle["state_dict"],
        bundle["pos_weight"],
        derive_seed(job["seed"], "priority6-ieee-local", job["group"]),
        bundle["local_lr"],
    )
    init_numeric, init_cats, init_label_logits = init_ieee(
        len(original), contract, derive_seed(job["seed"], "priority6-ieee-init", job["generation"], job["defense"], job["group"], job["restart"])
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
            delta = simulate(model, criterion, candidate_x, candidate_y, batches, rng, lr=bundle["local_lr"])
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
            "objective": "global_cosine_plus_range_only" if job["generation"] == "GEN_COSINE_TV" else "fixed_label_balanced_tensor",
            "lambda_tabular_tv": 0.0,
        }
        torch.save(artifact, folder / f"{method}.pt")
        rows.append({"method": method, "objective": best, "best_step": best_step, "metrics": metrics, "prior": prior_metrics})
    dump(folder / "results.json", rows)
    return {**{k: job[k] for k in ("dataset", "defense", "generation", "group", "restart")}, "status": "SUCCESS", "rows": rows}


def _sign_tail(wins: int, total: int) -> float:
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / 2**total if total else 1.0


def _evaluate(records: list[dict], dataset: str) -> dict:
    selected = []
    for group in sorted({row["group"] for row in records}):
        candidates = [row for row in records if row["group"] == group]
        baseline_pool = [(row, value) for row in candidates for value in row["rows"] if value["method"] == "baseline"]
        zero_pool = [(row, value) for row in candidates for value in row["rows"] if value["method"] == "zero_update"]
        baseline_job, baseline = min(baseline_pool, key=lambda pair: pair[1]["objective"])
        zero_job, zero = min(zero_pool, key=lambda pair: pair[1]["objective"])
        metric = (lambda result: result["metrics"]["classes"]["1"]["mean_mse"]) if dataset == "paysim" else (lambda result: result["metrics"]["mean_mse"])
        prior_get = (lambda result: result["prior"]["classes"]["1"]["mean_mse"]) if dataset == "paysim" else (lambda result: result["prior"]["mean_mse"])
        prior_values = [prior_get(value) for row in candidates for value in row["rows"] if value["method"] == "baseline"]
        prior = float(np.mean(prior_values))
        selected.append({
            "group": group,
            "baseline_restart": baseline_job["restart"],
            "zero_restart": zero_job["restart"],
            "baseline_mse": float(metric(baseline)),
            "prior_mse": prior,
            "zero_mse": float(metric(zero)),
            "baseline_minus_prior": float(metric(baseline) - prior),
            "baseline_minus_zero": float(metric(baseline) - metric(zero)),
        })
    gates = {}
    for control in ("prior", "zero"):
        values = np.asarray([row[f"baseline_minus_{control}"] for row in selected], dtype=float)
        non_ties = values[values != 0]
        wins = int((non_ties < 0).sum())
        gates[control] = {
            "wins": wins,
            "non_ties": int(len(non_ties)),
            "mean_difference": float(values.mean()),
            "median_difference": float(np.median(values)),
            "one_sided_exact_sign_p": _sign_tail(wins, len(non_ties)),
        }
        gates[control]["pass"] = bool(gates[control]["mean_difference"] < 0 and gates[control]["median_difference"] < 0 and gates[control]["one_sided_exact_sign_p"] < 0.05)
    gates["overall"] = bool(gates["prior"]["pass"] and gates["zero"]["pass"])
    return {"gate": gates, "selected": selected}


def execute(args) -> None:
    torch.set_num_threads(1)
    output = args.output.resolve()
    if args.resume:
        if not output.is_dir() or not (output / "execution_manifest.json").exists():
            raise RuntimeError("--resume requires an existing Priority 6 execution directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
    jobs = []
    defenses = tuple(args.defenses or DEFENSES)
    generations = tuple(args.generations or GENERATIONS)
    invalid_defenses = sorted(set(defenses) - set(DEFENSES))
    invalid_generations = sorted(set(generations) - set(GENERATIONS))
    if invalid_defenses or invalid_generations:
        raise ValueError(f"invalid defenses={invalid_defenses} generations={invalid_generations}")
    for defense in defenses:
        for generation in generations:
            for group in range(args.groups):
                for restart in range(4):
                    folder = output / args.dataset / defense / generation / f"group_{group}" / f"restart_{restart}"
                    jobs.append({
                        "dataset": args.dataset,
                        "defense": defense,
                        "generation": generation,
                        "group": group,
                        "restart": restart,
                        "seed": args.seed,
                        "folder": str(folder),
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
    }
    if not args.resume:
        dump(output / "execution_manifest.json", manifest)
    worker = _paysim_job if args.dataset == "paysim" else _ieee_job
    records = []
    pending = []
    for job in jobs:
        result_path = Path(job["folder"]) / "results.json"
        if args.resume and result_path.exists():
            records.append({
                **{key: job[key] for key in ("dataset", "defense", "generation", "group", "restart")},
                "status": "SUCCESS_REUSED",
                "rows": json.loads(result_path.read_text()),
            })
        else:
            pending.append(job)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, job) for job in pending]
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
    dump(output / "priority6_gate_report.json", report)
    print(json.dumps({"completed_jobs": len(records), "cells": {key: value["gate"] for key, value in summaries.items()}}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-ieee")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--groups", type=int, default=8)
    prepare.add_argument("--max-rows", type=int, default=50000)
    prepare.add_argument("--amendment", default=AMENDMENT)
    run = sub.add_parser("execute")
    run.add_argument("--dataset", choices=("paysim", "ieee"), required=True)
    run.add_argument("--target", type=Path)
    run.add_argument("--bundle", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--groups", type=int, default=8)
    run.add_argument("--defenses", nargs="+", choices=DEFENSES)
    run.add_argument("--generations", nargs="+", choices=GENERATIONS)
    run.add_argument("--amendment", default=AMENDMENT)
    args = parser.parse_args()
    if args.command == "prepare-ieee":
        prepare_ieee(
            args.output.resolve(),
            args.seed,
            args.groups,
            args.max_rows,
            args.amendment,
        )
    else:
        if not 1 <= args.workers <= 9:
            raise ValueError("workers must be in [1, 9]")
        if args.groups < 1:
            raise ValueError("--groups must be positive")
        if args.dataset == "paysim" and args.target is None:
            parser.error("--target is required for PaySim")
        if args.dataset == "ieee" and args.bundle is None:
            parser.error("--bundle is required for IEEE")
        execute(args)


if __name__ == "__main__":
    main()
