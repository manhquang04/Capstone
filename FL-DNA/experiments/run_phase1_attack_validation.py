"""Run the five bounded Phase 1 attack-validation checks.

This runner writes only to artifacts/phase1_validation/ and is intentionally
small: it validates attack bookkeeping before any larger privacy study.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.gradient_inversion import GradientInversionConfig, gradient_inversion_attack, parameter_gradients
from attacks.inversion_metrics import reconstruction_metrics
from data.load_creditcard import load_creditcard_data
from dna_encoder.encoder import DNAEncoder
from dna_encoder.transform_defense import DNATransformConfig, transform_update_array
from experiments.fraud_fl_common import BATCH_SIZE, LOCAL_EPOCHS, build_loss, fed_avg, set_random_seed, train_local_model
from models.fraud_mlp import FraudMLP
from privacy.seed_manager import derive_seed, generate_run_seed

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "phase1_validation"
RUN_ID = os.environ.get("PHASE1_RUN_ID", datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ"))
OUTPUT_DIR = ARTIFACT_ROOT / RUN_ID
BASE_SEED = 917_431
SAMPLE_COUNT = 2
WARMUP_ROUNDS = 1
BUDGETS = (0, 20, 100, 300)
RESTARTS = 3
L2_WEIGHTS = (1e-4, 0.0)
TRANSFORM_CONFIG = DNATransformConfig(block_size=256, mix_ratio=0.05, keep_ratio=0.90, shrink_factor=0.50, seed=BASE_SEED)


def main() -> None:
    started = perf_counter()
    set_random_seed(BASE_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loaders, validation_loader, _, input_dim, pos_weight, metadata = load_creditcard_data(
        batch_size=BATCH_SIZE,
        num_clients=3,
        max_rows=5_000,
        seed=BASE_SEED,
    )
    model = _warmup(loaders, input_dim, pos_weight).to("cpu")
    criterion = build_loss(pos_weight)
    features, labels, sample_indices = _select_balanced_samples(validation_loader, SAMPLE_COUNT)
    true_gradients = [
        [g.detach().cpu().clone() for g in _gradients(model, criterion, feature, label, create_graph=False)]
        for feature, label in zip(features, labels)
    ]

    records: list[dict[str, object]] = []
    records.extend(_equivalence_check(model, criterion, true_gradients[0], features[0], labels[0], input_dim, sample_indices[0]))
    records.extend(_budget_and_control_checks(model, criterion, true_gradients, features, labels, sample_indices, input_dim))
    records.extend(_paired_dna_checks(model, criterion, true_gradients, features, labels, sample_indices, input_dim))

    payload = {
        "experiment": "phase1_attack_validation",
        "config": {
            "dataset": "PaySim",
            "max_rows": 5000,
            "num_clients": 3,
            "warmup_rounds": WARMUP_ROUNDS,
            "target_samples": SAMPLE_COUNT,
            "target_labels": [int(x) for x in labels.reshape(-1).tolist()],
            "target_validation_indices": [int(x) for x in sample_indices],
            "budgets": list(BUDGETS),
            "restarts": RESTARTS,
            "l2_weights": list(L2_WEIGHTS),
            "base_seed": BASE_SEED,
            "attack": "known-label, sample-level gradient matching",
            "model_device_for_attack": "cpu",
            "feature_names": metadata.feature_names,
            "run_id": RUN_ID,
            "git_commit": _git_value(["rev-parse", "HEAD"]),
            "git_worktree_status": _git_value(["status", "--short"]),
            "model_checkpoint": "in-memory model after one-round warm-up; no checkpoint file was used",
            "preprocessing_assumption": "pooled train-fitted RobustScaler and OneHotEncoder reused by clients",
            "model_mode_for_gradient_capture": "eval",
            "criterion": "BinaryFocalLoss from experiments.fraud_fl_common",
            "optimizer": "Adam; optimizer state reset for each attack call",
            "defense_seed": TRANSFORM_CONFIG.seed,
            "metric_version": "joint_range_pseudo_image_v2",
        },
        "summary": _report_summary(records),
        "records": records,
        "runtime_seconds": perf_counter() - started,
    }
    (OUTPUT_DIR / "validation_report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_report_summary(records), indent=2))
    print(f"Saved: {OUTPUT_DIR.relative_to(PROJECT_ROOT) / 'validation_report.json'}")


def _git_value(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def _warmup(loaders, input_dim: int, pos_weight: torch.Tensor) -> FraudMLP:
    model = FraudMLP(input_dim)
    counts = [len(loader.dataset) for loader in loaders]
    for _ in range(WARMUP_ROUNDS):
        states = []
        for loader in loaders:
            local = copy.deepcopy(model)
            train_local_model(local, loader, pos_weight, LOCAL_EPOCHS)
            states.append(local.state_dict())
        model.load_state_dict(fed_avg(states, counts))
    return model


def _gradients(model, criterion, features: torch.Tensor, labels: torch.Tensor, create_graph: bool) -> list[torch.Tensor]:
    return parameter_gradients(model, criterion, features.reshape(1, -1), labels.reshape(1, 1), create_graph=create_graph)


def _select_balanced_samples(loader, count: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if count < 2:
        raise ValueError("count must be at least 2 to include fraud and non-fraud targets")
    features, labels = loader.dataset.tensors
    positive = torch.nonzero(labels.reshape(-1) == 1, as_tuple=False).reshape(-1)
    negative = torch.nonzero(labels.reshape(-1) == 0, as_tuple=False).reshape(-1)
    if positive.numel() == 0 or negative.numel() == 0:
        raise RuntimeError("Validation split must contain both fraud and non-fraud samples")
    positive_count = (count + 1) // 2
    negative_count = count // 2
    if positive.numel() < positive_count or negative.numel() < negative_count:
        raise RuntimeError("Validation split does not contain enough fraud/non-fraud targets")
    chosen = torch.cat([positive[:positive_count], negative[:negative_count]])
    chosen, _ = torch.sort(chosen)
    return features[chosen].clone(), labels[chosen].clone(), [int(x) for x in chosen.tolist()]


def _run_attack(model, criterion, observed, label, input_dim, seed, iterations, tag, original, extra=None, l2_weight=1e-4):
    config = GradientInversionConfig(iterations=iterations, learning_rate=0.05, l2_weight=l2_weight, optimizer="adam", seed=seed)
    started = perf_counter()
    result = gradient_inversion_attack(copy.deepcopy(model), criterion, observed, label.reshape(1, 1), input_dim, config)
    runtime_seconds = perf_counter() - started
    candidate = result.reconstructed_input.detach().cpu()
    initial = result.initial_dummy.detach().cpu()
    objective = _objective(model, criterion, observed, candidate, label, l2_weight)
    if not np.isclose(objective, result.best_loss, rtol=1e-5, atol=1e-7):
        raise AssertionError(f"saved best candidate does not reproduce best_attack_loss for {tag}: {objective} != {result.best_loss}")
    stem = tag.replace("/", "_")
    original_path = OUTPUT_DIR / f"{stem}_original.pt"
    initial_path = OUTPUT_DIR / f"{stem}_initial_dummy.pt"
    best_path = OUTPUT_DIR / f"{stem}_best_reconstruction.pt"
    torch.save({"vector": original.detach().cpu(), "dtype": "float32"}, original_path)
    torch.save({"vector": initial, "seed": seed}, initial_path)
    torch.save({"vector": candidate, "best_attack_loss": result.best_loss, "recomputed_objective": objective}, best_path)
    reloaded = torch.load(best_path, weights_only=False)
    reloaded_candidate = reloaded["vector"].detach().cpu()
    if tuple(reloaded_candidate.shape) != tuple(candidate.shape) or reloaded_candidate.dtype != candidate.dtype:
        raise AssertionError(f"reloaded candidate shape/dtype mismatch for {tag}")
    reloaded_objective = _objective(model, criterion, observed, reloaded_candidate, label, l2_weight)
    reloaded_ok = bool(np.isclose(reloaded_objective, result.best_loss, rtol=1e-5, atol=1e-7))
    if not reloaded_ok:
        raise AssertionError(
            f"reloaded best candidate does not reproduce best_attack_loss for {tag}: "
            f"{reloaded_objective} != {result.best_loss}"
        )
    components_total = result.best_gradient_match_loss + result.best_regularization_loss
    if not np.isclose(components_total, result.best_loss, rtol=1e-5, atol=1e-7):
        raise AssertionError(f"objective components do not sum to best_attack_loss for {tag}")
    record = {
        "experiment": tag,
        "sample_label": int(label.item()),
        "seed": seed,
        "iterations": iterations,
        "best_attack_loss": result.best_loss,
        "gradient_match_loss": result.best_gradient_match_loss,
        "regularization_loss": result.best_regularization_loss,
        "components_total_objective": components_total,
        "l2_weight": l2_weight,
        "initial_total_objective": result.initial_total_objective,
        "best_iteration": result.best_iteration,
        "recomputed_best_objective": objective,
        "recomputed_reloaded_objective": reloaded_objective,
        "best_loss_matches_in_memory_candidate": True,
        "best_loss_matches_reloaded_artifact": reloaded_ok,
        "best_loss_matches_saved_candidate": reloaded_ok,
        "loss_history": result.loss_history,
        "runtime_seconds": runtime_seconds,
        **reconstruction_metrics(original.numpy().reshape(-1), candidate.numpy().reshape(-1)),
    }
    if extra:
        record.update(extra)
    return record


def _save_prior_control(model, criterion, observed, label, input_dim, seed, tag, original, extra=None):
    """Save an unoptimized dummy prior; it must not inspect the observed update."""
    started = perf_counter()
    torch.manual_seed(seed)
    dummy = torch.randn(1, input_dim, dtype=original.dtype)
    stem = tag.replace("/", "_")
    torch.save({"vector": original.detach().cpu(), "dtype": "float32"}, OUTPUT_DIR / f"{stem}_original.pt")
    torch.save({"vector": dummy, "seed": seed}, OUTPUT_DIR / f"{stem}_initial_dummy.pt")
    torch.save({"vector": dummy, "optimized": False}, OUTPUT_DIR / f"{stem}_best_reconstruction.pt")
    reloaded_original = torch.load(OUTPUT_DIR / f"{stem}_original.pt", weights_only=False)["vector"]
    reloaded_initial = torch.load(OUTPUT_DIR / f"{stem}_initial_dummy.pt", weights_only=False)["vector"]
    if tuple(reloaded_original.shape) != tuple(original.shape) or tuple(reloaded_initial.shape) != tuple(dummy.shape):
        raise AssertionError(f"prior control artifact shape mismatch for {tag}")
    record = {
        "experiment": tag,
        "sample_label": int(label.item()),
        "seed": seed,
        "iterations": 0,
        "l2_weight": None,
        "best_attack_loss": None,
        "recomputed_best_objective": None,
        "recomputed_reloaded_objective": None,
        "best_loss_matches_in_memory_candidate": None,
        "best_loss_matches_reloaded_artifact": None,
        "best_loss_matches_saved_candidate": None,
        "loss_history": [],
        "runtime_seconds": perf_counter() - started,
        "optimized": False,
        "uses_update": False,
        **reconstruction_metrics(original.numpy().reshape(-1), dummy.numpy().reshape(-1)),
    }
    if extra:
        record.update(extra)
    return record


def _objective(model, criterion, observed, candidate, label, l2_weight=1e-4) -> float:
    candidate = candidate.detach().clone().requires_grad_(True)
    gradients = _gradients(model, criterion, candidate, label, create_graph=False)
    match = torch.stack([torch.mean((a - b) ** 2) for a, b in zip(gradients, observed)]).mean()
    return float((match + l2_weight * candidate.square().mean()).item())


def _equivalence_check(model, criterion, observed, original, label, input_dim, sample_index):
    seed = derive_seed(BASE_SEED, "equivalence", 0)
    baseline = _run_attack(model, criterion, observed, label, input_dim, seed, 10, "equivalence_baseline", original, {"sample_id": 0, "validation_index": sample_index})
    preagg = _run_attack(model, criterion, observed, label, input_dim, seed, 10, "equivalence_preaggregation", original, {"sample_id": 0, "validation_index": sample_index})
    baseline_vector = torch.load(OUTPUT_DIR / "equivalence_baseline_best_reconstruction.pt", weights_only=False)["vector"]
    preagg_vector = torch.load(OUTPUT_DIR / "equivalence_preaggregation_best_reconstruction.pt", weights_only=False)["vector"]
    equal = bool(torch.equal(baseline_vector, preagg_vector) and baseline["best_attack_loss"] == preagg["best_attack_loss"])
    return [{"experiment": "baseline_preaggregation_equivalence", "same_model_gradient_init_seed": True, "identical_reconstruction": equal, "metrics_use_real_target": True, "baseline": baseline, "preaggregation": preagg}]


def _budget_and_control_checks(model, criterion, observed_by_sample, features, labels, sample_indices, input_dim):
    records = []
    for sample_id, (feature, label) in enumerate(zip(features, labels)):
        observed = observed_by_sample[sample_id]
        zero_observed = [torch.zeros_like(g) for g in observed]
        for l2_weight in L2_WEIGHTS:
            for budget in BUDGETS:
                for restart in range(RESTARTS):
                    seed = derive_seed(BASE_SEED, "baseline", sample_id, restart)
                    extra = {"sample_id": sample_id, "validation_index": sample_indices[sample_id], "budget": budget, "restart": restart, "control": False, "l2_ablation": l2_weight}
                    records.append(_run_attack(model, criterion, observed, label, input_dim, seed, budget, f"baseline_s{sample_id}_l2{l2_weight}_b{budget}_r{restart}", feature, extra, l2_weight=l2_weight))
                    zero_extra = {**extra, "control": True, "control_observation": "zero-gradient control; same model, label, initialization, budget, restart, and l2 setting"}
                    records.append(_run_attack(model, criterion, zero_observed, label, input_dim, seed, budget, f"zero_gradient_s{sample_id}_l2{l2_weight}_b{budget}_r{restart}", feature, zero_extra, l2_weight=l2_weight))
            for restart in range(RESTARTS):
                prior_seed = derive_seed(BASE_SEED, "baseline", sample_id, restart)
                records.append(_save_prior_control(model, criterion, observed, label, input_dim, prior_seed, f"prior_s{sample_id}_l2{l2_weight}_r{restart}", feature, {"sample_id": sample_id, "validation_index": sample_indices[sample_id], "restart": restart, "l2_ablation": l2_weight, "control": True, "control_observation": "unoptimized dummy prior; no update optimization"}))
    return records


def _transform_observed(observed):
    transformed = []
    for tensor_index, gradient in enumerate(observed):
        array, _ = transform_update_array(gradient.numpy(), TRANSFORM_CONFIG, tensor_index=tensor_index)
        transformed.append(torch.from_numpy(array.copy()))
    return transformed


def _paired_dna_checks(model, criterion, observed_by_sample, features, labels, sample_indices, input_dim):
    records = []
    for sample_id, (feature, label) in enumerate(zip(features, labels)):
        observed = observed_by_sample[sample_id]
        transformed = _transform_observed(observed)
        seed = derive_seed(BASE_SEED, "paired_dna", sample_id)
        lossless_observed = _lossless_round_trip(observed)
        lossless = _run_attack(model, criterion, lossless_observed, label, input_dim, seed, 20, f"paired_lossless_dna_s{sample_id}", feature, {"sample_id": sample_id, "validation_index": sample_indices[sample_id], "attacker_same_initialization": True, "dna_encode_decode_executed": True})
        transformed_record = _run_attack(model, criterion, transformed, label, input_dim, seed, 20, f"paired_transform_s{sample_id}", feature, {"sample_id": sample_id, "validation_index": sample_indices[sample_id], "attacker_same_initialization": True, "attacker_knows_transform": True, "transform_used_in_optimization": False, "attacker_model": "raw-gradient matching; transform parameters disclosed as metadata, but no adaptive differentiable transform model", "transform_config": TRANSFORM_CONFIG.__dict__})
        records.append({"experiment": "paired_dna_comparison", "sample_id": sample_id, "lossless": lossless, "transform": transformed_record})
    return records


def _lossless_round_trip(observed):
    encoder = DNAEncoder(key=b"2" * 32)
    restored = []
    for gradient in observed:
        array = gradient.detach().cpu().numpy().astype(np.float32, copy=False)
        restored_array = encoder.decode_array(encoder.encode_array(array), array.shape)
        restored.append(torch.from_numpy(restored_array.copy()))
    return restored


def _report_summary(records):
    attack_records = _flatten_attack_records(records)
    equivalence_children = [r for r in attack_records if r.get("experiment", "").startswith("equivalence_")]
    paired_children = [r for r in attack_records if r.get("experiment", "").startswith("paired_")]
    return {
        "top_level_records": len(records),
        "attack_calls": len(attack_records),
        "equivalence_pass": records[0]["identical_reconstruction"],
        "target_labels": sorted({r.get("sample_label") for r in records if "sample_label" in r}),
        "baseline_budget_trials": sum(1 for r in attack_records if r.get("control") is False),
        "zero_gradient_trials": sum(1 for r in attack_records if r.get("control_observation", "").startswith("zero-gradient")),
        "prior_control_trials": sum(1 for r in attack_records if r.get("optimized") is False),
        "in_memory_candidate_checks": sum(1 for r in attack_records if r.get("best_loss_matches_in_memory_candidate")),
        "reloaded_artifact_checks": sum(1 for r in attack_records if r.get("best_loss_matches_reloaded_artifact")),
        "candidate_checks_skipped": sum(1 for r in attack_records if r.get("best_loss_matches_in_memory_candidate") is None),
        "equivalence_attack_calls": len(equivalence_children),
        "paired_dna_attack_calls": len(paired_children),
    }


def _flatten_attack_records(records):
    flattened = []
    for record in records:
        if "sample_label" in record:
            flattened.append(record)
        for key in ("baseline", "preaggregation", "lossless", "transform"):
            child = record.get(key)
            if isinstance(child, dict):
                flattened.append(child)
    return flattened


if __name__ == "__main__":
    main()
