"""Run the bounded Phase 2 held-out sample-gradient diagnostic."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.gradient_inversion import GradientInversionConfig, gradient_inversion_attack, parameter_gradients
from attacks.tabular_metrics import paysim_reconstruction_metrics
from data.load_creditcard import load_creditcard_data
from experiments.fraud_fl_common import BATCH_SIZE, LOCAL_EPOCHS, build_loss, fed_avg, set_random_seed, train_local_model
from models.fraud_mlp import FraudMLP
from privacy.seed_manager import derive_seed
from experiments.phase2_provenance import validation_source_rows
from attacks.gradient_inversion import _score_components

RUN_ID = os.environ.get("PHASE2_RUN_ID", datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ"))
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "phase2" / RUN_ID
MAX_ROWS = int(os.environ.get("PHASE2_MAX_ROWS", "50000"))
NUM_TARGETS = int(os.environ.get("PHASE2_NUM_TARGETS", "10"))
ITERATIONS = int(os.environ.get("PHASE2_ITERATIONS", "300"))
RESTARTS = int(os.environ.get("PHASE2_RESTARTS", "3"))
L2_WEIGHT = float(os.environ.get("PHASE2_L2_WEIGHT", "0"))
DATA_SEED = int(os.environ.get("PHASE2_DATA_SEED", "20260907"))
ATTACK_SEED = int(os.environ.get("PHASE2_ATTACK_SEED", "731921"))
WARMUP_ROUNDS = int(os.environ.get("PHASE2_WARMUP_ROUNDS", "1"))


def main() -> None:
    total_started = perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    dataset = PROJECT_ROOT / 'datasets' / 'creditcard.csv'
    source_labels = pd.read_csv(dataset, usecols=['isFraud']).isFraud.to_numpy()
    source_rows = validation_source_rows(source_labels, MAX_ROWS, DATA_SEED)
    development_rows = validation_source_rows(source_labels, 5000, 917431)[[0, 576]]
    set_random_seed(DATA_SEED)
    loaders, validation_loader, _, input_dim, pos_weight, metadata = load_creditcard_data(
        batch_size=BATCH_SIZE,
        num_clients=3,
        max_rows=MAX_ROWS,
        seed=DATA_SEED,
    )
    model = _warmup(loaders, input_dim, pos_weight).to("cpu")
    model.eval()
    criterion = build_loss(pos_weight)
    checkpoint_path = OUTPUT_DIR / "warmup_model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False))

    features, labels, indices = _select_targets(validation_loader, NUM_TARGETS, DATA_SEED)
    np.testing.assert_array_equal(source_labels[source_rows], validation_loader.dataset.tensors[1].numpy().ravel())
    target_source_rows = source_rows[indices]
    if set(target_source_rows) & set(development_rows):
        raise AssertionError('Evaluation targets overlap development; create an explicit exclusion protocol')
    torch.save({'features': features, 'labels': labels, 'source_rows': target_source_rows}, OUTPUT_DIR / 'targets.pt')
    (OUTPUT_DIR / 'protocol_lock.json').write_text(json.dumps(dict(
        iterations=ITERATIONS, restarts=RESTARTS, l2=L2_WEIGHT, learning_rate=0.05,
        source_rows=target_source_rows.tolist(), development_rows=development_rows.tolist(),
        rule='no extension after observing evaluation; compare MSE/MAE/type separately'), indent=2))
    records: list[dict[str, object]] = []
    failed_jobs: list[dict[str, object]] = []
    attack_runtime = 0.0

    for sample_id, (original, label, validation_index) in enumerate(zip(features, labels, indices)):
        observed = parameter_gradients(model, criterion, original.reshape(1, -1), label.reshape(1, 1))
        for restart in range(RESTARTS):
            seed = derive_seed(ATTACK_SEED, "phase2", sample_id, restart)
            config = GradientInversionConfig(
                iterations=ITERATIONS,
                learning_rate=0.05,
                l2_weight=L2_WEIGHT,
                optimizer="adam",
                seed=seed,
            )
            try:
                started = perf_counter()
                baseline = gradient_inversion_attack(copy.deepcopy(model), criterion, observed, label.reshape(1, 1), input_dim, config)
                zero = gradient_inversion_attack(
                    copy.deepcopy(model), criterion, [torch.zeros_like(value) for value in observed], label.reshape(1, 1), input_dim, config
                )
                attack_runtime += perf_counter() - started
                if not torch.equal(baseline.initial_dummy, zero.initial_dummy):
                    raise AssertionError("paired baseline and zero-gradient initial tensors differ")

                records.append(_record("prior", sample_id, validation_index, label, restart, seed, original, baseline.initial_dummy, None, metadata))
                records.append(_record("baseline", sample_id, validation_index, label, restart, seed, original, baseline.reconstructed_input, baseline, metadata))
                records.append(_record("zero_gradient", sample_id, validation_index, label, restart, seed, original, zero.reconstructed_input, zero, metadata))
                _save_vectors(sample_id, restart, original, baseline, zero, model, criterion, observed, label)
            except Exception as exc:  # preserve failed jobs in the report
                failed_jobs.append({"sample_id": sample_id, "restart": restart, "error": repr(exc)})

    selected = _select_by_attacker_objective(records)
    summary = _summarize_selected(selected)
    payload = {
        "experiment": "phase2_heldout_sample_gradient_diagnostic",
        "config": {
            "run_id": RUN_ID,
            "threat_model": "known-label, sample-level parameter-gradient observation",
            "model_mode": "eval",
            "max_rows": MAX_ROWS,
            "num_clients": 3,
            "warmup_rounds": WARMUP_ROUNDS,
            "targets_requested": NUM_TARGETS,
            "targets_evaluated": len(indices),
            "validation_indices": indices,
            "target_labels": [int(value.item()) for value in labels],
            "iterations": ITERATIONS,
            "restarts": RESTARTS,
            "l2_weight": L2_WEIGHT,
            "learning_rate": 0.05,
            "optimizer": "Adam",
            "candidate_selection": "minimum attacker total objective per method and sample; ground truth metrics are evaluation-only",
            "data_seed": DATA_SEED,
            "attack_seed_root": ATTACK_SEED,
            "feature_names": metadata.feature_names,
            "numeric_center": metadata.numeric_center,
            "numeric_scale": metadata.numeric_scale,
            "metric_version": "paysim_tabular_v1+joint_range_pseudo_image_v2",
            "categorical_metric_version": "strict_onehot_v2",
            "dataset_sha256": _sha256(dataset),
            "source_row_ids": target_source_rows.tolist(),
            "development_source_row_ids": development_rows.tolist(),
            "source_disjoint": True,
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_worktree_status": _git(["status", "--short"]),
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": _sha256(checkpoint_path),
        },
        "summary": summary,
        "selected_records": selected,
        "all_records": records,
        "failed_jobs": failed_jobs,
        "runtime": {
            "attack_seconds_including_baseline_and_zero_restarts": attack_runtime,
            "total_seconds_including_data_and_warmup": perf_counter() - total_started,
        },
    }
    report_path = OUTPUT_DIR / "heldout_diagnostic_results.json"
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(OUTPUT_DIR / "heldout_trials.csv", records)
    print(json.dumps({"summary": summary, "failed_jobs": len(failed_jobs), "output": str(report_path)}, indent=2))


def _warmup(loaders, input_dim, pos_weight):
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


def _select_targets(loader, count, seed):
    if count < 2:
        raise ValueError("Phase 2 needs at least two targets")
    features, labels = loader.dataset.tensors
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels.numpy().reshape(-1) == 1.0)
    negative = np.flatnonzero(labels.numpy().reshape(-1) == 0.0)
    positive_count = count // 2
    negative_count = count - positive_count
    if len(positive) < positive_count or len(negative) < negative_count:
        raise RuntimeError(f"insufficient targets: fraud={len(positive)}, non_fraud={len(negative)}")
    chosen = np.concatenate([
        rng.choice(positive, size=positive_count, replace=False),
        rng.choice(negative, size=negative_count, replace=False),
    ])
    rng.shuffle(chosen)
    index_tensor = torch.as_tensor(chosen, dtype=torch.long)
    return features[index_tensor].clone(), labels[index_tensor].clone(), [int(value) for value in chosen]


def _record(method, sample_id, validation_index, label, restart, seed, original, candidate, result, metadata):
    metrics = paysim_reconstruction_metrics(
        original.numpy(), candidate.detach().cpu().numpy(), metadata.feature_names, metadata.numeric_center, metadata.numeric_scale
    )
    record = {
        "method": method,
        "sample_id": sample_id,
        "validation_index": validation_index,
        "label": int(label.item()),
        "restart": restart,
        "seed": seed,
        **metrics,
    }
    if result is None:
        record.update({"iterations": 0, "best_iteration": 0, "gradient_match_loss": None, "regularization_loss": None, "total_objective": None})
    else:
        record.update(
            {
                "iterations": ITERATIONS,
                "best_iteration": result.best_iteration,
                "gradient_match_loss": result.best_gradient_match_loss,
                "regularization_loss": result.best_regularization_loss,
                "total_objective": result.best_loss,
                "initial_total_objective": result.initial_total_objective,
                "component_history": result.component_history,
            }
        )
    return record


def _save_vectors(sample_id, restart, original, baseline, zero, model, criterion, observed, label):
    path = OUTPUT_DIR / "vectors" / f"sample{sample_id:02d}_restart{restart}"
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"vector": original.detach().cpu()}, path / "original.pt")
    torch.save({"vector": baseline.initial_dummy.detach().cpu()}, path / "initial_dummy.pt")
    torch.save({"vector": baseline.reconstructed_input.detach().cpu(), "objective": baseline.best_loss}, path / "baseline_best.pt")
    torch.save({"vector": zero.reconstructed_input.detach().cpu(), "objective": zero.best_loss}, path / "zero_gradient_best.pt")
    torch.save(observed, path / 'observed.pt')
    for filename, expected in [('original.pt', original), ('initial_dummy.pt', baseline.initial_dummy),
                               ('baseline_best.pt', baseline.reconstructed_input), ('zero_gradient_best.pt', zero.reconstructed_input)]:
        actual = torch.load(path / filename, weights_only=False)['vector']
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.equal(actual, expected.cpu())
    for filename, result, signal in [('baseline_best.pt', baseline, observed),
                                      ('zero_gradient_best.pt', zero, [torch.zeros_like(g) for g in observed])]:
        candidate = torch.load(path / filename, weights_only=False)['vector']
        components = _score_components(model, criterion, candidate, signal, label.reshape(1, 1), L2_WEIGHT)
        np.testing.assert_allclose(components, [result.best_gradient_match_loss,
                                   result.best_regularization_loss, result.best_loss], rtol=1e-5, atol=1e-10)
    (path / 'checks.json').write_text(json.dumps({'vectors_reload': 4, 'objective_reload': 2, 'passed': True}))


def _select_by_attacker_objective(records):
    selected = []
    for sample_id in sorted({int(record["sample_id"]) for record in records}):
        priors = [record for record in records if record["sample_id"] == sample_id and record["method"] == "prior"]
        selected.extend(priors)
        for method in ("baseline", "zero_gradient"):
            candidates = [record for record in records if record["sample_id"] == sample_id and record["method"] == method]
            selected.append(min(candidates, key=lambda record: float(record["total_objective"])))
    return selected


def _summarize_selected(records):
    summary = {}
    for method in ("baseline", "zero_gradient", "prior"):
        items = [record for record in records if record["method"] == method]
        summary[method] = {
            "n_targets": len({record["sample_id"] for record in items}),
            "n_records": len(items),
            "mean_feature_mse": float(np.mean([record["feature_mse"] for record in items])),
            "mean_feature_mae": float(np.mean([record["feature_mae"] for record in items])),
            "mean_categorical_accuracy": float(np.mean([record["categorical_accuracy"] for record in items])),
            "valid_one_hot_fraction": float(np.mean([record["one_hot_valid_before_decode"] for record in items])),
        }
    baseline = {record["sample_id"]: record for record in records if record["method"] == "baseline"}
    zero = {record["sample_id"]: record for record in records if record["method"] == "zero_gradient"}
    differences = [baseline[key]["feature_mse"] - zero[key]["feature_mse"] for key in baseline]
    summary["paired_baseline_minus_zero_mse"] = {
        "mean": float(np.mean(differences)),
        "median": float(np.median(differences)),
        "baseline_better_count": int(sum(value < 0 for value in differences)),
        "targets": len(differences),
    }
    return summary


def _write_csv(path, records):
    scalar_records = [{key: value for key, value in record.items() if not isinstance(value, (dict, list))} for record in records]
    fields = sorted({key for record in scalar_records for key in record})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar_records)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args):
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


if __name__ == "__main__":
    main()
