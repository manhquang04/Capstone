"""Priority 23 BN-channel, DP recalibration, and RQ2 factual analysis.

This script intentionally does not modify existing artifacts.  It reads the
Priority-16 target bundle and existing RQ2 metrics, writes fresh Priority-23
diagnostic artifacts, and leaves CIFAR repaired-attack execution to a separate
runner because that part is compute-heavy.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.load_creditcard import (  # noqa: E402
    BASE_FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    _build_features,
    _fit_transform_features,
    load_creditcard_data,
)
from dna_encoder.transform_defense import DNATransformConfig, transform_update_array  # noqa: E402
from dna_encoder.transform_defense_v2 import DNATransformV2Config, transform_and_reconstruct_array_v2  # noqa: E402
from experiments.dp_update_accounting import alpha_grid, epsilon_from_rdp  # noqa: E402
from experiments.priority16_v1_medium_dna_vs_dp_probe import (  # noqa: E402
    _trainable_names,
    capture_paysim,
)
from experiments import fraud_fl_common as common  # noqa: E402
from models.fraud_mlp import FraudMLP  # noqa: E402
from privacy.dp_engine import apply_dp_to_local_state  # noqa: E402
from privacy.seed_manager import derive_seed  # noqa: E402


TARGET = ROOT / "artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_targets_20260927/paysim_priority16_v1_medium_confirmatory_targets.pt"
AMENDMENT = ROOT / "protocols/amendments/2026-09-28_priority23_bn_channel_and_cifar_repair.md"

V1_CONFIGS = {
    "v1_conservative": DNATransformConfig(block_size=256, mix_ratio=0.08, keep_ratio=0.88, shrink_factor=0.45, seed=681958327),
    "v1_medium": DNATransformConfig(block_size=256, mix_ratio=0.10, keep_ratio=0.85, shrink_factor=0.40, seed=681958327),
    "v1_stronger": DNATransformConfig(block_size=256, mix_ratio=0.12, keep_ratio=0.82, shrink_factor=0.35, seed=681958327),
}
V2_CONFIG = DNATransformV2Config(compression_ratio=0.95, quantization_eta=0.01, seed=20260916)
OLD_DP = {
    "dp_0p000315": 0.000315,
    "dp_0p0004": 0.0004,
    "dp_0p00105": 0.00105,
}
OLD_COMPARATOR_MULTIPLIERS = {
    "v1_conservative": 0.00025,
    "v1_medium": 0.000315,
    "v1_stronger": 0.0004,
    "v2_0p95_eta0p01": 0.00105,
}
CLIP_NORM = 100.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_population(seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return transformed train features, per-feature std, and feature names."""
    train_loaders, _, _, _, _, meta = load_creditcard_data(
        batch_size=1024,
        num_clients=3,
        seed=seed,
        max_rows=500000,
    )
    arrays = []
    for loader in train_loaders:
        dataset = loader.dataset
        arrays.append(dataset.tensors[0].detach().cpu().numpy())
    train = np.concatenate(arrays, axis=0).astype(np.float64)
    std = train.std(axis=0)
    std[std < 1e-12] = 1.0
    return train.mean(axis=0), std, list(meta.feature_names)


def state_from_update(global_state: dict[str, torch.Tensor], update: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for name, tensor in global_state.items():
        if name in update and torch.is_floating_point(tensor):
            out[name] = tensor.detach().cpu() + update[name].detach().cpu()
        else:
            out[name] = tensor.detach().cpu().clone()
    return out


def transform_full_v1(update: dict[str, torch.Tensor], config: DNATransformConfig) -> dict[str, torch.Tensor]:
    out = {}
    for tensor_index, (name, tensor) in enumerate(update.items()):
        if not torch.is_floating_point(tensor):
            out[name] = tensor.detach().cpu().clone()
            continue
        arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        transformed, _ = transform_update_array(arr, config, tensor_index=tensor_index)
        out[name] = torch.from_numpy(transformed).to(dtype=tensor.dtype)
    return out


def transform_full_v2(update: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for tensor_index, (name, tensor) in enumerate(update.items()):
        if not torch.is_floating_point(tensor):
            out[name] = tensor.detach().cpu().clone()
            continue
        arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        _, reconstructed, _, _ = transform_and_reconstruct_array_v2(
            arr,
            V2_CONFIG,
            tensor_index=tensor_index,
            quantization_seed=derive_seed(V2_CONFIG.seed, "priority23-bn", tensor_index),
        )
        out[name] = torch.from_numpy(reconstructed).to(dtype=tensor.dtype)
    return out


def dp_full_update(
    global_state: dict[str, torch.Tensor],
    update: dict[str, torch.Tensor],
    multiplier: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    local_state = state_from_update(global_state, update)
    dp_state, _, _, _ = apply_dp_to_local_state(
        local_state,
        {k: v.detach().cpu() for k, v in global_state.items()},
        CLIP_NORM,
        multiplier,
        noise_generator=torch.Generator().manual_seed(seed),
    )
    return {
        name: (dp_state[name].detach().cpu() - global_state[name].detach().cpu())
        for name in global_state
        if name in dp_state
    }


def recover_mean_from_running_mean(
    model: torch.nn.Module,
    delta_running_mean: torch.Tensor,
) -> np.ndarray:
    first_linear = model.network[0]
    first_bn = model.network[1]
    momentum = float(first_bn.momentum if first_bn.momentum is not None else 0.1)
    batch_mean_z = first_bn.running_mean.detach().cpu().double() + delta_running_mean.detach().cpu().double() / momentum
    rhs = batch_mean_z - first_linear.bias.detach().cpu().double()
    weight = first_linear.weight.detach().cpu().double()
    solution = torch.linalg.lstsq(weight, rhs).solution
    return solution.detach().cpu().numpy().astype(np.float64)


def bn_activation_var(model: torch.nn.Module, delta_running_var: torch.Tensor) -> np.ndarray:
    first_bn = model.network[1]
    momentum = float(first_bn.momentum if first_bn.momentum is not None else 0.1)
    before = first_bn.running_var.detach().cpu().double()
    # PyTorch BatchNorm running_var tracks the unbiased activation variance.
    return (before + delta_running_var.detach().cpu().double() / momentum).numpy().astype(np.float64)


def exact_greater_p(wins: int, non_tied: int) -> float:
    return float(binomtest(wins, non_tied, 0.5, alternative="greater").pvalue) if non_tied else 1.0


def summarize(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "iqr": float(q3 - q1),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def trainable_vector(update: dict[str, torch.Tensor], names: list[str]) -> np.ndarray:
    return np.concatenate([update[name].detach().cpu().numpy().reshape(-1).astype(np.float64) for name in names])


def run_part_c(groups: list[dict]) -> dict:
    rows = []
    orders = alpha_grid()
    for group_id, group in enumerate(groups):
        local_seed = derive_seed(2026092706, "priority14-local", "v1_medium_vs_dp_0p000315", group_id)
        model, criterion, x, y, batches, rng, observed_full = capture_paysim(group, local_seed, 4)
        names = _trainable_names(model)
        original = trainable_vector(observed_full, names)
        dim = int(original.size)
        for config_id, config in V1_CONFIGS.items():
            transformed = transform_full_v1({name: observed_full[name] for name in names}, config)
            diff = trainable_vector(transformed, names) - original
            rows.append(
                {
                    "group": group_id,
                    "config_id": config_id,
                    "update_l2": float(np.linalg.norm(original)),
                    "distortion_l2": float(np.linalg.norm(diff)),
                    "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(original), 1e-12)),
                    "trainable_dim": dim,
                }
            )
        transformed_v2 = transform_full_v2({name: observed_full[name] for name in names})
        diff = trainable_vector(transformed_v2, names) - original
        rows.append(
            {
                "group": group_id,
                "config_id": "v2_0p95_eta0p01",
                "update_l2": float(np.linalg.norm(original)),
                "distortion_l2": float(np.linalg.norm(diff)),
                "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(original), 1e-12)),
                "trainable_dim": dim,
            }
        )
    summary = {}
    for config_id in sorted({row["config_id"] for row in rows}):
        subset = [row for row in rows if row["config_id"] == config_id]
        distortions = np.asarray([row["distortion_l2"] for row in subset], dtype=np.float64)
        dims = int(subset[0]["trainable_dim"])
        multiplier_per_target = distortions / (CLIP_NORM * math.sqrt(dims))
        multiplier = float(np.median(multiplier_per_target))
        eps1 = epsilon_from_rdp(multiplier, 1.0, 1e-5, 1, orders)
        eps50 = epsilon_from_rdp(multiplier, 1.0, 1e-5, 50, orders)
        old = OLD_COMPARATOR_MULTIPLIERS[config_id]
        summary[config_id] = {
            "n_targets": len(subset),
            "trainable_dim": dims,
            "median_distortion_l2": float(np.median(distortions)),
            "mean_distortion_l2": float(np.mean(distortions)),
            "median_relative_l2": float(np.median([row["relative_l2"] for row in subset])),
            "new_noise_multiplier_median_l2": multiplier,
            "old_noise_multiplier": old,
            "new_to_old_multiplier_ratio": float(multiplier / old),
            "epsilon_single_release_delta_1e_minus_5": eps1["epsilon"],
            "epsilon_50_releases_delta_1e_minus_5": eps50["epsilon"],
            "optimal_alpha_single": eps1["alpha"],
            "optimal_alpha_50": eps50["alpha"],
        }
    return {"rows": rows, "summary": summary}


def run_part_a(groups: list[dict], recalibrated: dict) -> dict:
    pop_mean, pop_std, feature_names = load_population(seed=42)
    dp_multipliers = dict(OLD_DP)
    for config_id, item in recalibrated["summary"].items():
        dp_multipliers[f"dp_recalibrated_for_{config_id}"] = float(item["new_noise_multiplier_median_l2"])
    rows = []
    second_moment_rows = []
    for group_id, group in enumerate(groups):
        local_seed = derive_seed(2026092706, "priority14-local", "v1_medium_vs_dp_0p000315", group_id)
        model, criterion, x, y, batches, rng, observed_full = capture_paysim(group, local_seed, 4)
        true_mean = x.detach().cpu().numpy().astype(np.float64).mean(axis=0)
        global_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        defenses: dict[str, dict[str, torch.Tensor]] = {"none": observed_full}
        for name, config in V1_CONFIGS.items():
            defenses[name] = transform_full_v1(observed_full, config)
        defenses["v2_0p95_eta0p01_decoded"] = transform_full_v2(observed_full)
        for name, multiplier in dp_multipliers.items():
            defenses[name] = dp_full_update(global_state, observed_full, multiplier, derive_seed(20260928, "priority23-dp-bn", name, group_id))
        for defense, update in defenses.items():
            recovered = recover_mean_from_running_mean(model, update["network.1.running_mean"])
            raw_mse = float(np.mean((recovered - true_mean) ** 2))
            standardized_mse = float(np.mean(((recovered - true_mean) / pop_std) ** 2))
            pop_raw = float(np.mean((pop_mean - true_mean) ** 2))
            zero_raw = float(np.mean((np.zeros_like(true_mean) - true_mean) ** 2))
            pop_std_mse = float(np.mean(((pop_mean - true_mean) / pop_std) ** 2))
            zero_std_mse = float(np.mean(((np.zeros_like(true_mean) - true_mean) / pop_std) ** 2))
            rows.append(
                {
                    "group": group_id,
                    "defense": defense,
                    "raw_mse_recovered": raw_mse,
                    "raw_mse_population_mean": pop_raw,
                    "raw_mse_zero": zero_raw,
                    "standardized_mse_recovered": standardized_mse,
                    "standardized_mse_population_mean": pop_std_mse,
                    "standardized_mse_zero": zero_std_mse,
                    "win_vs_population_raw": raw_mse < pop_raw,
                    "win_vs_population_standardized": standardized_mse < pop_std_mse,
                    "source_ids": json.dumps(group["source_ids"]),
                }
            )
            activation_var = bn_activation_var(model, update["network.1.running_var"])
            second_moment_rows.append(
                {
                    "group": group_id,
                    "defense": defense,
                    "activation_var_mean": float(np.mean(activation_var)),
                    "activation_var_min": float(np.min(activation_var)),
                    "activation_var_max": float(np.max(activation_var)),
                }
            )
    summary = {}
    for defense in sorted({row["defense"] for row in rows}):
        subset = [row for row in rows if row["defense"] == defense]
        wins_raw = sum(bool(row["win_vs_population_raw"]) for row in subset)
        wins_std = sum(bool(row["win_vs_population_standardized"]) for row in subset)
        summary[defense] = {
            "n": len(subset),
            "raw_mse_recovered": summarize([row["raw_mse_recovered"] for row in subset]),
            "raw_mse_population_mean": summarize([row["raw_mse_population_mean"] for row in subset]),
            "standardized_mse_recovered": summarize([row["standardized_mse_recovered"] for row in subset]),
            "standardized_mse_population_mean": summarize([row["standardized_mse_population_mean"] for row in subset]),
            "wins_vs_population_raw": wins_raw,
            "p_raw_recovered_better_than_population": exact_greater_p(wins_raw, len(subset)),
            "wins_vs_population_standardized": wins_std,
            "p_standardized_recovered_better_than_population": exact_greater_p(wins_std, len(subset)),
        }
    return {
        "feature_names": feature_names,
        "population_feature_mean": pop_mean.tolist(),
        "population_feature_std": pop_std.tolist(),
        "rows": rows,
        "running_var_descriptive_rows": second_moment_rows,
        "summary": summary,
        "running_var_note": "running_var yields first-layer activation variance statistics; it does not uniquely identify input-feature second moments without additional assumptions on feature covariance.",
        "realistic_full_client_update_descriptive": run_realistic_full_client_check(),
    }


def train_one_client_record_batches(model: torch.nn.Module, loader, pos_weight: torch.Tensor) -> dict:
    model.to(torch.device("cpu"))
    model.train()
    criterion = common.build_loss(pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=common.LEARNING_RATE)
    batch_means = []
    batch_sizes = []
    for features, labels in loader:
        features = features.to(torch.device("cpu"))
        labels = labels.to(torch.device("cpu"))
        batch_means.append(features.detach().cpu().numpy().astype(np.float64).mean(axis=0))
        batch_sizes.append(int(features.shape[0]))
        optimizer.zero_grad()
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()
    return {"batch_means": batch_means, "batch_sizes": batch_sizes}


def recover_from_model_delta(before: torch.nn.Module, after: torch.nn.Module) -> np.ndarray:
    before_state = before.state_dict()
    after_state = after.state_dict()
    delta = after_state["network.1.running_mean"].detach().cpu() - before_state["network.1.running_mean"].detach().cpu()
    return recover_mean_from_running_mean(before, delta)


def run_realistic_full_client_check() -> dict:
    """One descriptive RQ2-style client update, full local epoch, batch 1024."""
    seed = 98089969
    common.set_random_seed(seed)
    common.DEVICE = torch.device("cpu")
    loaders, _, _, input_dim, pos_weight, metadata = load_creditcard_data(
        batch_size=1024,
        num_clients=3,
        seed=seed,
        max_rows=500000,
    )
    global_model = FraudMLP(input_dim).to(torch.device("cpu"))
    before = copy.deepcopy(global_model)
    local_model = copy.deepcopy(global_model)
    client_id = 0
    recorded = train_one_client_record_batches(local_model, loaders[client_id], pos_weight)
    recovered = recover_from_model_delta(before, local_model)
    dataset_features = loaders[client_id].dataset.tensors[0].detach().cpu().numpy().astype(np.float64)
    full_mean = dataset_features.mean(axis=0)
    last1 = recorded["batch_means"][-1]
    last5_count = min(5, len(recorded["batch_means"]))
    last5 = np.average(
        np.stack(recorded["batch_means"][-last5_count:]),
        axis=0,
        weights=np.asarray(recorded["batch_sizes"][-last5_count:], dtype=np.float64),
    )
    return {
        "seed": seed,
        "client_id": client_id,
        "local_epochs": 1,
        "batch_size": 1024,
        "n_client_samples": int(dataset_features.shape[0]),
        "n_batches": len(recorded["batch_means"]),
        "feature_names": list(metadata.feature_names),
        "recovered_effective_mean_mse_vs_client_full_mean": float(np.mean((recovered - full_mean) ** 2)),
        "recovered_effective_mean_mse_vs_last_batch_mean": float(np.mean((recovered - last1) ** 2)),
        "recovered_effective_mean_mse_vs_last5_batch_mean": float(np.mean((recovered - last5) ** 2)),
        "note": "For a full local epoch, BatchNorm running_mean is an EMA over mini-batches, so the recovered vector is an effective EMA feature mean, not the unweighted full-client mean.",
    }


def final_metric(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return payload["rounds"][-1], payload["config"]


def ci95(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n < 2:
        return (float(arr.mean()), float(arr.mean()))
    half = float(t.ppf(0.975, n - 1) * arr.std(ddof=1) / math.sqrt(n))
    mean = float(arr.mean())
    return mean - half, mean + half


def paired_summary(base_dir: Path, method: str) -> dict:
    rows = []
    for seed_dir in sorted(base_dir.glob("seed_*")):
        seed = int(seed_dir.name.split("_", 1)[1])
        b_metric, b_config = final_metric(seed_dir / "baseline" / "metrics.json")
        m_metric, m_config = final_metric(seed_dir / method / "metrics.json")
        row = {
            "seed": seed,
            "baseline_f1": b_metric["f1_score"],
            "method_f1": m_metric["f1_score"],
            "delta_f1": m_metric["f1_score"] - b_metric["f1_score"],
            "baseline_auc": b_metric["auc_roc"],
            "method_auc": m_metric["auc_roc"],
            "delta_auc": m_metric["auc_roc"] - b_metric["auc_roc"],
            "baseline_pr_auc": b_metric["pr_auc"],
            "method_pr_auc": m_metric["pr_auc"],
            "delta_pr_auc": m_metric["pr_auc"] - b_metric["pr_auc"],
        }
        rows.append(row)
    analyses = {}
    for endpoint in ("f1", "auc", "pr_auc"):
        deltas = [float(row[f"delta_{endpoint}"]) for row in rows]
        baseline_values = [float(row[f"baseline_{endpoint}"]) for row in rows]
        method_values = [float(row[f"method_{endpoint}"]) for row in rows]
        lo, hi = ci95(deltas)
        analyses[endpoint] = {
            "n": len(deltas),
            "baseline_mean": float(np.mean(baseline_values)),
            "method_mean": float(np.mean(method_values)),
            "mean_delta": float(np.mean(deltas)),
            "sd_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "ci95_delta": [lo, hi],
        }
    return {"method": method, "rows": rows, "analyses": analyses}


def run_part_d() -> dict:
    v1_base = ROOT / "artifacts/rq2/confirmatory_run_20260913"
    v2_base = ROOT / "artifacts/rq2_v2/confirmatory_20260916"
    v1 = paired_summary(v1_base, "dna_transform")
    v2 = paired_summary(v2_base, "dna_transform_v2")
    sample_v1_config = final_metric(next(v1_base.glob("seed_*")) / "dna_transform" / "metrics.json")[1]
    sample_v2_config = final_metric(next(v2_base.glob("seed_*")) / "dna_transform_v2" / "metrics.json")[1]
    evidence = {
        "v1_confirmatory_config": {
            "artifact": str((ROOT / "protocols/config/rq2_confirmatory.yaml").relative_to(ROOT)),
            "metrics_config_transform": sample_v1_config.get("dna_transform_config"),
            "config_id_in_yaml": "DNA-TRANSFORM-CONSERVATIVE-V1",
            "conclusion": "v1 RQ2 confirmatory used conservative, not medium or stronger",
        },
        "v2_confirmatory_config": {
            "metrics_config_transform_v2": sample_v2_config.get("dna_transform_v2_config"),
        },
        "bn_buffer_code_paths": {
            "v1_transform": "experiments/run_fraud_fl_dna_transform.py:dna_transform_state iterates state_dict.items() and transforms every floating tensor",
            "v2_transform": "experiments/run_fraud_fl_dna_transform_v2.py:dna_transform_v2_state iterates state_dict.items() and transforms every floating tensor",
            "dp": "privacy/dp_engine.py:apply_dp_to_local_state clips/noises every floating local_state item",
        },
        "v1_medium_or_stronger_rq2_exists": False,
        "search_basis": "No confirmatory RQ2 artifact/config for v1-medium or v1-stronger was found; rq2_confirmatory.yaml freezes conservative values mix=0.08/keep=0.88/shrink=0.45.",
    }
    return {"v1_confirmatory": v1, "v2_confirmatory": v2, "evidence": evidence}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/priority23_bn_channel_rq2_20260928")
    args = parser.parse_args()
    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    groups = torch.load(TARGET, map_location="cpu", weights_only=False)
    part_c = run_part_c(groups)
    part_a = run_part_a(groups, part_c)
    part_d = run_part_d()
    write_csv(out / "part_c_trainable_l2_calibration_per_target.csv", part_c["rows"])
    write_csv(out / "part_a_bn_mean_recovery_per_target.csv", part_a["rows"])
    write_csv(out / "part_a_running_var_descriptive_per_target.csv", part_a["running_var_descriptive_rows"])
    write_json(out / "priority23_part_a_c_d_summary.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": str(AMENDMENT.relative_to(ROOT)),
        "amendment_sha256": sha256(AMENDMENT),
        "target": str(TARGET.relative_to(ROOT)),
        "target_sha256": sha256(TARGET),
        "torch_num_threads": torch.get_num_threads(),
        "part_a_bn_channel": part_a,
        "part_c_trainable_only_dp_recalibration": part_c["summary"],
        "part_d_rq2_facts": part_d,
    })
    manifest = {
        "summary_json": "priority23_part_a_c_d_summary.json",
        "summary_json_sha256": sha256(out / "priority23_part_a_c_d_summary.json"),
        "part_a_csv_sha256": sha256(out / "part_a_bn_mean_recovery_per_target.csv"),
        "part_a_running_var_csv_sha256": sha256(out / "part_a_running_var_descriptive_per_target.csv"),
        "part_c_csv_sha256": sha256(out / "part_c_trainable_l2_calibration_per_target.csv"),
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
