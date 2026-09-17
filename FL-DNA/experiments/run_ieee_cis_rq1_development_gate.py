"""Priority 5.3 IEEE-CIS raw RQ1 development gate.

This is a development-only gate for the smallest IEEE-CIS scope:
four records per group, one fraud record per group, and one Adam local step.
The attacker is label-free: it optimizes features and soft labels jointly and
uses only public feature ranges/normalization plus categorical simplex
structure.  True labels are used only for target construction and reporting.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.local_update import simulate
from experiments.phase3_bounded_validation import update_objective
from experiments.run_ieee_cis_fl_baseline_smoke import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, fed_avg, train_local
from models.fraud_mlp import FraudMLP
from privacy.seed_manager import derive_seed, generate_run_seed


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass(frozen=True)
class FeatureContract:
    numeric: list[str]
    categorical: list[str]
    categories: list[list[str]]
    feature_names: list[str]
    numeric_lower: list[float]
    numeric_upper: list[float]
    numeric_center: list[float]
    numeric_scale: list[float]

    @property
    def input_dim(self) -> int:
        return len(self.feature_names)

    @property
    def category_slices(self) -> list[tuple[int, int]]:
        start = len(self.numeric)
        slices = []
        for values in self.categories:
            end = start + len(values)
            slices.append((start, end))
            start = end
        return slices


def read_ieee_frame(max_rows: int, seed: int) -> pd.DataFrame:
    usecols = ["TransactionID", "isFraud", *NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS]
    path = ROOT / "datasets/ieee-fraud-detection/train_transaction.csv"
    frame = pd.read_csv(path, usecols=lambda c: c in set(usecols))
    frame["_source_row"] = np.arange(len(frame), dtype=np.int64)
    if max_rows and max_rows < len(frame):
        _, frame = train_test_split(
            frame,
            test_size=max_rows,
            random_state=seed,
            stratify=frame["isFraud"],
        )
    return frame.reset_index(drop=True)


def fit_contract(frame: pd.DataFrame) -> tuple[FeatureContract, RobustScaler, OneHotEncoder, pd.Series]:
    numeric = [c for c in NUMERIC_COLUMNS if c in frame.columns]
    categorical = [c for c in CATEGORICAL_COLUMNS if c in frame.columns]
    medians = frame[numeric].median()
    scaler = RobustScaler()
    numeric_scaled = scaler.fit_transform(frame[numeric].fillna(medians))
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    encoder.fit(frame[categorical].fillna("__MISSING__").astype(str))
    categories = [[str(v) for v in values] for values in encoder.categories_]
    feature_names = list(numeric)
    for column, values in zip(categorical, categories):
        feature_names.extend([f"{column}={value}" for value in values])
    lower = np.quantile(numeric_scaled, 0.001, axis=0)
    upper = np.quantile(numeric_scaled, 0.999, axis=0)
    lower = np.maximum(lower, -20.0)
    upper = np.minimum(upper, 20.0)
    contract = FeatureContract(
        numeric=numeric,
        categorical=categorical,
        categories=categories,
        feature_names=feature_names,
        numeric_lower=lower.astype(float).tolist(),
        numeric_upper=upper.astype(float).tolist(),
        numeric_center=scaler.center_.astype(float).tolist(),
        numeric_scale=scaler.scale_.astype(float).tolist(),
    )
    return contract, scaler, encoder, medians


def transform_frame(frame: pd.DataFrame, contract: FeatureContract, scaler: RobustScaler, encoder: OneHotEncoder, medians: pd.Series) -> np.ndarray:
    numeric = scaler.transform(frame[contract.numeric].fillna(medians))
    categorical = encoder.transform(frame[contract.categorical].fillna("__MISSING__").astype(str))
    return np.concatenate([numeric, categorical], axis=1).astype(np.float32)


def make_warmup_loaders(x: np.ndarray, y: np.ndarray, clients: int, seed: int, batch_size: int) -> tuple[list[DataLoader], list[int]]:
    rng = np.random.default_rng(seed)
    fraud = np.where(y == 1)[0]
    non = np.where(y == 0)[0]
    rng.shuffle(fraud)
    rng.shuffle(non)
    client_indices = [[] for _ in range(clients)]
    for i, idx in enumerate(fraud):
        client_indices[i % clients].append(int(idx))
    for i, idx in enumerate(non):
        client_indices[i % clients].append(int(idx))
    loaders = []
    for client_id, indices in enumerate(client_indices):
        rng.shuffle(indices)
        gen = torch.Generator().manual_seed(seed + client_id)
        loaders.append(
            DataLoader(
                TensorDataset(torch.from_numpy(x[indices]), torch.from_numpy(y[indices]).reshape(-1, 1)),
                batch_size=batch_size,
                shuffle=True,
                generator=gen,
            )
        )
    return loaders, [len(indices) for indices in client_indices]


def warmup_model(x: np.ndarray, y: np.ndarray, input_dim: int, seed: int, clients: int, rounds: int, batch_size: int) -> tuple[dict, torch.Tensor]:
    positive = float(np.count_nonzero(y == 1))
    negative = float(np.count_nonzero(y == 0))
    pos_weight = torch.tensor([negative / max(positive, 1.0)], dtype=torch.float32)
    loaders, counts = make_warmup_loaders(x, y, clients, seed, batch_size)
    model = FraudMLP(input_dim).train()
    for _ in range(rounds):
        states = []
        for loader in loaders:
            local = FraudMLP(input_dim).train()
            local.load_state_dict(model.state_dict())
            train_local(local, loader, pos_weight, epochs=1)
            states.append(local.state_dict())
        model.load_state_dict(fed_avg(states, counts))
    return model.state_dict(), pos_weight


def make_groups(frame: pd.DataFrame, features: np.ndarray, count: int, seed: int) -> list[dict]:
    labels = frame["isFraud"].to_numpy().astype(np.float32)
    fraud_ids = np.where(labels == 1)[0]
    non_ids = np.where(labels == 0)[0]
    rng = np.random.default_rng(seed)
    if len(fraud_ids) < count or len(non_ids) < count * 3:
        raise ValueError("Not enough IEEE-CIS records for requested target groups")
    selected_fraud = rng.choice(fraud_ids, count, replace=False)
    selected_non = rng.choice(non_ids, count * 3, replace=False).reshape(count, 3)
    groups = []
    used_source_rows: set[int] = set()
    for group_id in range(count):
        ids = np.concatenate([[selected_fraud[group_id]], selected_non[group_id]])
        rng.shuffle(ids)
        source_rows = frame.iloc[ids]["_source_row"].astype(int).to_list()
        if used_source_rows & set(source_rows):
            raise AssertionError("Internal IEEE-CIS target overlap")
        used_source_rows.update(source_rows)
        groups.append(
            {
                "group_id": group_id,
                "frame_indices": ids.astype(int).tolist(),
                "source_rows": source_rows,
                "transaction_ids": frame.iloc[ids]["TransactionID"].astype(int).to_list(),
                "x": features[ids].astype(np.float32),
                "y": labels[ids].reshape(-1, 1).astype(np.float32),
            }
        )
    return groups


def decode(numeric: torch.Tensor, cat_logits: list[torch.Tensor], label_logits: torch.Tensor, contract: FeatureContract) -> tuple[torch.Tensor, torch.Tensor]:
    lower = torch.tensor(contract.numeric_lower, dtype=numeric.dtype, device=numeric.device)
    upper = torch.tensor(contract.numeric_upper, dtype=numeric.dtype, device=numeric.device)
    parts = [numeric.clamp(lower, upper)]
    parts.extend(logits.softmax(dim=1) for logits in cat_logits)
    return torch.cat(parts, dim=1), torch.sigmoid(label_logits)


def init_latent(records: int, contract: FeatureContract, seed: int) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    numeric = torch.randn((records, len(contract.numeric)), generator=gen, dtype=torch.float32)
    cat_logits = [torch.randn((records, len(values)), generator=gen, dtype=torch.float32) for values in contract.categories]
    label_logits = torch.randn((records, 1), generator=gen, dtype=torch.float32)
    return numeric, cat_logits, label_logits


def align_all_records(original: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    costs = torch.cdist(original.double(), candidate.double()).square().cpu().numpy()
    rows, cols = linear_sum_assignment(costs)
    aligned = torch.empty_like(candidate)
    aligned[torch.as_tensor(rows)] = candidate[torch.as_tensor(cols)]
    return aligned


def score_reconstruction(original: torch.Tensor, candidate: torch.Tensor, labels: torch.Tensor, contract: FeatureContract, path: Path) -> dict:
    aligned = align_all_records(original, candidate)
    x = original.detach().cpu().numpy().astype(np.float64)
    z = aligned.detach().cpu().numpy().astype(np.float64)
    y = labels.detach().cpu().numpy().reshape(-1).astype(int)
    mse = ((x - z) ** 2).mean(axis=1)
    mae = np.abs(x - z).mean(axis=1)
    cat_acc = []
    for start, end in contract.category_slices:
        cat_acc.append((x[:, start:end].argmax(axis=1) == z[:, start:end].argmax(axis=1)).astype(float))
    cat_acc_arr = np.vstack(cat_acc).mean(axis=0) if cat_acc else np.ones(len(mse))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ordered_record", "label", "mse", "mae", "categorical_argmax_accuracy"])
        writer.writerows(zip(range(len(mse)), y.tolist(), mse.tolist(), mae.tolist(), cat_acc_arr.tolist()))
    classes = {}
    for label in (0, 1):
        mask = y == label
        classes[str(label)] = {
            "n": int(mask.sum()),
            "mean_mse": float(mse[mask].mean()) if mask.any() else None,
            "median_mse": float(np.median(mse[mask])) if mask.any() else None,
            "mean_mae": float(mae[mask].mean()) if mask.any() else None,
            "categorical_argmax_accuracy": float(cat_acc_arr[mask].mean()) if mask.any() else None,
        }
    return {
        "mean_mse": float(mse.mean()),
        "median_mse": float(np.median(mse)),
        "mean_mae": float(mae.mean()),
        "categorical_argmax_accuracy": float(cat_acc_arr.mean()),
        "classes": classes,
    }


def capture(group: dict, state_dict: dict, pos_weight: torch.Tensor, seed: int, lr: float) -> tuple[FraudMLP, nn.Module, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    x = torch.from_numpy(group["x"])
    y = torch.from_numpy(group["y"])
    model = FraudMLP(x.shape[1]).train()
    model.load_state_dict(state_dict)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rng_state = torch.Generator().manual_seed(seed).get_state()
    batches = [torch.arange(len(x))]
    observed = simulate(model, criterion, x, y, batches, rng_state, lr=lr, create_graph=False)
    return model, criterion, x, y, rng_state, observed


def run_pair(
    folder: Path,
    group: dict,
    contract: FeatureContract,
    state_dict: dict,
    pos_weight: torch.Tensor,
    run_seed: int,
    restart: int,
    iterations: int,
    attack_lr: float,
    local_lr: float,
) -> list[dict]:
    folder.mkdir(parents=True, exist_ok=False)
    group_id = int(group["group_id"])
    model, criterion, original, true_labels, rng_state, observed = capture(
        group, state_dict, pos_weight, derive_seed(run_seed, "ieee-local", group_id), local_lr
    )
    keys = [key for key, value in observed.items() if value.is_floating_point()]
    init_numeric, init_cats, init_labels = init_latent(
        len(original), contract, derive_seed(run_seed, "ieee-init", group_id, restart)
    )
    with torch.no_grad():
        prior_x, prior_y = decode(init_numeric, init_cats, init_labels, contract)
    prior_metrics = score_reconstruction(original, prior_x, true_labels, contract, folder / "prior.csv")
    rows = []
    for method in ("baseline", "zero_update"):
        signal = observed if method == "baseline" else {key: torch.zeros_like(value) for key, value in observed.items()}
        numeric = init_numeric.detach().clone().requires_grad_(True)
        cat_logits = [item.detach().clone().requires_grad_(True) for item in init_cats]
        label_logits = init_labels.detach().clone().requires_grad_(True)
        params = [numeric, *cat_logits, label_logits]
        optimizer = torch.optim.Adam(params, lr=attack_lr)
        best = float("inf")
        best_step = 0
        best_state = None
        history = []
        started = time.perf_counter()
        batches = [torch.arange(len(original))]
        for step in range(iterations + 1):
            candidate_x, candidate_y = decode(numeric, cat_logits, label_logits, contract)
            simulated = simulate(model, criterion, candidate_x, candidate_y, batches, rng_state, lr=local_lr)
            loss = update_objective(simulated, signal, keys, reference=observed, mode="balanced_tensor")
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite IEEE-CIS attack objective at group={group_id} restart={restart} step={step}")
            value = float(loss.detach())
            history.append(value)
            if value < best:
                best = value
                best_step = step
                best_state = (
                    numeric.detach().clone(),
                    [item.detach().clone() for item in cat_logits],
                    label_logits.detach().clone(),
                )
            if step < iterations:
                gradients = torch.autograd.grad(loss, params)
                optimizer.zero_grad()
                for parameter, gradient in zip(params, gradients):
                    parameter.grad = gradient
                optimizer.step()
        assert best_state is not None
        reconstructed_x, reconstructed_y = decode(best_state[0], best_state[1], best_state[2], contract)
        metrics = score_reconstruction(original, reconstructed_x.detach(), true_labels, contract, folder / f"{method}.csv")
        artifact = {
            "method": method,
            "group_id": group_id,
            "restart": restart,
            "source_rows": group["source_rows"],
            "transaction_ids": group["transaction_ids"],
            "original": original,
            "true_labels": true_labels,
            "initial_numeric": init_numeric,
            "initial_cat_logits": init_cats,
            "initial_label_logits": init_labels,
            "reconstruction": reconstructed_x.detach(),
            "soft_labels": reconstructed_y.detach(),
            "observed": observed,
            "keys": keys,
            "best_objective": best,
            "best_step": best_step,
            "history": history,
            "objective_mode": "balanced_tensor",
            "attacker_label_oracle": False,
        }
        torch.save(artifact, folder / f"{method}.pt")
        rows.append(
            {
                "method": method,
                "group_id": group_id,
                "restart": restart,
                "objective": best,
                "best_step": best_step,
                "metrics": metrics,
                "prior": prior_metrics,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    dump(folder / "results.json", rows)
    return rows


def sign_tail(wins: int, total: int) -> float:
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / (2**total) if total else 1.0


def evaluate(records: list[dict]) -> tuple[dict, list[dict]]:
    selected = []
    for group_id in sorted({int(row["group_id"]) for row in records}):
        baseline = min(
            (row for row in records if int(row["group_id"]) == group_id and row["method"] == "baseline"),
            key=lambda row: row["objective"],
        )
        zero = min(
            (row for row in records if int(row["group_id"]) == group_id and row["method"] == "zero_update"),
            key=lambda row: row["objective"],
        )
        prior_mse = float(
            np.mean([row["prior"]["mean_mse"] for row in records if int(row["group_id"]) == group_id and row["method"] == "baseline"])
        )
        selected.append(
            {
                "group_id": group_id,
                "baseline_restart": baseline["restart"],
                "zero_restart": zero["restart"],
                "baseline_mse": baseline["metrics"]["mean_mse"],
                "prior_mse": prior_mse,
                "zero_mse": zero["metrics"]["mean_mse"],
                "baseline_minus_prior": baseline["metrics"]["mean_mse"] - prior_mse,
                "baseline_minus_zero": baseline["metrics"]["mean_mse"] - zero["metrics"]["mean_mse"],
                "fraud_baseline_mse": baseline["metrics"]["classes"]["1"]["mean_mse"],
                "fraud_prior_mse": float(
                    np.mean([row["prior"]["classes"]["1"]["mean_mse"] for row in records if int(row["group_id"]) == group_id and row["method"] == "baseline"])
                ),
                "fraud_zero_mse": zero["metrics"]["classes"]["1"]["mean_mse"],
            }
        )
    comparisons = {}
    for control in ("prior", "zero"):
        deltas = np.asarray([row[f"baseline_minus_{control}"] for row in selected], dtype=float)
        non_ties = deltas[deltas != 0]
        wins = int((non_ties < 0).sum())
        comparisons[control] = {
            "n": len(selected),
            "wins": wins,
            "non_ties": int(len(non_ties)),
            "mean_difference": float(deltas.mean()),
            "median_difference": float(np.median(deltas)),
            "one_sided_sign_p": sign_tail(wins, len(non_ties)),
        }
        comparisons[control]["gate"] = bool(
            comparisons[control]["mean_difference"] < 0
            and comparisons[control]["median_difference"] < 0
            and comparisons[control]["one_sided_sign_p"] < 0.05
        )
    comparisons["raw_gate"] = bool(comparisons["prior"]["gate"] and comparisons["zero"]["gate"])
    return comparisons, selected


def write_selected(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--development-groups", type=int, default=8)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--attack-lr", type=float, default=0.05)
    parser.add_argument("--local-lr", type=float, default=1e-3)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--warmup-clients", type=int, default=3)
    parser.add_argument("--warmup-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_seed = args.seed if args.seed is not None else generate_run_seed()
    set_seed(run_seed)
    out = args.output or ROOT / "artifacts/priority5_ieee_cis" / datetime.now(timezone.utc).strftime(
        "phase53_rq1_development_gate_%Y%m%dT%H%M%S%fZ"
    )
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    frame = read_ieee_frame(args.max_rows, derive_seed(run_seed, "ieee-frame-sample"))
    warmup_frame, target_pool = train_test_split(
        frame,
        test_size=0.20,
        random_state=derive_seed(run_seed, "ieee-target-pool"),
        stratify=frame["isFraud"],
    )
    warmup_frame = warmup_frame.reset_index(drop=True)
    target_pool = target_pool.reset_index(drop=True)
    contract, scaler, encoder, medians = fit_contract(warmup_frame)
    warmup_x = transform_frame(warmup_frame, contract, scaler, encoder, medians)
    warmup_y = warmup_frame["isFraud"].astype(np.float32).to_numpy()
    target_x = transform_frame(target_pool, contract, scaler, encoder, medians)
    state_dict, pos_weight = warmup_model(
        warmup_x,
        warmup_y,
        contract.input_dim,
        derive_seed(run_seed, "ieee-warmup"),
        args.warmup_clients,
        args.warmup_rounds,
        args.warmup_batch_size,
    )
    torch.save(state_dict, out / "pre_local_state.pt")
    groups = make_groups(target_pool, target_x, args.development_groups, derive_seed(run_seed, "ieee-development-targets"))
    torch.save(groups, out / "development_targets.pt")

    protocol = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": "protocols/amendments/2026-09-16_priority5_ieee_cis_rq1_development_gate.md",
        "run_seed": run_seed,
        "dataset": "IEEE-CIS Fraud Detection",
        "dataset_sha256": checksum(ROOT / "datasets/ieee-fraud-detection/train_transaction.csv"),
        "scope": "development-only raw attacker gate; four records/group; one fraud/group; one Adam step",
        "records_per_group": 4,
        "fraud_records_per_group": 1,
        "local_steps": 1,
        "local_optimizer": "Adam",
        "local_lr": args.local_lr,
        "development_groups": args.development_groups,
        "restarts": args.restarts,
        "iterations": args.iterations,
        "attack_lr": args.attack_lr,
        "attacker_label_oracle": False,
        "attacker_parameterization": "scaled numeric features clipped to development quantile range; categorical softmax simplexes; soft labels optimized jointly",
        "attacker_constraints": "public normalization/range/categorical structure only; no PaySim balance constraints",
        "evaluation_alignment": "Hungarian matching over all records by transformed feature MSE; true labels used only for reporting class-wise MSE",
        "gate": "all-record feature-MSE baseline must beat Prior and Zero-update controls: negative mean, negative median, one-sided exact sign p<0.05",
        "fraud_mse": "reported descriptively only because labels are not part of attacker knowledge",
        "feature_contract": {
            "numeric": contract.numeric,
            "categorical": contract.categorical,
            "categories": contract.categories,
            "input_dim": contract.input_dim,
            "numeric_lower": contract.numeric_lower,
            "numeric_upper": contract.numeric_upper,
        },
        "warmup": {
            "source": "source-disjoint warmup split from the sampled IEEE-CIS development frame",
            "max_rows": args.max_rows,
            "warmup_rows": int(len(warmup_frame)),
            "target_pool_rows": int(len(target_pool)),
            "warmup_rounds": args.warmup_rounds,
            "warmup_clients": args.warmup_clients,
            "warmup_batch_size": args.warmup_batch_size,
            "pos_weight": float(pos_weight.item()),
        },
        "source_disjointness": {
            "target_rows_unique": len({row for group in groups for row in group["source_rows"]}) == args.development_groups * 4,
            "warmup_target_overlap": int(
                len(set(warmup_frame["_source_row"].astype(int).tolist()) & {row for group in groups for row in group["source_rows"]})
            ),
        },
    }
    dump(out / "protocol_lock.json", protocol)
    dump(
        out / "target_manifest.json",
        {
            "groups": [
                {
                    "group_id": group["group_id"],
                    "source_rows": group["source_rows"],
                    "transaction_ids": group["transaction_ids"],
                    "labels": group["y"].reshape(-1).astype(int).tolist(),
                }
                for group in groups
            ],
            "source_rows_sha256": hashlib.sha256(
                json.dumps([group["source_rows"] for group in groups], sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
    )

    records = []
    for group in groups:
        group_id = int(group["group_id"])
        for restart in range(args.restarts):
            records.extend(
                run_pair(
                    out / "development" / f"group_{group_id}" / f"restart_{restart}",
                    group,
                    contract,
                    state_dict,
                    pos_weight,
                    run_seed,
                    restart,
                    args.iterations,
                    args.attack_lr,
                    args.local_lr,
                )
            )
    comparisons, selected = evaluate(records)
    dump(out / "development_records.json", records)
    dump(out / "development_summary.json", comparisons)
    write_selected(out / "development_selected.csv", selected)
    report = {
        "status": "IEEE_CIS_RAW_GATE_PASSED" if comparisons["raw_gate"] else "IEEE_CIS_RAW_GATE_FAILED",
        "development_gate_passed": bool(comparisons["raw_gate"]),
        "protocol_sha256": checksum(out / "protocol_lock.json"),
        "targets_sha256": checksum(out / "development_targets.pt"),
        "comparisons": comparisons,
        "selected": selected,
        "elapsed_seconds": time.perf_counter() - started,
        "next_per_protocol": (
            "run the predeclared 8-record/one-fraud screen"
            if comparisons["raw_gate"]
            else "stop Priority 5.3 and report that the present attacker did not validate at IEEE-CIS smallest scope"
        ),
    }
    dump(out / "baseline_gate.json", report)
    print(json.dumps({"output": str(out), "status": report["status"], "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
