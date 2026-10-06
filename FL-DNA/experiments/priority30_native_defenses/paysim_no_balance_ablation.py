"""Priority 30 Part C: PaySim no-balance-feature RQ2 ablation.

This driver intentionally loads PaySim through the frozen existing loader and
then projects out exactly the six balance-derived columns.  It does not modify
data/load_creditcard.py.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.load_creditcard import load_creditcard_data  # noqa: E402
from dna_encoder.transform_defense import DNATransformConfig  # noqa: E402
from dna_encoder.transform_defense_v2 import DNATransformV2Config  # noqa: E402
from experiments.fraud_fl_common import (  # noqa: E402
    BATCH_SIZE,
    DEVICE,
    LOCAL_EPOCHS,
    NUM_CLIENTS,
    NUM_ROUNDS,
    evaluate_model,
    fed_avg,
    set_random_seed,
    train_local_model,
)
from experiments.run_fraud_fl_dna_transform import dna_transform_state  # noqa: E402
from experiments.run_fraud_fl_dna_transform_v2 import dna_transform_v2_state  # noqa: E402
from models.fraud_mlp import FraudMLP  # noqa: E402
from privacy.seed_manager import derive_seed  # noqa: E402


DROP_FEATURES = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_diff_orig",
    "balance_diff_dest",
]


def keep_indices(feature_names: list[str]) -> list[int]:
    drop = set(DROP_FEATURES)
    missing = sorted(drop - set(feature_names))
    if missing:
        raise ValueError(f"missing expected balance features: {missing}")
    return [idx for idx, name in enumerate(feature_names) if name not in drop]


def project_loader(loader: DataLoader, indices: list[int], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    features, labels = loader.dataset.tensors
    dataset = TensorDataset(features[:, indices].clone(), labels.clone())
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def load_projected(seed: int, drop_balance: bool) -> tuple[list[DataLoader], DataLoader, DataLoader, int, torch.Tensor, object]:
    clients, val, test, input_dim, pos_weight, metadata = load_creditcard_data(
        batch_size=BATCH_SIZE,
        num_clients=NUM_CLIENTS,
        seed=seed,
    )
    if not drop_balance:
        return clients, val, test, input_dim, pos_weight, metadata
    indices = keep_indices(metadata.feature_names)
    feature_names = [metadata.feature_names[i] for i in indices]
    clients2 = [
        project_loader(loader, indices, BATCH_SIZE, True, seed + client_id)
        for client_id, loader in enumerate(clients)
    ]
    val2 = project_loader(val, indices, BATCH_SIZE, False, seed)
    test2 = project_loader(test, indices, BATCH_SIZE, False, seed)
    meta2 = replace(
        metadata,
        feature_names=feature_names,
        numeric_center=[metadata.numeric_center[i] for i, name in enumerate(metadata.feature_names[: len(metadata.numeric_center)]) if name in feature_names],
        numeric_scale=[metadata.numeric_scale[i] for i, name in enumerate(metadata.feature_names[: len(metadata.numeric_scale)]) if name in feature_names],
    )
    return clients2, val2, test2, len(indices), pos_weight, meta2


def run_method(seed: int, method: str, drop_balance: bool) -> dict[str, object]:
    set_random_seed(seed)
    clients, val, test, input_dim, pos_weight, metadata = load_projected(seed, drop_balance)
    model = FraudMLP(input_dim).to(DEVICE)
    sample_counts = [len(loader.dataset) for loader in clients]
    v1_config = DNATransformConfig(mix_ratio=0.08, keep_ratio=0.88, shrink_factor=0.45, seed=derive_seed(seed, "p30-v1"))
    v2_config = DNATransformV2Config(compression_ratio=0.95, quantization_eta=0.01, seed=derive_seed(seed, "p30-v2"))
    rounds = []
    for round_number in range(1, NUM_ROUNDS + 1):
        global_state = model.state_dict()
        local_states = []
        losses = []
        for client_id, loader in enumerate(clients):
            local = copy.deepcopy(model)
            losses.append(train_local_model(local, loader, pos_weight, LOCAL_EPOCHS))
            state = local.state_dict()
            if method == "v1_conservative":
                state, _stats = dna_transform_state(state, global_state, v1_config)
            elif method == "v2_0p95":
                state, _stats = dna_transform_v2_state(
                    state,
                    global_state,
                    v2_config,
                    quantization_seed=derive_seed(seed, "p30-v2-q", round_number, client_id),
                )
            elif method == "baseline":
                pass
            else:
                raise ValueError(method)
            local_states.append(state)
        model.load_state_dict(fed_avg(local_states, sample_counts))
        val_metrics = evaluate_model(model, val)
        test_metrics = evaluate_model(model, test, threshold=float(val_metrics["optimal_threshold"]))
        rounds.append({"round": round_number, "train_loss": sum(losses) / len(losses), **test_metrics})
    final = rounds[-1]
    return {
        "seed": seed,
        "method": method,
        "drop_balance": drop_balance,
        "input_dim": input_dim,
        "features": metadata.feature_names,
        "f1_score": final["f1_score"],
        "auc_roc": final["auc_roc"],
        "pr_auc": final["pr_auc"],
        "rounds": rounds,
    }


def result_row(path: Path) -> dict[str, object]:
    result = json.loads(path.read_text())
    return {k: result[k] for k in ["seed", "method", "drop_balance", "input_dim", "f1_score", "auc_roc", "pr_auc"]}


def run_job(seed: int, method: str, drop: bool, path: str) -> dict[str, object]:
    torch.set_num_threads(1)
    out_path = Path(path)
    if out_path.exists():
        return result_row(out_path)
    result = run_method(seed, method, drop)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    return {k: result[k] for k in ["seed", "method", "drop_balance", "input_dim", "f1_score", "auc_roc", "pr_auc"]}


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    methods = ["baseline", "v1_conservative", "v2_0p95"]
    jobs = [
        (args.seed_start + i, method, drop, out / f"seed_{args.seed_start + i}_{method}_{'nobalance' if drop else 'full'}.json")
        for i in range(args.replicates)
        for drop in [False, True]
        for method in methods
    ]

    rows = []
    pending = [job for job in jobs if not job[3].exists()]
    for _seed, _method, _drop, path in jobs:
        if path.exists():
            row = result_row(path)
            rows.append(row)
            print(json.dumps({**row, "resumed": True}), flush=True)
    if args.workers <= 1:
        for seed, method, drop, path in pending:
            row = run_job(seed, method, drop, str(path))
            rows.append(row)
            print(json.dumps(row), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_job, seed, method, drop, str(path)) for seed, method, drop, path in pending]
            for fut in as_completed(futures):
                row = fut.result()
                rows.append(row)
                print(json.dumps(row), flush=True)
    rows = sorted(rows, key=lambda r: (int(r["seed"]), bool(r["drop_balance"]), str(r["method"])))
    csv_path = out / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def check_projection() -> dict[str, object]:
    clients, val, test, input_dim, _pos_weight, metadata = load_creditcard_data(
        batch_size=BATCH_SIZE,
        num_clients=NUM_CLIENTS,
        seed=42,
    )
    indices = keep_indices(metadata.feature_names)
    kept = [metadata.feature_names[i] for i in indices]
    return {
        "original_dim": input_dim,
        "projected_dim": len(indices),
        "removed": [name for name in metadata.feature_names if name in DROP_FEATURES],
        "kept": kept,
        "exact_removed_match": [name for name in metadata.feature_names if name in DROP_FEATURES] == DROP_FEATURES,
        "projected_client0_shape": list(project_loader(clients[0], indices, BATCH_SIZE, True, 42).dataset.tensors[0].shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/priority30_native_defenses/paysim_no_balance_ablation")
    parser.add_argument("--replicates", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=30_300)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(check_projection(), indent=2), flush=True)
    else:
        run(args)


if __name__ == "__main__":
    main()
