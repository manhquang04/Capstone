"""Run simulated FL with DNA-based update transformation defense."""

from __future__ import annotations

import copy
import os
import sys
from collections import OrderedDict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_creditcard import load_creditcard_data
from dna_encoder.transform_defense import DNATransformConfig, transform_update_array
from experiments.fraud_fl_common import (
    BATCH_SIZE,
    LOCAL_EPOCHS,
    NUM_CLIENTS,
    NUM_ROUNDS,
    RANDOM_SEED,
    evaluate_model,
    fed_avg,
    print_round_header,
    print_round_metrics,
    save_metrics,
    set_random_seed,
    train_local_model,
)
from models.fraud_mlp import FraudMLP

OUTPUT_PATH = Path(
    os.environ.get(
        "DNA_TRANSFORM_OUTPUT_PATH",
        str(PROJECT_ROOT / "results" / "fraud" / "dna_transform_metrics.json"),
    )
).resolve()
TRANSFORM_CONFIG = DNATransformConfig(
    block_size=int(os.environ.get("DNA_TRANSFORM_BLOCK_SIZE", "256")),
    mix_ratio=float(os.environ.get("DNA_TRANSFORM_MIX", "0.05")),
    keep_ratio=float(os.environ.get("DNA_TRANSFORM_KEEP", "0.90")),
    shrink_factor=float(os.environ.get("DNA_TRANSFORM_SHRINK", "0.50")),
    seed=RANDOM_SEED,
)


def dna_transform_state(
    state_dict: OrderedDict[str, torch.Tensor],
    global_state: OrderedDict[str, torch.Tensor],
    config: DNATransformConfig,
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, float | int]]:
    """Apply DNA transform defense to each floating model update."""
    transformed_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    tensor_count = 0
    element_count = 0
    mean_abs_delta_sum = 0.0
    max_abs_delta = 0.0
    relative_l2_sum = 0.0
    cosine_sum = 0.0

    for tensor_index, (name, tensor) in enumerate(state_dict.items()):
        if not torch.is_floating_point(tensor):
            transformed_state[name] = tensor.clone()
            continue

        update = tensor - global_state[name]
        update_array = update.detach().cpu().numpy().astype(np.float32, copy=False)
        transformed_array, stats = transform_update_array(
            update_array,
            config,
            tensor_index=tensor_index,
        )
        transformed_update = torch.from_numpy(transformed_array.copy()).to(dtype=tensor.dtype)
        transformed_state[name] = global_state[name] + transformed_update

        tensor_count += 1
        element_count += stats.transformed_elements
        mean_abs_delta_sum += stats.mean_abs_delta
        max_abs_delta = max(max_abs_delta, stats.max_abs_delta)
        relative_l2_sum += stats.relative_l2_delta
        cosine_sum += stats.cosine_similarity

    divisor = max(tensor_count, 1)
    return transformed_state, {
        "dna_transform_tensors": tensor_count,
        "dna_transform_elements": element_count,
        "dna_transform_mean_abs_delta": mean_abs_delta_sum / divisor,
        "dna_transform_max_abs_delta": max_abs_delta,
        "dna_transform_relative_l2_delta": relative_l2_sum / divisor,
        "dna_transform_cosine_similarity": cosine_sum / divisor,
    }


def main() -> None:
    set_random_seed()
    client_loaders, validation_loader, test_loader, input_dim, pos_weight, metadata = load_creditcard_data(
        batch_size=BATCH_SIZE,
        num_clients=NUM_CLIENTS,
    )
    global_model = FraudMLP(input_dim)
    sample_counts = [len(loader.dataset) for loader in client_loaders]
    round_metrics = []

    print("Simulated FL + DNA-based Gradient Transformation Defense")
    print(f"DNA transform config: {TRANSFORM_CONFIG}")
    print_round_header(extra_columns=True)
    for round_number in range(1, NUM_ROUNDS + 1):
        global_state = global_model.state_dict()
        local_states = []
        local_losses = []
        transform_seconds = 0.0
        transform_stats: list[dict[str, float | int]] = []

        for loader in client_loaders:
            local_model = copy.deepcopy(global_model)
            local_losses.append(train_local_model(local_model, loader, pos_weight, LOCAL_EPOCHS))

            started = perf_counter()
            transformed_state, stats = dna_transform_state(
                local_model.state_dict(),
                global_state,
                TRANSFORM_CONFIG,
            )
            transform_seconds += perf_counter() - started
            transform_stats.append(stats)
            local_states.append(transformed_state)

        global_model.load_state_dict(fed_avg(local_states, sample_counts))
        validation_metrics = evaluate_model(global_model, validation_loader)
        metrics = {
            "round": round_number,
            "train_loss": sum(local_losses) / len(local_losses),
            **evaluate_model(
                global_model,
                test_loader,
                threshold=float(validation_metrics["optimal_threshold"]),
            ),
            "dna_encode_decode_ms": transform_seconds * 1_000,
            "encoded_tensors": int(sum(s["dna_transform_tensors"] for s in transform_stats)),
            "encoded_elements": int(sum(s["dna_transform_elements"] for s in transform_stats)),
            **_average_transform_stats(transform_stats),
        }
        round_metrics.append(metrics)
        print_round_metrics(metrics)

    save_metrics(
        OUTPUT_PATH,
        "DNA_TransformDefense",
        round_metrics,
        extra_config={
            "dataset": "PaySim",
            "target": "isFraud",
            "features": metadata.feature_names,
            "train_fraud_rate": metadata.train_fraud_rate,
            "validation_fraud_rate": metadata.validation_fraud_rate,
            "test_fraud_rate": metadata.test_fraud_rate,
            "client_sample_counts": metadata.client_sample_counts,
            "client_fraud_rates": metadata.client_fraud_rates,
            "client_type_distributions": metadata.client_type_distributions,
            "threshold_tuning": "F1 on validation split",
            "dna_transform_defense": "DNA-seeded block permutation, selective attenuation, and residual update mixing",
            "dna_transform_config": TRANSFORM_CONFIG.__dict__,
        },
    )
    print(f"Saved metrics: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


def _average_transform_stats(stats: list[dict[str, float | int]]) -> dict[str, float]:
    if not stats:
        return {}
    keys = [
        "dna_transform_mean_abs_delta",
        "dna_transform_max_abs_delta",
        "dna_transform_relative_l2_delta",
        "dna_transform_cosine_similarity",
    ]
    return {
        key: float(sum(float(item[key]) for item in stats) / len(stats))
        for key in keys
    }


if __name__ == "__main__":
    main()
