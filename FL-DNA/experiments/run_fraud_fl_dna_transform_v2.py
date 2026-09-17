"""Run simulated FL with DNA Transform v2 lossy sketch defense.

Step 4 only: development-scale utility smoke.  This runner is intentionally
separate from the v1 DNA Transform runner and does not modify v1 code.
"""

from __future__ import annotations

import copy
import os
import sys
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_creditcard import load_creditcard_data
from dna_encoder.transform_defense_v2 import (
    DNATransformV2Config,
    transform_and_reconstruct_array_v2,
)
from experiments.fraud_fl_common import (
    BATCH_SIZE,
    DEVICE,
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
from privacy.seed_manager import derive_seed


OUTPUT_PATH = Path(
    os.environ.get(
        "DNA_TRANSFORM_V2_OUTPUT_PATH",
        str(PROJECT_ROOT / "results" / "fraud" / "dna_transform_v2_metrics.json"),
    )
).resolve()

TRANSFORM_V2_CONFIG = DNATransformV2Config(
    compression_ratio=float(os.environ.get("DNA_TRANSFORM_V2_COMPRESSION_RATIO", "0.95")),
    quantization_eta=float(os.environ.get("DNA_TRANSFORM_V2_QUANTIZATION_ETA", "0.01")),
)

DNA_TRANSFORM_V2_RUN_SEED = int(
    os.environ.get("DNA_TRANSFORM_V2_RUN_SEED", derive_seed(RANDOM_SEED, "dna-transform-v2"))
)


def dna_transform_v2_state(
    state_dict: OrderedDict[str, torch.Tensor],
    global_state: OrderedDict[str, torch.Tensor],
    config: DNATransformV2Config,
    quantization_seed: int,
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, float | int]]:
    """Sketch and reconstruct each floating local update before FedAvg."""

    transformed_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    tensor_count = 0
    original_elements = 0
    sketch_elements = 0
    relative_l2_sum = 0.0
    max_relative_l2 = 0.0
    rank_retained_sum = 0.0
    quantization_delta_sum = 0.0
    original_sq_sum = 0.0
    diff_sq_sum = 0.0

    for tensor_index, (name, tensor) in enumerate(state_dict.items()):
        if not torch.is_floating_point(tensor):
            transformed_state[name] = tensor.clone()
            continue

        update = tensor - global_state[name]
        update_array = update.detach().cpu().numpy().astype(np.float32, copy=False)
        _, reconstructed_array, metadata, stats = transform_and_reconstruct_array_v2(
            update_array,
            config,
            tensor_index=tensor_index,
            quantization_seed=quantization_seed,
        )
        reconstructed_update = torch.from_numpy(reconstructed_array.copy()).to(
            dtype=tensor.dtype,
            device=tensor.device,
        )
        transformed_state[name] = global_state[name] + reconstructed_update

        original = update_array.reshape(-1).astype(np.float64)
        recovered = reconstructed_array.reshape(-1).astype(np.float64)
        diff = recovered - original
        original_sq = float(np.dot(original, original))
        diff_sq = float(np.dot(diff, diff))
        relative_l2 = diff_sq**0.5 / max(original_sq**0.5, np.finfo(float).tiny)

        tensor_count += 1
        original_elements += int(metadata.original_size)
        sketch_elements += int(metadata.sketch_size)
        relative_l2_sum += relative_l2
        max_relative_l2 = max(max_relative_l2, relative_l2)
        rank_retained_sum += metadata.sketch_size / max(metadata.padded_size, 1)
        quantization_delta_sum += float(stats.quantization_delta)
        original_sq_sum += original_sq
        diff_sq_sum += diff_sq

    divisor = max(tensor_count, 1)
    aggregate_relative_l2 = diff_sq_sum**0.5 / max(original_sq_sum**0.5, np.finfo(float).tiny)
    return transformed_state, {
        "dna_transform_v2_tensors": tensor_count,
        "dna_transform_v2_original_elements": original_elements,
        "dna_transform_v2_sketch_elements": sketch_elements,
        "dna_transform_v2_sketch_ratio": sketch_elements / max(original_elements, 1),
        "dna_transform_v2_mean_relative_l2_error": relative_l2_sum / divisor,
        "dna_transform_v2_max_relative_l2_error": max_relative_l2,
        "dna_transform_v2_aggregate_relative_l2_error": aggregate_relative_l2,
        "dna_transform_v2_mean_rank_retained_fraction": rank_retained_sum / divisor,
        "dna_transform_v2_mean_quantization_delta": quantization_delta_sum / divisor,
    }


def main() -> None:
    set_random_seed()
    client_loaders, validation_loader, test_loader, input_dim, pos_weight, metadata = load_creditcard_data(
        batch_size=BATCH_SIZE,
        num_clients=NUM_CLIENTS,
    )
    global_model = FraudMLP(input_dim).to(DEVICE)
    sample_counts = [len(loader.dataset) for loader in client_loaders]
    round_metrics = []

    print("Simulated FL + DNA Transform v2 lossy sketch")
    print(f"DNA Transform v2 config: {TRANSFORM_V2_CONFIG}")
    print_round_header(extra_columns=True)
    for round_number in range(1, NUM_ROUNDS + 1):
        global_state = global_model.state_dict()
        local_states = []
        local_losses = []
        transform_seconds = 0.0
        transform_stats: list[dict[str, float | int]] = []
        client_round_seeds: list[int] = []

        for client_index, loader in enumerate(client_loaders):
            local_model = copy.deepcopy(global_model)
            local_losses.append(train_local_model(local_model, loader, pos_weight, LOCAL_EPOCHS))

            client_seed = derive_seed(
                DNA_TRANSFORM_V2_RUN_SEED,
                "dna_transform_v2",
                round_number,
                client_index,
            )
            client_round_seeds.append(client_seed)
            started = perf_counter()
            transformed_state, stats = dna_transform_v2_state(
                local_model.state_dict(),
                global_state,
                replace(TRANSFORM_V2_CONFIG, seed=client_seed),
                quantization_seed=derive_seed(client_seed, "quantization"),
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
            "encoded_tensors": int(sum(s["dna_transform_v2_tensors"] for s in transform_stats)),
            "encoded_elements": int(sum(s["dna_transform_v2_sketch_elements"] for s in transform_stats)),
            "dna_transform_v2_client_seeds": client_round_seeds,
            **_average_transform_stats(transform_stats),
        }
        round_metrics.append(metrics)
        print_round_metrics(metrics)

    save_metrics(
        OUTPUT_PATH,
        "DNA_Transform_v2_LossySketch",
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
            "dna_transform_v2_defense": "lossy DNA-seeded random projection sketch plus stochastic quantization; server lifts each client update before FedAvg",
            "dna_transform_v2_config": TRANSFORM_V2_CONFIG.__dict__,
            "dna_transform_v2_seed_strategy": "deterministic client-round seeds derived with BLAKE2b from a run seed",
            "dna_transform_v2_seed_scope": "client_round",
            "dna_transform_v2_run_seed": DNA_TRANSFORM_V2_RUN_SEED,
        },
    )
    print(f"Saved metrics: {_display_path(OUTPUT_PATH)}")


def _average_transform_stats(stats: list[dict[str, float | int]]) -> dict[str, float]:
    if not stats:
        return {}
    keys = [
        "dna_transform_v2_sketch_ratio",
        "dna_transform_v2_mean_relative_l2_error",
        "dna_transform_v2_max_relative_l2_error",
        "dna_transform_v2_aggregate_relative_l2_error",
        "dna_transform_v2_mean_rank_retained_fraction",
        "dna_transform_v2_mean_quantization_delta",
    ]
    return {
        key: float(sum(float(item[key]) for item in stats) / len(stats))
        for key in keys
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
