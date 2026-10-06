"""Run simulated FL with DNA Transform v2-SB server-blind sketch aggregation."""

from __future__ import annotations

import copy
import hashlib
import os
import sys
from collections import OrderedDict, defaultdict
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_creditcard import load_creditcard_data
from dna_encoder.transform_defense_v2_server_blind import (
    ServerBlindAggregator,
    ServerBlindV2Config,
    lift_sketch_array_v2sb,
    sketch_update_array_v2sb,
)
from experiments.fraud_fl_common import (
    BATCH_SIZE,
    DEVICE,
    LOCAL_EPOCHS,
    NUM_CLIENTS,
    NUM_ROUNDS,
    RANDOM_SEED,
    evaluate_model,
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
        "DNA_TRANSFORM_V2SB_OUTPUT_PATH",
        str(PROJECT_ROOT / "results" / "fraud" / "dna_transform_v2sb_metrics.json"),
    )
).resolve()

TRANSFORM_V2SB_CONFIG = ServerBlindV2Config(
    compression_ratio=float(os.environ.get("DNA_TRANSFORM_V2SB_COMPRESSION_RATIO", "0.95")),
    quantization_eta=float(os.environ.get("DNA_TRANSFORM_V2SB_QUANTIZATION_ETA", "0.01")),
)

DNA_TRANSFORM_V2SB_RUN_SEED = int(
    os.environ.get("DNA_TRANSFORM_V2SB_RUN_SEED", derive_seed(RANDOM_SEED, "dna-transform-v2sb"))
)


def _round_key(run_seed: int, round_number: int) -> bytes:
    payload = f"FL-DNA-v2SB|{run_seed}|round|{round_number}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def _aggregate_v2sb_updates(
    local_states: list[OrderedDict[str, torch.Tensor]],
    global_state: OrderedDict[str, torch.Tensor],
    sample_counts: list[int],
    config: ServerBlindV2Config,
    *,
    round_key: bytes,
    round_number: int,
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, float | int]]:
    total_samples = sum(sample_counts)
    weights = [count / total_samples for count in sample_counts]
    per_tensor_sketches: dict[str, list] = defaultdict(list)
    transformed_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    tensor_count = 0
    original_elements = 0
    sketch_elements = 0
    original_sq_sum = 0.0
    diff_sq_sum = 0.0
    client_encode_seconds = 0.0
    server_aggregate_seconds = 0.0
    client_decode_seconds = 0.0

    for client_id, local_state in enumerate(local_states):
        started = perf_counter()
        for tensor_index, (name, tensor) in enumerate(local_state.items()):
            if not torch.is_floating_point(tensor):
                continue
            update = tensor - global_state[name]
            update_array = update.detach().cpu().numpy().astype(np.float32, copy=False)
            sketch = sketch_update_array_v2sb(
                update_array,
                config,
                round_key=round_key,
                round_number=round_number,
                tensor_index=tensor_index,
                client_id=client_id,
            )
            per_tensor_sketches[name].append(sketch)
        client_encode_seconds += perf_counter() - started

    aggregator = ServerBlindAggregator()
    for tensor_index, (name, global_tensor) in enumerate(global_state.items()):
        if not torch.is_floating_point(global_tensor):
            transformed_state[name] = local_states[0][name].clone()
            continue
        started = perf_counter()
        aggregate_sketch = aggregator.aggregate(per_tensor_sketches[name], weights)
        server_aggregate_seconds += perf_counter() - started

        started = perf_counter()
        recovered_array = lift_sketch_array_v2sb(aggregate_sketch, round_key=round_key)
        client_decode_seconds += perf_counter() - started
        recovered_update = torch.from_numpy(recovered_array.copy()).to(
            dtype=global_tensor.dtype,
            device=global_tensor.device,
        )
        transformed_state[name] = global_tensor + recovered_update

        original_update = sum(
            weights[i] * (local_states[i][name] - global_tensor).detach().cpu().numpy().astype(np.float64)
            for i in range(len(local_states))
        )
        recovered = recovered_array.astype(np.float64)
        diff = recovered - original_update
        original_sq_sum += float(np.dot(original_update.reshape(-1), original_update.reshape(-1)))
        diff_sq_sum += float(np.dot(diff.reshape(-1), diff.reshape(-1)))
        tensor_count += 1
        original_elements += int(aggregate_sketch.metadata.original_size)
        sketch_elements += int(aggregate_sketch.metadata.sketch_size)

    if getattr(aggregator, "key") is not None or getattr(aggregator, "round_key") is not None:
        raise RuntimeError("ServerBlindAggregator touched key material")
    aggregate_relative_l2 = diff_sq_sum**0.5 / max(original_sq_sum**0.5, np.finfo(float).tiny)
    return transformed_state, {
        "dna_transform_v2sb_tensors": tensor_count,
        "dna_transform_v2sb_original_elements": original_elements,
        "dna_transform_v2sb_sketch_elements": sketch_elements,
        "dna_transform_v2sb_sketch_ratio": sketch_elements / max(original_elements, 1),
        "dna_transform_v2sb_aggregate_relative_l2_error": aggregate_relative_l2,
        "dna_transform_v2sb_client_encode_ms": client_encode_seconds * 1000.0,
        "dna_transform_v2sb_server_aggregate_ms": server_aggregate_seconds * 1000.0,
        "dna_transform_v2sb_client_decode_ms": client_decode_seconds * 1000.0,
        "dna_transform_v2sb_total_ms": (client_encode_seconds + server_aggregate_seconds + client_decode_seconds)
        * 1000.0,
    }


def main() -> None:
    torch.set_num_threads(1)
    set_random_seed()
    client_loaders, validation_loader, test_loader, input_dim, pos_weight, metadata = load_creditcard_data(
        batch_size=BATCH_SIZE,
        num_clients=NUM_CLIENTS,
    )
    global_model = FraudMLP(input_dim).to(DEVICE)
    sample_counts = [len(loader.dataset) for loader in client_loaders]
    round_metrics = []

    print("Simulated FL + DNA Transform v2-SB server-blind sketch aggregation")
    print(f"DNA Transform v2-SB config: {TRANSFORM_V2SB_CONFIG}")
    print_round_header(extra_columns=True)
    for round_number in range(1, NUM_ROUNDS + 1):
        global_state = global_model.state_dict()
        local_states = []
        local_losses = []
        for loader in client_loaders:
            local_model = copy.deepcopy(global_model)
            local_losses.append(train_local_model(local_model, loader, pos_weight, LOCAL_EPOCHS))
            local_states.append(local_model.state_dict())

        transformed_state, transform_stats = _aggregate_v2sb_updates(
            local_states,
            global_state,
            sample_counts,
            TRANSFORM_V2SB_CONFIG,
            round_key=_round_key(DNA_TRANSFORM_V2SB_RUN_SEED, round_number),
            round_number=round_number,
        )
        global_model.load_state_dict(transformed_state)
        validation_metrics = evaluate_model(global_model, validation_loader)
        metrics = {
            "round": round_number,
            "train_loss": sum(local_losses) / len(local_losses),
            **evaluate_model(
                global_model,
                test_loader,
                threshold=float(validation_metrics["optimal_threshold"]),
            ),
            "dna_encode_decode_ms": transform_stats["dna_transform_v2sb_total_ms"],
            "encoded_tensors": int(transform_stats["dna_transform_v2sb_tensors"] * NUM_CLIENTS),
            "encoded_elements": int(transform_stats["dna_transform_v2sb_sketch_elements"] * NUM_CLIENTS),
            **transform_stats,
        }
        round_metrics.append(metrics)
        print_round_metrics(metrics)

    save_metrics(
        OUTPUT_PATH,
        "DNA_Transform_v2SB_ServerBlind",
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
            "dna_transform_v2sb_defense": "server-blind shared-key sketch aggregation; server aggregates sketches without decode key; clients decode aggregate",
            "dna_transform_v2sb_config": asdict(TRANSFORM_V2SB_CONFIG),
            "dna_transform_v2sb_key_strategy": "deterministic 256-bit round key from SHA-256(run_seed, round) for reproducible simulated key agreement",
            "dna_transform_v2sb_run_seed": DNA_TRANSFORM_V2SB_RUN_SEED,
        },
    )
    print(f"Saved metrics: {_display_path(OUTPUT_PATH)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
