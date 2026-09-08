"""Run simulated FL with DNA-protected local model updates."""

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
from dna_encoder.encoder import DNAEncoder
from experiments.fraud_fl_common import (
    BATCH_SIZE,
    DEVICE,
    LOCAL_EPOCHS,
    NUM_CLIENTS,
    NUM_ROUNDS,
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
        "DNA_OUTPUT_PATH",
        str(PROJECT_ROOT / "results" / "fraud" / "dna_metrics.json"),
    )
).resolve()


def dna_round_trip_state(
    state_dict: OrderedDict[str, torch.Tensor],
    global_state: OrderedDict[str, torch.Tensor],
    encoder: DNAEncoder,
) -> tuple[OrderedDict[str, torch.Tensor], int, int]:
    """Encode/decode each floating model update before server aggregation."""
    decoded_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    encoded_tensors = 0
    encoded_elements = 0

    for name, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            decoded_state[name] = tensor.clone()
            continue

        update = tensor - global_state[name]
        float32_array = update.detach().cpu().numpy().astype(np.float32, copy=False)
        payload = encoder.encode_array(float32_array)
        decoded_array = encoder.decode_array(payload, tensor.shape)
        decoded_update = torch.from_numpy(decoded_array.copy()).to(dtype=tensor.dtype, device=tensor.device)
        decoded_tensor = global_state[name] + decoded_update
        if decoded_tensor.shape != tensor.shape:
            raise ValueError(f"DNA round trip changed tensor shape for '{name}'")

        decoded_state[name] = decoded_tensor
        encoded_tensors += 1
        encoded_elements += tensor.numel()

    return decoded_state, encoded_tensors, encoded_elements


def main() -> None:
    set_random_seed()
    client_loaders, validation_loader, test_loader, input_dim, pos_weight, metadata = load_creditcard_data(
        batch_size=BATCH_SIZE,
        num_clients=NUM_CLIENTS,
    )
    global_model = FraudMLP(input_dim).to(DEVICE)
    encoder = DNAEncoder()
    sample_counts = [len(loader.dataset) for loader in client_loaders]
    round_metrics = []

    print("Simulated FL + DNA Encoder")
    print_round_header(extra_columns=True)
    for round_number in range(1, NUM_ROUNDS + 1):
        global_state = global_model.state_dict()
        local_states = []
        local_losses = []
        dna_seconds = 0.0
        encoded_tensors = 0
        encoded_elements = 0

        for loader in client_loaders:
            local_model = copy.deepcopy(global_model)
            local_losses.append(train_local_model(local_model, loader, pos_weight, LOCAL_EPOCHS))

            started = perf_counter()
            decoded_state, tensor_count, element_count = dna_round_trip_state(
                local_model.state_dict(),
                global_state,
                encoder,
            )
            dna_seconds += perf_counter() - started
            encoded_tensors += tensor_count
            encoded_elements += element_count
            local_states.append(decoded_state)

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
            "dna_encode_decode_ms": dna_seconds * 1_000,
            "encoded_tensors": encoded_tensors,
            "encoded_elements": encoded_elements,
        }
        round_metrics.append(metrics)
        print_round_metrics(metrics)

    save_metrics(
        OUTPUT_PATH,
        "DNA",
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
            "dna_protection": "DNA encode/decode of local model updates + AES-256-GCM",
        },
    )
    print(f"Saved metrics: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
