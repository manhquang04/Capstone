"""Run simulated FL with DNA-protected updates and secure aggregation."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from time import perf_counter

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
    print_round_header,
    print_round_metrics,
    save_metrics,
    set_random_seed,
    train_local_model,
)
from experiments.run_fraud_fl_dna import dna_round_trip_state
from models.fraud_mlp import FraudMLP
from privacy.seed_manager import derive_seed, generate_run_seed
from privacy.secure_agg import secure_aggregate_states

OUTPUT_PATH = PROJECT_ROOT / "results" / "fraud" / "dna_secureagg_metrics.json"
SECURE_AGG_RUN_SEED = int(os.environ.get("SECURE_AGG_RUN_SEED", generate_run_seed()))


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

    print("Simulated FL + DNA Encoder + Secure Aggregation")
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

        aggregated_state, secureagg_metadata = secure_aggregate_states(
            local_states,
            global_state,
            sample_counts,
            seed=derive_seed(SECURE_AGG_RUN_SEED, "secureagg_dna", round_number),
        )
        global_model.load_state_dict(aggregated_state)
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
            **secureagg_metadata,
        }
        round_metrics.append(metrics)
        print_round_metrics(metrics)

    save_metrics(
        OUTPUT_PATH,
        "DNA+SecureAgg",
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
            "secure_aggregation": "Pairwise mask simulation over weighted local model updates",
            "secure_agg_seed_strategy": "fresh run seed from secure randomness; deterministic round seeds derived with BLAKE2b",
            "secure_agg_seed_scope": "round",
            "secure_agg_run_seed": SECURE_AGG_RUN_SEED,
            "server_sees_individual_raw_updates": False,
        },
    )
    print(f"Saved metrics: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
