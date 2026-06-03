"""Run FL with DNA transform defense followed by secure aggregation."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_creditcard import load_creditcard_data
from experiments.fraud_fl_common import (
    BATCH_SIZE,
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
from experiments.run_fraud_fl_dna_transform import (
    TRANSFORM_CONFIG,
    _average_transform_stats,
    dna_transform_state,
)
from models.fraud_mlp import FraudMLP
from privacy.secure_agg import secure_aggregate_states

OUTPUT_PATH = PROJECT_ROOT / "results" / "fraud" / "dna_transform_secureagg_metrics.json"


def main() -> None:
    set_random_seed()
    client_loaders, validation_loader, test_loader, input_dim, pos_weight, metadata = load_creditcard_data(
        batch_size=BATCH_SIZE,
        num_clients=NUM_CLIENTS,
    )
    global_model = FraudMLP(input_dim)
    sample_counts = [len(loader.dataset) for loader in client_loaders]
    round_metrics = []

    print("Simulated FL + DNA Transform Defense + Secure Aggregation")
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

        aggregated_state, secureagg_metadata = secure_aggregate_states(
            local_states,
            global_state,
            sample_counts,
            seed=RANDOM_SEED + round_number,
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
            "dna_encode_decode_ms": transform_seconds * 1_000,
            "encoded_tensors": int(sum(s["dna_transform_tensors"] for s in transform_stats)),
            "encoded_elements": int(sum(s["dna_transform_elements"] for s in transform_stats)),
            **_average_transform_stats(transform_stats),
            **secureagg_metadata,
        }
        round_metrics.append(metrics)
        print_round_metrics(metrics)

    save_metrics(
        OUTPUT_PATH,
        "DNA_TransformDefense_SecureAgg",
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
            "secure_aggregation": "Pairwise mask simulation over transformed weighted local model updates",
            "server_sees_individual_raw_updates": False,
        },
    )
    print(f"Saved metrics: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
