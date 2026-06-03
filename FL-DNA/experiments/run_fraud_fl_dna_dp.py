"""Run simulated FL with DP-protected updates transported through DNA encoding."""

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
from experiments.run_fraud_fl_dna import dna_round_trip_state
from models.fraud_mlp import FraudMLP
from privacy.dp_engine import apply_dp_to_local_state

OUTPUT_PATH = PROJECT_ROOT / "results" / "fraud" / "dna_dp_metrics.json"
CLIP_NORM = float(os.environ.get("DP_CLIP_NORM", "100.0"))
NOISE_MULTIPLIER = float(os.environ.get("DP_NOISE_MULTIPLIER", "0.0005"))


def main() -> None:
    set_random_seed()
    client_loaders, validation_loader, test_loader, input_dim, pos_weight, metadata = load_creditcard_data(
        batch_size=BATCH_SIZE,
        num_clients=NUM_CLIENTS,
    )
    global_model = FraudMLP(input_dim)
    encoder = DNAEncoder()
    sample_counts = [len(loader.dataset) for loader in client_loaders]
    round_metrics = []

    print("Simulated FL + DNA Encoder + Differential Privacy")
    print(f"DP config: clip_norm={CLIP_NORM}, noise_multiplier={NOISE_MULTIPLIER}")
    print_round_header(extra_columns=True)
    for round_number in range(1, NUM_ROUNDS + 1):
        global_state = global_model.state_dict()
        local_states = []
        local_losses = []
        dna_seconds = 0.0
        encoded_tensors = 0
        encoded_elements = 0
        norms_before_clip = []
        norms_after_clip = []
        noise_std = NOISE_MULTIPLIER * CLIP_NORM

        for loader in client_loaders:
            local_model = copy.deepcopy(global_model)
            local_losses.append(train_local_model(local_model, loader, pos_weight, LOCAL_EPOCHS))

            dp_state, norm_before, norm_after, noise_std = apply_dp_to_local_state(
                local_model.state_dict(),
                global_state,
                CLIP_NORM,
                NOISE_MULTIPLIER,
            )
            norms_before_clip.append(norm_before)
            norms_after_clip.append(norm_after)

            started = perf_counter()
            decoded_state, tensor_count, element_count = dna_round_trip_state(
                dp_state,
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
            "dp_clip_norm": CLIP_NORM,
            "dp_noise_multiplier": NOISE_MULTIPLIER,
            "dp_noise_std": noise_std,
            "avg_update_norm_before_clip": sum(norms_before_clip) / len(norms_before_clip),
            "avg_update_norm_after_clip": sum(norms_after_clip) / len(norms_after_clip),
        }
        round_metrics.append(metrics)
        print_round_metrics(metrics)

    save_metrics(
        OUTPUT_PATH,
        "DNA+DP",
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
            "dp_protection": "Client update L2 clipping + Gaussian noise",
            "dna_protection": "DNA encode/decode of DP-protected model updates + AES-256-GCM",
            "dp_clip_norm": CLIP_NORM,
            "dp_noise_multiplier": NOISE_MULTIPLIER,
        },
    )
    print(f"Saved metrics: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
