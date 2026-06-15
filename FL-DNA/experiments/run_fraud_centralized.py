"""Run centralized PaySim MLP as an upper-bound reference."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_creditcard import load_paysim_splits
from experiments.fraud_fl_common import (
    BATCH_SIZE,
    DEVICE,
    LOCAL_EPOCHS,
    NUM_ROUNDS,
    evaluate_model,
    print_round_header,
    print_round_metrics,
    save_metrics,
    set_random_seed,
    train_local_model,
)
from models.fraud_mlp import FraudMLP

OUTPUT_PATH = PROJECT_ROOT / "results" / "fraud" / "centralized_metrics.json"


def main() -> None:
    set_random_seed()
    train_loader, validation_loader, test_loader, input_dim, pos_weight, metadata = (
        load_paysim_splits(batch_size=BATCH_SIZE)
    )
    model = FraudMLP(input_dim).to(DEVICE)
    round_metrics = []

    print("Centralized PaySim MLP")
    print_round_header()
    for epoch in range(1, NUM_ROUNDS + 1):
        train_loss = train_local_model(model, train_loader, pos_weight, LOCAL_EPOCHS)
        validation_metrics = evaluate_model(model, validation_loader)
        metrics = {
            "round": epoch,
            "train_loss": train_loss,
            **evaluate_model(
                model,
                test_loader,
                threshold=float(validation_metrics["optimal_threshold"]),
            ),
        }
        round_metrics.append(metrics)
        print_round_metrics(metrics)

    save_metrics(
        OUTPUT_PATH,
        "Centralized",
        round_metrics,
        extra_config={
            "dataset": "PaySim",
            "target": "isFraud",
            "features": metadata.feature_names,
            "train_fraud_rate": metadata.train_fraud_rate,
            "validation_fraud_rate": metadata.validation_fraud_rate,
            "test_fraud_rate": metadata.test_fraud_rate,
            "threshold_tuning": "F1 on validation split",
            "training_mode": "centralized pooled train set",
        },
    )
    print(f"Saved metrics: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
