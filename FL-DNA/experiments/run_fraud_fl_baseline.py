"""Run simulated FL baseline on Credit Card Fraud Detection."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_creditcard import load_creditcard_data
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
from models.fraud_mlp import FraudMLP

OUTPUT_PATH = PROJECT_ROOT / "results" / "fraud" / "baseline_metrics.json"


def main() -> None:
    set_random_seed()
    client_loaders, test_loader, input_dim, pos_weight = load_creditcard_data(
        batch_size=BATCH_SIZE,
        num_clients=NUM_CLIENTS,
    )
    global_model = FraudMLP(input_dim)
    sample_counts = [len(loader.dataset) for loader in client_loaders]
    round_metrics = []

    print("Simulated FL Baseline")
    print_round_header()
    for round_number in range(1, NUM_ROUNDS + 1):
        local_states = []
        for loader in client_loaders:
            local_model = copy.deepcopy(global_model)
            train_local_model(local_model, loader, pos_weight, LOCAL_EPOCHS)
            local_states.append(local_model.state_dict())

        global_model.load_state_dict(fed_avg(local_states, sample_counts))
        metrics = {"round": round_number, **evaluate_model(global_model, test_loader)}
        round_metrics.append(metrics)
        print_round_metrics(metrics)

    save_metrics(OUTPUT_PATH, "Baseline", round_metrics)
    print(f"Saved metrics: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
