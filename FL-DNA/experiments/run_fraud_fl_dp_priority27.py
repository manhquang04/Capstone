"""Run simulated FL with Priority 27 DP variants."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_creditcard import load_creditcard_data
from experiments.fraud_fl_common import (
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
from experiments.priority27_dp_variants import apply_dp_variant_to_local_state
from models.fraud_mlp import FraudMLP
from privacy.seed_manager import derive_seed


OUTPUT_PATH = Path(os.environ.get("PRIORITY27_DP_OUTPUT_PATH", str(PROJECT_ROOT / "results/priority27/dp_variant_metrics.json"))).resolve()
VARIANT = os.environ["PRIORITY27_DP_VARIANT"]
NOISE_MULTIPLIER = float(os.environ["PRIORITY27_DP_NOISE_MULTIPLIER"])
CLIP_SPEC = json.loads(os.environ["PRIORITY27_DP_CLIP_SPEC_JSON"])


def main() -> None:
    set_random_seed()
    client_loaders, validation_loader, test_loader, input_dim, pos_weight, metadata = load_creditcard_data(
        batch_size=1024,
        num_clients=NUM_CLIENTS,
        seed=RANDOM_SEED,
        max_rows=int(os.environ.get("MAX_ROWS", "500000")),
    )
    global_model = FraudMLP(input_dim).to(DEVICE)
    sample_counts = [len(loader.dataset) for loader in client_loaders]
    round_metrics = []
    print(f"Priority 27 DP variant: {VARIANT}, sigma={NOISE_MULTIPLIER}")
    print_round_header()
    for round_number in range(1, NUM_ROUNDS + 1):
        global_state = global_model.state_dict()
        local_states = []
        local_losses = []
        diagnostics = []
        for client_index, loader in enumerate(client_loaders):
            local_model = copy.deepcopy(global_model)
            local_losses.append(train_local_model(local_model, loader, pos_weight, LOCAL_EPOCHS))
            seed = derive_seed(RANDOM_SEED, "priority27-dp-variant", VARIANT, round_number, client_index)
            dp_state, diag = apply_dp_variant_to_local_state(
                local_model.state_dict(),
                global_state,
                variant=VARIANT,
                clip_spec=CLIP_SPEC,
                noise_multiplier=NOISE_MULTIPLIER,
                noise_generator=torch.Generator().manual_seed(seed),
            )
            local_states.append(dp_state)
            diagnostics.append(diag)
        global_model.load_state_dict(fed_avg(local_states, sample_counts))
        validation_metrics = evaluate_model(global_model, validation_loader)
        metrics = {
            "round": round_number,
            "train_loss": sum(local_losses) / len(local_losses),
            **evaluate_model(global_model, test_loader, threshold=float(validation_metrics["optimal_threshold"])),
            "priority27_dp_variant": VARIANT,
            "dp_noise_multiplier": NOISE_MULTIPLIER,
            "dp_clip_spec": CLIP_SPEC,
            "mean_noise_std": float(sum(d.get("noise_std", 0.0) for d in diagnostics) / len(diagnostics)),
            "mean_effective_sensitivity": float(sum(d.get("effective_sensitivity", 0.0) for d in diagnostics) / len(diagnostics)),
        }
        round_metrics.append(metrics)
        print_round_metrics(metrics)
    save_metrics(
        OUTPUT_PATH,
        f"Priority27DP:{VARIANT}",
        round_metrics,
        extra_config={
            "dataset": "PaySim",
            "target": "isFraud",
            "features": metadata.feature_names,
            "client_sample_counts": metadata.client_sample_counts,
            "priority27_dp_variant": VARIANT,
            "dp_noise_multiplier": NOISE_MULTIPLIER,
            "dp_clip_spec": CLIP_SPEC,
        },
    )
    print(f"Saved metrics: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

