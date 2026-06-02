"""Run simulated FL with local-update clipping and Gaussian noise."""

from __future__ import annotations

import copy
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_creditcard import load_creditcard_data
from experiments.fraud_fl_common import (
    BATCH_SIZE,
    evaluate_model,
    fed_avg,
    save_metrics,
    set_random_seed,
    train_local_model,
)
from models.fraud_mlp import FraudMLP

NUM_CLIENTS = 3
QUICK_MODE = os.environ.get("QUICK") == "1"
NUM_ROUNDS = 3 if QUICK_MODE else 5
LOCAL_EPOCHS = 1
CLIP_NORM = 1.0
NOISE_MULTIPLIER = 0.05
SEED = 42
OUTPUT_PATH = PROJECT_ROOT / "results" / "fraud" / "dp_metrics.json"


def apply_dp_to_local_state(
    local_state: OrderedDict[str, torch.Tensor],
    global_state: OrderedDict[str, torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
) -> tuple[OrderedDict[str, torch.Tensor], float, float, float]:
    """Clip one client's complete floating update and add Gaussian noise."""
    floating_updates = [
        local_state[name] - global_state[name]
        for name in local_state
        if torch.is_floating_point(local_state[name])
    ]
    if not floating_updates:
        raise ValueError("local_state contains no floating-point tensors")

    update_norm = torch.sqrt(
        sum(torch.sum(update.square()) for update in floating_updates)
    )
    update_norm_value = float(update_norm.item())
    clip_factor = min(1.0, clip_norm / max(update_norm_value, 1e-12))
    clipped_norm_value = update_norm_value * clip_factor
    noise_std = noise_multiplier * clip_norm

    dp_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, local_tensor in local_state.items():
        global_tensor = global_state[name]
        if torch.is_floating_point(local_tensor):
            clipped_update = (local_tensor - global_tensor) * clip_factor
            gaussian_noise = torch.randn_like(clipped_update) * noise_std
            dp_state[name] = global_tensor + clipped_update + gaussian_noise
        else:
            dp_state[name] = local_tensor.clone()

    return dp_state, update_norm_value, clipped_norm_value, noise_std


def main() -> None:
    set_random_seed(SEED)
    client_loaders, test_loader, input_dim, pos_weight = load_creditcard_data(
        batch_size=BATCH_SIZE,
        num_clients=NUM_CLIENTS,
        seed=SEED,
    )
    global_model = FraudMLP(input_dim)
    sample_counts = [len(loader.dataset) for loader in client_loaders]
    round_metrics = []

    print("Simulated FL + Differential Privacy")
    for round_number in range(1, NUM_ROUNDS + 1):
        global_state = global_model.state_dict()
        dp_local_states = []
        norms_before_clip = []
        norms_after_clip = []
        noise_std = NOISE_MULTIPLIER * CLIP_NORM

        for loader in client_loaders:
            local_model = copy.deepcopy(global_model)
            train_local_model(local_model, loader, pos_weight, LOCAL_EPOCHS)
            dp_state, norm_before, norm_after, noise_std = apply_dp_to_local_state(
                local_model.state_dict(),
                global_state,
                CLIP_NORM,
                NOISE_MULTIPLIER,
            )
            dp_local_states.append(dp_state)
            norms_before_clip.append(norm_before)
            norms_after_clip.append(norm_after)

        global_model.load_state_dict(fed_avg(dp_local_states, sample_counts))
        evaluated = evaluate_model(global_model, test_loader)
        metrics = {
            "round": round_number,
            "f1": evaluated["f1_score"],
            "auc": evaluated["auc_roc"],
            "accuracy": evaluated["accuracy"],
            "precision": evaluated["precision"],
            "recall": evaluated["recall"],
            "dp_noise_std": noise_std,
            "avg_update_norm_before_clip": sum(norms_before_clip)
            / len(norms_before_clip),
            "avg_update_norm_after_clip": sum(norms_after_clip)
            / len(norms_after_clip),
        }
        round_metrics.append(metrics)

        auc_text = "N/A" if metrics["auc"] is None else f"{metrics['auc']:.6f}"
        print(
            f"Round {round_number}/{NUM_ROUNDS} | F1: {metrics['f1']:.6f} | "
            f"AUC: {auc_text} | Acc: {metrics['accuracy']:.6f} | "
            f"Precision: {metrics['precision']:.6f} | Recall: {metrics['recall']:.6f}"
        )
        print(
            "DP info | "
            f"Avg norm before clip: {metrics['avg_update_norm_before_clip']:.6f} | "
            f"Avg norm after clip: {metrics['avg_update_norm_after_clip']:.6f} | "
            f"Noise std: {metrics['dp_noise_std']:.6f}"
        )

    save_metrics(
        OUTPUT_PATH,
        "DP",
        round_metrics,
        extra_config={
            "seed": SEED,
            "num_clients": NUM_CLIENTS,
            "num_rounds": NUM_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "clip_norm": CLIP_NORM,
            "noise_multiplier": NOISE_MULTIPLIER,
            "dp_mechanism": "Local state update L2 clipping + Gaussian noise",
        },
    )
    print(f"Saved metrics: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
