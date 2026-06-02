"""Shared utilities for simulated Credit Card Fraud federated learning."""

from __future__ import annotations

import json
import os
import random
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

RANDOM_SEED = 42
NUM_CLIENTS = 3
QUICK_MODE = os.environ.get("QUICK") == "1"
NUM_ROUNDS = 3 if QUICK_MODE else 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 256
LEARNING_RATE = 1e-3


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Configure deterministic random seeds for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_local_model(
    model: nn.Module,
    loader: DataLoader,
    pos_weight: torch.Tensor,
    local_epochs: int = LOCAL_EPOCHS,
) -> None:
    """Train one client model from the current global state."""
    model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for _ in range(local_epochs):
        for features, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()


def fed_avg(
    state_dicts: Sequence[OrderedDict[str, torch.Tensor]],
    sample_counts: Sequence[int],
) -> OrderedDict[str, torch.Tensor]:
    """Compute sample-count weighted FedAvg across local model states."""
    if not state_dicts or len(state_dicts) != len(sample_counts):
        raise ValueError("state_dicts and sample_counts must have equal non-zero length")

    total_samples = sum(sample_counts)
    averaged_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name in state_dicts[0]:
        reference = state_dicts[0][name]
        if torch.is_floating_point(reference):
            averaged = torch.zeros_like(reference)
            for state_dict, sample_count in zip(state_dicts, sample_counts):
                averaged += state_dict[name] * (sample_count / total_samples)
            averaged_state[name] = averaged
        else:
            averaged_state[name] = reference.clone()
    return averaged_state


def evaluate_model(model: nn.Module, loader: DataLoader) -> dict[str, float | None]:
    """Evaluate a binary classifier without crashing on undefined AUC."""
    model.eval()
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in loader:
            probabilities = torch.sigmoid(model(features))
            all_labels.append(labels.numpy().reshape(-1))
            all_probabilities.append(probabilities.numpy().reshape(-1))

    y_true = np.concatenate(all_labels).astype(np.int64)
    y_score = np.concatenate(all_probabilities)
    y_pred = (y_score >= 0.5).astype(np.int64)

    auc_roc: float | None = None
    if np.unique(y_true).size >= 2:
        try:
            auc_roc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            auc_roc = None

    return {
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": auc_roc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def save_metrics(
    output_path: Path,
    method: str,
    round_metrics: list[dict[str, float | int | None]],
    extra_config: dict[str, object] | None = None,
) -> None:
    """Save experiment configuration and metrics as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config: dict[str, object] = {
        "seed": RANDOM_SEED,
        "num_clients": NUM_CLIENTS,
        "num_rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "BCEWithLogitsLoss(pos_weight)",
    }
    if extra_config:
        config.update(extra_config)

    output_path.write_text(
        json.dumps(
            {"method": method, "config": config, "rounds": round_metrics},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def print_round_header(extra_columns: bool = False) -> None:
    """Print a compact metrics table header."""
    header = (
        f"{'Round':>5} | {'F1':>8} | {'AUC':>8} | {'Accuracy':>8} | "
        f"{'Precision':>9} | {'Recall':>8}"
    )
    if extra_columns:
        header += (
            f" | {'DNA ms':>10} | {'Tensors':>7} | {'Elements':>10}"
        )
    print(header)
    print("-" * len(header))


def print_round_metrics(metrics: dict[str, float | int | None]) -> None:
    """Print one metrics table row."""
    auc = metrics["auc_roc"]
    auc_text = "N/A" if auc is None else f"{auc:.6f}"
    row = (
        f"{metrics['round']:>5} | {metrics['f1_score']:>8.6f} | "
        f"{auc_text:>8} | {metrics['accuracy']:>8.6f} | "
        f"{metrics['precision']:>9.6f} | {metrics['recall']:>8.6f}"
    )
    if "dna_encode_decode_ms" in metrics:
        row += (
            f" | {metrics['dna_encode_decode_ms']:>10.3f} | "
            f"{metrics['encoded_tensors']:>7} | {metrics['encoded_elements']:>10}"
        )
    print(row)
