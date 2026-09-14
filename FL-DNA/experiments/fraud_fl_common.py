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
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

# FL_RUN_SEED is the protocol-level paired seed.  Every method launched with
# the same value must reuse the same split, client partition, initialization,
# and data-loader order.  Keep 42 as the backwards-compatible default.
RANDOM_SEED = int(os.environ.get("FL_RUN_SEED", "42"))
NUM_CLIENTS = int(os.environ.get("FL_NUM_CLIENTS", "3"))
QUICK_MODE = os.environ.get("QUICK") == "1"
NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", 3 if QUICK_MODE else 15))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", 1))
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
LOSS_TYPE = os.environ.get("LOSS_TYPE", "focal").lower()
FOCAL_GAMMA = float(os.environ.get("FOCAL_GAMMA", "2.0"))
FOCAL_ALPHA = float(os.environ.get("FOCAL_ALPHA", "0.95"))


def _get_device() -> torch.device:
    """Prefer Apple Metal acceleration when available, otherwise use CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _get_device()


class BinaryFocalLoss(nn.Module):
    """Focal loss for highly imbalanced binary classification with logits."""

    def __init__(self, alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        return loss.mean()


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Configure repeatable seeds without disabling Apple Silicon acceleration."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_local_model(
    model: nn.Module,
    loader: DataLoader,
    pos_weight: torch.Tensor,
    local_epochs: int = LOCAL_EPOCHS,
) -> float:
    """Train one client model from the current global state."""
    model.to(DEVICE)
    model.train()
    criterion = build_loss(pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    total_loss = 0.0
    total_samples = 0

    for _ in range(local_epochs):
        for features, labels in loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)


def build_loss(pos_weight: torch.Tensor) -> nn.Module:
    """Build the configured imbalance-aware loss."""
    if LOSS_TYPE == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    if LOSS_TYPE == "focal":
        return BinaryFocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    raise ValueError("LOSS_TYPE must be 'focal' or 'weighted_bce'")


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


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    threshold: float | None = None,
) -> dict[str, float | int | None]:
    """Evaluate a binary classifier, optionally tuning threshold for F1."""
    model.to(DEVICE)
    model.eval()
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            probabilities = torch.sigmoid(model(features))
            all_labels.append(labels.detach().cpu().numpy().reshape(-1))
            all_probabilities.append(probabilities.detach().cpu().numpy().reshape(-1))

    y_true = np.concatenate(all_labels).astype(np.int64)
    y_score = np.concatenate(all_probabilities)

    if threshold is None:
        best_thresh = tune_threshold(y_true, y_score)
    else:
        best_thresh = float(threshold)

    y_pred = (y_score >= best_thresh).astype(np.int64)

    auc_roc: float | None = None
    pr_auc: float | None = None
    if np.unique(y_true).size >= 2:
        try:
            auc_roc = float(roc_auc_score(y_true, y_score))
            pr_auc = float(average_precision_score(y_true, y_score))
        except ValueError:
            auc_roc = None
            pr_auc = None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "optimal_threshold": float(best_thresh),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Tune a decision threshold for fraud F1 on validation probabilities."""
    linear_grid = np.linspace(0.001, 0.999, 999)
    quantile_grid = np.quantile(y_score, np.linspace(0.001, 0.999, 999))
    thresholds = np.unique(np.concatenate([linear_grid, quantile_grid]))

    y_true = y_true.astype(np.int64, copy=False)
    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    sorted_true = y_true[order]
    positive_prefix = np.concatenate([[0], np.cumsum(sorted_true)])

    first_predicted_index = np.searchsorted(sorted_scores, thresholds, side="left")
    predicted_positive = y_true.size - first_predicted_index
    total_positive = int(sorted_true.sum())
    true_positive = total_positive - positive_prefix[first_predicted_index]
    false_positive = predicted_positive - true_positive
    false_negative = total_positive - true_positive

    denominator = (2 * true_positive) + false_positive + false_negative
    f1_scores = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(f1_scores))])


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
        "loss": LOSS_TYPE,
        "focal_alpha": FOCAL_ALPHA if LOSS_TYPE == "focal" else None,
        "focal_gamma": FOCAL_GAMMA if LOSS_TYPE == "focal" else None,
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
        f"{'Round':>5} | {'Loss':>8} | {'F1':>8} | {'ROC-AUC':>8} | "
        f"{'PR-AUC':>8} | {'Precision':>9} | {'Recall':>8} | {'TP':>5} | {'FP':>5} | {'FN':>5}"
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
    pr_auc = metrics["pr_auc"]
    pr_auc_text = "N/A" if pr_auc is None else f"{pr_auc:.6f}"
    row = (
        f"{metrics['round']:>5} | {metrics['train_loss']:>8.5f} | "
        f"{metrics['f1_score']:>8.6f} | {auc_text:>8} | {pr_auc_text:>8} | "
        f"{metrics['precision']:>9.6f} | {metrics['recall']:>8.6f}"
        f" | {metrics['tp']:>5} | {metrics['fp']:>5} | {metrics['fn']:>5}"
    )
    if "dna_encode_decode_ms" in metrics:
        row += (
            f" | {metrics['dna_encode_decode_ms']:>10.3f} | "
            f"{metrics['encoded_tensors']:>7} | {metrics['encoded_elements']:>10}"
        )
    print(row)
