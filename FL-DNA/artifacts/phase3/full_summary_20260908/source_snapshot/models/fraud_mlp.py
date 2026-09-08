"""Improved binary classifier for Credit Card Fraud Detection."""

from __future__ import annotations

from torch import nn


class FraudMLP(nn.Module):
    """
    Improved MLP with deeper architecture and batch normalization.
    Returns raw logits for BCEWithLogitsLoss.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 1),
        )

    def forward(self, inputs):
        return self.network(inputs)
