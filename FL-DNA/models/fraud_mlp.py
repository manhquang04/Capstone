"""Simple binary classifier for Credit Card Fraud Detection."""

from __future__ import annotations

from torch import nn


class FraudMLP(nn.Module):
    """MLP that returns raw logits for BCEWithLogitsLoss."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, inputs):
        return self.network(inputs)
