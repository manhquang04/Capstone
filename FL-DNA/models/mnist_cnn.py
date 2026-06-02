"""Compact CNN for MNIST classification and gradient inversion experiments."""

from __future__ import annotations

from torch import nn


class MNISTCNN(nn.Module):
    """Two-block CNN that returns raw class logits."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, inputs):
        features = self.features(inputs)
        return self.classifier(features.reshape(features.size(0), -1))
