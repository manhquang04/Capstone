"""Train and save the compact MNIST CNN used by privacy experiments."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_mnist import load_mnist_data
from models.mnist_cnn import MNISTCNN

SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 2
TARGET_ACCURACY = 0.95
RESULTS_DIR = PROJECT_ROOT / "results" / "mnist"
MODEL_PATH = RESULTS_DIR / "base_model.pt"
METRICS_PATH = RESULTS_DIR / "train_metrics.json"


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def evaluate(model: nn.Module, test_loader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += labels.numel()
    return correct / total


def main() -> None:
    set_seed()
    train_loader, test_loader = load_mnist_data(batch_size=BATCH_SIZE)
    model = MNISTCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = []

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)

        accuracy = evaluate(model, test_loader)
        average_loss = running_loss / len(train_loader.dataset)
        epochs.append(
            {"epoch": epoch, "train_loss": average_loss, "test_accuracy": accuracy}
        )
        print(
            f"Epoch {epoch}/{MAX_EPOCHS} | Train loss: {average_loss:.6f} | "
            f"Test accuracy: {accuracy:.4%}"
        )
        if accuracy >= TARGET_ACCURACY:
            break

    torch.save(model.state_dict(), MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "model": "MNISTCNN",
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "target_accuracy": TARGET_ACCURACY,
                "epochs": epochs,
                "final_test_accuracy": epochs[-1]["test_accuracy"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved metrics: {METRICS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
