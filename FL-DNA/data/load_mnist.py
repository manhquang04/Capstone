"""Load the existing local MNIST dataset without downloading files."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT / "datasets"
MNIST_RAW_DIR = DATASETS_ROOT / "MNIST" / "raw"


def load_mnist_data(batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    """Return train and test loaders from the immutable local MNIST files."""
    if not MNIST_RAW_DIR.is_dir():
        raise FileNotFoundError(
            "MNIST dataset not found. Place the existing MNIST files under "
            f"'{MNIST_RAW_DIR}'. This loader does not download datasets."
        )

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root=DATASETS_ROOT,
        train=True,
        transform=transform,
        download=False,
    )
    test_dataset = datasets.MNIST(
        root=DATASETS_ROOT,
        train=False,
        transform=transform,
        download=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
