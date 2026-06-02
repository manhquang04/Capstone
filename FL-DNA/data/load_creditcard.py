"""Load and partition the Credit Card Fraud Detection dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "datasets" / "creditcard.csv"
RANDOM_SEED = 42


def load_creditcard_data(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    batch_size: int = 256,
    num_clients: int = 3,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> tuple[list[DataLoader], DataLoader, int, torch.Tensor]:
    """Load immutable CSV data and return stratified client loaders."""
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            "Credit Card Fraud dataset not found. Place the original CSV at "
            f"'{dataset_path}'."
        )
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")

    dataframe = pd.read_csv(dataset_path)
    if "Class" not in dataframe.columns:
        raise ValueError("Dataset must contain the target column 'Class'")

    features = dataframe.drop(columns=["Class"]).copy()
    labels = dataframe["Class"].to_numpy(dtype=np.float32)
    if not {"Amount", "Time"}.issubset(features.columns):
        raise ValueError("Dataset features must include 'Amount' and 'Time'")

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    scaler = StandardScaler()
    train_features.loc[:, ["Amount", "Time"]] = scaler.fit_transform(
        train_features[["Amount", "Time"]]
    )
    test_features.loc[:, ["Amount", "Time"]] = scaler.transform(
        test_features[["Amount", "Time"]]
    )

    train_array = train_features.to_numpy(dtype=np.float32)
    test_array = test_features.to_numpy(dtype=np.float32)
    client_indices = _stratified_client_indices(train_labels, num_clients, seed)
    client_loaders = [
        _create_loader(
            train_array[indices],
            train_labels[indices],
            batch_size=batch_size,
            shuffle=True,
            seed=seed + client_id,
        )
        for client_id, indices in enumerate(client_indices)
    ]
    test_loader = _create_loader(
        test_array,
        test_labels,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    positive_count = float(np.count_nonzero(train_labels == 1.0))
    negative_count = float(np.count_nonzero(train_labels == 0.0))
    if positive_count == 0:
        raise ValueError("Training split contains no positive fraud samples")
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32)
    return client_loaders, test_loader, train_array.shape[1], pos_weight


def _stratified_client_indices(
    labels: np.ndarray,
    num_clients: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    partitions: list[list[np.ndarray]] = [[] for _ in range(num_clients)]

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for client_id, chunk in enumerate(np.array_split(indices, num_clients)):
            partitions[client_id].append(chunk)

    client_indices = []
    for chunks in partitions:
        indices = np.concatenate(chunks)
        rng.shuffle(indices)
        client_indices.append(indices)
    return client_indices


def _create_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    tensor_features = np.array(features, dtype=np.float32, copy=True, order="C")
    tensor_labels = np.array(labels, dtype=np.float32, copy=True, order="C")
    dataset = TensorDataset(
        torch.from_numpy(tensor_features),
        torch.from_numpy(tensor_labels).reshape(-1, 1),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
