"""Load, preprocess, and federated-partition the PaySim fraud dataset."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "datasets" / "creditcard.csv"
RANDOM_SEED = 42
TARGET_COLUMN = "isFraud"
DEFAULT_NUM_WORKERS = int(os.environ.get("DATALOADER_NUM_WORKERS", "4"))
BASE_FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_diff_orig",
    "balance_diff_dest",
]
CATEGORICAL_COLUMNS = ["type"]


@dataclass(frozen=True)
class PaySimMetadata:
    """Small metadata bundle for reporting experiment setup."""

    feature_names: list[str]
    train_samples: int
    validation_samples: int
    test_samples: int
    train_fraud_rate: float
    validation_fraud_rate: float
    test_fraud_rate: float
    client_sample_counts: list[int]
    client_fraud_rates: list[float]
    client_type_distributions: list[dict[str, float]]
    numeric_center: list[float]
    numeric_scale: list[float]
    type_categories: list[str]


def load_creditcard_data(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    batch_size: int = 512,
    num_clients: int = 3,
    validation_size: float = 0.15,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
    max_rows: int | None = None,
) -> tuple[list[DataLoader], DataLoader, DataLoader, int, torch.Tensor, PaySimMetadata]:
    """
    Load PaySim and return client, validation, and test loaders.

    The file is named creditcard.csv in this project, but the expected schema is
    PaySim: step/type/amount/balance columns with target isFraud. nameOrig,
    nameDest, and isFlaggedFraud are intentionally excluded.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"PaySim CSV not found at '{dataset_path}'")
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")

    dataframe = _read_dataset(dataset_path, seed, max_rows)
    missing = [col for col in BASE_FEATURE_COLUMNS + [TARGET_COLUMN] if col not in dataframe.columns]
    if missing:
        raise ValueError(f"PaySim dataset is missing required columns: {missing}")

    features = _build_features(dataframe)
    labels = dataframe[TARGET_COLUMN].astype(np.float32).to_numpy()

    train_features, temp_features, train_labels, temp_labels = train_test_split(
        features,
        labels,
        test_size=validation_size + test_size,
        random_state=seed,
        stratify=labels,
    )
    relative_test_size = test_size / (validation_size + test_size)
    validation_features, test_features, validation_labels, test_labels = train_test_split(
        temp_features,
        temp_labels,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_labels,
    )

    train_array, validation_array, test_array, feature_names, numeric_center, numeric_scale, type_categories = _fit_transform_features(
        train_features,
        validation_features,
        test_features,
    )

    client_indices = _mild_non_iid_client_indices(train_features, train_labels, num_clients, seed)
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
    validation_loader = _create_loader(
        validation_array,
        validation_labels,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
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

    metadata = PaySimMetadata(
        feature_names=feature_names,
        train_samples=int(train_labels.size),
        validation_samples=int(validation_labels.size),
        test_samples=int(test_labels.size),
        train_fraud_rate=float(np.mean(train_labels)),
        validation_fraud_rate=float(np.mean(validation_labels)),
        test_fraud_rate=float(np.mean(test_labels)),
        client_sample_counts=[len(indices) for indices in client_indices],
        client_fraud_rates=[float(np.mean(train_labels[indices])) for indices in client_indices],
        client_type_distributions=[
            _type_distribution(train_features.iloc[indices]["type"]) for indices in client_indices
        ],
        numeric_center=numeric_center,
        numeric_scale=numeric_scale,
        type_categories=type_categories,
    )
    return client_loaders, validation_loader, test_loader, train_array.shape[1], pos_weight, metadata


def load_paysim_splits(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    batch_size: int = 512,
    validation_size: float = 0.15,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
    max_rows: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, int, torch.Tensor, PaySimMetadata]:
    """Load PaySim and return pooled train, validation, and test loaders."""
    dataset_path = Path(dataset_path)
    dataframe = _read_dataset(dataset_path, seed, max_rows)
    missing = [col for col in BASE_FEATURE_COLUMNS + [TARGET_COLUMN] if col not in dataframe.columns]
    if missing:
        raise ValueError(f"PaySim dataset is missing required columns: {missing}")

    features = _build_features(dataframe)
    labels = dataframe[TARGET_COLUMN].astype(np.float32).to_numpy()

    train_features, temp_features, train_labels, temp_labels = train_test_split(
        features,
        labels,
        test_size=validation_size + test_size,
        random_state=seed,
        stratify=labels,
    )
    relative_test_size = test_size / (validation_size + test_size)
    validation_features, test_features, validation_labels, test_labels = train_test_split(
        temp_features,
        temp_labels,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_labels,
    )
    train_array, validation_array, test_array, feature_names, numeric_center, numeric_scale, type_categories = _fit_transform_features(
        train_features,
        validation_features,
        test_features,
    )

    train_loader = _create_loader(train_array, train_labels, batch_size, True, seed)
    validation_loader = _create_loader(validation_array, validation_labels, batch_size, False, seed)
    test_loader = _create_loader(test_array, test_labels, batch_size, False, seed)

    positive_count = float(np.count_nonzero(train_labels == 1.0))
    negative_count = float(np.count_nonzero(train_labels == 0.0))
    if positive_count == 0:
        raise ValueError("Training split contains no positive fraud samples")
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32)

    metadata = PaySimMetadata(
        feature_names=feature_names,
        train_samples=int(train_labels.size),
        validation_samples=int(validation_labels.size),
        test_samples=int(test_labels.size),
        train_fraud_rate=float(np.mean(train_labels)),
        validation_fraud_rate=float(np.mean(validation_labels)),
        test_fraud_rate=float(np.mean(test_labels)),
        client_sample_counts=[int(train_labels.size)],
        client_fraud_rates=[float(np.mean(train_labels))],
        client_type_distributions=[_type_distribution(train_features["type"])],
        numeric_center=numeric_center,
        numeric_scale=numeric_scale,
        type_categories=type_categories,
    )
    return train_loader, validation_loader, test_loader, train_array.shape[1], pos_weight, metadata


def _type_distribution(types: pd.Series) -> dict[str, float]:
    counts = types.value_counts(normalize=True).sort_index()
    return {str(name): float(value) for name, value in counts.items()}


def _read_dataset(dataset_path: Path, seed: int, max_rows: int | None) -> pd.DataFrame:
    use_columns = BASE_FEATURE_COLUMNS + [TARGET_COLUMN]
    env_max_rows = os.environ.get("MAX_ROWS")
    if max_rows is None and env_max_rows:
        max_rows = int(env_max_rows)

    if max_rows is None:
        return pd.read_csv(dataset_path, usecols=use_columns)

    # Keep quick prototype runs stratified instead of taking only early time steps.
    full = pd.read_csv(dataset_path, usecols=use_columns)
    if max_rows >= len(full):
        return full
    _, sampled = train_test_split(
        full,
        test_size=max_rows,
        random_state=seed,
        stratify=full[TARGET_COLUMN],
    )
    return sampled.reset_index(drop=True)


def _build_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    features = dataframe[BASE_FEATURE_COLUMNS].copy()
    features["balance_diff_orig"] = (
        features["oldbalanceOrg"] - features["newbalanceOrig"]
    )
    features["balance_diff_dest"] = (
        features["newbalanceDest"] - features["oldbalanceDest"]
    )

    for col in NUMERIC_COLUMNS:
        features[col] = pd.to_numeric(features[col], errors="coerce")
        features[col] = features[col].fillna(features[col].median())
    features["type"] = features["type"].fillna("UNKNOWN").astype(str)
    return features


def _fit_transform_features(
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[float], list[float], list[str]]:
    scaler = RobustScaler()
    train_num = scaler.fit_transform(train_features[NUMERIC_COLUMNS])
    validation_num = scaler.transform(validation_features[NUMERIC_COLUMNS])
    test_num = scaler.transform(test_features[NUMERIC_COLUMNS])

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    train_cat = encoder.fit_transform(train_features[CATEGORICAL_COLUMNS])
    validation_cat = encoder.transform(validation_features[CATEGORICAL_COLUMNS])
    test_cat = encoder.transform(test_features[CATEGORICAL_COLUMNS])

    feature_names = NUMERIC_COLUMNS + [
        f"type={category}" for category in encoder.categories_[0].tolist()
    ]
    train_array = np.hstack([train_num, train_cat]).astype(np.float32)
    validation_array = np.hstack([validation_num, validation_cat]).astype(np.float32)
    test_array = np.hstack([test_num, test_cat]).astype(np.float32)
    return (
        train_array,
        validation_array,
        test_array,
        feature_names,
        [float(value) for value in scaler.center_],
        [float(value) for value in scaler.scale_],
        [str(value) for value in encoder.categories_[0].tolist()],
    )


def _mild_non_iid_client_indices(
    features: pd.DataFrame,
    labels: np.ndarray,
    num_clients: int,
    seed: int,
) -> list[np.ndarray]:
    """
    Create mild non-IID clients by assigning transaction types unevenly while
    still mixing every type across clients. Fraud labels remain present via
    stratified fallback for tiny chunks.
    """
    rng = np.random.default_rng(seed)
    partitions: list[list[np.ndarray]] = [[] for _ in range(num_clients)]

    positive_indices = np.flatnonzero(labels == 1.0)
    rng.shuffle(positive_indices)
    for client_id, chunk in enumerate(np.array_split(positive_indices, num_clients)):
        partitions[client_id].append(chunk)

    feature_types = features["type"].to_numpy()
    for type_index, transaction_type in enumerate(sorted(features["type"].unique())):
        type_indices = np.flatnonzero((feature_types == transaction_type) & (labels == 0.0))
        rng.shuffle(type_indices)
        primary_client = type_index % num_clients
        weights = np.full(num_clients, 0.45 / max(num_clients - 1, 1))
        weights[primary_client] = 0.55
        if num_clients == 1:
            weights[0] = 1.0

        split_points = np.cumsum(weights)[:-1]
        for client_id, chunk in enumerate(np.array_split(type_indices, (split_points * len(type_indices)).astype(int))):
            if len(chunk):
                partitions[client_id].append(chunk)

    all_clients = []
    seen: set[int] = set()
    for client_id, chunks in enumerate(partitions):
        if chunks:
            indices = np.concatenate(chunks)
        else:
            indices = np.array([], dtype=np.int64)
        indices = np.array([idx for idx in indices if idx not in seen], dtype=np.int64)
        seen.update(indices.tolist())
        rng.shuffle(indices)
        all_clients.append(indices)
    return all_clients


def _create_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.array(features, dtype=np.float32, copy=True, order="C")),
        torch.from_numpy(np.array(labels, dtype=np.float32, copy=True, order="C")).reshape(-1, 1),
    )
    generator = torch.Generator().manual_seed(seed)
    num_workers = max(DEFAULT_NUM_WORKERS, 0)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "generator": generator,
        "num_workers": num_workers,
        "pin_memory": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)
