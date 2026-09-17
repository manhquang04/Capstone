"""Priority 5.2 IEEE-CIS development-scale FL baseline smoke.

This script is independent from the DNA Transform v2 attacker track.  It only
checks whether a small FL baseline can learn on IEEE-CIS; it does not create
attack targets.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.fraud_mlp import FraudMLP


NUMERIC_COLUMNS = [
    "TransactionAmt",
    "TransactionDT",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
]
CATEGORICAL_COLUMNS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "M1", "M2", "M3", "M4", "M5", "M6"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_ieee_subset(path: Path, max_rows: int, seed: int) -> pd.DataFrame:
    usecols = ["TransactionID", "isFraud", *NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS]
    frame = pd.read_csv(path, usecols=lambda c: c in set(usecols))
    if max_rows and max_rows < len(frame):
        _, frame = train_test_split(
            frame,
            test_size=max_rows,
            random_state=seed,
            stratify=frame["isFraud"],
        )
    return frame.reset_index(drop=True)


def build_arrays(frame: pd.DataFrame, seed: int):
    labels = frame["isFraud"].astype(np.float32).to_numpy()
    features = frame.drop(columns=["isFraud", "TransactionID"])
    train_x, temp_x, train_y, temp_y = train_test_split(
        features,
        labels,
        test_size=0.35,
        random_state=seed,
        stratify=labels,
    )
    val_x, test_x, val_y, test_y = train_test_split(
        temp_x,
        temp_y,
        test_size=0.2 / 0.35,
        random_state=seed,
        stratify=temp_y,
    )
    numeric = [c for c in NUMERIC_COLUMNS if c in train_x.columns]
    categorical = [c for c in CATEGORICAL_COLUMNS if c in train_x.columns]
    scaler = RobustScaler()
    train_num = scaler.fit_transform(train_x[numeric].fillna(train_x[numeric].median()))
    val_num = scaler.transform(val_x[numeric].fillna(train_x[numeric].median()))
    test_num = scaler.transform(test_x[numeric].fillna(train_x[numeric].median()))
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    train_cat = encoder.fit_transform(train_x[categorical].fillna("__MISSING__").astype(str))
    val_cat = encoder.transform(val_x[categorical].fillna("__MISSING__").astype(str))
    test_cat = encoder.transform(test_x[categorical].fillna("__MISSING__").astype(str))
    train_arr = np.concatenate([train_num, train_cat], axis=1).astype(np.float32)
    val_arr = np.concatenate([val_num, val_cat], axis=1).astype(np.float32)
    test_arr = np.concatenate([test_num, test_cat], axis=1).astype(np.float32)
    metadata = {
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "input_dim": int(train_arr.shape[1]),
        "train_samples": int(train_y.size),
        "validation_samples": int(val_y.size),
        "test_samples": int(test_y.size),
        "train_fraud_rate": float(train_y.mean()),
        "validation_fraud_rate": float(val_y.mean()),
        "test_fraud_rate": float(test_y.mean()),
    }
    return train_arr, train_y, val_arr, val_y, test_arr, test_y, metadata


def make_loaders(train_x, train_y, val_x, val_y, test_x, test_y, num_clients: int, seed: int, batch_size: int):
    fraud = np.where(train_y == 1)[0]
    non = np.where(train_y == 0)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(fraud)
    rng.shuffle(non)
    client_indices = [[] for _ in range(num_clients)]
    for i, idx in enumerate(fraud):
        client_indices[i % num_clients].append(int(idx))
    for i, idx in enumerate(non):
        client_indices[i % num_clients].append(int(idx))
    loaders = []
    for client_id, indices in enumerate(client_indices):
        rng.shuffle(indices)
        gen = torch.Generator().manual_seed(seed + client_id)
        loaders.append(
            DataLoader(
                TensorDataset(torch.from_numpy(train_x[indices]), torch.from_numpy(train_y[indices]).reshape(-1, 1)),
                batch_size=batch_size,
                shuffle=True,
                generator=gen,
            )
        )
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y).reshape(-1, 1)), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y).reshape(-1, 1)), batch_size=batch_size)
    return loaders, val_loader, test_loader, [len(x) for x in client_indices], [float(train_y[x].mean()) for x in client_indices]


def train_local(model, loader, pos_weight, epochs: int) -> float:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total = 0.0
    count = 0
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss.item()) * y.size(0)
            count += y.size(0)
    return total / max(count, 1)


def fed_avg(states: list[OrderedDict[str, torch.Tensor]], counts: list[int]) -> OrderedDict[str, torch.Tensor]:
    total = sum(counts)
    out = OrderedDict()
    for key in states[0]:
        if torch.is_floating_point(states[0][key]):
            value = torch.zeros_like(states[0][key])
            for state, count in zip(states, counts):
                value += state[key] * (count / total)
            out[key] = value
        else:
            out[key] = states[0][key].clone()
    return out


def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            prob = torch.sigmoid(model(x)).reshape(-1).numpy()
            ps.append(prob)
            ys.append(y.reshape(-1).numpy())
    y_true = np.concatenate(ys).astype(int)
    y_score = np.concatenate(ps)
    thresholds = np.unique(np.quantile(y_score, np.linspace(0.001, 0.999, 200)))
    f1s = [f1_score(y_true, y_score >= t, zero_division=0) for t in thresholds]
    best_t = float(thresholds[int(np.argmax(f1s))])
    y_pred = y_score >= best_t
    return {
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "threshold": best_t,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/priority5_ieee_cis/phase52_fl_baseline_smoke.json")
    args = parser.parse_args()
    set_seed(args.seed)
    train_path = ROOT / "datasets/ieee-fraud-detection/train_transaction.csv"
    frame = read_ieee_subset(train_path, args.max_rows, args.seed)
    train_x, train_y, val_x, val_y, test_x, test_y, metadata = build_arrays(frame, args.seed)
    loaders, val_loader, test_loader, counts, fraud_rates = make_loaders(
        train_x, train_y, val_x, val_y, test_x, test_y, args.clients, args.seed, args.batch_size
    )
    metadata["client_sample_counts"] = counts
    metadata["client_fraud_rates"] = fraud_rates
    positive = float(np.count_nonzero(train_y == 1))
    negative = float(np.count_nonzero(train_y == 0))
    pos_weight = torch.tensor([negative / max(positive, 1.0)], dtype=torch.float32)
    model = FraudMLP(metadata["input_dim"])
    rounds = []
    for round_id in range(1, args.rounds + 1):
        states, losses = [], []
        for loader in loaders:
            local = FraudMLP(metadata["input_dim"])
            local.load_state_dict(model.state_dict())
            losses.append(train_local(local, loader, pos_weight, 1))
            states.append(local.state_dict())
        model.load_state_dict(fed_avg(states, counts))
        val = evaluate(model, val_loader)
        test = evaluate(model, test_loader)
        row = {"round": round_id, "train_loss": float(np.mean(losses)), "validation": val, "test": test}
        rounds.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Priority 5.2 IEEE-CIS development-scale FL baseline smoke; no attack targets",
        "config": vars(args) | {"output": str(args.output)},
        "metadata": metadata,
        "rounds": rounds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
