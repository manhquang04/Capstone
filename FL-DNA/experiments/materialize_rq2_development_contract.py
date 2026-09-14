"""Materialize per-seed split/partition/model-init checksums for RQ2 Stage 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.load_creditcard import DEFAULT_DATASET_PATH, load_creditcard_data
from experiments.fraud_fl_common import set_random_seed
from models.fraud_mlp import FraudMLP


def hash_tensors(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def hash_state(state: OrderedDict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode())
        digest.update(hash_tensors([tensor]).encode())
    return digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 505, 606, 707, 808])
    parser.add_argument("--max-rows", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/rq2/stage1_development_20260912/per_seed_contract.json",
    )
    args = parser.parse_args()

    records = []
    for seed in args.seeds:
        set_random_seed(seed)
        clients, validation, test, input_dim, _, metadata = load_creditcard_data(
            batch_size=args.batch_size,
            num_clients=args.num_clients,
            seed=seed,
            max_rows=args.max_rows,
        )
        model = FraudMLP(input_dim)
        client_hashes = [hash_tensors(list(loader.dataset.tensors)) for loader in clients]
        combined = hashlib.sha256()
        for label, value in [
            *(('client', item) for item in client_hashes),
            ("validation", hash_tensors(list(validation.dataset.tensors))),
            ("test", hash_tensors(list(test.dataset.tensors))),
        ]:
            combined.update(label.encode())
            combined.update(value.encode())
        records.append(
            {
                "seed": seed,
                "max_rows": args.max_rows,
                "client_dataset_sha256": client_hashes,
                "validation_dataset_sha256": hash_tensors(list(validation.dataset.tensors)),
                "test_dataset_sha256": hash_tensors(list(test.dataset.tensors)),
                "split_partition_sha256": combined.hexdigest(),
                "model_initial_state_sha256": hash_state(model.state_dict()),
                "loader_order_seed_by_client": [seed + index for index in range(args.num_clients)],
                "feature_order": metadata.feature_names,
                "numeric_center": metadata.numeric_center,
                "numeric_scale": metadata.numeric_scale,
                "type_categories": metadata.type_categories,
                "sample_counts": {
                    "clients": metadata.client_sample_counts,
                    "validation": metadata.validation_samples,
                    "test": metadata.test_samples,
                },
            }
        )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "RQ2 Stage-1 shared development batch",
        "seed_contract": "each seed controls sample/split, partition, model init, and loader order; all methods reuse it",
        "dataset": str(DEFAULT_DATASET_PATH),
        "dataset_sha256": sha256(DEFAULT_DATASET_PATH),
        "max_rows": args.max_rows,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "seeds": args.seeds, "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
