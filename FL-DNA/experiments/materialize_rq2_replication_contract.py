"""Materialize a frozen RQ2 replication seed/split/partition/init contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.load_creditcard import DEFAULT_DATASET_PATH, load_creditcard_data
from experiments.fraud_fl_common import set_random_seed
from experiments.materialize_rq2_development_contract import hash_state, hash_tensors, sha256
from models.fraud_mlp import FraudMLP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replication-id", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-clients", type=int, default=3)
    args = parser.parse_args()
    if len(args.seeds) != 21 or len(set(args.seeds)) != 21:
        raise ValueError("replication contract requires exactly 21 unique seeds")

    records = []
    for seed in args.seeds:
        set_random_seed(seed)
        clients, validation, test, input_dim, _, metadata = load_creditcard_data(
            batch_size=args.batch_size, num_clients=args.num_clients,
            seed=seed, max_rows=args.max_rows,
        )
        model = FraudMLP(input_dim)
        client_hashes = [hash_tensors(list(loader.dataset.tensors)) for loader in clients]
        combined = hashlib.sha256()
        for label, value in [
            *(("client", item) for item in client_hashes),
            ("validation", hash_tensors(list(validation.dataset.tensors))),
            ("test", hash_tensors(list(test.dataset.tensors))),
        ]:
            combined.update(label.encode()); combined.update(value.encode())
        records.append({
            "seed": seed,
            "client_dataset_sha256": client_hashes,
            "validation_dataset_sha256": hash_tensors(list(validation.dataset.tensors)),
            "test_dataset_sha256": hash_tensors(list(test.dataset.tensors)),
            "split_partition_sha256": combined.hexdigest(),
            "model_initial_state_sha256": hash_state(model.state_dict()),
            "loader_order_seed_by_client": [seed + index for index in range(args.num_clients)],
            "sample_counts": {"clients": metadata.client_sample_counts,
                              "validation": metadata.validation_samples,
                              "test": metadata.test_samples},
        })
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": f"RQ2 Group 3 {args.replication_id} frozen paired contract",
        "dataset": str(DEFAULT_DATASET_PATH),
        "dataset_sha256": sha256(DEFAULT_DATASET_PATH),
        "max_rows": args.max_rows,
        "seeds": args.seeds,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
