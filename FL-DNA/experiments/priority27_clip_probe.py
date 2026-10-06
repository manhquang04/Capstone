"""Measure Priority 27 DP clip norms on two development training replicates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.load_creditcard import load_creditcard_data  # noqa: E402
from experiments import fraud_fl_common as common  # noqa: E402
from experiments.priority27_dp_variants import floating_names, tensor_update_norms, trainable_names_from_state, update_norm  # noqa: E402
from models.fraud_mlp import FraudMLP  # noqa: E402


AMENDMENT = ROOT / "protocols/amendments/2026-09-29_priority27_gaps_and_harness.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(seed: int, max_rows: int, num_rounds: int, num_clients: int) -> dict:
    os.environ["FL_RUN_SEED"] = str(seed)
    common.RANDOM_SEED = seed
    common.NUM_ROUNDS = num_rounds
    common.NUM_CLIENTS = num_clients
    common.DEVICE = torch.device("cpu")
    common.set_random_seed(seed)
    loaders, _validation_loader, _test_loader, input_dim, pos_weight, _metadata = load_creditcard_data(
        batch_size=common.BATCH_SIZE,
        num_clients=num_clients,
        seed=seed,
        max_rows=max_rows,
    )
    global_model = FraudMLP(input_dim).to(common.DEVICE)
    full_norms = []
    trainable_norms = []
    per_tensor: dict[str, list[float]] = {}
    for _round in range(1, num_rounds + 1):
        global_state = global_model.state_dict()
        local_states = []
        for loader in loaders:
            local_model = copy.deepcopy(global_model)
            common.train_local_model(local_model, loader, pos_weight, common.LOCAL_EPOCHS)
            local_state = local_model.state_dict()
            full_names = floating_names(local_state)
            train_names = trainable_names_from_state(local_state, global_state)
            full_norms.append(update_norm(local_state, global_state, full_names))
            trainable_norms.append(update_norm(local_state, global_state, train_names))
            for name, norm in tensor_update_norms(local_state, global_state, full_names).items():
                per_tensor.setdefault(name, []).append(norm)
            local_states.append(local_state)
        global_model.load_state_dict(common.fed_avg(local_states, [len(loader.dataset) for loader in loaders]))
    return {"full_state_norms": full_norms, "trainable_norms": trainable_norms, "per_tensor_norms": per_tensor}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=500000)
    parser.add_argument("--num-rounds", type=int, default=50)
    parser.add_argument("--num-clients", type=int, default=3)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    combined_full = []
    combined_train = []
    combined_tensor: dict[str, list[float]] = {}
    per_seed = {}
    for seed in args.seeds:
        result = collect(seed, args.max_rows, args.num_rounds, args.num_clients)
        per_seed[str(seed)] = result
        combined_full.extend(result["full_state_norms"])
        combined_train.extend(result["trainable_norms"])
        for name, values in result["per_tensor_norms"].items():
            combined_tensor.setdefault(name, []).extend(values)
    per_tensor_clip = {name: float(np.percentile(values, 95)) for name, values in combined_tensor.items()}
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": str(AMENDMENT.relative_to(ROOT)),
        "amendment_sha256": sha256(AMENDMENT),
        "seeds": args.seeds,
        "max_rows": args.max_rows,
        "num_rounds": args.num_rounds,
        "num_clients": args.num_clients,
        "full_state_clip_p95": float(np.percentile(combined_full, 95)),
        "trainable_only_clip_p95": float(np.percentile(combined_train, 95)),
        "per_tensor_clip_p95": per_tensor_clip,
        "per_seed_counts": {seed: {"full": len(v["full_state_norms"]), "trainable": len(v["trainable_norms"])} for seed, v in per_seed.items()},
    }
    (args.output_dir / "clip_probe_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

