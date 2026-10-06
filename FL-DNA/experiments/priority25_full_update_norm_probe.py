"""Priority 25 full transmitted-update norm probe.

This script is calibration-only.  It measures per-client, per-round norms of
the exact floating state_dict update that RQ2 DP clips/noises.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_creditcard import load_creditcard_data
from experiments.fraud_fl_common import (
    BATCH_SIZE,
    DEVICE,
    LOCAL_EPOCHS,
    NUM_CLIENTS,
    NUM_ROUNDS,
    evaluate_model,
    fed_avg,
    set_random_seed,
    train_local_model,
)
from models.fraud_mlp import FraudMLP


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def floating_update_norm(local_state: dict[str, torch.Tensor], global_state: dict[str, torch.Tensor]) -> float:
    total = torch.tensor(0.0, device=DEVICE)
    for name, tensor in local_state.items():
        if torch.is_floating_point(tensor):
            diff = tensor - global_state[name]
            total = total + torch.sum(diff.square())
    return float(torch.sqrt(total).detach().cpu().item())


def run_seed(seed: int) -> list[dict[str, float | int]]:
    os.environ["FL_RUN_SEED"] = str(seed)
    set_random_seed(seed)
    client_loaders, validation_loader, test_loader, input_dim, pos_weight, _metadata = load_creditcard_data(
        batch_size=BATCH_SIZE,
        num_clients=NUM_CLIENTS,
    )
    global_model = FraudMLP(input_dim).to(DEVICE)
    sample_counts = [len(loader.dataset) for loader in client_loaders]
    rows: list[dict[str, float | int]] = []
    for round_number in range(1, NUM_ROUNDS + 1):
        global_state = global_model.state_dict()
        local_states = []
        local_losses = []
        for client_index, loader in enumerate(client_loaders):
            local_model = copy.deepcopy(global_model)
            local_losses.append(train_local_model(local_model, loader, pos_weight, LOCAL_EPOCHS))
            local_state = local_model.state_dict()
            rows.append(
                {
                    "seed": int(seed),
                    "round": int(round_number),
                    "client": int(client_index),
                    "full_floating_update_norm": floating_update_norm(local_state, global_state),
                }
            )
            local_states.append(local_state)
        global_model.load_state_dict(fed_avg(local_states, sample_counts))
        validation_metrics = evaluate_model(global_model, validation_loader)
        _ = evaluate_model(global_model, test_loader, threshold=float(validation_metrics["optimal_threshold"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    all_rows: list[dict[str, float | int]] = []
    for seed in args.seeds:
        all_rows.extend(run_seed(seed))

    csv_path = output / "full_update_norms.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "round", "client", "full_floating_update_norm"])
        writer.writeheader()
        writer.writerows(all_rows)

    norms = np.asarray([float(row["full_floating_update_norm"]) for row in all_rows], dtype=float)
    percentile_95 = float(np.percentile(norms, 95))
    clip_norm = float(1.01 * percentile_95)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": "protocols/amendments/2026-09-29_priority25_rq1_extension.md",
        "amendment_sha256": sha256(PROJECT_ROOT / "protocols/amendments/2026-09-29_priority25_rq1_extension.md"),
        "script": "experiments/priority25_full_update_norm_probe.py",
        "seeds": args.seeds,
        "n_norms": int(norms.size),
        "min_norm": float(norms.min()),
        "median_norm": float(np.median(norms)),
        "p95_norm": percentile_95,
        "max_norm": float(norms.max()),
        "clip_norm_rule": "1.01 * empirical_p95_full_floating_update_norm",
        "clip_norm": clip_norm,
        "csv": str(csv_path.relative_to(PROJECT_ROOT)),
    }
    (output / "norm_probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

