"""Projection sanity check for Phase 4 Level-B architecture-uncertain protocol.

This diagnostic verifies that the public projected observation does not make
candidate architectures trivially separable by scale alone before any Level-B
grid attack is run.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from attacks.local_update import simulate
from experiments import fraud_fl_common as common
from experiments.run_phase3_adam_ladder import capture
from experiments.run_phase3_full_client import checksum
from privacy.seed_manager import derive_seed


ROOT = Path(__file__).resolve().parents[1]
PHASE4_ROOT = ROOT / "artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z"

PROJECTION_INPUT_DIM = 131_072
PROJECTION_OUTPUT_DIM = 4_096
PROJECTION_SEED = 202_609_104_091
SURROGATE_INIT_SEED = 202_609_107_337


ARCHITECTURES = {
    "B1": {"hidden_widths": [64, 32], "batch_norm": False, "dropout": [0.0, 0.0]},
    "B2": {"hidden_widths": [128, 64], "batch_norm": False, "dropout": [0.0, 0.0]},
    "B3": {"hidden_widths": [128, 64, 32], "batch_norm": True, "dropout": [0.3, 0.2, 0.1]},
    "B4": {"hidden_widths": [256, 128, 64], "batch_norm": True, "dropout": [0.3, 0.2, 0.1]},
    "B5": {"hidden_widths": [128, 128, 64, 32], "batch_norm": True, "dropout": [0.3, 0.2, 0.2, 0.1]},
    "B6": {"hidden_widths": [64, 64, 32], "batch_norm": True, "dropout": [0.2, 0.2, 0.1]},
}


class CandidateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_widths: list[int], batch_norm: bool, dropout: list[float]):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for index, width in enumerate(hidden_widths):
            layers.append(nn.Linear(previous, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            rate = dropout[index] if index < len(dropout) else 0.0
            if rate > 0:
                layers.append(nn.Dropout(rate))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def floating_delta_items(delta: dict[str, torch.Tensor]) -> list[tuple[str, torch.Tensor]]:
    return [(key, value) for key, value in sorted(delta.items()) if torch.is_floating_point(value)]


def flatten_delta(delta: dict[str, torch.Tensor]) -> torch.Tensor:
    parts = [value.detach().reshape(-1).float().cpu() for _, value in floating_delta_items(delta)]
    if not parts:
        raise ValueError("delta has no floating tensors")
    return torch.cat(parts)


def pad_or_truncate(vector: torch.Tensor, length: int) -> torch.Tensor:
    if vector.numel() > length:
        return vector[:length].clone()
    if vector.numel() == length:
        return vector.clone()
    padded = torch.zeros(length, dtype=vector.dtype)
    padded[: vector.numel()] = vector
    return padded


def fast_walsh_hadamard(vector: torch.Tensor) -> torch.Tensor:
    """Unnormalized Walsh-Hadamard transform for a 1D power-of-two vector."""
    if vector.ndim != 1 or vector.numel() & (vector.numel() - 1):
        raise ValueError("Walsh-Hadamard input must be 1D with power-of-two length")
    output = vector.clone()
    width = 1
    while width < output.numel():
        view = output.view(-1, width * 2)
        left = view[:, :width].clone()
        right = view[:, width:].clone()
        view[:, :width] = left + right
        view[:, width:] = left - right
        width *= 2
    return output


def signed_hadamard_jl(vector: torch.Tensor) -> torch.Tensor:
    """Apply a fixed dense structured JL projection without storing a matrix."""
    vector = pad_or_truncate(vector.float().cpu(), PROJECTION_INPUT_DIM)
    generator = torch.Generator().manual_seed(PROJECTION_SEED)
    signs = torch.randint(0, 2, (PROJECTION_INPUT_DIM,), generator=generator, dtype=torch.int8)
    signed = vector * (signs.float().mul_(2).sub_(1))
    mixed = fast_walsh_hadamard(signed) / math.sqrt(PROJECTION_INPUT_DIM)
    indices = torch.randperm(PROJECTION_INPUT_DIM, generator=generator)[:PROJECTION_OUTPUT_DIM]
    return mixed[indices] * math.sqrt(PROJECTION_INPUT_DIM / PROJECTION_OUTPUT_DIM)


def vector_stats(vector: torch.Tensor) -> dict[str, float | int]:
    abs_vector = vector.abs()
    return {
        "length": int(vector.numel()),
        "norm": float(vector.norm().item()),
        "mean": float(vector.mean().item()),
        "std": float(vector.std(unbiased=False).item()),
        "min": float(vector.min().item()),
        "max": float(vector.max().item()),
        "abs_mean": float(abs_vector.mean().item()),
        "abs_max": float(abs_vector.max().item()),
        "near_zero_fraction": float((abs_vector < 1e-8).float().mean().item()),
    }


def safe_l2_normalize(vector: torch.Tensor) -> torch.Tensor:
    norm = vector.norm()
    if float(norm) == 0.0:
        return vector.clone()
    return vector / norm


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def floating_state_count(model: nn.Module) -> int:
    state = model.state_dict()
    return sum(value.numel() for value in state.values() if torch.is_floating_point(value))


def make_surrogate(arch_id: str, input_dim: int) -> CandidateMLP:
    config = ARCHITECTURES[arch_id]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(SURROGATE_INIT_SEED, "architecture", arch_id))
        model = CandidateMLP(
            input_dim=input_dim,
            hidden_widths=config["hidden_widths"],
            batch_norm=config["batch_norm"],
            dropout=config["dropout"],
        )
    return model.train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase4-root", type=Path, default=PHASE4_ROOT)
    parser.add_argument("--target-file", default="development_gate_targets.pt")
    parser.add_argument("--group-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    start = time.perf_counter()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.phase4_root / datetime.now(timezone.utc).strftime("level_b_projection_sanity_%Y%m%dT%H%M%S%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)

    groups = torch.load(args.phase4_root / args.target_file, weights_only=False)
    group = groups[args.group_index]
    model, criterion, x, y, batches, rng_state, observed = capture(
        group,
        derive_seed(202_609_104_091, "level_b_projection_sanity", args.group_index),
        batch_size=4,
    )

    true_vector = flatten_delta(observed)
    true_projected = signed_hadamard_jl(true_vector)
    true_projected_normalized = safe_l2_normalize(true_projected)

    rows: list[dict[str, object]] = []
    rows.append(
        {
            "label": "true_bounded_phase4_update",
            "architecture_id": "B3_true_checkpoint",
            "parameter_count": parameter_count(model),
            "floating_state_count": floating_state_count(model),
            "flattened_delta_length": int(true_vector.numel()),
            "padding_zeros": int(max(PROJECTION_INPUT_DIM - true_vector.numel(), 0)),
            "raw_stats": vector_stats(true_vector),
            "projected_stats": vector_stats(true_projected),
            "normalized_projected_stats": vector_stats(true_projected_normalized),
        }
    )

    for arch_id in ("B1", "B3", "B4"):
        surrogate = make_surrogate(arch_id, input_dim=x.shape[1])
        surrogate_criterion = common.BinaryFocalLoss()
        delta = simulate(copy.deepcopy(surrogate), surrogate_criterion, x, y, batches, rng_state, create_graph=False)
        raw_vector = flatten_delta(delta)
        projected = signed_hadamard_jl(raw_vector)
        projected_normalized = safe_l2_normalize(projected)
        rows.append(
            {
                "label": "surrogate_candidate_update",
                "architecture_id": arch_id,
                "parameter_count": parameter_count(surrogate),
                "floating_state_count": floating_state_count(surrogate),
                "flattened_delta_length": int(raw_vector.numel()),
                "padding_zeros": int(max(PROJECTION_INPUT_DIM - raw_vector.numel(), 0)),
                "raw_stats": vector_stats(raw_vector),
                "projected_stats": vector_stats(projected),
                "normalized_projected_stats": vector_stats(projected_normalized),
            }
        )

    true_norm = rows[0]["projected_stats"]["norm"]  # type: ignore[index]
    true_normalized_abs_mean = rows[0]["normalized_projected_stats"]["abs_mean"]  # type: ignore[index]
    for row in rows:
        projected_stats = row["projected_stats"]  # type: ignore[assignment]
        row["projected_norm_ratio_to_true"] = float(projected_stats["norm"] / true_norm)  # type: ignore[index]
        normalized_stats = row["normalized_projected_stats"]  # type: ignore[assignment]
        row["normalized_abs_mean_ratio_to_true"] = float(normalized_stats["abs_mean"] / true_normalized_abs_mean)  # type: ignore[index]

    projected_norms = [float(row["projected_stats"]["norm"]) for row in rows]  # type: ignore[index]
    normalized_abs_means = [float(row["normalized_projected_stats"]["abs_mean"]) for row in rows]  # type: ignore[index]
    report = {
        "protocol": {
            "observation_mode": "public_projection_matrix",
            "projection_input_dim": PROJECTION_INPUT_DIM,
            "projection_output_dim": PROJECTION_OUTPUT_DIM,
            "projection_seed": PROJECTION_SEED,
            "projection_distribution": "signed_hadamard_jl",
            "projection_objective_normalization": "l2_normalize_projected_update",
            "surrogate_init_seed": SURROGATE_INIT_SEED,
            "target_file": args.target_file,
            "group_index": args.group_index,
            "source_ids": group["source_ids"],
            "projection_input_dim_sha256_context": checksum(args.phase4_root / args.target_file),
        },
        "rows": rows,
        "summary": {
            "projected_norm_min": min(projected_norms),
            "projected_norm_max": max(projected_norms),
            "projected_norm_max_over_min": max(projected_norms) / min(projected_norms),
            "normalized_abs_mean_min": min(normalized_abs_means),
            "normalized_abs_mean_max": max(normalized_abs_means),
            "normalized_abs_mean_max_over_min": max(normalized_abs_means) / min(normalized_abs_means),
            "status": "DIAGNOSTIC_ONLY",
            "interpretation": (
                "Raw projection scale is logged for leakage checks. Candidate selection must use the "
                "L2-normalized projected objective; this artifact does not select an architecture or attack result."
            ),
        },
        "seconds": time.perf_counter() - start,
    }
    dump(output_dir / "level_b_projection_sanity_report.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
