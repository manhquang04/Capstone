"""Inspect DNA Transform mechanics before Phase 4 adaptive DNA evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dna_encoder.transform_defense import DNATransformConfig, _dna_block_seed_from_float32_block, transform_update_array
from experiments.run_phase3_adam_ladder import capture
from experiments.run_phase3_full_client import checksum, dump
from privacy.seed_manager import derive_seed, generate_run_seed


ROOT = Path(__file__).resolve().parents[1]


def _fixed_realization_matrix(block, config, tensor_index, block_index):
    block_seed = _dna_block_seed_from_float32_block(block, config.seed, tensor_index, block_index)
    rng = np.random.default_rng(block_seed)
    permutation = rng.permutation(block.size)
    permuted = block[permutation]
    shrink = np.ones(block.size, dtype=np.float64)
    if block.size > 1:
        keep_ratio = min(max(config.keep_ratio, 0.0), 1.0)
        threshold = np.quantile(np.abs(permuted), 1.0 - keep_ratio)
        shrink[np.abs(permuted) < threshold] = config.shrink_factor
    perm_matrix = np.zeros((block.size, block.size), dtype=np.float64)
    perm_matrix[np.arange(block.size), permutation] = 1.0
    attenuate_permute = np.diag(shrink) @ perm_matrix
    matrix = (1.0 - config.mix_ratio) * np.eye(block.size) + config.mix_ratio * attenuate_permute
    return matrix, block_seed, int((shrink != 1.0).sum())


def _analyze_state_delta(state_delta, config):
    rows = []
    for tensor_index, (name, value) in enumerate(state_delta.items()):
        if not value.is_floating_point():
            continue
        flat = value.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        for block_index, start in enumerate(range(0, flat.size, config.block_size)):
            block = flat[start : min(start + config.block_size, flat.size)]
            if block.size == 0:
                continue
            matrix, block_seed, attenuated = _fixed_realization_matrix(block, config, tensor_index, block_index)
            cond = float(np.linalg.cond(matrix))
            rank = int(np.linalg.matrix_rank(matrix))
            rows.append(
                {
                    "tensor": name,
                    "tensor_index": tensor_index,
                    "block_index": block_index,
                    "block_size": int(block.size),
                    "block_seed": int(block_seed),
                    "attenuated_elements": attenuated,
                    "rank": rank,
                    "full_rank": bool(rank == block.size),
                    "condition_number": cond,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase4_run", type=Path)
    parser.add_argument("--target-file", default="fresh_final_targets.pt")
    parser.add_argument("--group-id", type=int, default=0)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--mix-ratio", type=float, default=0.08)
    parser.add_argument("--keep-ratio", type=float, default=0.88)
    parser.add_argument("--shrink-factor", type=float, default=0.45)
    args = parser.parse_args()

    run = args.phase4_run.resolve()
    baseline = json.loads((run / "baseline_gate.json").read_text())
    groups = torch.load(run / args.target_file, weights_only=False)
    run_seed = args.base_seed if args.base_seed is not None else generate_run_seed()
    config = DNATransformConfig(
        block_size=args.block_size,
        mix_ratio=args.mix_ratio,
        keep_ratio=args.keep_ratio,
        shrink_factor=args.shrink_factor,
        seed=run_seed,
    )
    group = groups[args.group_id]
    _, _, _, _, _, _, observed = capture(
        group,
        derive_seed(baseline["protocol"]["run_seed"], "local", args.group_id, baseline["protocol"]["batch_size"]),
        baseline["protocol"]["batch_size"],
    )
    transform_stats = []
    transformed_state = {}
    for tensor_index, (name, delta) in enumerate(observed.items()):
        if not delta.is_floating_point():
            continue
        transformed, stats = transform_update_array(delta.detach().cpu().numpy(), config, tensor_index=tensor_index)
        transformed_state[name] = torch.from_numpy(transformed)
        transform_stats.append({"tensor": name, **stats.__dict__})
    matrix_rows = _analyze_state_delta(observed, config)
    condition_numbers = [row["condition_number"] for row in matrix_rows]
    summary = {
        "phase4_run": str(run),
        "target_file": args.target_file,
        "group_id": args.group_id,
        "source_ids": group["source_ids"],
        "dna_transform_config": config.__dict__,
        "fixed_realization_blocks": len(matrix_rows),
        "fixed_realization_full_rank_blocks": int(sum(row["full_rank"] for row in matrix_rows)),
        "fixed_realization_max_condition_number": float(max(condition_numbers)) if condition_numbers else None,
        "fixed_realization_median_condition_number": float(np.median(condition_numbers)) if condition_numbers else None,
        "transform_mean_relative_l2_delta": float(np.mean([row["relative_l2_delta"] for row in transform_stats])),
        "transform_mean_cosine_similarity": float(np.mean([row["cosine_similarity"] for row in transform_stats])),
        "interpretation": (
            "The actual forward transform is input-dependent because block seed, permutation, and "
            "attenuation mask are derived from the raw update block. If an exact block realization "
            "were known, each block is a linear full-rank map under this inspected configuration."
        ),
        "dataset_sha256": checksum(ROOT / "datasets/creditcard.csv"),
    }
    out = run / "dna_transform_mechanics_fresh_group0.json"
    dump(out, {"summary": summary, "block_rows": matrix_rows, "transform_stats": transform_stats})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
