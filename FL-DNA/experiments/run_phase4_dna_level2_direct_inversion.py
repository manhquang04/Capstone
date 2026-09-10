"""Phase 4 DNA Level-2 direct inversion under exact-realization knowledge.

This is an attacker-favorable upper bound.  The attacker is given the exact
per-block DNA Transform realization, not only the public algorithm.  With that
realization fixed, each block is a linear map and can be inverted directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from attacks.local_update import simulate
from dna_encoder.transform_defense import DNATransformConfig, transform_update_array
from experiments import fraud_fl_common as common
from experiments.analyze_dna_transform_mechanics import _fixed_realization_matrix
from experiments.phase3_bounded_validation import update_objective
from experiments.run_phase3_adam_ladder import capture
from experiments.run_phase3_full_client import checksum, dump
from privacy.seed_manager import derive_seed, generate_run_seed


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sign_tail(wins, total):
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / 2**total if total else 1.0


def _invert_array_with_exact_realization(array, transformed, config, tensor_index):
    original = np.asarray(array, dtype=np.float32)
    flat = original.reshape(-1).astype(np.float64, copy=False)
    y = np.asarray(transformed, dtype=np.float32).reshape(-1).astype(np.float64, copy=False)
    recovered = np.empty_like(flat)
    block_rows = []
    for block_index, start in enumerate(range(0, flat.size, config.block_size)):
        stop = min(start + config.block_size, flat.size)
        block = flat[start:stop]
        matrix, block_seed, attenuated = _fixed_realization_matrix(
            block.astype(np.float32, copy=False),
            config,
            tensor_index,
            block_index,
        )
        recovered[start:stop] = np.linalg.solve(matrix, y[start:stop])
        block_rows.append(
            {
                "tensor_index": tensor_index,
                "block_index": block_index,
                "block_size": int(block.size),
                "block_seed": int(block_seed),
                "attenuated_elements": int(attenuated),
                "rank": int(np.linalg.matrix_rank(matrix)),
                "condition_number": float(np.linalg.cond(matrix)),
            }
        )
    return recovered.reshape(original.shape).astype(np.float32), block_rows


def _recover_state(observed, config):
    recovered = {}
    transform_rows = []
    block_rows = []
    for tensor_index, (name, delta) in enumerate(observed.items()):
        if not delta.is_floating_point():
            recovered[name] = delta.clone()
            continue
        original_array = delta.detach().cpu().numpy().astype(np.float32, copy=False)
        transformed_array, stats = transform_update_array(original_array, config, tensor_index=tensor_index)
        recovered_array, rows = _invert_array_with_exact_realization(
            original_array,
            transformed_array,
            config,
            tensor_index,
        )
        recovered_tensor = torch.from_numpy(recovered_array).to(dtype=delta.dtype)
        recovered[name] = recovered_tensor
        abs_error = (recovered_tensor - delta.detach().cpu()).abs()
        transform_rows.append(
            {
                "tensor": name,
                **stats.__dict__,
                "max_abs_recovery_error": float(abs_error.max()),
                "mean_abs_recovery_error": float(abs_error.mean()),
                "relative_l2_recovery_error": float(
                    torch.linalg.vector_norm(recovered_tensor - delta.detach().cpu())
                    / torch.linalg.vector_norm(delta.detach().cpu()).clamp_min(torch.finfo(delta.dtype).tiny)
                ),
            }
        )
        for row in rows:
            block_rows.append({"tensor": name, **row})
    return recovered, transform_rows, block_rows


def _load_records(report_dir, group_id, restarts):
    rows = []
    for restart in range(restarts):
        result_path = report_dir / f"group_{group_id}" / f"restart_{restart}" / "results.json"
        rows.extend(json.loads(result_path.read_text()))
    return rows


def _select(rows, method, key):
    return min((row for row in rows if row["method"] == method), key=lambda row: row[key])


def _evaluate_group_with_recovered_signal(report_dir, group, group_id, restarts, model, criterion, y, batches, rng, observed, recovered):
    keys = [key for key, value in observed.items() if value.is_floating_point()]
    rows = _load_records(report_dir, group_id, restarts)
    scored = []
    for restart in range(restarts):
        folder = report_dir / f"group_{group_id}" / f"restart_{restart}"
        for method, signal in (
            ("baseline", recovered),
            ("zero_update", {key: torch.zeros_like(value) for key, value in observed.items()}),
        ):
            loaded = torch.load(folder / f"{method}.pt", weights_only=False)
            delta = simulate(model, criterion, loaded["reconstruction"], y, batches, rng)
            objective = update_objective(delta, signal, keys, reference=recovered, mode=loaded["objective_mode"])
            lambda_value = float(loaded.get("nonnegative_lambda", 0.0))
            if lambda_value:
                # The saved best objective included the non-negative latent penalty.
                from experiments.run_phase4_harddiff_reparam_for_misselected import _nonnegative_penalty, metadata

                objective = objective + lambda_value * _nonnegative_penalty(loaded["latent"], metadata())
            metrics = next(row["metrics"] for row in rows if row["method"] == method and row["restart"] == restart)
            prior = next(row["prior"] for row in rows if row["method"] == method and row["restart"] == restart)
            scored.append(
                {
                    "method": method,
                    "restart": restart,
                    "objective": float(objective.detach()),
                    "fraud_mse": float(metrics["classes"]["1"]["mean_mse"]),
                    "non_fraud_mse": float(metrics["classes"]["0"]["mean_mse"]),
                    "prior_fraud_mse": float(prior["classes"]["1"]["mean_mse"]),
                }
            )
    baseline = _select(scored, "baseline", "objective")
    zero = _select(scored, "zero_update", "objective")
    prior_mean = float(np.mean([row["prior_fraud_mse"] for row in scored if row["method"] == "baseline"]))
    return {
        "group_id": group_id,
        "source_ids": group["source_ids"],
        "baseline_restart": baseline["restart"],
        "zero_restart": zero["restart"],
        "baseline_fraud_mse": baseline["fraud_mse"],
        "prior_fraud_mse": prior_mean,
        "zero_fraud_mse": zero["fraud_mse"],
        "baseline_minus_prior": baseline["fraud_mse"] - prior_mean,
        "baseline_minus_zero": baseline["fraud_mse"] - zero["fraud_mse"],
    }


def _gate(selected):
    result = {}
    for control in ("prior", "zero"):
        key = f"baseline_minus_{control}"
        values = np.asarray([row[key] for row in selected], dtype=float)
        wins = int((values < 0).sum())
        result[control] = {
            "n": len(values),
            "wins": wins,
            "mean_difference": float(values.mean()),
            "median_difference": float(np.median(values)),
            "one_sided_sign_p": _sign_tail(wins, len(values)),
            "gate": bool(values.mean() < 0 and np.median(values) < 0 and _sign_tail(wins, len(values)) < 0.05),
        }
    result["gate"] = bool(result["prior"]["gate"] and result["zero"]["gate"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase4_run", type=Path)
    parser.add_argument("--target-file", default="fresh_final_targets.pt")
    parser.add_argument("--baseline-attack-dir", default="harddiff_reparam_fresh_final_standard_nonneg_0p001")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--mix-ratio", type=float, default=0.08)
    parser.add_argument("--keep-ratio", type=float, default=0.88)
    parser.add_argument("--shrink-factor", type=float, default=0.45)
    parser.add_argument("--dna-run-seed", type=int)
    args = parser.parse_args()

    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    run = args.phase4_run.resolve()
    baseline = json.loads((run / "baseline_gate.json").read_text())
    target_groups = torch.load(run / args.target_file, weights_only=False)
    baseline_dir = run / args.baseline_attack_dir
    dna_run_seed = args.dna_run_seed if args.dna_run_seed is not None else generate_run_seed()
    restarts = json.loads((baseline_dir / "harddiff_reparam_report.json").read_text())["restarts"]

    selected = []
    transform_rows = []
    block_rows = []
    for group_id, group in enumerate(target_groups):
        local_seed = derive_seed(baseline["protocol"]["run_seed"], "local", group_id, baseline["protocol"]["batch_size"])
        model, criterion, _, y, batches, rng, observed = capture(group, local_seed, baseline["protocol"]["batch_size"])
        config = DNATransformConfig(
            block_size=args.block_size,
            mix_ratio=args.mix_ratio,
            keep_ratio=args.keep_ratio,
            shrink_factor=args.shrink_factor,
            seed=derive_seed(dna_run_seed, "phase4-dna-transform", args.target_file, group_id),
        )
        recovered, tensor_rows, matrix_rows = _recover_state(observed, config)
        for row in tensor_rows:
            transform_rows.append({"group_id": group_id, **row})
        for row in matrix_rows:
            block_rows.append({"group_id": group_id, **row})
        selected.append(
            _evaluate_group_with_recovered_signal(
                baseline_dir,
                group,
                group_id,
                restarts,
                model,
                criterion,
                y,
                batches,
                rng,
                observed,
                recovered,
            )
        )

    recovery_errors = [row["relative_l2_recovery_error"] for row in transform_rows]
    condition_numbers = [row["condition_number"] for row in block_rows]
    report = {
        "target_file": args.target_file,
        "baseline_attack_dir": args.baseline_attack_dir,
        "attacker_level": "Level 2 realization-known direct inversion upper bound",
        "dna_transform_config": {
            "block_size": args.block_size,
            "mix_ratio": args.mix_ratio,
            "keep_ratio": args.keep_ratio,
            "shrink_factor": args.shrink_factor,
        },
        "dna_run_seed": dna_run_seed,
        "gate": _gate(selected),
        "recovery": {
            "tensors": len(transform_rows),
            "blocks": len(block_rows),
            "full_rank_blocks": int(sum(row["rank"] == row["block_size"] for row in block_rows)),
            "max_condition_number": float(max(condition_numbers)),
            "median_condition_number": float(np.median(condition_numbers)),
            "max_relative_l2_recovery_error": float(max(recovery_errors)),
            "mean_relative_l2_recovery_error": float(np.mean(recovery_errors)),
            "max_abs_recovery_error": float(max(row["max_abs_recovery_error"] for row in transform_rows)),
        },
        "selected": selected,
        "interpretation": (
            "Realization-known direct inversion reconstructs the raw transmitted update before data inversion. "
            "This is an attacker-favorable upper bound and is not the same as seed-unknown Level 1."
        ),
        "dataset_sha256": checksum(ROOT / "datasets/creditcard.csv"),
    }
    out = run / "dna_level2_realization_known_direct"
    out.mkdir(exist_ok=True)
    dump(out / "dna_level2_direct_report.json", report)
    _write_csv(out / "dna_level2_direct_selected.csv", selected)
    _write_csv(out / "dna_level2_direct_transform_stats.csv", transform_rows)
    _write_csv(out / "dna_level2_direct_block_stats.csv", block_rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
