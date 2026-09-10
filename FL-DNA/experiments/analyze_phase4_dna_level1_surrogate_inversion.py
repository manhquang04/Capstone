"""Phase 4 Level-1 surrogate-realization inversion diagnostic.

This diagnostic checks whether the Level-2 direct inversion idea remains useful
when the attacker does not know the exact DNA Transform realization.  It does
not optimize data reconstructions and does not open a DNA defense gate by
itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from attacks.local_update import simulate
from dna_encoder.transform_defense import DNATransformConfig, transform_update_array
from experiments import fraud_fl_common as common
from experiments.phase3_bounded_validation import update_objective
from experiments.run_phase3_adam_ladder import capture
from experiments.run_phase3_full_client import checksum, dump
from privacy.seed_manager import derive_seed, generate_run_seed


ROOT = Path(__file__).resolve().parents[1]
MAX_SEED = 2**31 - 1


def _surrogate_seed(config: DNATransformConfig, group_id: int, candidate_id: int, tensor_index: int, block_index: int) -> int:
    return derive_seed(
        config.seed or 1,
        "level1-surrogate-realization",
        group_id,
        candidate_id,
        tensor_index,
        block_index,
    )


def _candidate_realization_matrix(block_size: int, config: DNATransformConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(block_size)
    shrink = np.ones(block_size, dtype=np.float64)
    low_count = int(round((1.0 - min(max(config.keep_ratio, 0.0), 1.0)) * block_size))
    if low_count:
        low_positions = rng.choice(block_size, size=min(low_count, block_size), replace=False)
        shrink[low_positions] = config.shrink_factor
    perm_matrix = np.zeros((block_size, block_size), dtype=np.float64)
    perm_matrix[np.arange(block_size), permutation] = 1.0
    return (1.0 - config.mix_ratio) * np.eye(block_size) + config.mix_ratio * (np.diag(shrink) @ perm_matrix)


def _state_rel_l2(recovered, observed):
    numerator = 0.0
    denominator = 0.0
    dot = 0.0
    rec_norm = 0.0
    obs_norm = 0.0
    for key, value in observed.items():
        if not value.is_floating_point():
            continue
        rec = recovered[key].detach().cpu().double().reshape(-1)
        obs = value.detach().cpu().double().reshape(-1)
        numerator += float(torch.sum((rec - obs) ** 2))
        denominator += float(torch.sum(obs ** 2))
        dot += float(torch.dot(rec, obs))
        rec_norm += float(torch.sum(rec ** 2))
        obs_norm += float(torch.sum(obs ** 2))
    rel_l2 = float(np.sqrt(numerator / max(denominator, np.finfo(float).tiny)))
    cosine = float(dot / max(np.sqrt(rec_norm * obs_norm), np.finfo(float).tiny))
    return rel_l2, cosine


def _invert_with_surrogate_realization(transformed_state, config, group_id, candidate_id):
    recovered = {}
    block_count = 0
    for tensor_index, (name, transformed) in enumerate(transformed_state.items()):
        if not transformed.is_floating_point():
            recovered[name] = transformed.clone()
            continue
        flat = transformed.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
        rec = np.empty_like(flat)
        for block_index, start in enumerate(range(0, flat.size, config.block_size)):
            stop = min(start + config.block_size, flat.size)
            block_size = stop - start
            seed = _surrogate_seed(config, group_id, candidate_id, tensor_index, block_index)
            matrix = _candidate_realization_matrix(block_size, config, seed)
            rec[start:stop] = np.linalg.solve(matrix, flat[start:stop])
            block_count += 1
        recovered[name] = torch.from_numpy(rec.reshape(transformed.shape).astype(np.float32))
    return recovered, block_count


def _apply_surrogate_realization(state_delta, config, group_id, candidate_id):
    transformed = {}
    for tensor_index, (name, value) in enumerate(state_delta.items()):
        if not value.is_floating_point():
            transformed[name] = value.clone()
            continue
        flat = value.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
        out = np.empty_like(flat)
        for block_index, start in enumerate(range(0, flat.size, config.block_size)):
            stop = min(start + config.block_size, flat.size)
            block_size = stop - start
            seed = _surrogate_seed(config, group_id, candidate_id, tensor_index, block_index)
            matrix = _candidate_realization_matrix(block_size, config, seed)
            out[start:stop] = matrix @ flat[start:stop]
        transformed[name] = torch.from_numpy(out.reshape(value.shape).astype(np.float32))
    return transformed


def _load_candidate_pool(baseline_dir, group_id, restarts):
    rows = []
    for restart in range(restarts):
        folder = baseline_dir / f"group_{group_id}" / f"restart_{restart}"
        for method in ("baseline", "zero_update"):
            rows.append(torch.load(folder / f"{method}.pt", weights_only=False))
    return rows


def _precompute_candidate_deltas(model, criterion, y, batches, rng, candidate_pool):
    rows = []
    for candidate in candidate_pool:
        rows.append(
            {
                "delta": simulate(model, criterion, candidate["reconstruction"], y, batches, rng),
                "objective_mode": candidate["objective_mode"],
                "method": candidate["method"],
                "restart": int(candidate["restart"]),
            }
        )
    return rows


def _candidate_pool_objective(candidate_deltas, signal, reference, keys):
    best = None
    for candidate in candidate_deltas:
        objective = update_objective(candidate["delta"], signal, keys, reference=reference, mode=candidate["objective_mode"])
        value = float(objective.detach())
        if best is None or value < best["objective"]:
            best = {
                "objective": value,
                "pool_method": candidate["method"],
                "pool_restart": candidate["restart"],
            }
    return best


def _candidate_pool_forward_transmitted_objective(candidate_deltas, transmitted, config, group_id, candidate_id, keys):
    best = None
    for candidate in candidate_deltas:
        candidate_transmitted = _apply_surrogate_realization(candidate["delta"], config, group_id, candidate_id)
        objective = update_objective(
            candidate_transmitted,
            transmitted,
            keys,
            reference=transmitted,
            mode=candidate["objective_mode"],
        )
        value = float(objective.detach())
        if best is None or value < best["objective"]:
            best = {
                "objective": value,
                "pool_method": candidate["method"],
                "pool_restart": candidate["restart"],
            }
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase4_run", type=Path)
    parser.add_argument("--target-file", default="fresh_final_targets.pt")
    parser.add_argument("--baseline-attack-dir", default="harddiff_reparam_fresh_final_standard_nonneg_0p001")
    parser.add_argument("--candidates", type=int, default=8)
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
    restarts = json.loads((baseline_dir / "harddiff_reparam_report.json").read_text())["restarts"]
    dna_run_seed = args.dna_run_seed if args.dna_run_seed is not None else generate_run_seed()

    rows = []
    inverse_pool_selection_rows = []
    forward_transmitted_selection_rows = []
    for group_id, group in enumerate(target_groups):
        local_seed = derive_seed(baseline["protocol"]["run_seed"], "local", group_id, baseline["protocol"]["batch_size"])
        model, criterion, _, y, batches, rng, observed = capture(group, local_seed, baseline["protocol"]["batch_size"])
        keys = [key for key, value in observed.items() if value.is_floating_point()]
        candidate_pool = _load_candidate_pool(baseline_dir, group_id, restarts)
        candidate_deltas = _precompute_candidate_deltas(model, criterion, y, batches, rng, candidate_pool)
        config = DNATransformConfig(
            block_size=args.block_size,
            mix_ratio=args.mix_ratio,
            keep_ratio=args.keep_ratio,
            shrink_factor=args.shrink_factor,
            seed=derive_seed(dna_run_seed, "phase4-dna-transform", args.target_file, group_id),
        )
        transformed_state = {}
        for tensor_index, (name, delta) in enumerate(observed.items()):
            if delta.is_floating_point():
                transformed, _ = transform_update_array(delta.detach().cpu().numpy(), config, tensor_index=tensor_index)
                transformed_state[name] = torch.from_numpy(transformed)
            else:
                transformed_state[name] = delta.detach().cpu()

        candidate_rows = []
        for candidate_id in range(args.candidates):
            recovered, blocks = _invert_with_surrogate_realization(transformed_state, config, group_id, candidate_id)
            rel_l2, cosine = _state_rel_l2(recovered, observed)
            visible = _candidate_pool_objective(
                candidate_deltas,
                recovered,
                recovered,
                keys,
            )
            forward_visible = _candidate_pool_forward_transmitted_objective(
                candidate_deltas,
                transformed_state,
                config,
                group_id,
                candidate_id,
                keys,
            )
            candidate_rows.append(
                {
                    "group_id": group_id,
                    "candidate_id": candidate_id,
                    "blocks": blocks,
                    "relative_l2_recovery_error": rel_l2,
                    "cosine_to_raw_update": cosine,
                    "self_referential_inverse_pool_objective": visible["objective"],
                    "self_referential_inverse_pool_method": visible["pool_method"],
                    "self_referential_inverse_pool_restart": visible["pool_restart"],
                    "attacker_visible_forward_transmitted_objective": forward_visible["objective"],
                    "attacker_visible_forward_transmitted_method": forward_visible["pool_method"],
                    "attacker_visible_forward_transmitted_restart": forward_visible["pool_restart"],
                }
            )
        rows.extend(candidate_rows)
        inverse_pool_selection_rows.append(min(candidate_rows, key=lambda row: row["self_referential_inverse_pool_objective"]))
        forward_transmitted_selection_rows.append(
            min(candidate_rows, key=lambda row: row["attacker_visible_forward_transmitted_objective"])
        )

    best_by_group = []
    for group_id in range(len(target_groups)):
        group_rows = [row for row in rows if row["group_id"] == group_id]
        best_by_group.append(min(group_rows, key=lambda row: row["relative_l2_recovery_error"]))

    rel_values = [row["relative_l2_recovery_error"] for row in rows]
    best_rel_values = [row["relative_l2_recovery_error"] for row in best_by_group]
    best_cosines = [row["cosine_to_raw_update"] for row in best_by_group]
    inverse_rel_values = [row["relative_l2_recovery_error"] for row in inverse_pool_selection_rows]
    inverse_cosines = [row["cosine_to_raw_update"] for row in inverse_pool_selection_rows]
    forward_rel_values = [row["relative_l2_recovery_error"] for row in forward_transmitted_selection_rows]
    forward_cosines = [row["cosine_to_raw_update"] for row in forward_transmitted_selection_rows]
    report = {
        "target_file": args.target_file,
        "diagnostic": "Level 1 surrogate-realization direct inversion diagnostics",
        "baseline_attack_dir": args.baseline_attack_dir,
        "dna_transform_config": {
            "block_size": args.block_size,
            "mix_ratio": args.mix_ratio,
            "keep_ratio": args.keep_ratio,
            "shrink_factor": args.shrink_factor,
        },
        "dna_run_seed": dna_run_seed,
        "surrogate_candidates_per_group": args.candidates,
        "seed_space_audit": {
            "run_seed_bits": 31,
            "base_seed_source": "generate_run_seed uses secrets.randbelow(2**31 - 1) + 1 unless provided",
            "client_round_seed": "derive_seed uses BLAKE2b over the logged run seed and public parts",
            "block_seed": "base seed plus tensor/block offsets plus DNA counts and rolling hash from the unknown raw block",
            "bruteforce_assessment": "not enumerable in this scope when the run seed or raw-block-derived realization is unknown",
        },
        "all_surrogate_recovery": {
            "candidates": len(rows),
            "mean_relative_l2_error": float(np.mean(rel_values)),
            "median_relative_l2_error": float(np.median(rel_values)),
            "min_relative_l2_error": float(np.min(rel_values)),
            "max_relative_l2_error": float(np.max(rel_values)),
        },
        "oracle_realization_selection_diagnostic_only": {
            "groups": len(best_by_group),
            "mean_relative_l2_error": float(np.mean(best_rel_values)),
            "median_relative_l2_error": float(np.median(best_rel_values)),
            "min_relative_l2_error": float(np.min(best_rel_values)),
            "max_relative_l2_error": float(np.max(best_rel_values)),
            "mean_cosine_to_raw_update": float(np.mean(best_cosines)),
            "median_cosine_to_raw_update": float(np.median(best_cosines)),
        },
        "self_referential_inverse_pool_selection_deprecated": {
            "groups": len(inverse_pool_selection_rows),
            "mean_relative_l2_error": float(np.mean(inverse_rel_values)),
            "median_relative_l2_error": float(np.median(inverse_rel_values)),
            "min_relative_l2_error": float(np.min(inverse_rel_values)),
            "max_relative_l2_error": float(np.max(inverse_rel_values)),
            "mean_cosine_to_raw_update": float(np.mean(inverse_cosines)),
            "median_cosine_to_raw_update": float(np.median(inverse_cosines)),
            "selection_rule": (
                "choose the surrogate realization whose recovered signal has the lowest "
                "balanced-tensor objective over the saved attacker candidate pool"
            ),
            "selection_formula": (
                "argmin_r min_c L_balanced(delta(c), recovered_signal_r; reference=recovered_signal_r), "
                "where recovered_signal_r = M_r^{-1} transmitted_update"
            ),
            "caveat": (
                "Deprecated diagnostic: this objective is self-referential because recovered_signal_r "
                "is both the target signal and the scaling reference."
            ),
        },
        "attacker_visible_forward_transmitted_selection": {
            "groups": len(forward_transmitted_selection_rows),
            "mean_relative_l2_error": float(np.mean(forward_rel_values)),
            "median_relative_l2_error": float(np.median(forward_rel_values)),
            "min_relative_l2_error": float(np.min(forward_rel_values)),
            "max_relative_l2_error": float(np.max(forward_rel_values)),
            "mean_cosine_to_raw_update": float(np.mean(forward_cosines)),
            "median_cosine_to_raw_update": float(np.median(forward_cosines)),
            "selection_rule": (
                "choose the surrogate realization whose forward-transformed saved candidate delta "
                "best matches the transmitted DNA update"
            ),
            "selection_formula": (
                "argmin_r min_c L_balanced(M_r delta(c), transmitted_update; reference=transmitted_update)"
            ),
            "caveat": (
                "This uses only the transmitted update for realization selection, but it still reuses "
                "a saved baseline candidate pool rather than optimizing fresh against the DNA objective."
            ),
        },
        "interpretation": (
            "Random surrogate realizations are a cheap Level-1 diagnostic, not a full adaptive attack. "
            "Oracle selection uses raw-update error and must not be reported as a valid Level-1 result. "
            "The deprecated inverse-pool row is self-referential and should not be used as Level-1 evidence. "
            "The forward-transmitted row is attacker-visible but still reuses an existing candidate pool; "
            "it is a selection-rule diagnostic, not a fresh optimized Level-1 attack."
        ),
        "dataset_sha256": checksum(ROOT / "datasets/creditcard.csv"),
    }
    target_tag = (run / args.target_file).stem.replace("_targets", "")
    out = run / f"dna_level1_surrogate_realization_{target_tag}_c{args.candidates}"
    out.mkdir(exist_ok=True)
    dump(out / "dna_level1_surrogate_report.json", report)
    dump(out / "dna_level1_surrogate_rows.json", rows)
    dump(out / "dna_level1_surrogate_oracle_best_by_group.json", best_by_group)
    dump(out / "dna_level1_surrogate_inverse_pool_selection_by_group_deprecated.json", inverse_pool_selection_rows)
    dump(out / "dna_level1_surrogate_forward_transmitted_selection_by_group.json", forward_transmitted_selection_rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
