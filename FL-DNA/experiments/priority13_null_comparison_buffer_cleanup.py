"""Priority 13 buffer-exclusion replay for null RQ1 comparisons.

This script does not optimize any attacker and does not create any target set.
It reuses the frozen selected candidates from contaminated tabular artifacts,
recomputes their defended/update-space signals, excludes BatchNorm buffers by
keeping only trainable parameter keys, and re-applies the paired sign-test
structure to the existing null comparisons.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.local_update import simulate
from experiments.run_phase4_dna_level1_forward_attack import (
    _apply_surrogate_realization_torch,
    _surrogate_plan_from_state,
)
from experiments.run_phase4_harddiff_reparam_for_misselected import capture as capture_paysim
from experiments.run_phase4_simple_defense_attack import (
    _apply_candidate_defense,
    _plan as simple_defense_plan,
)


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _best(root: Path, method: str = "baseline") -> tuple[Path, dict]:
    rows = []
    for path in root.glob("**/*.pt"):
        if path.name.endswith("targets.pt"):
            continue
        artifact = _load(path)
        if artifact.get("method") == method:
            rows.append((float(artifact["best_objective"]), str(path), path, artifact))
    if not rows:
        raise RuntimeError(f"no {method} artifact under {root}")
    _, _, path, artifact = min(rows, key=lambda row: (row[0], row[1]))
    return path, artifact


def _report(root: Path, preferred: str | None = None) -> dict:
    matches = list(root.glob(f"**/{preferred}")) if preferred else []
    if not matches:
        matches = list(root.glob("**/*_report.json"))
    if not matches:
        raise RuntimeError(f"no report under {root}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def _trainable_keys(update: dict[str, torch.Tensor]) -> list[str]:
    return [
        key
        for key, value in update.items()
        if value.is_floating_point()
        and "running_mean" not in key
        and "running_var" not in key
        and "num_batches_tracked" not in key
    ]


def _mse(candidate: dict[str, torch.Tensor], target: dict[str, torch.Tensor], keys: list[str]) -> float:
    values = []
    for key in keys:
        if key not in candidate or key not in target:
            raise KeyError(f"missing key {key}")
        values.append((candidate[key].detach().double() - target[key].detach().double()).reshape(-1).square())
    flat = torch.cat(values)
    return float(flat.mean())


def _flat(update: dict[str, torch.Tensor], keys: list[str]) -> torch.Tensor:
    return torch.cat([update[key].detach().double().reshape(-1) for key in keys])


def _sorted_mse(candidate: dict[str, torch.Tensor], target: dict[str, torch.Tensor], keys: list[str]) -> float:
    """Permutation-invariant descriptive MSE over sorted trainable values."""
    left = torch.sort(_flat(candidate, keys)).values
    right = torch.sort(_flat(target, keys)).values
    return float((left - right).square().mean())


def _paired_summary(rows: list[dict], diff_key: str, threshold: float) -> dict:
    diffs = np.asarray([row[diff_key] for row in rows], dtype=float)
    non_tied = np.asarray([x for x in diffs if abs(x) > threshold], dtype=float)
    wins = int((non_tied > 0).sum())
    p = float(binomtest(wins, len(non_tied), 0.5, alternative="greater").pvalue) if len(non_tied) else 1.0
    zero_wins = int((diffs > 0).sum())
    zero_losses = int((diffs < 0).sum())
    zero_ties = int((diffs == 0).sum())
    zero_n = zero_wins + zero_losses
    zero_p = float(binomtest(zero_wins, zero_n, 0.5, alternative="greater").pvalue) if zero_n else 1.0
    zero_dp_p = float(binomtest(zero_losses, zero_n, 0.5, alternative="greater").pvalue) if zero_n else 1.0
    return {
        "threshold": threshold,
        "wins": wins,
        "losses": int(len(non_tied) - wins),
        "ties": int(len(diffs) - len(non_tied)),
        "non_tied_n": int(len(non_tied)),
        "one_sided_exact_sign_p": p,
        "reject_h0_alpha_0p05": bool(len(non_tied) and p < 0.05),
        "mean_difference": float(diffs.mean()) if len(diffs) else None,
        "median_difference": float(np.median(diffs)) if len(diffs) else None,
        "zero_threshold_descriptive": {
            "wins": zero_wins,
            "losses": zero_losses,
            "exact_ties": zero_ties,
            "non_tied_n": int(zero_n),
            "total_n": int(len(diffs)),
            "one_sided_exact_sign_p_dna_advantage": zero_p,
            "one_sided_exact_sign_p_dp_advantage": zero_dp_p,
        },
    }


def _dp_plan_from_report(observed: dict[str, torch.Tensor], keys: list[str], group: int, report: dict) -> dict:
    args = SimpleNamespace(
        target_rel_l2=float(report.get("target_relative_l2_delta", 0.1)),
        clip_factor=float(report.get("clip_factor", 0.95)),
        clip_norm=float(report["clip_norm"]),
        noise_multiplier=float(report["noise_multiplier"]),
        defense_seed=int(report["defense_seed"]),
        mc_noise_samples=int(report["mc_noise_samples"]),
        topk_temperature=float(report.get("topk_temperature", 1e-3) or 1e-3),
    )
    return simple_defense_plan("clipping_noise_mc", observed, keys, group, args)


def _candidate_delta_from_paysim(artifact: dict, target_path: Path) -> dict[str, torch.Tensor]:
    groups = torch.load(target_path, map_location="cpu", weights_only=False)
    group = groups[int(artifact["group_id"])]
    model, criterion, _x, y, batches, rng, _observed = capture_paysim(group, int(artifact["local_seed"]), 4)
    return simulate(model, criterion, artifact["reconstruction"], y, batches, rng)


def replay_rq1_group2_stronger() -> dict:
    run_dir = ROOT / "artifacts/rq1/group2_stronger_confirmatory_run_20260913"
    target = ROOT / "artifacts/rq1/group2_stronger_confirmatory_freeze_20260913/rq1_stronger_confirmatory_targets.pt"
    if not target.exists():
        # Target is copied into every group run; use the first copy if the freeze
        # directory is not present in this checkout.
        target = next((run_dir / "dna/group_0_run").glob("*targets.pt"))
    threshold = 0.0390625
    rows = []
    for group in range(44):
        dna_root = run_dir / "dna" / f"group_{group}_run"
        dp_root = run_dir / "dp" / f"group_{group}_run"
        dna_path, dna = _best(dna_root, "baseline")
        dp_path, dp = _best(dp_root, "baseline")
        if list(map(int, dna["source_ids"])) != list(map(int, dp["source_ids"])):
            raise AssertionError(f"source mismatch group {group}")
        dna_delta = _candidate_delta_from_paysim(dna, target)
        dna_plan = _surrogate_plan_from_state(dna["observed_raw_update"], SimpleNamespace(**dna["dna_transform_config"]), group, int(dna["realization_id"]))
        dna_candidate = _apply_surrogate_realization_torch(dna_delta, dna_plan)
        dp_delta = _candidate_delta_from_paysim(dp, target)
        dp_report = _report(dp_root, "simple_defense_attack_report.json")
        dp_keys_full = [key for key, value in dp["observed_raw_update"].items() if value.is_floating_point()]
        dp_plan = _dp_plan_from_report(dp["observed_raw_update"], dp_keys_full, group, dp_report)
        dp_candidate = _apply_candidate_defense(dp_delta, dp_keys_full, "clipping_noise_mc", dp_plan)
        dna_keys = _trainable_keys(dna["transmitted_update"])
        dp_keys = _trainable_keys(dp["defended_update"])
        common_keys = [key for key in dna_keys if key in dp_keys]
        dna_mse = _mse(dna_candidate, dna["transmitted_update"], common_keys)
        dp_mse = _mse(dp_candidate, dp["defended_update"], common_keys)
        dna_sorted_mse = _sorted_mse(dna_candidate, dna["transmitted_update"], common_keys)
        dp_sorted_mse = _sorted_mse(dp_candidate, dp["defended_update"], common_keys)
        rows.append(
            {
                "group_id": group,
                "dna_artifact": str(dna_path.relative_to(ROOT)),
                "dp_artifact": str(dp_path.relative_to(ROOT)),
                "mse_dna_trainable_update": dna_mse,
                "mse_dp_trainable_update": dp_mse,
                "D_dna_minus_dp": dna_mse - dp_mse,
                "sorted_mse_dna_trainable_update": dna_sorted_mse,
                "sorted_mse_dp_trainable_update": dp_sorted_mse,
                "D_sorted_dna_minus_dp": dna_sorted_mse - dp_sorted_mse,
            }
        )
    return {
        "test": "RQ1 Group 2 stronger v1 vs DP distortion-matched",
        "old_report_result": {"wins": 25, "non_tied_n": 44, "one_sided_exact_sign_p": 0.2257},
        "run_dir": str(run_dir.relative_to(ROOT)),
        "target_source": str(target.relative_to(ROOT)),
        "metric": "trainable-update MSE after excluding BatchNorm buffers",
        "limitation": "candidate reconstructions were selected by the original contaminated objective; this replay only rescored saved candidates",
        "paired_summary": _paired_summary(rows, "D_dna_minus_dp", threshold),
        "permutation_invariant_sorted_value_summary": _paired_summary(rows, "D_sorted_dna_minus_dp", 0.0),
        "rows": rows,
    }


def _candidate_delta_from_v2(artifact: dict, target_path: Path) -> dict[str, torch.Tensor]:
    # RQ1-v2 is PaySim scope with the same saved local_seed/capture contract.
    groups = torch.load(target_path, map_location="cpu", weights_only=False)
    group = groups[int(artifact["group_id"])]
    model, criterion, _x, y, batches, rng, _observed = capture_paysim(group, int(artifact["local_seed"]), 4)
    return simulate(model, criterion, artifact["reconstruction"], y, batches, rng)


def replay_rq1_v2() -> dict:
    run_dir = ROOT / "artifacts/rq1_v2/confirmatory_run_20260916"
    target = ROOT / "artifacts/rq1_v2/confirmatory_freeze_20260916/rq1_v2_confirmatory_targets.pt"
    threshold = 0.01953125
    rows = []
    for group in range(176):
        dna_root = run_dir / "dna_v2" / f"group_{group}_run"
        dp_root = run_dir / "dp_v2" / f"group_{group}_run"
        dna_path, dna = _best(dna_root, "baseline")
        dp_path, dp = _best(dp_root, "baseline")
        if list(map(int, dna["source_ids"])) != list(map(int, dp["source_ids"])):
            raise AssertionError(f"source mismatch group {group}")
        dna_delta = _candidate_delta_from_v2(dna, target)
        dp_delta = _candidate_delta_from_v2(dp, target)
        dp_report = _report(dp_root, "simple_defense_attack_report.json")
        dp_keys_full = [key for key, value in dp["observed_raw_update"].items() if value.is_floating_point()]
        dp_plan = _dp_plan_from_report(dp["observed_raw_update"], dp_keys_full, group, dp_report)
        dp_candidate = _apply_candidate_defense(dp_delta, dp_keys_full, "clipping_noise_mc", dp_plan)
        dna_keys = _trainable_keys(dna["iht_recovered_update"])
        dp_keys = _trainable_keys(dp["defended_update"])
        common_keys = [key for key in dna_keys if key in dp_keys]
        dna_mse = _mse(dna_delta, dna["iht_recovered_update"], common_keys)
        dp_mse = _mse(dp_candidate, dp["defended_update"], common_keys)
        rows.append(
            {
                "group_id": group,
                "dna_v2_artifact": str(dna_path.relative_to(ROOT)),
                "dp_v2_artifact": str(dp_path.relative_to(ROOT)),
                "mse_dna_v2_trainable_update": dna_mse,
                "mse_dp_v2_trainable_update": dp_mse,
                "D_dna_v2_minus_dp_v2": dna_mse - dp_mse,
            }
        )
    return {
        "test": "RQ1-v2 confirmatory DNA-v2 vs DP-v2",
        "old_report_result": {"wins": 85, "non_tied_n": 176, "one_sided_exact_sign_p": 0.70106},
        "run_dir": str(run_dir.relative_to(ROOT)),
        "target_source": str(target.relative_to(ROOT)),
        "metric": "trainable-update MSE after excluding BatchNorm buffers",
        "limitation": "candidate reconstructions were selected by the original contaminated objective; this replay only rescored saved candidates",
        "paired_summary": _paired_summary(rows, "D_dna_v2_minus_dp_v2", threshold),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/priority13_null_cleanup/step1_replay_20260921.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {
        "created_at": "2026-09-21",
        "scope": "Priority 13 Step 1 replay only; no attacker optimization; no new targets",
        "vector_scope": "trainable parameters only; BatchNorm buffers excluded",
        "tests": [replay_rq1_group2_stronger(), replay_rq1_v2()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({test["test"]: test["paired_summary"] for test in result["tests"]}, indent=2))


if __name__ == "__main__":
    main()
