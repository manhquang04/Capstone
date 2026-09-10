"""Verify Gate B implementation integrity for a Phase 4 attack run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from attacks.local_update import simulate
from experiments import fraud_fl_common as common
from experiments.phase3_bounded_validation import update_objective
from experiments.run_phase3_adam_ladder import capture
from experiments.run_phase3_full_client import checksum, dump
from experiments.run_phase4_harddiff_reparam_for_misselected import _nonnegative_penalty
from privacy.seed_manager import derive_seed


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path):
    return json.loads(path.read_text())


def _check_close(name, actual, expected, atol=1e-5, rtol=1e-5):
    diff = abs(float(actual) - float(expected))
    limit = atol + rtol * abs(float(expected))
    if diff > limit:
        raise AssertionError(f"{name} mismatch: actual={actual} expected={expected} diff={diff}")
    return diff


def _objective_for(method, loaded, model, criterion, y, batches, rng, observed, keys):
    signal = observed if method == "baseline" else {key: torch.zeros_like(value) for key, value in observed.items()}
    delta = simulate(model, criterion, loaded["reconstruction"], y, batches, rng)
    objective = update_objective(delta, signal, keys, reference=observed, mode=loaded["objective_mode"])
    lambda_value = float(loaded.get("nonnegative_lambda", 0.0))
    if lambda_value:
        objective = objective + lambda_value * _nonnegative_penalty(loaded["latent"], type("Meta", (), {
            "numeric_center": _METADATA["numeric_center"],
            "numeric_scale": _METADATA["numeric_scale"],
        })())
    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--attack-dir", required=True)
    parser.add_argument("--target-file", default="fresh_final_targets.pt")
    args = parser.parse_args()

    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    run = args.run.resolve()
    attack_dir = run / args.attack_dir
    report = _load_json(attack_dir / "harddiff_reparam_report.json")
    protocol = _load_json(run / "baseline_gate.json")["protocol"]
    targets = torch.load(run / args.target_file, weights_only=False)

    global _METADATA
    _METADATA = _load_json(ROOT / "artifacts/phase3/full_20260908T143837588533Z/preprocessing.json")

    checks = []
    failures = []
    expected_files = 0
    for group_id in report["groups"]:
        group = targets[group_id]
        local_seed = derive_seed(protocol["run_seed"], "local", group_id, protocol["batch_size"])
        model, criterion, x, y, batches, rng, observed = capture(group, local_seed, protocol["batch_size"])
        keys = [key for key, value in observed.items() if value.is_floating_point()]
        for restart in range(report["restarts"]):
            folder = attack_dir / f"group_{group_id}" / f"restart_{restart}"
            baseline_path = folder / "baseline.pt"
            zero_path = folder / "zero_update.pt"
            expected_files += 2
            if not baseline_path.exists() or not zero_path.exists():
                failures.append({"group_id": group_id, "restart": restart, "error": "missing baseline/zero artifact"})
                continue
            baseline = torch.load(baseline_path, weights_only=False)
            zero = torch.load(zero_path, weights_only=False)
            if not torch.equal(baseline["initial"], zero["initial"]):
                failures.append({"group_id": group_id, "restart": restart, "error": "baseline/zero initial mismatch"})
            for method, loaded in (("baseline", baseline), ("zero_update", zero)):
                objective = _objective_for(method, loaded, model, criterion, y, batches, rng, observed, keys)
                diff = _check_close(
                    f"group {group_id} restart {restart} {method}",
                    float(objective.detach()),
                    float(loaded["best_objective"]),
                )
                checks.append(
                    {
                        "group_id": group_id,
                        "restart": restart,
                        "method": method,
                        "local_seed": local_seed,
                        "initial_seed": derive_seed(
                            protocol["run_seed"],
                            "harddiff-initial",
                            report["initialization"],
                            group_id,
                            restart,
                            protocol["batch_size"],
                        ),
                        "recorded_best_objective": float(loaded["best_objective"]),
                        "recomputed_best_objective": float(objective.detach()),
                        "objective_abs_diff": diff,
                    }
                )

    summary = {
        "gate": "PASS" if not failures else "FAIL",
        "run": str(run),
        "attack_dir": str(attack_dir),
        "target_file": args.target_file,
        "expected_method_artifacts": expected_files,
        "checked_method_artifacts": len(checks),
        "paired_initial_checks": report["restarts"] * len(report["groups"]),
        "reload_objective_checks": len(checks),
        "max_objective_abs_diff": max((row["objective_abs_diff"] for row in checks), default=None),
        "native_replay_check": "capture() asserts differentiable replay against native Adam for every group",
        "seed_logging": (
            "run_seed is stored in baseline_gate.json; local_seed and initial_seed are derived with BLAKE2b "
            "and written per checked artifact in this integrity report"
        ),
        "failed_jobs": failures,
        "dataset_sha256": checksum(ROOT / "datasets/creditcard.csv"),
    }
    output = attack_dir / "gate_b_integrity_report.json"
    dump(output, {"summary": summary, "checks": checks})
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
