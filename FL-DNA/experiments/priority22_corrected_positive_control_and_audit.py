from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attacks.local_update import simulate  # noqa: E402
from experiments.phase3_bounded_validation import _align_for_evaluation, update_objective  # noqa: E402
from experiments.run_phase3_full_client import score  # noqa: E402
from experiments.run_phase4_harddiff_reparam_for_misselected import (  # noqa: E402
    _decode_harddiff,
    _feature_distribution,
    _initial,
    _nonnegative_penalty,
)
from privacy.seed_manager import derive_seed  # noqa: E402

import experiments.priority16_v1_medium_dna_vs_dp_probe as p16  # noqa: E402


AMENDMENT = "protocols/amendments/2026-09-28_priority22_corrected_positive_control.md"
TARGET = ROOT / "artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_targets_20260927/paysim_priority16_v1_medium_confirmatory_targets.pt"
TARGET_SHA = "645fe7ff62403e0bce81a3fc6cbed4d857306d36719cc2e3bad740acb73297fe"
P16_RUN = ROOT / "artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_run_20260927"
P16_REPORT = P16_RUN / "priority14_probe_report.json"
CELL = "v1_medium_vs_dp_0p000315"
RUN_SEED = 2026092706
TIE_BAND = 0.0390625


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, sort_keys=True) + "\n")


def binom_tail(wins: int, total: int) -> float:
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / (2**total) if total else 1.0


def sign_counts(values: list[float], tie_band: float) -> dict[str, Any]:
    wins = sum(1 for value in values if value > tie_band)
    losses = sum(1 for value in values if value < -tie_band)
    ties = len(values) - wins - losses
    non_tied = wins + losses
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "non_tied": non_tied,
        "one_sided_p": binom_tail(wins, non_tied),
    }


def summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


def clean_mse(candidate: dict[str, torch.Tensor], signal: dict[str, torch.Tensor]) -> float:
    parts = [(candidate[key].detach().double() - signal[key].detach().double()).reshape(-1).square() for key in signal]
    return float(torch.cat(parts).mean())


def optimize_signal(
    *,
    job: dict[str, Any],
    method: str,
    signal: dict[str, torch.Tensor],
    candidate_fn,
    model,
    criterion,
    original: torch.Tensor,
    labels: torch.Tensor,
    batches,
    rng,
    trainable: list[str],
    distribution,
    folder: Path,
) -> dict[str, Any]:
    """Priority-16-equivalent optimization loop for a caller-supplied signal."""

    folder.mkdir(parents=True, exist_ok=True)
    meta = distribution[0]
    initial = _initial(
        original.shape,
        derive_seed(job["seed"], "priority14-init", job["cell"], method, job["group"], job["restart"]),
        "standard",
        distribution,
    )
    latent = initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=0.1)
    best = float("inf")
    best_step = 0
    best_latent = latent.detach().clone()
    history: list[float] = []
    for step in range(601):
        reconstruction = _decode_harddiff(latent, meta)
        delta_full = simulate(model, criterion, reconstruction, labels, batches, rng)
        delta = p16._filter_trainable(delta_full, trainable)
        candidate = candidate_fn(delta)
        penalty = 0.001 * _nonnegative_penalty(latent, meta)
        loss = update_objective(candidate, signal, list(signal), reference=signal, mode="balanced_tensor") + penalty
        value = float(loss.detach())
        history.append(value)
        if value < best:
            best = value
            best_step = step
            best_latent = latent.detach().clone()
        if step < 600:
            (gradient,) = torch.autograd.grad(loss, latent)
            optimizer.zero_grad()
            latent.grad = gradient
            optimizer.step()
    reconstruction = _decode_harddiff(best_latent, meta).detach()
    aligned = _align_for_evaluation(original, reconstruction, labels)
    delta_full = simulate(model, criterion, reconstruction, labels, batches, rng)
    delta = p16._filter_trainable(delta_full, trainable)
    candidate = candidate_fn(delta)
    mse = clean_mse(candidate, signal)
    metrics = score(original, aligned, labels, meta, folder / f"{method}.csv")
    prior = _align_for_evaluation(original, _decode_harddiff(initial, meta), labels)
    prior_metrics = score(original, prior, labels, meta, folder / "prior.csv")
    artifact = {
        "method": method,
        "cell": "priority22_corrected_positive_control",
        "group_id": job["group"],
        "restart": job["restart"],
        "source_ids": job["source_ids"],
        "original": original,
        "labels": labels,
        "initial": initial,
        "reconstruction": reconstruction,
        "aligned": aligned,
        "best_objective": best,
        "best_step": best_step,
        "history": history,
        "clean_vector_mse": mse,
        "feature_metrics": metrics,
        "prior_metrics": prior_metrics,
        "vector_scope": "trainable_parameters_only",
        "attacker_equivalence_contract": "matches experiments/priority16_v1_medium_dna_vs_dp_probe.py::_optimize_branch init=standard lr=0.1 steps=601 penalty=0.001",
    }
    torch.save(artifact, folder / f"{method}.pt")
    return artifact


def prepare_context(targets: list[dict[str, Any]], group_id: int):
    group = targets[group_id]
    distribution = _feature_distribution()
    local_seed = derive_seed(RUN_SEED, "priority14-local", CELL, group_id)
    model, criterion, original, labels, batches, rng, observed_full = p16.capture_paysim(group, local_seed, 4)
    trainable = p16._trainable_names(model)
    observed = p16._filter_trainable(observed_full, trainable)
    return group, distribution, model, criterion, original, labels, batches, rng, trainable, observed


def equivalence_check(output: Path, targets: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for selected_row in selected[:2]:
        group_id = int(selected_row["group_id"])
        restart = int(selected_row["dna_restart"])
        group, distribution, model, criterion, original, labels, batches, rng, trainable, observed = prepare_context(targets, group_id)
        cell = p16.CELLS[CELL]
        signal, payload = p16._dna_payload(observed, cell, group_id, restart)
        job = {"seed": RUN_SEED, "cell": CELL, "group": group_id, "restart": restart, "source_ids": group["source_ids"]}
        artifact = optimize_signal(
            job=job,
            method="dna",
            signal=signal,
            candidate_fn=lambda delta, payload=payload: p16._candidate_dna(delta, payload),
            model=model,
            criterion=criterion,
            original=original,
            labels=labels,
            batches=batches,
            rng=rng,
            trainable=trainable,
            distribution=distribution,
            folder=output / "equivalence" / f"group_{group_id}" / f"restart_{restart}",
        )
        saved_path = P16_RUN / CELL / f"group_{group_id}" / f"restart_{restart}" / "dna.pt"
        saved = torch.load(saved_path, weights_only=False, map_location="cpu")
        rows.append(
            {
                "group_id": group_id,
                "restart": restart,
                "saved_clean_vector_mse": float(saved["clean_vector_mse"]),
                "reproduced_clean_vector_mse": float(artifact["clean_vector_mse"]),
                "bit_exact": float(saved["clean_vector_mse"]) == float(artifact["clean_vector_mse"]),
                "saved_path": str(saved_path.relative_to(ROOT)),
                "reproduced_path": str((output / "equivalence" / f"group_{group_id}" / f"restart_{restart}" / "dna.pt").relative_to(ROOT)),
            }
        )
    passed = all(row["bit_exact"] for row in rows)
    result = {"passed": passed, "rows": rows}
    dump(output / "equivalence_check.json", result)
    if not passed:
        raise RuntimeError(f"equivalence check failed: {rows}")
    return result


def corrected_positive_control(output: Path) -> dict[str, Any]:
    if sha256(TARGET) != TARGET_SHA:
        raise RuntimeError("Priority 16 target hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    targets = torch.load(TARGET, weights_only=False, map_location="cpu")
    p16_report = json.loads(P16_REPORT.read_text())
    selected = p16_report["summary"]["selected"]
    eq = equivalence_check(output, targets, selected)
    rows = []
    started = time.perf_counter()
    for group_id, group in enumerate(targets):
        group, distribution, model, criterion, original, labels, batches, rng, trainable, observed = prepare_context(targets, group_id)
        zero_signal = {key: torch.zeros_like(value) for key, value in observed.items()}
        branch_results = {}
        for method, signal in (("none", observed), ("zero_update", zero_signal)):
            restart_results = []
            for restart in range(4):
                job = {"seed": RUN_SEED, "cell": CELL, "group": group_id, "restart": restart, "source_ids": group["source_ids"]}
                artifact = optimize_signal(
                    job=job,
                    method=method,
                    signal=signal,
                    candidate_fn=lambda delta: delta,
                    model=model,
                    criterion=criterion,
                    original=original,
                    labels=labels,
                    batches=batches,
                    rng=rng,
                    trainable=trainable,
                    distribution=distribution,
                    folder=output / "branches" / method / f"group_{group_id}" / f"restart_{restart}",
                )
                restart_results.append(
                    {
                        "restart": restart,
                        "objective": float(artifact["best_objective"]),
                        "input_mse": float(artifact["feature_metrics"]["mean_mse"]),
                        "prior_input_mse": float(artifact["prior_metrics"]["mean_mse"]),
                        "artifact": str((output / "branches" / method / f"group_{group_id}" / f"restart_{restart}" / f"{method}.pt").relative_to(ROOT)),
                    }
                )
            branch_results[method] = min(restart_results, key=lambda row: row["objective"])
        none = branch_results["none"]
        zero = branch_results["zero_update"]
        rows.append(
            {
                "group_id": group_id,
                "source_ids": json.dumps(group["source_ids"]),
                "none_restart": none["restart"],
                "zero_restart": zero["restart"],
                "none_objective": none["objective"],
                "zero_objective": zero["objective"],
                "none_input_mse": none["input_mse"],
                "none_prior_input_mse": none["prior_input_mse"],
                "zero_input_mse": zero["input_mse"],
                "zero_prior_input_mse": zero["prior_input_mse"],
                "prior_minus_none": none["prior_input_mse"] - none["input_mse"],
                "zero_minus_none": zero["input_mse"] - none["input_mse"],
                "none_artifact": none["artifact"],
                "zero_artifact": zero["artifact"],
            }
        )
        print(json.dumps({"group": group_id, "status": "done"}), flush=True)
    csv_path = output / "priority22_corrected_positive_control_per_target.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    prior_values = [float(row["prior_minus_none"]) for row in rows]
    zero_values = [float(row["zero_minus_none"]) for row in rows]
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": AMENDMENT,
        "target": str(TARGET.relative_to(ROOT)),
        "target_sha256": sha256(TARGET),
        "p16_report": str(P16_REPORT.relative_to(ROOT)),
        "n": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "equivalence_check": eq,
        "tie_band": TIE_BAND,
        "none_input_mse_summary": summary([float(row["none_input_mse"]) for row in rows]),
        "prior_input_mse_summary": summary([float(row["none_prior_input_mse"]) for row in rows]),
        "zero_input_mse_summary": summary([float(row["zero_input_mse"]) for row in rows]),
        "none_beats_prior": sign_counts(prior_values, TIE_BAND),
        "none_beats_zero": sign_counts(zero_values, TIE_BAND),
        "csv": str(csv_path.relative_to(ROOT)),
        "csv_sha256": sha256(csv_path),
    }
    dump(output / "priority22_corrected_positive_control_summary.json", result)
    return result


def audit_cifar() -> dict[str, Any]:
    report_path = ROOT / "artifacts/priority7_image_domain/confirmatory_v2_cosine_gate_20260917/priority7_image_gate_report.json"
    payload = json.loads(report_path.read_text())
    cell = payload["cells_summary"]["v2_ratio0p95_eta0p01__GEN_COSINE_TV"]
    selected = cell["selected"]
    return {
        "report": str(report_path.relative_to(ROOT)),
        "report_sha256": sha256(report_path),
        "metric_space": "input-image space image_mse; _score computes MSE/PSNR/SSIM and _evaluate gates on baseline_mse-prior_mse and baseline_mse-zero_mse",
        "win_formula_code": [
            "experiments/run_priority7_image_domain_gate.py:370-379",
            "experiments/run_priority7_image_domain_gate.py:387-430",
        ],
        "attacker_objective": "GEN_COSINE_TV minimizes global cosine in defended-update sketch space; selected restart is minimum objective",
        "attacker_objective_code": [
            "experiments/run_priority7_image_domain_gate.py:210-218",
            "experiments/run_priority7_image_domain_gate.py:323-346",
            "experiments/run_priority7_image_domain_gate.py:391-394",
        ],
        "attacker_knowledge": "Level 2 / seed-known for v2: candidate uses true v2 base seed plus observed sketch payload containing tensor_index and quantization_delta; sampled/sign pattern derives from fixed seed.",
        "attacker_knowledge_code": [
            "experiments/run_priority7_image_domain_gate.py:135-153",
            "experiments/run_phase4_dna_v2_sketch_space_attack.py:107-147",
        ],
        "controls": "Prior is sigmoid(random initial_logits) using the same init family as the attack restarts. Zero-update starts from the same initial logits per restart and optimizes against zero defended-signal. _evaluate averages prior MSE over baseline restarts, rather than using only the selected baseline restart's prior.",
        "controls_code": [
            "experiments/run_priority7_image_domain_gate.py:316-325",
            "experiments/run_priority7_image_domain_gate.py:391-396",
        ],
        "mechanical_floor_or_asymmetry": "No DP-style branch noise floor exists. Baseline and zero controls use the same candidate defended transform; the main asymmetry is that v2 is evaluated seed-known.",
        "gate": cell["gate"],
        "selected_summary": {
            "baseline_mse": summary([float(row["baseline_mse"]) for row in selected]),
            "prior_mse": summary([float(row["prior_mse"]) for row in selected]),
            "zero_mse": summary([float(row["zero_mse"]) for row in selected]),
            "baseline_psnr": summary([float(row["baseline_psnr"]) for row in selected]),
        },
    }


def tabular_gate_audit() -> list[dict[str, Any]]:
    return [
        {
            "family": "Priority 6/9 PaySim GEN_IDLG_STYLE/GEN_COSINE_TV attacker-vs-control gates",
            "examples": "Priority 6 PaySim v1-medium; Priority 9 v1-stronger/v1-conservative development/confirmatory",
            "metric_space": "input-space PaySim fraud-class mean MSE (`classes['1']['mean_mse']`)",
            "prior_init_mode": "standard hard-diff latent init",
            "attacker_init_lr": "same initial tensor as Prior; Adam lr=0.1; 601 evaluations",
            "same_init_as_prior": True,
            "code": "experiments/run_priority6_sota_style_attackers.py:356-423 and :522-558",
            "flag": "valid same-init gate design, but historical results may still be affected by other previously documented contamination unless rerun under corrected-vector pipeline",
        },
        {
            "family": "Priority 6 IEEE-CIS GEN_COSINE_TV attacker-vs-control gates",
            "examples": "Priority 6 IEEE-CIS v2 ratio0.95/eta0.01 confirmatory",
            "metric_space": "input-space IEEE feature mean MSE",
            "prior_init_mode": "IEEE numeric/categorical/label latent init from `init_ieee`",
            "attacker_init_lr": "same numeric/categorical/label initial state as Prior; Adam lr=0.05; 301 evaluations",
            "same_init_as_prior": True,
            "code": "experiments/run_priority6_sota_style_attackers.py:427-515 and :522-558",
            "flag": "valid same-init gate design, but historical result was later questioned by corrected-vector development gate; interpret with its own reports",
        },
        {
            "family": "Legacy Phase-4/RQ1 variant confirmatory gates",
            "examples": "RQ1 primary, Group 2/3, Priority 2 clean confirmatory branch gates",
            "metric_space": "PaySim fraud-class MSE / reconstruction metrics from selected artifacts",
            "prior_init_mode": "standard hard-diff init in command wrappers; branch scripts compute Prior from the same initial tensor",
            "attacker_init_lr": "same initial tensor as Prior; branch-specific Adam lr/iterations from config",
            "same_init_as_prior": True,
            "code": "experiments/run_phase4_harddiff_reparam_for_misselected.py:176-215; experiments/run_phase4_dna_level1_forward_attack.py:118-160; experiments/run_phase4_simple_defense_attack.py:339-378; experiments/analyze_rq1_variant_confirmatory.py:41-68",
            "flag": "same-init Prior design; many legacy gates are contaminated by the separate BN-buffer issue documented in Priority 11",
        },
        {
            "family": "Priority 14/16/20 DP-vs-transform head-to-head",
            "examples": "v1 stronger/medium/v2 DP-vs-DNA",
            "metric_space": "not attacker-vs-Prior/Zero gate; original D was update-space defended residual",
            "prior_init_mode": "not applicable for published D; P21 re-analysis compared saved branch reconstructions against their own initial tensors",
            "attacker_init_lr": "standard init; Adam lr=0.1; 601 evaluations",
            "same_init_as_prior": None,
            "code": "experiments/priority16_v1_medium_dna_vs_dp_probe.py:137-175; reports/priority21_head_to_head_metric_validity_report_20260928.md",
            "flag": "not a valid Prior/Zero gate; P21 showed update-space D dominated by DP floor and input-space comparison does not support DP wins",
        },
        {
            "family": "Priority 21 positive control (invalidated)",
            "examples": "none vs Prior/Zero from P21 Step 4",
            "metric_space": "input-space mean MSE",
            "prior_init_mode": "plausible init, not matching P16",
            "attacker_init_lr": "plausible init + Adam lr=0.08",
            "same_init_as_prior": True,
            "code": "experiments/priority21_metric_validity.py prior invalid version",
            "flag": "invalid because it does not match P16 init/lr; superseded by Priority 22 corrected positive control",
        },
    ]


def audit(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cifar": audit_cifar(),
        "tabular_gates": tabular_gate_audit(),
    }
    dump(output / "priority22_audit_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("positive-control")
    pc.add_argument("--output-dir", type=Path, required=True)
    au = sub.add_parser("audit")
    au.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.cmd == "positive-control":
        print(json.dumps(corrected_positive_control(args.output_dir.resolve()), indent=2, allow_nan=False))
    else:
        print(json.dumps(audit(args.output_dir.resolve()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
