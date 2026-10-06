from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attacks.local_update import simulate  # noqa: E402
from experiments.phase3_bounded_validation import _align_for_evaluation, update_objective  # noqa: E402
from experiments.run_phase3_full_client import score  # noqa: E402
from experiments.run_phase4_dna_level1_forward_attack import (  # noqa: E402
    _apply_surrogate_realization_torch,
    _surrogate_plan_from_state,
    _transmit_observed,
)
from experiments.run_phase4_dna_v2_iht_attack import _capture_for_iht as capture_paysim  # noqa: E402
from experiments.run_phase4_dna_v2_sketch_space_attack import (  # noqa: E402
    _candidate_sketches,
    _observed_sketches,
)
from experiments.run_phase4_harddiff_reparam_for_misselected import (  # noqa: E402
    _decode_harddiff,
    _feature_distribution,
    _initial,
    _nonnegative_penalty,
)
from experiments.run_phase4_simple_defense_attack import (  # noqa: E402
    _apply_candidate_defense,
    _apply_observed_defense,
    _plan as simple_defense_plan,
)
from privacy.seed_manager import derive_seed  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def binom_tail(wins: int, total: int) -> float:
    if total <= 0:
        return 1.0
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / (2**total)


def sign_counts(values: list[float], tie_band: float = 0.0) -> dict[str, Any]:
    wins = sum(1 for v in values if v > tie_band)
    losses = sum(1 for v in values if v < -tie_band)
    ties = len(values) - wins - losses
    n = wins + losses
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "non_tied": n,
        "one_sided_p": binom_tail(wins, n),
    }


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "sd": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


def trainable_names(model: torch.nn.Module) -> list[str]:
    return [name for name, _ in model.named_parameters()]


def filter_trainable(update: dict[str, torch.Tensor], names: list[str]) -> dict[str, torch.Tensor]:
    return {name: update[name] for name in names}


def mse(candidate: dict[str, torch.Tensor], signal: dict[str, torch.Tensor]) -> float:
    flat = torch.cat([(candidate[key] - signal[key]).reshape(-1).detach().cpu() for key in signal])
    return float(flat.double().square().mean().item())


def weighted_type_accuracy(metrics: dict[str, Any]) -> float | None:
    classes = metrics.get("classes", {})
    total = sum(int(v.get("n", 0)) for v in classes.values())
    if total == 0:
        return None
    return float(sum(int(v.get("n", 0)) * float(v.get("type_accuracy", 0.0)) for v in classes.values()) / total)


def fraud_mean_mse(metrics: dict[str, Any]) -> float | None:
    fraud = metrics.get("classes", {}).get("1")
    return None if fraud is None else float(fraud.get("mean_mse"))


@dataclass(frozen=True)
class DnaConfig:
    kind: str
    block_size: int | None = None
    mix_ratio: float | None = None
    keep_ratio: float | None = None
    shrink_factor: float | None = None
    seed: int = 681958327
    compression_ratio: float | None = None
    eta: float | None = None


@dataclass(frozen=True)
class DpConfig:
    clip_norm: float
    noise_multiplier: float
    mc_noise_samples: int = 100
    defense_seed: int = 314159265


@dataclass(frozen=True)
class Cell:
    name: str
    run_dir: Path
    report_path: Path
    target_path: Path
    seed: int
    dna: DnaConfig
    dp: DpConfig
    tie_band: float
    kind: str = "priority14"
    checkpoint_id: int | None = None
    checkpoint_path: Path | None = None


def cells() -> list[Cell]:
    return [
        Cell(
            name="priority14_v1_stronger_vs_dp_0p0004",
            run_dir=ROOT / "artifacts/priority14_fresh_probe/v1_stronger_confirmatory_20260921",
            report_path=ROOT / "artifacts/priority14_fresh_probe/v1_stronger_confirmatory_20260921/priority14_probe_report.json",
            target_path=ROOT / "artifacts/priority14_fresh_probe/v1_stronger_confirmatory_targets_20260921/paysim_priority14_v1_stronger_confirmatory_targets.pt",
            seed=2026092128,
            dna=DnaConfig(kind="v1", block_size=256, mix_ratio=0.12, keep_ratio=0.82, shrink_factor=0.35),
            dp=DpConfig(clip_norm=100.0, noise_multiplier=0.0004),
            tie_band=0.0390625,
        ),
        Cell(
            name="priority14_v2_ratio0p95_vs_dp_0p00105",
            run_dir=ROOT / "artifacts/priority14_fresh_probe/v2_ratio0p95_confirmatory_20260921",
            report_path=ROOT / "artifacts/priority14_fresh_probe/v2_ratio0p95_confirmatory_20260921/priority14_probe_report.json",
            target_path=ROOT / "artifacts/priority14_fresh_probe/v2_ratio0p95_confirmatory_targets_20260921/paysim_priority14_v2_ratio0p95_confirmatory_targets.pt",
            seed=2026092129,
            dna=DnaConfig(kind="v2", compression_ratio=0.95, eta=0.01, seed=20260916),
            dp=DpConfig(clip_norm=100.0, noise_multiplier=0.00105),
            tie_band=0.01953125,
        ),
        Cell(
            name="priority16_v1_medium_vs_dp_0p000315",
            run_dir=ROOT / "artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_run_20260927",
            report_path=ROOT / "artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_run_20260927/priority14_probe_report.json",
            target_path=ROOT / "artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_targets_20260927/paysim_priority16_v1_medium_confirmatory_targets.pt",
            seed=2026092706,
            dna=DnaConfig(kind="v1", block_size=256, mix_ratio=0.10, keep_ratio=0.85, shrink_factor=0.40),
            dp=DpConfig(clip_norm=100.0, noise_multiplier=0.000315),
            tie_band=0.0390625,
        ),
    ]


def priority20_cells() -> list[Cell]:
    base = ROOT / "artifacts/priority20_multi_checkpoint"
    out = []
    for ckpt in range(5):
        out.append(
            Cell(
                name=f"priority20_ckpt{ckpt}_v1_medium_vs_dp_0p000315",
                run_dir=base / f"ckpt_{ckpt}_run_20260927",
                report_path=base / f"ckpt_{ckpt}_run_20260927/priority20_probe_report.json",
                target_path=base / f"ckpt_{ckpt}_targets_20260927/paysim_priority20_ckpt_{ckpt}_targets.pt",
                seed=2026092821 + ckpt,
                dna=DnaConfig(kind="v1", block_size=256, mix_ratio=0.10, keep_ratio=0.85, shrink_factor=0.40),
                dp=DpConfig(clip_norm=100.0, noise_multiplier=0.000315),
                tie_band=0.0390625,
                kind="priority20",
                checkpoint_id=ckpt,
                checkpoint_path=base / f"checkpoints_20260927/ckpt_{ckpt}/pre_local.pt",
            )
        )
    return out


def dna_args(config: DnaConfig) -> Any:
    if config.kind == "v1":
        return SimpleNamespace(
            block_size=config.block_size,
            mix_ratio=config.mix_ratio,
            keep_ratio=config.keep_ratio,
            shrink_factor=config.shrink_factor,
            seed=config.seed,
        )
    return SimpleNamespace(
        compression_ratio=config.compression_ratio,
        quantization_eta=config.eta,
        seed=int(config.seed),
        v2_base_seed=int(config.seed),
        ste_quantization=True,
    )


def dp_args(config: DpConfig) -> Any:
    return SimpleNamespace(
        clip_factor=0.95,
        clip_norm=config.clip_norm,
        noise_multiplier=config.noise_multiplier,
        defense_seed=config.defense_seed,
        mc_noise_samples=config.mc_noise_samples,
    )


def dna_payload(observed: dict[str, torch.Tensor], config: DnaConfig, group: int, restart: int) -> tuple[dict[str, torch.Tensor], dict[str, Any], Any]:
    args = dna_args(config)
    if config.kind == "v1":
        signal = _transmit_observed(observed, args)
        plan = _surrogate_plan_from_state(observed, args, group, restart)
        return signal, plan, args
    plan = _observed_sketches(observed, args, group)
    return {key: value["sketch"] for key, value in plan.items()}, plan, args


def candidate_dna(delta: dict[str, torch.Tensor], plan: dict[str, Any], args: Any, kind: str) -> dict[str, torch.Tensor]:
    if kind == "v1":
        return _apply_surrogate_realization_torch(delta, plan)
    return _candidate_sketches(delta, plan, args)


def dp_payload(observed: dict[str, torch.Tensor], config: DpConfig, group: int) -> tuple[dict[str, torch.Tensor], dict[str, Any], Any]:
    args = dp_args(config)
    keys = list(observed)
    plan = simple_defense_plan("clipping_noise_mc", observed, keys, group, args)
    signal = _apply_observed_defense(observed, keys, "clipping_noise_mc", plan)
    return signal, plan, args


def candidate_dp(delta: dict[str, torch.Tensor], plan: dict[str, Any]) -> dict[str, torch.Tensor]:
    return _apply_candidate_defense(delta, list(delta), "clipping_noise_mc", plan)


def load_metadata() -> SimpleNamespace:
    return SimpleNamespace(**json.loads((ROOT / "artifacts/phase3/full_20260908T143837588533Z/preprocessing.json").read_text()))


def capture_for_cell(cell: Cell, group_id: int, group: dict[str, Any]) -> tuple[Any, Any, torch.Tensor, torch.Tensor, Any, Any, dict[str, torch.Tensor]]:
    if cell.kind == "priority20":
        from experiments.priority20_multi_checkpoint_head_to_head import _capture_with_checkpoint

        local_seed = derive_seed(cell.seed, "priority20-local", int(cell.checkpoint_id), group_id)
        return _capture_with_checkpoint(group, local_seed, 4, cell.checkpoint_path)
    local_seed = derive_seed(cell.seed, "priority14-local", cell.report_cell_name if hasattr(cell, "report_cell_name") else cell_json_cell(cell), group_id)
    return capture_paysim(group, local_seed, 4)


def cell_json_cell(cell: Cell) -> str:
    report = json.loads(cell.report_path.read_text())
    return str(report["cell"])


def selected_rows(cell: Cell) -> list[dict[str, Any]]:
    return json.loads(cell.report_path.read_text())["summary"]["selected"]


def branch_path(cell: Cell, branch: str, group: int, restart: int) -> Path:
    report_cell = cell_json_cell(cell)
    return cell.run_dir / report_cell / f"group_{group}" / f"restart_{restart}" / f"{branch}.pt"


def reanalysis(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_cells = cells() + [cell for cell in priority20_cells() if cell.report_path.exists()]
    results = {"created_at": datetime.now(timezone.utc).isoformat(), "cells": []}
    for cell in all_cells:
        targets = torch.load(cell.target_path, weights_only=False, map_location="cpu")
        selected = selected_rows(cell)
        rows = []
        d_in_values = []
        d_in_oracle_values = []
        d_update_values = []
        corrected_values = []
        dp_minus_floor = []
        dna_minus_floor = []
        explained = []
        prior_dna = []
        prior_dp = []
        for row in selected:
            group = int(row["group_id"])
            target = targets[group]
            model, criterion, x, y, batches, rng, observed_full = capture_for_cell(cell, group, target)
            names = trainable_names(model)
            observed = filter_trainable(observed_full, names)

            dna_restart = int(row["dna_restart"])
            dp_restart = int(row["dp_restart"])
            dna_signal, dna_plan, dna_a = dna_payload(observed, cell.dna, group, dna_restart)
            dp_signal, dp_plan, _ = dp_payload(observed, cell.dp, group)
            floor_dna = mse(candidate_dna(observed, dna_plan, dna_a, cell.dna.kind), dna_signal)
            floor_dp = mse(candidate_dp(observed, dp_plan), dp_signal)
            dna_mse = float(row["dna_mse"])
            dp_mse = float(row["dp_mse"])
            d_update = dna_mse - dp_mse
            d_corr = (dna_mse - floor_dna) - (dp_mse - floor_dp)
            d_update_values.append(d_update)
            corrected_values.append(d_corr)
            dp_minus_floor.append(dp_mse - floor_dp)
            dna_minus_floor.append(dna_mse - floor_dna)
            explained.append(((floor_dna - floor_dp) / d_update) if d_update != 0 else None)

            dna_art = torch.load(branch_path(cell, "dna", group, dna_restart), weights_only=False, map_location="cpu")
            dp_art = torch.load(branch_path(cell, "dp", group, dp_restart), weights_only=False, map_location="cpu")
            dna_f = float(dna_art["feature_metrics"]["mean_mse"])
            dp_f = float(dp_art["feature_metrics"]["mean_mse"])
            d_in = dna_f - dp_f
            d_in_values.append(d_in)

            # Branch-specific priors, from the selected restarts' initial tensors.
            metadata = load_metadata()
            dna_prior_aligned = _align_for_evaluation(dna_art["original"], _decode_harddiff(dna_art["initial"], metadata), dna_art["labels"])
            dp_prior_aligned = _align_for_evaluation(dp_art["original"], _decode_harddiff(dp_art["initial"], metadata), dp_art["labels"])
            prior_dna.append(float((dna_art["original"].double() - dna_prior_aligned.double()).square().mean().item()))
            prior_dp.append(float((dp_art["original"].double() - dp_prior_aligned.double()).square().mean().item()))

            # Oracle: select lowest input-space MSE per branch over saved restarts.
            dna_oracle = []
            dp_oracle = []
            for restart in range(4):
                da = torch.load(branch_path(cell, "dna", group, restart), weights_only=False, map_location="cpu")
                pa = torch.load(branch_path(cell, "dp", group, restart), weights_only=False, map_location="cpu")
                dna_oracle.append(float(da["feature_metrics"]["mean_mse"]))
                dp_oracle.append(float(pa["feature_metrics"]["mean_mse"]))
            d_oracle = min(dna_oracle) - min(dp_oracle)
            d_in_oracle_values.append(d_oracle)
            rows.append(
                {
                    "group_id": group,
                    "dna_restart": dna_restart,
                    "dp_restart": dp_restart,
                    "dna_update_mse": dna_mse,
                    "dp_update_mse": dp_mse,
                    "D_update": d_update,
                    "floor_dna": floor_dna,
                    "floor_dp": floor_dp,
                    "dna_mse_minus_floor": dna_mse - floor_dna,
                    "dp_mse_minus_floor": dp_mse - floor_dp,
                    "D_floor_corrected": d_corr,
                    "floor_explained_fraction": ((floor_dna - floor_dp) / d_update) if d_update != 0 else None,
                    "dna_feature_mse": dna_f,
                    "dp_feature_mse": dp_f,
                    "D_input": d_in,
                    "D_input_oracle": d_oracle,
                    "dna_type_accuracy": weighted_type_accuracy(dna_art["feature_metrics"]),
                    "dp_type_accuracy": weighted_type_accuracy(dp_art["feature_metrics"]),
                    "dna_fraud_mean_mse": fraud_mean_mse(dna_art["feature_metrics"]),
                    "dp_fraud_mean_mse": fraud_mean_mse(dp_art["feature_metrics"]),
                    "dna_prior_input_mse": prior_dna[-1],
                    "dp_prior_input_mse": prior_dp[-1],
                }
            )
        csv_path = output_dir / f"{cell.name}_priority21_reanalysis.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        cell_summary = {
            "name": cell.name,
            "report_path": str(cell.report_path.relative_to(ROOT)),
            "target_path": str(cell.target_path.relative_to(ROOT)),
            "n": len(rows),
            "update_D_summary": summarize(d_update_values),
            "floor_dp_summary": summarize([r["floor_dp"] for r in rows]),
            "floor_dna_summary": summarize([r["floor_dna"] for r in rows]),
            "dp_mse_minus_floor_summary": summarize(dp_minus_floor),
            "dna_mse_minus_floor_summary": summarize(dna_minus_floor),
            "floor_explained_fraction_summary": summarize([v for v in explained if v is not None]),
            "floor_corrected_D_summary": summarize(corrected_values),
            "floor_corrected_sign_test_dp_wins_positive": sign_counts([-v for v in corrected_values], 0.0),
            "input_D_summary": summarize(d_in_values),
            "input_sign_test_dp_wins_positive": sign_counts(d_in_values, 0.0),
            "input_tie_banded_sign_test_dp_wins_positive": sign_counts(d_in_values, cell.tie_band),
            "input_oracle_D_summary": summarize(d_in_oracle_values),
            "input_oracle_sign_test_dp_wins_positive": sign_counts(d_in_oracle_values, 0.0),
            "input_oracle_tie_banded_sign_test_dp_wins_positive": sign_counts(d_in_oracle_values, cell.tie_band),
            "selected_dna_prior_input_mse_summary": summarize(prior_dna),
            "selected_dp_prior_input_mse_summary": summarize(prior_dp),
            "csv": str(csv_path.relative_to(ROOT)),
            "csv_sha256": sha256(csv_path),
        }
        results["cells"].append(cell_summary)
    json_path = output_dir / "priority21_reanalysis_summary.json"
    dump_json(json_path, results)
    results["summary_json"] = str(json_path.relative_to(ROOT))
    results["summary_json_sha256"] = sha256(json_path)
    return results


def positive_control(output_dir: Path, workers: int = 1) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    del workers  # Intentionally sequential for reproducibility/instrument validity.
    output_dir.mkdir(parents=True, exist_ok=True)
    cell = cells()[2]
    targets = torch.load(cell.target_path, weights_only=False, map_location="cpu")
    metadata = load_metadata()
    distribution = _feature_distribution()
    tie_band = cell.tie_band
    rows = []
    start = time.perf_counter()
    for group, target in enumerate(targets):
        model, criterion, x, y, batches, rng, observed_full = capture_for_cell(cell, group, target)
        names = trainable_names(model)
        observed = {key: value.detach() for key, value in filter_trainable(observed_full, names).items()}
        zero_signal = {key: torch.zeros_like(value) for key, value in observed.items()}
        branch_outputs = {}
        for method, signal in (("none", observed), ("zero_update", zero_signal)):
            restart_rows = []
            for restart in range(4):
                folder = output_dir / "branches" / method / f"group_{group}" / f"restart_{restart}"
                folder.mkdir(parents=True, exist_ok=True)
                initial = _initial(
                    x.shape,
                    derive_seed(cell.seed, "priority21-initial", method, group, restart),
                    "plausible",
                    distribution,
                )
                latent = initial.clone().requires_grad_(True)
                optimizer = torch.optim.Adam([latent], lr=0.08)
                best = float("inf")
                best_step = 0
                best_latent = initial.clone()
                history = []
                for step in range(601):
                    decoded = _decode_harddiff(latent, metadata)
                    delta = simulate(model, criterion, decoded, y, batches, rng, create_graph=True)
                    delta = filter_trainable(delta, names)
                    loss = update_objective(delta, signal, list(signal), reference=signal, mode="balanced_tensor")
                    loss = loss + 0.001 * _nonnegative_penalty(latent, metadata)
                    value = float(loss.detach().cpu())
                    history.append(value)
                    if value < best:
                        best = value
                        best_step = step
                        best_latent = latent.detach().clone()
                    if step < 600 and torch.isfinite(loss):
                        (gradient,) = torch.autograd.grad(loss, latent)
                        optimizer.zero_grad()
                        latent.grad = gradient
                        optimizer.step()
                reconstruction = _decode_harddiff(best_latent, metadata)
                aligned = _align_for_evaluation(x, reconstruction, y)
                metrics = score(x, aligned, y, metadata, folder / "features.csv")
                prior = _align_for_evaluation(x, _decode_harddiff(initial, metadata), y)
                prior_metrics = score(x, prior, y, metadata, folder / "prior.csv")
                candidate_update = simulate(model, criterion, reconstruction, y, batches, rng, create_graph=False)
                candidate_update = filter_trainable(candidate_update, names)
                clean_vector_mse = mse(candidate_update, signal)
                artifact = {
                    "method": method,
                    "cell": "priority21_positive_control_priority16_targets",
                    "group_id": group,
                    "restart": restart,
                    "source_ids": target["source_ids"],
                    "original": x.detach().cpu(),
                    "labels": y.detach().cpu(),
                    "initial": initial.detach().cpu(),
                    "reconstruction": reconstruction.detach().cpu(),
                    "aligned": aligned.detach().cpu(),
                    "best_objective": best,
                    "best_step": best_step,
                    "history": history,
                    "clean_vector_mse": clean_vector_mse,
                    "feature_metrics": metrics,
                    "prior_metrics": prior_metrics,
                    "vector_scope": "trainable_parameters_only",
                }
                torch.save(artifact, folder / f"{method}.pt")
                restart_rows.append(
                    {
                        "restart": restart,
                        "objective": best,
                        "input_mse": float(metrics["mean_mse"]),
                        "prior_input_mse": float(prior_metrics["mean_mse"]),
                        "path": str((folder / f"{method}.pt").relative_to(ROOT)),
                    }
                )
            branch_outputs[method] = min(restart_rows, key=lambda r: r["objective"])
        none = branch_outputs["none"]
        zero = branch_outputs["zero_update"]
        rows.append(
            {
                "group_id": group,
                "source_ids": json.dumps(target["source_ids"]),
                "none_restart": none["restart"],
                "zero_restart": zero["restart"],
                "none_input_mse": none["input_mse"],
                "prior_input_mse": none["prior_input_mse"],
                "zero_input_mse": zero["input_mse"],
                "none_minus_prior": none["input_mse"] - none["prior_input_mse"],
                "none_minus_zero": none["input_mse"] - zero["input_mse"],
                "none_artifact": none["path"],
                "zero_artifact": zero["path"],
            }
        )
        print(f"[priority21 positive] group {group+1}/{len(targets)} done", flush=True)
    csv_path = output_dir / "priority21_positive_control_per_target.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    none_vs_prior = [float(r["prior_input_mse"]) - float(r["none_input_mse"]) for r in rows]
    none_vs_zero = [float(r["zero_input_mse"]) - float(r["none_input_mse"]) for r in rows]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": "protocols/amendments/2026-09-28_priority21_positive_control_protocol.md",
        "target_path": str(cell.target_path.relative_to(ROOT)),
        "target_sha256": sha256(cell.target_path),
        "n": len(rows),
        "elapsed_seconds": time.perf_counter() - start,
        "tie_band": tie_band,
        "none_input_mse_summary": summarize([float(r["none_input_mse"]) for r in rows]),
        "prior_input_mse_summary": summarize([float(r["prior_input_mse"]) for r in rows]),
        "zero_input_mse_summary": summarize([float(r["zero_input_mse"]) for r in rows]),
        "none_beats_prior_tie_banded": sign_counts(none_vs_prior, tie_band),
        "none_beats_zero_tie_banded": sign_counts(none_vs_zero, tie_band),
        "none_beats_prior_strict": sign_counts(none_vs_prior, 0.0),
        "none_beats_zero_strict": sign_counts(none_vs_zero, 0.0),
        "csv": str(csv_path.relative_to(ROOT)),
        "csv_sha256": sha256(csv_path),
    }
    json_path = output_dir / "priority21_positive_control_summary.json"
    dump_json(json_path, summary)
    summary["summary_json"] = str(json_path.relative_to(ROOT))
    summary["summary_json_sha256"] = sha256(json_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_re = sub.add_parser("reanalyze")
    p_re.add_argument("--output-dir", type=Path, required=True)
    p_pc = sub.add_parser("positive-control")
    p_pc.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.cmd == "reanalyze":
        print(json.dumps(reanalysis(args.output_dir), indent=2, allow_nan=False))
    elif args.cmd == "positive-control":
        print(json.dumps(positive_control(args.output_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
