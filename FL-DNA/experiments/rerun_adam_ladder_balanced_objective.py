"""Rerun the 4-record Adam confirmation with balanced per-tensor objective.

The targets are reused from adam_ladder_v2 so this run isolates the objective
change.  Old artifacts remain untouched.
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from attacks.local_update import simulate
from experiments import fraud_fl_common as common
from experiments.phase3_bounded_validation import _align_for_evaluation, update_objective
from experiments.run_phase3_adam_ladder import capture, metadata
from experiments.run_phase3_full_client import checksum, dump, score
from privacy.seed_manager import derive_seed


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/phase3_closure/adam_ladder_v2"


def _decode(latent):
    return torch.cat((latent[:, :8], latent[:, 8:].softmax(-1)), dim=1)


def _run_pair(folder, group, group_id, restart, run_seed, lr, iterations, batch_size):
    folder.mkdir(parents=True, exist_ok=False)
    model, criterion, x, y, batches, rng, observed = capture(
        group, derive_seed(run_seed, "local", group_id, batch_size), batch_size
    )
    keys = [key for key, value in observed.items() if value.is_floating_point()]
    initial = torch.randn(
        x.shape,
        generator=torch.Generator().manual_seed(derive_seed(run_seed, "initial", group_id, restart, batch_size)),
    )
    meta = metadata()
    prior = _align_for_evaluation(x, _decode(initial), y)
    prior_metrics = score(x, prior, y, meta, folder / "prior.csv")
    rows = []
    for method in ("baseline", "zero_update"):
        signal = observed if method == "baseline" else {key: torch.zeros_like(value) for key, value in observed.items()}
        latent = initial.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=lr)
        best = float("inf")
        best_step = 0
        best_latent = latent.detach().clone()
        history = []
        started = time.perf_counter()
        for step in range(iterations + 1):
            delta = simulate(model, criterion, _decode(latent), y, batches, rng)
            loss = update_objective(delta, signal, keys, reference=observed, mode="balanced_tensor")
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite objective at step {step}")
            value = float(loss.detach())
            history.append(value)
            if value < best:
                best = value
                best_step = step
                best_latent = latent.detach().clone()
            if step < iterations:
                gradient, = torch.autograd.grad(loss, latent)
                if not torch.isfinite(gradient).all():
                    raise RuntimeError(f"Non-finite input gradient at step {step}")
                optimizer.zero_grad()
                latent.grad = gradient
                optimizer.step()
        reconstruction = _decode(best_latent).detach()
        aligned = _align_for_evaluation(x, reconstruction, y)
        metrics = score(x, aligned, y, meta, folder / f"{method}.csv")
        artifact_path = folder / f"{method}.pt"
        torch.save(
            {
                "method": method,
                "group_id": group_id,
                "restart": restart,
                "source_ids": group["source_ids"],
                "original": x,
                "labels": y,
                "initial": initial,
                "reconstruction": reconstruction,
                "aligned": aligned,
                "best_objective": best,
                "best_step": best_step,
                "history": history,
                "objective_mode": "balanced_tensor",
                "observed": observed,
                "keys": keys,
            },
            artifact_path,
        )
        loaded = torch.load(artifact_path, weights_only=False)
        checked = update_objective(
            simulate(model, criterion, loaded["reconstruction"], y, batches, rng),
            signal,
            keys,
            reference=observed,
            mode="balanced_tensor",
        )
        torch.testing.assert_close(checked.detach(), torch.tensor(best), rtol=1e-5, atol=1e-10)
        rows.append(
            {
                "method": method,
                "group_id": group_id,
                "restart": restart,
                "objective": best,
                "best_step": best_step,
                "metrics": metrics,
                "prior": prior_metrics,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    dump(folder / "results.json", rows)
    return rows


def _sign_tail(wins, total):
    return sum(__import__("math").comb(total, k) for k in range(wins, total + 1)) / 2**total if total else 1.0


def _evaluate(rows):
    selected = []
    for group_id in sorted({row["group_id"] for row in rows}):
        baseline = min(
            (row for row in rows if row["group_id"] == group_id and row["method"] == "baseline"),
            key=lambda row: row["objective"],
        )
        zero = min(
            (row for row in rows if row["group_id"] == group_id and row["method"] == "zero_update"),
            key=lambda row: row["objective"],
        )
        prior = float(np.mean([row["prior"]["mean_mse"] for row in rows if row["group_id"] == group_id and row["method"] == "baseline"]))
        selected.append(
            {
                "group_id": group_id,
                "baseline_restart": baseline["restart"],
                "zero_restart": zero["restart"],
                "baseline_mse": baseline["metrics"]["mean_mse"],
                "prior_mse": prior,
                "zero_mse": zero["metrics"]["mean_mse"],
                "baseline_minus_prior": baseline["metrics"]["mean_mse"] - prior,
                "baseline_minus_zero": baseline["metrics"]["mean_mse"] - zero["metrics"]["mean_mse"],
            }
        )
    comparisons = {}
    for control in ("prior", "zero"):
        values = np.asarray([row[f"baseline_minus_{control}"] for row in selected])
        non_ties = values[values != 0]
        wins = int((non_ties < 0).sum())
        comparisons[control] = {
            "n": len(selected),
            "wins": wins,
            "non_ties": int(len(non_ties)),
            "mean_difference": float(values.mean()),
            "median_difference": float(np.median(values)),
            "one_sided_sign_p": _sign_tail(wins, len(non_ties)),
        }
        comparisons[control]["gate"] = bool(
            comparisons[control]["mean_difference"] < 0
            and comparisons[control]["median_difference"] < 0
            and comparisons[control]["one_sided_sign_p"] < 0.05
        )
    return comparisons, selected


def _write_selected(path, selected):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)


def _tensor_norm_report(group, run_seed, batch_size):
    _, _, _, _, _, _, observed = capture(group, derive_seed(run_seed, "local", 0, batch_size), batch_size)
    rows = []
    for key, value in observed.items():
        if value.is_floating_point():
            rows.append(
                {
                    "tensor": key,
                    "numel": int(value.numel()),
                    "norm": float(value.norm()),
                    "max_abs": float(value.abs().max()),
                    "mse_to_zero": float(value.square().mean()),
                    "is_running_buffer": bool("running_" in key),
                }
            )
    rows.sort(key=lambda row: row["norm"], reverse=True)
    param_norm = sum(row["norm"] ** 2 for row in rows if not row["is_running_buffer"]) ** 0.5
    buffer_norm = sum(row["norm"] ** 2 for row in rows if row["is_running_buffer"]) ** 0.5
    return {
        "rows": rows,
        "param_like_norm": param_norm,
        "running_buffer_norm": buffer_norm,
        "running_buffer_to_param_like_norm_ratio": buffer_norm / max(param_norm, 1e-30),
    }


def main():
    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    started = time.perf_counter()
    source_protocol = json.loads((SOURCE / "protocol_lock.json").read_text())
    source_frozen = json.loads((SOURCE / "frozen.json").read_text())
    groups = torch.load(SOURCE / "confirmation_targets.pt", weights_only=False)
    out = ROOT / "artifacts/phase3_closure" / datetime.now(timezone.utc).strftime("adam_ladder_balanced_objective_%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    torch.save(groups, out / "confirmation_targets.pt")
    protocol = {
        "source_run": str(SOURCE),
        "source_protocol_sha256": checksum(SOURCE / "protocol_lock.json"),
        "source_confirmation_targets_sha256": checksum(SOURCE / "confirmation_targets.pt"),
        "run_seed": source_protocol["run_seed"],
        "records_per_group": 4,
        "fraud_per_group": 1,
        "batch_size": 4,
        "confirmation_groups": len(groups),
        "iterations": source_frozen["iterations"],
        "restarts": source_frozen["restarts"],
        "attack_lr": source_frozen["lr"],
        "objective_mode": "balanced_tensor",
        "purpose": "Rerun old 4-record/1-step Adam confirmation targets after replacing flattened cosine/magnitude with equal per-tensor normalized objective.",
    }
    dump(out / "protocol_lock.json", protocol)
    norm_report = _tensor_norm_report(groups[0], source_protocol["run_seed"], protocol["batch_size"])
    dump(out / "observed_tensor_norms_group0.json", norm_report)
    with (out / "observed_tensor_norms_group0.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(norm_report["rows"][0]))
        writer.writeheader()
        writer.writerows(norm_report["rows"])
    records = []
    for group_id, group in enumerate(groups):
        for restart in range(source_frozen["restarts"]):
            records.extend(
                _run_pair(
                    out / "confirmation" / f"group_{group_id}" / f"restart_{restart}",
                    group,
                    group_id,
                    restart,
                    source_protocol["run_seed"],
                    source_frozen["lr"],
                    source_frozen["iterations"],
                    protocol["batch_size"],
                )
            )
    comparisons, selected = _evaluate(records)
    _write_selected(out / "confirmation_selected.csv", selected)
    dump(out / "confirmation_records.json", records)
    report = {
        "status": "BALANCED_OBJECTIVE_GATE_PASSED" if comparisons["prior"]["gate"] and comparisons["zero"]["gate"] else "BALANCED_OBJECTIVE_GATE_FAILED",
        "protocol": protocol,
        "tensor_norm_diagnostic": {
            "running_buffer_norm": norm_report["running_buffer_norm"],
            "param_like_norm": norm_report["param_like_norm"],
            "running_buffer_to_param_like_norm_ratio": norm_report["running_buffer_to_param_like_norm_ratio"],
            "largest_tensor": norm_report["rows"][0],
        },
        "confirmation": comparisons,
        "selected_confirmation": selected,
        "elapsed_seconds": time.perf_counter() - started,
    }
    dump(out / "balanced_objective_report.json", report)
    print(json.dumps({"output": str(out), "status": report["status"], "confirmation": comparisons, "tensor_norm_diagnostic": report["tensor_norm_diagnostic"]}, indent=2))


if __name__ == "__main__":
    main()
