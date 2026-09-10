"""Post-hoc diagnostics for attack gates that repeatedly miss controls.

This script does not change candidate selection or rerun attacks. It reads the
locked confirmation artifacts, splits reconstruction error by class, and checks
which update tensors dominate the matching objective for selected baseline
reconstructions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from attacks.local_update import simulate
from experiments.run_phase3_adam_ladder import capture
from experiments.run_phase3_full_client import dump
from privacy.seed_manager import derive_seed


def _load_json(path: Path):
    return json.loads(path.read_text())


def _report_path(run: Path):
    for name in (
        "attack_scale_redevelopment_report.json",
        "attack_redevelopment_report.json",
        "balanced_objective_report.json",
        "class_decomposed_attack_report.json",
    ):
        path = run / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No attack redevelopment report found in {run}")


def _select(rows, group_id, method):
    return min(
        (row for row in rows if row["group_id"] == group_id and row["method"] == method),
        key=lambda row: row["objective"],
    )


def _mean_prior(rows, group_id, label):
    values = [
        row["prior"]["classes"][label]["mean_mse"]
        for row in rows
        if row["group_id"] == group_id and row["method"] == "baseline"
    ]
    return float(np.mean(values))


def _class_diagnostics(rows):
    selected = []
    for group_id in sorted({row["group_id"] for row in rows}):
        baseline = _select(rows, group_id, "baseline")
        zero = _select(rows, group_id, "zero_update")
        for label in ("0", "1"):
            baseline_mse = baseline["metrics"]["classes"][label]["mean_mse"]
            prior_mse = _mean_prior(rows, group_id, label)
            zero_mse = zero["metrics"]["classes"][label]["mean_mse"]
            selected.append(
                {
                    "group_id": group_id,
                    "label": label,
                    "label_name": "non_fraud" if label == "0" else "fraud",
                    "baseline_restart": baseline["restart"],
                    "zero_restart": zero["restart"],
                    "baseline_mse": baseline_mse,
                    "prior_mse": prior_mse,
                    "zero_mse": zero_mse,
                    "baseline_minus_prior": baseline_mse - prior_mse,
                    "baseline_minus_zero": baseline_mse - zero_mse,
                }
            )
    summary = []
    for label in ("0", "1"):
        label_rows = [row for row in selected if row["label"] == label]
        for control in ("prior", "zero"):
            deltas = np.asarray([row[f"baseline_minus_{control}"] for row in label_rows])
            summary.append(
                {
                    "label": label,
                    "label_name": "non_fraud" if label == "0" else "fraud",
                    "control": control,
                    "n_groups": len(label_rows),
                    "wins": int((deltas < 0).sum()),
                    "mean_delta": float(deltas.mean()),
                    "median_delta": float(np.median(deltas)),
                    "min_delta": float(deltas.min()),
                    "max_delta": float(deltas.max()),
                }
            )
    return selected, summary


def _tensor_term(candidate_value, observed_value):
    candidate_flat = candidate_value.reshape(-1)
    observed_flat = observed_value.reshape(-1)
    observed_energy = observed_flat.square().sum().clamp_min(1e-20)
    if float(observed_flat.square().sum().detach()) == 0.0:
        return float(candidate_flat.square().mean().detach()), 0.0, float(candidate_flat.square().mean().detach())
    cosine_loss = 1 - F.cosine_similarity(candidate_flat, observed_flat, dim=0, eps=1e-12)
    relative_mse = (candidate_flat - observed_flat).square().sum() / observed_energy
    total = cosine_loss + 0.1 * relative_mse
    return float(total.detach()), float(cosine_loss.detach()), float(relative_mse.detach())


def _tensor_diagnostics(run, protocol, selected_rows):
    groups = torch.load(run / "confirmation_targets.pt", weights_only=False)
    rows = []
    for item in selected_rows:
        group_id = item["group_id"]
        restart = item["baseline_restart"]
        group = groups[group_id]
        model, criterion, _, labels, batches, rng, observed = capture(
            group, derive_seed(protocol["run_seed"], "local", group_id), 4
        )
        artifact = torch.load(
            run / "confirmation" / f"group_{group_id}" / f"restart_{restart}" / "baseline.pt",
            weights_only=False,
        )
        reconstruction = artifact.get("best_reconstruction", artifact.get("reconstruction"))
        if reconstruction is None:
            raise KeyError("Baseline artifact has no reconstruction tensor")
        candidate = simulate(model, criterion, reconstruction, labels, batches, rng)
        for key, observed_value in observed.items():
            if not observed_value.is_floating_point():
                continue
            total, cosine, relative_mse = _tensor_term(candidate[key], observed_value)
            rows.append(
                {
                    "group_id": group_id,
                    "tensor": key,
                    "elements": int(observed_value.numel()),
                    "kind": "batchnorm_buffer" if "running_" in key else "parameter",
                    "term": total,
                    "cosine_loss": cosine,
                    "relative_mse": relative_mse,
                    "observed_norm": float(observed_value.reshape(-1).norm().detach()),
                }
            )
    grouped = {}
    for row in rows:
        grouped.setdefault(row["tensor"], []).append(row)
    summary = []
    for tensor, tensor_rows in grouped.items():
        summary.append(
            {
                "tensor": tensor,
                "kind": tensor_rows[0]["kind"],
                "elements": tensor_rows[0]["elements"],
                "mean_term": float(np.mean([row["term"] for row in tensor_rows])),
                "mean_cosine_loss": float(np.mean([row["cosine_loss"] for row in tensor_rows])),
                "mean_relative_mse": float(np.mean([row["relative_mse"] for row in tensor_rows])),
                "mean_observed_norm": float(np.mean([row["observed_norm"] for row in tensor_rows])),
            }
        )
    summary.sort(key=lambda row: row["mean_term"], reverse=True)
    return rows, summary


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path, report):
    class_rows = report["class_summary"]
    tensor_rows = report["tensor_summary"][:10]
    lines = [
        "# Attack Failure Pattern Diagnostics",
        "",
        f"Run: `{report['run']}`",
        "",
        "## Class Split",
        "",
        "| Class | Control | Wins | Mean delta | Median delta |",
        "|---|---|---:|---:|---:|",
    ]
    for row in class_rows:
        lines.append(
            f"| {row['label_name']} | {row['control']} | {row['wins']}/{row['n_groups']} | "
            f"{row['mean_delta']:.6f} | {row['median_delta']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Negative delta means the baseline attack has lower reconstruction MSE than the control.",
            "",
            "## Top Tensor Terms",
            "",
            "| Tensor | Kind | Mean term | Mean cosine loss | Mean relative MSE |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in tensor_rows:
        lines.append(
            f"| `{row['tensor']}` | {row['kind']} | {row['mean_term']:.6f} | "
            f"{row['mean_cosine_loss']:.6f} | {row['mean_relative_mse']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: these tensor terms describe where the selected reconstruction still fails to match the observed update. They are diagnostic only; they do not alter the locked confirmation result.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    report = _load_json(_report_path(run))
    protocol = report.get("protocol") or _load_json(run / "protocol_lock.json")
    rows = _load_json(run / "confirmation_records.json")
    selected_class, class_summary = _class_diagnostics(rows)
    tensor_rows, tensor_summary = _tensor_diagnostics(run, protocol, report["selected_confirmation"])
    output = {
        "run": str(run),
        "baseline_gate_passed": report.get("baseline_gate_passed"),
        "status": report["status"],
        "class_summary": class_summary,
        "selected_class_rows": selected_class,
        "tensor_summary": tensor_summary,
        "note": "Diagnostics only; confirmation selection and gates are unchanged.",
    }
    dump(run / "failure_pattern_diagnostics.json", output)
    _write_csv(run / "failure_pattern_by_class.csv", selected_class)
    _write_csv(run / "failure_pattern_class_summary.csv", class_summary)
    _write_csv(run / "failure_pattern_tensor_summary.csv", tensor_summary)
    _write_markdown(run / "failure_pattern_diagnostics.md", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
