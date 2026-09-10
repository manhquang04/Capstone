"""Measure fraud and non-fraud gradient contribution on locked attack targets.

This diagnostic answers whether the observed failure pattern is likely caused by
the attack objective or by the update signal itself being class-skewed. It does
not run inversion and does not change any confirmation gate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments import fraud_fl_common as common
from experiments.run_phase3_adam_ladder import REFERENCE
from experiments.run_phase3_full_client import dump
from models.fraud_mlp import FraudMLP


def _load_json(path: Path):
    return json.loads(path.read_text())


def _protocol(run: Path):
    for name in ("attack_scale_redevelopment_report.json", "attack_redevelopment_report.json"):
        path = run / name
        if path.exists():
            report = _load_json(path)
            if "protocol" in report:
                return report["protocol"]
    return _load_json(run / "protocol_lock.json")


def _focal_per_sample(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_t = common.FOCAL_ALPHA * targets + (1.0 - common.FOCAL_ALPHA) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(common.FOCAL_GAMMA) * bce).reshape(-1)


def _flat_gradients(model, criterion, x, y, *, train_mode, seed):
    model.zero_grad(set_to_none=True)
    model.train(train_mode)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        loss = criterion(model(x), y)
    parameters = [param for param in model.parameters() if param.requires_grad]
    gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
    return torch.cat([gradient.detach().reshape(-1) for gradient in gradients]), float(loss.detach())


def _shared_train_bn_gradients(model, x, y, *, seed):
    model.zero_grad(set_to_none=True)
    model.train(True)
    parameters = [param for param in model.parameters() if param.requires_grad]
    labels = y.reshape(-1)
    masks = {"non_fraud": labels == 0, "fraud": labels == 1}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        per_sample = _focal_per_sample(model(x), y)
    full_loss = per_sample.mean()
    full_grads = torch.autograd.grad(full_loss, parameters, retain_graph=True)
    full_gradient = torch.cat([gradient.detach().reshape(-1) for gradient in full_grads])
    total = len(x)
    class_payload = {}
    contribution_sum = torch.zeros_like(full_gradient)
    for label_name, mask in masks.items():
        contribution_loss = per_sample[mask].sum() / total
        mean_loss = per_sample[mask].mean()
        contribution_grads = torch.autograd.grad(contribution_loss, parameters, retain_graph=True)
        mean_grads = torch.autograd.grad(mean_loss, parameters, retain_graph=True)
        contribution_gradient = torch.cat([gradient.detach().reshape(-1) for gradient in contribution_grads])
        mean_gradient = torch.cat([gradient.detach().reshape(-1) for gradient in mean_grads])
        contribution_sum += contribution_gradient
        class_payload[label_name] = {
            "count": int(mask.sum()),
            "norm": float(mean_gradient.norm()),
            "weighted_norm": float(contribution_gradient.norm()),
            "cosine_to_full": _safe_cosine(contribution_gradient, full_gradient),
            "weighted_cosine_to_full": _safe_cosine(contribution_gradient, full_gradient),
            "loss": float(mean_loss.detach()),
        }
    return full_gradient, float(full_loss.detach()), class_payload, float((contribution_sum - full_gradient).norm())


def _safe_cosine(a, b):
    if float(a.square().sum()) == 0.0 or float(b.square().sum()) == 0.0:
        return 0.0
    return float(F.cosine_similarity(a, b, dim=0, eps=1e-12))


def _measure_group(group, group_id, protocol, mode):
    x = torch.from_numpy(group["x"])
    y = torch.from_numpy(group["y"])
    labels = y.reshape(-1)
    masks = {"non_fraud": labels == 0, "fraud": labels == 1}
    if mode == "train" and any(int(mask.sum()) < 2 for mask in masks.values()):
        return None
    model = FraudMLP(x.shape[1])
    model.load_state_dict(torch.load(REFERENCE / "pre_local.pt", weights_only=False))
    criterion = common.BinaryFocalLoss()
    seed = int(protocol["run_seed"]) + group_id * 1009
    if mode == "train_shared_bn":
        full_gradient, full_loss, class_payload, additivity_error = _shared_train_bn_gradients(model, x, y, seed=seed)
    else:
        full_gradient, full_loss = _flat_gradients(model, criterion, x, y, train_mode=(mode == "train"), seed=seed)
        total = len(x)
        class_payload = {}
        for label_name, mask in masks.items():
            count = int(mask.sum())
            gradient, loss = _flat_gradients(
                model,
                criterion,
                x[mask],
                y[mask],
                train_mode=(mode == "train"),
                seed=seed,
            )
            weighted = gradient * (count / total)
            class_payload[label_name] = {
                "count": count,
                "norm": float(gradient.norm()),
                "weighted_norm": float(weighted.norm()),
                "cosine_to_full": _safe_cosine(weighted, full_gradient),
                "weighted_cosine_to_full": _safe_cosine(weighted, full_gradient),
                "loss": loss,
            }
        additivity_error = None
    fraud = class_payload["fraud"]
    non = class_payload["non_fraud"]
    return {
        "group_id": group_id,
        "mode": mode,
        "full_loss": full_loss,
        "full_norm": float(full_gradient.norm()),
        "fraud_count": fraud["count"],
        "non_fraud_count": non["count"],
        "fraud_norm": fraud["norm"],
        "non_fraud_norm": non["norm"],
        "fraud_weighted_norm": fraud["weighted_norm"],
        "non_fraud_weighted_norm": non["weighted_norm"],
        "fraud_to_non_norm_ratio": fraud["norm"] / max(non["norm"], 1e-20),
        "fraud_to_non_weighted_norm_ratio": fraud["weighted_norm"] / max(non["weighted_norm"], 1e-20),
        "fraud_cosine_to_full": fraud["cosine_to_full"],
        "non_fraud_cosine_to_full": non["cosine_to_full"],
        "fraud_loss": fraud["loss"],
        "non_fraud_loss": non["loss"],
        "class_gradient_additivity_error": additivity_error,
    }


def _summarize(rows):
    summary = []
    for mode in sorted({row["mode"] for row in rows}):
        mode_rows = [row for row in rows if row["mode"] == mode]
        ratios = np.asarray([row["fraud_to_non_weighted_norm_ratio"] for row in mode_rows])
        raw_ratios = np.asarray([row["fraud_to_non_norm_ratio"] for row in mode_rows])
        summary.append(
            {
                "mode": mode,
                "n_groups": len(mode_rows),
                "fraud_weighted_norm_gt_non_fraud": int((ratios > 1.0).sum()),
                "mean_weighted_ratio": float(ratios.mean()),
                "median_weighted_ratio": float(np.median(ratios)),
                "min_weighted_ratio": float(ratios.min()),
                "max_weighted_ratio": float(ratios.max()),
                "mean_raw_ratio": float(raw_ratios.mean()),
                "median_raw_ratio": float(np.median(raw_ratios)),
                "mean_fraud_cosine_to_full": float(np.mean([row["fraud_cosine_to_full"] for row in mode_rows])),
                "mean_non_fraud_cosine_to_full": float(
                    np.mean([row["non_fraud_cosine_to_full"] for row in mode_rows])
                ),
            }
        )
    return summary


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path, report):
    lines = [
        "# Class Gradient Contribution Diagnostic",
        "",
        f"Run: `{report['run']}`",
        "",
        "This diagnostic computes fraud and non-fraud gradient contributions on the locked confirmation targets. It does not run inversion and does not change attack gates.",
        "",
        "| Mode | Groups | Fraud weighted norm > non-fraud | Mean weighted ratio | Median weighted ratio | Mean fraud cosine/full | Mean non-fraud cosine/full |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        lines.append(
            f"| {row['mode']} | {row['n_groups']} | {row['fraud_weighted_norm_gt_non_fraud']}/{row['n_groups']} | "
            f"{row['mean_weighted_ratio']:.6f} | {row['median_weighted_ratio']:.6f} | "
            f"{row['mean_fraud_cosine_to_full']:.6f} | {row['mean_non_fraud_cosine_to_full']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Weighted ratio uses class-gradient norm multiplied by class count / group size, matching mean-reduction loss scale. A ratio above 1 indicates fraud contributes a larger weighted gradient norm than non-fraud despite fewer records.",
            "`train_shared_bn` is the closest diagnostic to the attack simulator: it uses one full train-mode forward pass and splits the per-sample loss on the same BatchNorm/dropout graph.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["eval", "train", "train_shared_bn"],
        choices=["eval", "train", "train_shared_bn"],
    )
    parser.add_argument(
        "--targets",
        default="confirmation_targets.pt",
        help="Target group artifact inside the run directory.",
    )
    args = parser.parse_args()

    common.DEVICE = torch.device("cpu")
    torch.set_num_threads(1)
    run = args.run.resolve()
    protocol = _protocol(run)
    groups = torch.load(run / args.targets, weights_only=False)
    rows = []
    skipped = []
    for mode in args.modes:
        for group_id, group in enumerate(groups):
            row = _measure_group(group, group_id, protocol, mode)
            if row is None:
                skipped.append({"group_id": group_id, "mode": mode, "reason": "class subset has fewer than 2 records"})
            else:
                rows.append(row)
    report = {
        "run": str(run),
        "protocol_run_seed": protocol["run_seed"],
        "summary": _summarize(rows),
        "rows": rows,
        "skipped": skipped,
        "note": "Diagnostic only; does not rerun attacks or alter gates.",
    }
    dump(run / "class_gradient_contribution.json", report)
    _write_csv(run / "class_gradient_contribution.csv", rows)
    _write_csv(run / "class_gradient_contribution_summary.csv", report["summary"])
    _write_markdown(run / "class_gradient_contribution.md", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
