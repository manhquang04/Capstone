"""Priority 25 descriptive diagnostic: Adam step vs -lr*sign(gradient)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.priority16_v1_medium_dna_vs_dp_probe import capture_paysim  # noqa: E402
from privacy.seed_manager import derive_seed  # noqa: E402


AMENDMENT = ROOT / "protocols/amendments/2026-09-29_priority25_rq1_extension.md"
ADAM_LR = 0.001


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def flatten_trainable_update(model: torch.nn.Module, observed: dict[str, torch.Tensor]) -> np.ndarray:
    parts = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parts.append(observed[name].detach().cpu().reshape(-1).double().numpy())
    return np.concatenate(parts).astype(np.float64)


def flatten_gradient(model: torch.nn.Module, criterion: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, batches: list[slice], rng) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(rng)
        total_loss = None
        for ids in batches:
            loss = criterion(model(x[ids]), y[ids])
            total_loss = loss if total_loss is None else total_loss + loss
        assert total_loss is not None
        total_loss.backward()
    parts = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                parts.append(torch.zeros_like(parameter).detach().cpu().reshape(-1).double().numpy())
            else:
                parts.append(parameter.grad.detach().cpu().reshape(-1).double().numpy())
    return np.concatenate(parts).astype(np.float64)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or float(np.std(a)) < 1e-18 or float(np.std(b)) < 1e-18:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026092961)
    args = parser.parse_args()

    torch.set_num_threads(1)
    groups = torch.load(args.target, map_location="cpu", weights_only=False)
    rows = []
    for group_id, group in enumerate(groups):
        local_seed = derive_seed(args.seed, "priority25-adam-sign", group_id)
        model, criterion, x, y, batches, rng, observed = capture_paysim(group, local_seed, 4)
        step = flatten_trainable_update(model, observed)
        grad = flatten_gradient(model, criterion, x, y, batches, rng)
        sign_step = -ADAM_LR * np.sign(grad)
        nonzero = np.abs(grad) > 1e-12
        cosine = float(np.dot(step, sign_step) / (max(np.linalg.norm(step), 1e-18) * max(np.linalg.norm(sign_step), 1e-18)))
        sign_agreement = float(np.mean(np.sign(step[nonzero]) == np.sign(sign_step[nonzero]))) if np.any(nonzero) else float("nan")
        abs_corr = safe_corr(np.abs(step[nonzero]), np.abs(grad[nonzero])) if np.any(nonzero) else float("nan")
        rows.append(
            {
                "group": group_id,
                "source_ids": json.dumps(group["source_ids"]),
                "trainable_dim": int(step.size),
                "nonzero_gradient_coordinates": int(np.sum(nonzero)),
                "cosine_step_vs_neg_lr_sign_grad": cosine,
                "sign_agreement_nonzero_grad": sign_agreement,
                "pearson_abs_step_abs_grad_nonzero": abs_corr,
                "step_abs_mean": float(np.mean(np.abs(step))),
                "step_abs_sd": float(np.std(np.abs(step))),
                "step_abs_min": float(np.min(np.abs(step))),
                "step_abs_max": float(np.max(np.abs(step))),
                "grad_abs_mean_nonzero": float(np.mean(np.abs(grad[nonzero]))) if np.any(nonzero) else float("nan"),
                "grad_abs_sd_nonzero": float(np.std(np.abs(grad[nonzero]))) if np.any(nonzero) else float("nan"),
            }
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "adam_sign_diagnostic.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": str(AMENDMENT.relative_to(ROOT)),
        "amendment_sha256": sha256(AMENDMENT),
        "target": str(args.target.resolve().relative_to(ROOT)),
        "target_sha256": sha256(args.target.resolve()),
        "seed": args.seed,
        "n": len(rows),
        "adam_lr": ADAM_LR,
        "mean_cosine_step_vs_neg_lr_sign_grad": float(np.mean([r["cosine_step_vs_neg_lr_sign_grad"] for r in rows])),
        "median_cosine_step_vs_neg_lr_sign_grad": float(np.median([r["cosine_step_vs_neg_lr_sign_grad"] for r in rows])),
        "mean_sign_agreement_nonzero_grad": float(np.mean([r["sign_agreement_nonzero_grad"] for r in rows])),
        "median_sign_agreement_nonzero_grad": float(np.median([r["sign_agreement_nonzero_grad"] for r in rows])),
        "mean_abs_magnitude_corr": float(np.nanmean([r["pearson_abs_step_abs_grad_nonzero"] for r in rows])),
        "median_abs_magnitude_corr": float(np.nanmedian([r["pearson_abs_step_abs_grad_nonzero"] for r in rows])),
        "interpretation": "High cosine/sign agreement with low abs-magnitude correlation indicates a fresh-state Adam step behaves approximately like a sign update and weakly preserves gradient magnitudes.",
        "csv": str(csv_path.relative_to(ROOT)),
    }
    (output / "adam_sign_diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

