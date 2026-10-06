"""Priority 28 CIFAR positive-control runner.

This runner is restricted to the undefended FedSGD image instrument.  It uses
fresh CIFAR target bundles produced by ``run_priority7_image_domain_gate.py`` and
checks whether the attack beats Prior, decoy, constant gray, and CIFAR mean
baselines.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.priority23_repaired_cifar_attack import (  # noqa: E402
    cifar_mean_image,
    image_metric_dict,
    make_model,
    optimize_one,
)
from experiments.run_priority7_image_domain_gate import _load_targets, _named_update  # noqa: E402
from privacy.seed_manager import derive_seed  # noqa: E402


AMENDMENT = ROOT / "protocols/amendments/2026-09-30_priority28_defense_audit.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def exact_p(wins: int, n: int) -> float:
    return float(binomtest(wins, n, 0.5, alternative="greater").pvalue) if n else 1.0


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    groups = _load_targets(args.target.resolve())[: args.groups]
    mean_image = cifar_mean_image()
    budget = {
        "iterations": int(args.iterations),
        "attack_lr": float(args.attack_lr),
        "tv_lambda": float(args.tv_lambda),
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": str(AMENDMENT.relative_to(ROOT)),
        "amendment_sha256": sha256(AMENDMENT),
        "target": str(args.target.resolve().relative_to(ROOT)),
        "target_sha256": sha256(args.target.resolve()),
        "groups": len(groups),
        "restarts": int(args.restarts),
        "budget": budget,
        "controls": ["prior", "decoy", "gray_0p5", "cifar_train_mean"],
        "torch_num_threads": torch.get_num_threads(),
        "instrument": "I1_CIFAR_FedSGD_none_positive_control",
    }
    dump(out / "execution_manifest.json", manifest)
    rows = []
    for group_id, group in enumerate(groups):
        images = group["images"].to(dtype=torch.float32)
        labels = group["labels"].to(dtype=torch.long)
        decoy_group = groups[(group_id + 1) % len(groups)]
        decoy_images = decoy_group["images"].to(dtype=torch.float32)
        decoy_labels = decoy_group["labels"].to(dtype=torch.long)
        model = make_model(derive_seed(args.seed, "model", group_id))
        observed = {name: value.detach() for name, value in _named_update(model, images, labels, args.local_lr).items()}
        decoy_model = make_model(derive_seed(args.seed, "model", (group_id + 1) % len(groups)))
        decoy_observed = {
            name: value.detach()
            for name, value in _named_update(decoy_model, decoy_images, decoy_labels, args.local_lr).items()
        }
        best_own = None
        best_decoy = None
        for restart in range(args.restarts):
            own = optimize_one(
                model=model,
                labels=labels,
                target_images=images,
                signal=observed,
                payload=("none", None),
                branch="none",
                budget=budget,
                seed=args.seed,
                group=group_id,
                restart=restart,
                local_lr=args.local_lr,
            )
            decoy = optimize_one(
                model=model,
                labels=labels,
                target_images=images,
                signal=decoy_observed,
                payload=("none", None),
                branch="none",
                budget=budget,
                seed=args.seed,
                group=group_id,
                restart=restart,
                local_lr=args.local_lr,
            )
            if best_own is None or own["best_objective"] < best_own["best_objective"]:
                best_own = own | {"restart": restart}
            if best_decoy is None or decoy["best_objective"] < best_decoy["best_objective"]:
                best_decoy = decoy | {"restart": restart}
        assert best_own is not None and best_decoy is not None
        gray = torch.full_like(images, 0.5)
        mean = mean_image.expand_as(images)
        metrics = {
            "own": image_metric_dict(images, best_own["reconstruction"]),
            "decoy": image_metric_dict(images, best_decoy["reconstruction"]),
            "prior": image_metric_dict(images, best_own["prior"]),
            "gray_0p5": image_metric_dict(images, gray),
            "cifar_train_mean": image_metric_dict(images, mean),
        }
        row = {
            "group": group_id,
            "source_indices": json.dumps(group["source_indices"]),
            "own_restart": int(best_own["restart"]),
            "decoy_restart": int(best_decoy["restart"]),
            "own_objective": float(best_own["best_objective"]),
            "decoy_objective": float(best_decoy["best_objective"]),
        }
        for name, vals in metrics.items():
            for metric, value in vals.items():
                row[f"{name}_{metric}"] = float(value)
        rows.append(row)
        print(json.dumps({"group": group_id, "status": "done"}), flush=True)
    csv_path = out / "cifar_positive_control.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    own = np.asarray([row["own_mse"] for row in rows], dtype=float)
    summary = {
        "n": len(rows),
        "own_mse_mean": float(own.mean()),
        "wins_vs_prior": int(np.sum(own < np.asarray([row["prior_mse"] for row in rows], dtype=float))),
        "wins_vs_decoy": int(np.sum(own < np.asarray([row["decoy_mse"] for row in rows], dtype=float))),
        "wins_vs_gray": int(np.sum(own < np.asarray([row["gray_0p5_mse"] for row in rows], dtype=float))),
        "wins_vs_mean": int(np.sum(own < np.asarray([row["cifar_train_mean_mse"] for row in rows], dtype=float))),
    }
    for key in ("prior", "decoy", "gray", "mean"):
        summary[f"p_vs_{key}"] = exact_p(int(summary[f"wins_vs_{key}"]), len(rows))
    summary["qualified"] = all(summary[f"p_vs_{key}"] < 0.05 for key in ("prior", "decoy", "gray", "mean"))
    dump(out / "cifar_positive_control_summary.json", summary)
    dump(
        out / "manifest.json",
        manifest
        | {
            "csv_sha256": sha256(csv_path),
            "summary_sha256": sha256(out / "cifar_positive_control_summary.json"),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026093001)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--attack-lr", type=float, default=0.05)
    parser.add_argument("--tv-lambda", type=float, default=1e-4)
    parser.add_argument("--local-lr", type=float, default=0.01)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
