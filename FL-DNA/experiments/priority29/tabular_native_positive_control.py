"""Priority 29 Adult/TabLeak native positive-control gate.

This ports the validated Adult native TabLeak run into the project artifact
tree, with outputs outside external_defenses/ and torch threads fixed to one.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("DATALOADER_NUM_WORKERS", "0")
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "external_defenses" / "tableak"
sys.path.insert(0, str(OFFICIAL))

from attacks import invert_grad  # noqa: E402
from datasets import ADULT  # noqa: E402
from models import FullyConnected  # noqa: E402
from utils import batch_feature_wise_accuracy_score, match_reconstruction_ground_truth, post_process_continuous  # noqa: E402
from utils.encoder_decoder import to_categorical  # noqa: E402
import attacks.gradient_inversion_attack as official_attack  # noqa: E402


_official_sigmoid_bound = official_attack.continuous_sigmoid_bound


def nonleaf_sigmoid_bound(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    return _official_sigmoid_bound(x.clone(), *args, **kwargs)


official_attack.continuous_sigmoid_bound = nonleaf_sigmoid_bound


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_one_sided_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n))


def official_config() -> dict[str, Any]:
    tree = ast.parse((OFFICIAL / "run_inversion_attacks.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "configs" for t in node.targets):
            entry = next(v for k, v in zip(node.value.keys, node.value.values) if ast.literal_eval(k) == 46)
            config = {
                ast.literal_eval(k): ("cpu" if isinstance(v, ast.Attribute) else ast.literal_eval(v))
                for k, v in zip(entry.keys, entry.values)
            }
            config.pop("invert_labels", None)
            config["return_all"] = True
            config["return_all_reconstruction_losses"] = True
            return config
    raise RuntimeError("Official experiment 46 absent")


def measure(dataset: ADULT, x: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, Any]:
    rec = post_process_continuous(reconstruction.detach().clone(), dataset)
    truth = dataset.decode_batch(x, standardized=True)
    guess = dataset.decode_batch(rec, standardized=True)
    aligned, all_error, cat_error, cont_error = match_reconstruction_ground_truth(
        truth, guess, dataset.create_tolerance_map()
    )
    return {
        "accuracy_percent": 100.0 * (1.0 - float(np.mean(all_error))),
        "categorical_accuracy_percent": 100.0 * (1.0 - float(np.mean(cat_error))),
        "continuous_accuracy_percent": 100.0 * (1.0 - float(np.mean(cont_error))),
        "per_feature_accuracy_percent": {
            k: 100.0 * (1.0 - float(v))
            for k, v in batch_feature_wise_accuracy_score(
                truth, aligned, dataset.create_tolerance_map(), dataset.train_features
            ).items()
        },
    }


def mean_mode_baseline(dataset: ADULT, truth: torch.Tensor) -> torch.Tensor:
    center = dataset.Xtrain.mean(0).clone()
    raw = dataset.de_standardize(dataset.Xtrain)
    ptr = 0
    for _, categories in dataset.train_features.items():
        if categories is None:
            ptr += 1
            continue
        width = len(categories)
        mode = int(raw[:, ptr : ptr + width].mean(0).argmax())
        unscaled = torch.zeros(width)
        unscaled[mode] = 1
        center[ptr : ptr + width] = (unscaled - dataset.mean[ptr : ptr + width]) / dataset.std[ptr : ptr + width]
        ptr += width
    return center.expand_as(truth).clone()


def empirical_marginal_baseline(dataset: ADULT, truth: torch.Tensor, seed: int) -> tuple[torch.Tensor, float, float]:
    rng = np.random.default_rng(seed)
    scores = []
    first_guess: torch.Tensor | None = None
    for _ in range(30):
        guess = torch.empty_like(truth)
        ptr = 0
        for _, categories in dataset.train_features.items():
            width = 1 if categories is None else len(categories)
            indices = rng.integers(0, len(dataset.Xtrain), len(truth))
            guess[:, ptr : ptr + width] = dataset.Xtrain[indices, ptr : ptr + width]
            ptr += width
        if first_guess is None:
            first_guess = guess.clone()
        scores.append(measure(dataset, truth, guess)["accuracy_percent"])
    return first_guess if first_guess is not None else torch.empty_like(truth), float(np.mean(scores)), float(np.std(scores, ddof=1))


def gradient(net: torch.nn.Module, criterion: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
    return [g.detach() for g in torch.autograd.grad(criterion(net(x), y), net.parameters())]


def target_batches(stage: str, n: int, dataset_size: int, batch_size: int = 8) -> list[list[int]]:
    population = list(range(dataset_size))
    rng = random.Random(29_020_801)
    rng.shuffle(population)
    offsets = {"n8": 0, "n24": 2_000, "confirmatory": 6_000}
    offset = offsets[stage]
    batches = []
    for group in range(n):
        start = offset + group * batch_size
        batches.append(population[start : start + batch_size])
    return batches


def run_stage(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    out_dir = ROOT / args.output_dir / args.stage
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_existing:
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    os.chdir(OFFICIAL)
    dataset = ADULT()
    dataset.standardize()
    config = official_config()
    batches = target_batches(args.stage, args.n, len(dataset.Xtrain))
    flat_ids = [idx for batch in batches for idx in batch]
    if len(flat_ids) != len(set(flat_ids)):
        raise RuntimeError("duplicate Adult row id in target batches")

    rows: list[dict[str, Any]] = []
    run_start = time.monotonic()
    try:
        for group, indices in enumerate(batches):
            target_dir = out_dir / f"group_{group:03d}"
            target_dir.mkdir(parents=True, exist_ok=True)
            np.random.seed(29_020_000 + group)
            torch.manual_seed(29_020_000 + group)
            net = FullyConnected(dataset.num_features, [100, 100, 2])
            criterion = torch.nn.CrossEntropyLoss()
            x = dataset.Xtrain[indices].clone()
            y = dataset.ytrain[indices].clone()
            true_grad = gradient(net, criterion, x, y)
            torch.save(net.state_dict(), target_dir / "model.pt")
            np.savez(target_dir / "inputs.npz", truth=x.numpy(), labels=y.numpy(), indices=np.array(indices))
            start = time.monotonic()
            rec, ensemble, losses = invert_grad(
                net=net,
                training_criterion=criterion,
                true_grad=true_grad,
                true_label=y,
                true_data=torch.empty_like(x),
                dataset=dataset,
                **config,
            )
            elapsed = time.monotonic() - start
            mean_mode = mean_mode_baseline(dataset, x)
            empirical, empirical_mean, empirical_sd = empirical_marginal_baseline(dataset, x, seed=29_021_000 + group)
            attack_metric = measure(dataset, x, rec)
            mean_mode_metric = measure(dataset, x, mean_mode)
            empirical_metric = measure(dataset, x, empirical)
            np.savez(
                target_dir / "arrays.npz",
                truth=x.numpy(),
                labels=y.numpy(),
                reconstruction=rec.detach().numpy(),
                ensemble=np.stack([r.numpy() for r in ensemble]),
                objective_losses=np.array(losses),
                mean_mode=mean_mode.numpy(),
                empirical=empirical.numpy(),
            )
            row = {
                "stage": args.stage,
                "group": group,
                "adult_indices": " ".join(str(i) for i in indices),
                "runtime_seconds": elapsed,
                "attack_accuracy_percent": attack_metric["accuracy_percent"],
                "mean_mode_accuracy_percent": mean_mode_metric["accuracy_percent"],
                "empirical_single_accuracy_percent": empirical_metric["accuracy_percent"],
                "empirical_marginal_mean_accuracy_percent": empirical_mean,
                "empirical_marginal_sd_accuracy_percent": empirical_sd,
                "win_vs_mean_mode": int(attack_metric["accuracy_percent"] > mean_mode_metric["accuracy_percent"]),
                "win_vs_empirical_marginal_mean": int(attack_metric["accuracy_percent"] > empirical_mean),
                "tie_vs_mean_mode": int(attack_metric["accuracy_percent"] == mean_mode_metric["accuracy_percent"]),
                "tie_vs_empirical_marginal_mean": int(attack_metric["accuracy_percent"] == empirical_mean),
            }
            rows.append(row)
            (target_dir / "result.json").write_text(
                json.dumps(
                    {
                        "row": row,
                        "attack_metric": attack_metric,
                        "mean_mode_metric": mean_mode_metric,
                        "empirical_single_metric": empirical_metric,
                        "config": config,
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
            print(json.dumps({"completed_group": group, "accuracy": row["attack_accuracy_percent"], "seconds": elapsed}), flush=True)
    finally:
        os.chdir(old_cwd)

    per_target = out_dir / "per_target.csv"
    with per_target.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for baseline, win_key, tie_key in [
        ("mean_mode", "win_vs_mean_mode", "tie_vs_mean_mode"),
        ("empirical_marginal_mean", "win_vs_empirical_marginal_mean", "tie_vs_empirical_marginal_mean"),
    ]:
        wins = sum(int(row[win_key]) for row in rows)
        ties = sum(int(row[tie_key]) for row in rows)
        losses = len(rows) - wins - ties
        summary[baseline] = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "one_sided_exact_p": exact_one_sided_p(wins, losses),
        }
    manifest = {
        "priority": 29,
        "stage": args.stage,
        "n": args.n,
        "dataset": "Adult native TabLeak",
        "batch_size": 8,
        "target_batches": batches,
        "source_disjoint_evidence": {
            "within_stage_unique_rows": len(flat_ids) == len(set(flat_ids)),
            "stage_offsets": {"n8": 0, "n24": 2000, "confirmatory": 6000},
        },
        "summary": summary,
        "config": config,
        "elapsed_seconds": time.monotonic() - run_start,
        "threads": torch.get_num_threads(),
        "outputs": {"per_target_csv": str(per_target)},
    }
    (out_dir / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["n8", "n24", "confirmatory"], required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--output-dir", default="artifacts/priority29_native_audit/tabular_positive_control")
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()
    run_stage(args)


if __name__ == "__main__":
    main()

