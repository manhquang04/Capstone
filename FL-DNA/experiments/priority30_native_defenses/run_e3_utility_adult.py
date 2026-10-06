#!/usr/bin/env python3
"""Priority 30 E3(iii): Adult utility-matched DP comparator.

This runner is intentionally separate from the E3(i)/(ii) runner.  It fills
the tabular-only utility-matched DP arm requested after S2/S2b/S2c/S2d, for
the Adult defenses that survived E2 and were explicitly named for this pass:
PRECODE and DNA v1 conservative.

It is resumable at both the utility-grid and per-target attack levels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("DATALOADER_NUM_WORKERS", "0")
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_defenses" / "tableak"))

from datasets import ADULT  # noqa: E402
from models import FullyConnected  # noqa: E402

from experiments.dp_update_accounting import alpha_grid, epsilon_from_rdp  # noqa: E402
from experiments.priority30_native_defenses.native_adapters import PrecodeAdultFC, dna_v1_gradient  # noqa: E402
from experiments.priority30_native_defenses.run_audit import (  # noqa: E402
    adult_targets,
    gradient_dict,
    measure_adult,
    run_tableak_with_optional_adaptive,
)
from experiments.priority30_native_defenses.run_e3_dp_comparators import add_clipped_noise, exact_p, holm  # noqa: E402


DEFENSES = ["precode", "dna_v1_conservative"]
GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
GRID_EXTENSION = [1e-1, 3e-1, 1.0, 3.0, 10.0]
SEEDS = list(range(41_000, 41_016))
TRAIN_STEPS = 50
LR = 0.05


def _adult_data() -> ADULT:
    old = Path.cwd()
    os.chdir(ROOT / "external_defenses" / "tableak")
    try:
        dataset = ADULT()
        dataset.standardize()
        return dataset
    finally:
        os.chdir(old)


def _new_model(dataset: ADULT, defense: str | None = None) -> torch.nn.Module:
    base = FullyConnected(dataset.num_features, [100, 100, 2])
    if defense == "precode":
        return PrecodeAdultFC(base)
    return base


def _flatten(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.detach().reshape(-1).float().cpu() for t in tensors])


def _assign_grads(model: torch.nn.Module, grads: list[torch.Tensor]) -> None:
    for param, grad in zip([p for p in model.parameters() if p.requires_grad], grads):
        param.grad = grad.detach().clone().to(param.device, dtype=param.dtype)


def _clip_and_noise_grads(grads: list[torch.Tensor], clip_norm: float, sigma: float, seed: int) -> list[torch.Tensor]:
    flat = _flatten(grads)
    norm = float(flat.norm().item())
    factor = 1.0 if norm <= clip_norm or norm == 0 else clip_norm / norm
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out: list[torch.Tensor] = []
    for grad in grads:
        noise = torch.randn(grad.shape, generator=gen, dtype=grad.dtype) * (sigma * clip_norm)
        out.append(grad.detach().cpu() * factor + noise)
    return out


def _accuracy(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    return float((pred == y).float().mean().item())


def _grad_norm_for_seed(seed: int) -> float:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = _adult_data()
    model = _new_model(dataset)
    criterion = torch.nn.CrossEntropyLoss()
    grads = list(gradient_dict(model, dataset.Xtrain, dataset.ytrain, criterion).values())
    return float(_flatten(grads).norm().item())


def estimate_clip_norm(out_dir: Path) -> float:
    path = out_dir / "S4" / "adult_utility_clip_norm.json"
    if path.exists():
        return float(json.loads(path.read_text())["clip_norm"])
    norms = [_grad_norm_for_seed(seed) for seed in SEEDS]
    clip_norm = float(np.percentile(norms, 95))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "clip_norm": clip_norm,
                "source": "95th percentile of full-train FedSGD gradient norms over the 16 Priority 30 Adult utility seeds",
                "seeds": SEEDS,
                "norms": norms,
            },
            indent=2,
        )
        + "\n"
    )
    return clip_norm


def run_training(seed: int, branch: str, sigma: float | None, clip_norm: float, out_file: str) -> dict[str, Any]:
    torch.set_num_threads(1)
    out = Path(out_file)
    if out.exists():
        return json.loads(out.read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = _adult_data()
    criterion = torch.nn.CrossEntropyLoss()
    if branch == "precode":
        model = _new_model(dataset, "precode")
    else:
        model = _new_model(dataset)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    for step in range(TRAIN_STEPS):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(dataset.Xtrain), dataset.ytrain)
        params = [p for p in model.parameters() if p.requires_grad]
        grads = list(torch.autograd.grad(loss, params, create_graph=False))
        if branch == "v1":
            named = {name: grad.detach() for (name, param), grad in zip([(n, p) for n, p in model.named_parameters() if p.requires_grad], grads)}
            defended = dna_v1_gradient(named)
            grads = [defended[name] for name in named.keys()]
        elif branch == "dp":
            if sigma is None:
                raise ValueError("DP branch requires sigma")
            grads = _clip_and_noise_grads(grads, clip_norm, sigma, seed=seed * 1000 + step)
        elif branch in {"baseline", "precode"}:
            pass
        else:
            raise ValueError(branch)
        _assign_grads(model, grads)
        opt.step()
    result = {
        "seed": seed,
        "branch": branch,
        "sigma": sigma,
        "clip_norm": clip_norm,
        "train_steps": TRAIN_STEPS,
        "lr": LR,
        "test_accuracy": _accuracy(model, dataset.Xtest, dataset.ytest),
    }
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def calibrate(output_dir: Path, workers: int) -> dict[str, Any]:
    out_dir = output_dir / "S4" / "adult_utility_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "S4" / "adult_utility_selected.json"
    if selected_path.exists():
        return json.loads(selected_path.read_text())
    clip_norm = estimate_clip_norm(output_dir)
    grid = list(GRID)
    rows: list[dict[str, Any]] = []

    def collect_for_grid(current_grid: list[float]) -> list[dict[str, Any]]:
        jobs: list[tuple[int, str, float | None, Path]] = []
        for seed in SEEDS:
            for branch, sigma in [("baseline", None), ("precode", None), ("v1", None)]:
                jobs.append((seed, branch, sigma, out_dir / f"seed_{seed}_{branch}.json"))
            for sigma in current_grid:
                jobs.append((seed, "dp", sigma, out_dir / f"seed_{seed}_dp_{sigma:.0e}.json"))
        pending = [(seed, branch, sigma, path) for seed, branch, sigma, path in jobs if not path.exists()]
        if workers <= 1:
            for seed, branch, sigma, path in pending:
                run_training(seed, branch, sigma, clip_norm, str(path))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(run_training, seed, branch, sigma, clip_norm, str(path)) for seed, branch, sigma, path in pending]
                for fut in as_completed(futs):
                    fut.result()
        return [json.loads(path.read_text()) for _seed, _branch, _sigma, path in jobs]

    rows = collect_for_grid(grid)
    df_rows = rows
    base = {r["seed"]: r["test_accuracy"] for r in df_rows if r["branch"] == "baseline"}

    def mean_delta(branch: str, sigma: float | None = None) -> float:
        vals = []
        for r in df_rows:
            if r["branch"] != branch:
                continue
            if branch == "dp" and abs(float(r["sigma"]) - float(sigma)) > 1e-15:
                continue
            vals.append(float(r["test_accuracy"]) - base[int(r["seed"])])
        return float(np.mean(vals))

    thresholds = {
        "precode": mean_delta("precode") - 0.005,
        "dna_v1_conservative": mean_delta("v1") - 0.005,
    }

    def select_sigma(threshold: float, current_grid: list[float]) -> tuple[float | None, bool]:
        eligible = [sigma for sigma in current_grid if mean_delta("dp", sigma) >= threshold]
        if not eligible:
            return current_grid[0], False
        chosen = max(eligible)
        idx = current_grid.index(chosen)
        bracketed = idx < len(current_grid) - 1 and mean_delta("dp", current_grid[idx + 1]) < threshold
        return chosen, bracketed

    selected = {defense: select_sigma(threshold, grid) for defense, threshold in thresholds.items()}
    if any(not bracketed for _sigma, bracketed in selected.values()):
        grid = grid + list(GRID_EXTENSION)
        rows = collect_for_grid(grid)
        df_rows = rows
        base = {r["seed"]: r["test_accuracy"] for r in df_rows if r["branch"] == "baseline"}
        thresholds = {
            "precode": mean_delta("precode") - 0.005,
            "dna_v1_conservative": mean_delta("v1") - 0.005,
        }
        selected = {defense: select_sigma(threshold, grid) for defense, threshold in thresholds.items()}

    grid_csv = output_dir / "S4" / "adult_utility_grid.csv"
    with grid_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["seed", "branch", "sigma", "clip_norm", "train_steps", "lr", "test_accuracy", "delta_vs_baseline"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(df_rows, key=lambda x: (int(x["seed"]), str(x["branch"]), float(x["sigma"] or 0.0))):
            rr = dict(r)
            rr["delta_vs_baseline"] = float(rr["test_accuracy"]) - base[int(rr["seed"])]
            writer.writerow(rr)

    orders = alpha_grid()
    selection: dict[str, Any] = {
        "clip_norm": clip_norm,
        "grid": grid,
        "grid_extension_used": len(grid) > len(GRID),
        "seeds": SEEDS,
        "train_steps": TRAIN_STEPS,
        "lr": LR,
        "thresholds": thresholds,
        "selected": {},
    }
    for defense, (sigma, bracketed) in selected.items():
        eps = epsilon_from_rdp(max(float(sigma), 1e-12), 1.0, 1e-5, 1, orders)
        selection["selected"][defense] = {
            "sigma": sigma,
            "bracketed": bracketed,
            "mean_delta_accuracy": mean_delta("dp", sigma),
            "threshold": thresholds[defense],
            "epsilon_delta_1e_minus_5_one_release": eps["epsilon"],
            "optimal_alpha": eps["alpha"],
        }
    selected_path.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    return selection


def run_adult_utility_attack(stage: str, defense: str, target_id: int, indices: list[int], out_file: str, selected: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    start = time.monotonic()
    out = Path(out_file)
    if out.exists():
        return {"skipped": True, "path": str(out)}
    out.parent.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    os.chdir(ROOT / "external_defenses" / "tableak")
    try:
        dataset = _adult_data()
        x = dataset.Xtrain[indices].clone()
        y = dataset.ytrain[indices].clone()
        net = FullyConnected(dataset.num_features, [100, 100, 2])
        criterion = torch.nn.CrossEntropyLoss()
        base_grad = list(gradient_dict(net, x, y, criterion).values())
        spec = selected["selected"][defense]
        observed = add_clipped_noise(
            base_grad,
            clip_norm=float(selected["clip_norm"]),
            sigma=float(spec["sigma"]),
            seed=45_000 + 101 * target_id + (17 if defense == "precode" else 31),
        )
        torch.manual_seed(30_700 + target_id)
        np.random.seed(30_700 + target_id)
        rec, _ensemble, losses = run_tableak_with_optional_adaptive(
            net=net,
            criterion=criterion,
            observed=observed,
            labels=y,
            x=x,
            dataset=dataset,
            adaptive_context=None,
        )
        metric = measure_adult(dataset, x, rec)
        np.savez(out.parent / "arrays.npz", truth=x.numpy(), labels=y.numpy(), reconstruction=rec.detach().numpy(), objective_losses=np.asarray(losses))
        result = {
            "status": "ran",
            "stage": stage,
            "domain": "adult",
            "eval": "E3",
            "defense": defense,
            "comparator": "utility_matched_clipped",
            "target_id": target_id,
            "adult_indices": indices,
            "metric": metric,
            "dp_spec": {"clip_norm": selected["clip_norm"], **spec},
            "elapsed_seconds": time.monotonic() - start,
        }
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result
    finally:
        os.chdir(old)


def build_tasks(n: int, output_dir: Path, selected: dict[str, Any]) -> list[tuple[tuple[Any, ...], Path]]:
    tasks: list[tuple[tuple[Any, ...], Path]] = []
    targets = adult_targets("S2", n)
    for defense in DEFENSES:
        if not bool(selected["selected"][defense]["bracketed"]):
            continue
        for target_id, target in enumerate(targets):
            path = output_dir / "S4" / "adult" / "utility_matched_clipped" / defense / f"target_{target_id:03d}" / "result.json"
            tasks.append((("S4", defense, target_id, list(target), str(path), selected), path))
    return tasks


def summarize(output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_dir / "S4" / "adult" / "utility_matched_clipped").glob("*/target_*/result.json")):
        data = json.loads(path.read_text())
        if data.get("status") != "ran":
            continue
        defense = data["defense"]
        target_id = int(data["target_id"])
        if defense == "precode":
            defense_path = output_dir / "S2c" / "adult" / "E2" / defense / f"target_{target_id:03d}" / "result.json"
        else:
            defense_path = output_dir / "S2b" / "adult" / "E2" / defense / f"target_{target_id:03d}" / "result.json"
        if not defense_path.exists():
            continue
        defended = json.loads(defense_path.read_text())
        if defended.get("status") != "ran":
            continue
        dp_acc = float(data["metric"]["accuracy_percent"])
        def_acc = float(defended["metric"]["accuracy_percent"])
        dp_error = 100.0 - dp_acc
        def_error = 100.0 - def_acc
        rows.append(
            {
                "domain": "adult",
                "defense": defense,
                "comparator": "utility_matched_clipped",
                "target_id": target_id,
                "defense_accuracy": def_acc,
                "dp_accuracy": dp_acc,
                "defense_error_score": def_error,
                "dp_error_score": dp_error,
                "diff_defense_minus_dp": def_error - dp_error,
                "defense_wins": def_error > dp_error,
                "dp_wins": dp_error > def_error,
                "tie": def_error == dp_error,
            }
        )
    out_dir = output_dir / "S6"
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out_dir / "e3_utility_adult_per_target.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summaries: list[dict[str, Any]] = []
    for defense in sorted({r["defense"] for r in rows}):
        cell = [r for r in rows if r["defense"] == defense]
        wins_def = sum(1 for r in cell if r["defense_wins"])
        wins_dp = sum(1 for r in cell if r["dp_wins"])
        ties = sum(1 for r in cell if r["tie"])
        summaries.append(
            {
                "domain": "adult",
                "defense": defense,
                "comparator": "utility_matched_clipped",
                "n": len(cell),
                "defense_wins": wins_def,
                "dp_wins": wins_dp,
                "ties": ties,
                "p_defense_greater_error": exact_p(wins_def, wins_dp),
                "p_dp_greater_error": exact_p(wins_dp, wins_def),
                "median_defense_accuracy": float(np.median([r["defense_accuracy"] for r in cell])),
                "median_dp_accuracy": float(np.median([r["dp_accuracy"] for r in cell])),
                "median_paired_accuracy_diff_defense_minus_dp": float(np.median([r["defense_accuracy"] - r["dp_accuracy"] for r in cell])),
                "median_defense_error_minus_dp_error": float(np.median([r["diff_defense_minus_dp"] for r in cell])),
            }
        )
    left = [{"key": r["defense"], "p_value": r["p_defense_greater_error"]} for r in summaries]
    right = [{"key": r["defense"], "p_value": r["p_dp_greater_error"]} for r in summaries]
    holm(left)
    holm(right)
    left_map = {r["key"]: r["holm_p"] for r in left}
    right_map = {r["key"]: r["holm_p"] for r in right}
    for r in summaries:
        r["holm_p_defense_greater_error"] = left_map[r["defense"]]
        r["holm_p_dp_greater_error"] = right_map[r["defense"]]
    if summaries:
        with (out_dir / "e3_utility_adult_sign_tests.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = sorted({k for row in summaries for k in row})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["calibrate", "S4", "summarize", "all"], default="all")
    parser.add_argument("--n", type=int, default=39)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts/priority30_native_defenses/audit")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output_dir = ROOT / args.output_dir
    if args.stage == "summarize":
        summarize(output_dir)
        return
    selected = calibrate(output_dir, workers=args.workers)
    if args.stage == "calibrate":
        return
    tasks = build_tasks(args.n, output_dir, selected)
    pending = [task for task in tasks if not task[1].exists()]
    progress = output_dir / "S4_progress.log"
    with progress.open("a", encoding="utf-8") as log:
        log.write(f"stage={args.stage} total={len(tasks)} pending={len(pending)} workers={args.workers}\n")
    done = len(tasks) - len(pending)
    start = time.monotonic()
    if args.workers <= 1:
        for params, _path in pending:
            run_adult_utility_attack(*params)
            done += 1
            if done % 10 == 0:
                with progress.open("a", encoding="utf-8") as log:
                    log.write(f"done={done}/{len(tasks)} elapsed={time.monotonic()-start:.1f}s\n")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_adult_utility_attack, *params) for params, _path in pending]
            for fut in as_completed(futs):
                fut.result()
                done += 1
                if done % 10 == 0:
                    with progress.open("a", encoding="utf-8") as log:
                        log.write(f"done={done}/{len(tasks)} elapsed={time.monotonic()-start:.1f}s\n")
    summarize(output_dir)


if __name__ == "__main__":
    main()
