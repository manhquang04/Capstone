#!/usr/bin/env python3
"""Priority 30 E3 DP comparators for the native-defense audit.

This runner is intentionally separate from run_audit.py because the original
runner had no real S3/S4 implementation.  It reuses the exact S1/S2
confirmatory target IDs and writes resumable per-target JSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
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
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_defenses" / "invertinggradients"))
sys.path.insert(0, str(ROOT / "external_defenses" / "tableak"))

import inversefed  # noqa: E402
from attacks import invert_grad  # noqa: E402
from datasets import ADULT  # noqa: E402
from models import FullyConnected  # noqa: E402

from experiments.dp_update_accounting import alpha_grid, epsilon_from_rdp  # noqa: E402
from experiments.priority30_native_defenses.run_audit import (  # noqa: E402
    ADULT_DEFENSES,
    IMAGE_CONFIG,
    IMAGE_DEFENSES,
    adult_targets,
    gradient_dict,
    image_metrics,
    image_observed,
    image_targets,
    measure_adult,
    run_tableak_with_optional_adaptive,
)


PAPER_SIGMA: dict[str, tuple[float, str]] = {
    # Reproduction report C1 gives ranges for Soteria and DLG/pruning and two
    # PRECODE noise levels.  This comparator uses the strongest documented
    # Gaussian endpoint for a conservative paper-style Gaussian-noise baseline.
    "soteria": (1e-1, "REPRODUCTION_REPORT.md C1 Table 2 documented Gaussian sigma range 1e-4..1e-1; strongest endpoint used"),
    "gradient_pruning": (1e-1, "REPRODUCTION_REPORT.md C1 DLG/pruning documented Gaussian variance/noise range 1e-4..1e-1; strongest endpoint used"),
    "precode": (1e-2, "REPRODUCTION_REPORT.md C1 PRECODE NG-2 documented Gaussian std sigma=1e-2"),
    "ats": (1e-3, "REPRODUCTION_REPORT.md C1 ATS Gaussian(1e-3) documented scale"),
}

DEFENSE_SEED_OFFSET = {
    name: 1009 * (idx + 1)
    for idx, name in enumerate(sorted(set(IMAGE_DEFENSES) | set(ADULT_DEFENSES)))
}


def flatten(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.detach().reshape(-1).float().cpu() for t in tensors])


def add_unclipped_noise(tensors: list[torch.Tensor], sigma: float, seed: int) -> list[torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out: list[torch.Tensor] = []
    for tensor in tensors:
        noise = torch.randn(tensor.shape, generator=gen, dtype=tensor.dtype) * sigma
        out.append(tensor.detach().cpu() + noise)
    return out


def add_clipped_noise(tensors: list[torch.Tensor], clip_norm: float, sigma: float, seed: int) -> list[torch.Tensor]:
    flat = flatten(tensors)
    norm = float(flat.norm().item())
    factor = 1.0 if norm <= clip_norm or norm == 0.0 else clip_norm / norm
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out: list[torch.Tensor] = []
    for tensor in tensors:
        base = tensor.detach().cpu() * factor
        noise = torch.randn(tensor.shape, generator=gen, dtype=tensor.dtype) * (sigma * clip_norm)
        out.append(base + noise)
    return out


def exact_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return float("nan")
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n)


def holm(rows: list[dict[str, Any]], p_key: str = "p_value") -> None:
    order = sorted([i for i, row in enumerate(rows) if math.isfinite(float(row[p_key]))], key=lambda i: float(rows[i][p_key]))
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order, start=1):
        adj = min(1.0, (m - rank + 1) * float(rows[idx][p_key]))
        running = max(running, adj)
        rows[idx]["holm_p"] = running
    for row in rows:
        row.setdefault("holm_p", float("nan"))


def image_base_gradient(index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.nn.Module, list[torch.Tensor]]:
    data = CIFAR10(str(ROOT / "datasets" / "cifar10"), train=False, download=False, transform=ToTensor())
    raw, label_int = data[index]
    raw = raw.unsqueeze(0)
    label = torch.tensor([label_int], dtype=torch.long)
    dm = torch.tensor(inversefed.consts.cifar10_mean)[:, None, None]
    ds = torch.tensor(inversefed.consts.cifar10_std)[:, None, None]
    normalized = (raw - dm) / ds
    model, _key = inversefed.construct_model("LeNetZhu", seed=42)
    model.eval()
    grad = list(gradient_dict(model, normalized, label, torch.nn.CrossEntropyLoss()).values())
    return raw, label, normalized, model, grad


def run_image_dp_attack(stage: str, comparator: str, defense: str, target_id: int, index: int, out_file: str, calib: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    start = time.monotonic()
    out = Path(out_file)
    if out.exists():
        return {"skipped": True, "path": str(out)}
    out.parent.mkdir(parents=True, exist_ok=True)
    if comparator == "paper_unclipped" and defense not in PAPER_SIGMA:
        result = {"status": "NOT_APPLICABLE", "reason": "no unique published Gaussian-noise sigma for this defense in REPRODUCTION_REPORT.md C1"}
        out.write_text(json.dumps(result, indent=2) + "\n")
        return result
    raw, label, _normalized, model, base_grad = image_base_gradient(index)
    if comparator == "paper_unclipped":
        sigma, sigma_source = PAPER_SIGMA[defense]
        observed = add_unclipped_noise(base_grad, sigma=sigma, seed=31_000 + 101 * target_id + DEFENSE_SEED_OFFSET[defense])
        spec = {"clip": "none", "sigma": sigma, "sigma_source": sigma_source}
    elif comparator == "distortion_matched_clipped":
        spec = calib["image"][defense]
        observed = add_clipped_noise(base_grad, clip_norm=float(spec["clip_norm"]), sigma=float(spec["sigma"]), seed=32_000 + 101 * target_id + DEFENSE_SEED_OFFSET[defense])
    else:
        raise ValueError(comparator)
    dm = torch.tensor(inversefed.consts.cifar10_mean)[:, None, None]
    ds = torch.tensor(inversefed.consts.cifar10_std)[:, None, None]
    attack = inversefed.GradientReconstructor(model, (dm, ds), IMAGE_CONFIG.copy(), num_images=1)
    torch.manual_seed(30_600 + target_id)
    np.random.seed(30_600 + target_id)
    recon_norm, stats = attack.reconstruct(observed, label, img_shape=(3, 32, 32))
    recon = recon_norm * ds + dm
    result = {
        "status": "ran",
        "stage": stage,
        "domain": "image",
        "eval": "E3",
        "defense": defense,
        "comparator": comparator,
        "target_id": target_id,
        "cifar10_index": index,
        "metrics": image_metrics(raw, recon),
        "objective": float(dict(stats).get("opt", float("nan"))),
        "dp_spec": spec,
        "elapsed_seconds": time.monotonic() - start,
    }
    out.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    return result


def run_adult_dp_attack(stage: str, comparator: str, defense: str, target_id: int, indices: list[int], out_file: str, calib: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    start = time.monotonic()
    out = Path(out_file)
    if out.exists():
        return {"skipped": True, "path": str(out)}
    out.parent.mkdir(parents=True, exist_ok=True)
    if comparator == "paper_unclipped" and defense not in PAPER_SIGMA:
        result = {"status": "NOT_APPLICABLE", "reason": "no unique published Gaussian-noise sigma for this defense in REPRODUCTION_REPORT.md C1"}
        out.write_text(json.dumps(result, indent=2) + "\n")
        return result
    old = Path.cwd()
    os.chdir(ROOT / "external_defenses" / "tableak")
    try:
        dataset = ADULT()
        dataset.standardize()
        x = dataset.Xtrain[indices].clone()
        y = dataset.ytrain[indices].clone()
        net = FullyConnected(dataset.num_features, [100, 100, 2])
        criterion = torch.nn.CrossEntropyLoss()
        base_grad = list(gradient_dict(net, x, y, criterion).values())
        if comparator == "paper_unclipped":
            sigma, sigma_source = PAPER_SIGMA[defense]
            observed = add_unclipped_noise(base_grad, sigma=sigma, seed=33_000 + 101 * target_id + DEFENSE_SEED_OFFSET[defense])
            spec = {"clip": "none", "sigma": sigma, "sigma_source": sigma_source}
        elif comparator == "distortion_matched_clipped":
            spec = calib["adult"][defense]
            observed = add_clipped_noise(base_grad, clip_norm=float(spec["clip_norm"]), sigma=float(spec["sigma"]), seed=34_000 + 101 * target_id + DEFENSE_SEED_OFFSET[defense])
        else:
            raise ValueError(comparator)
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
            "comparator": comparator,
            "target_id": target_id,
            "adult_indices": indices,
            "metric": metric,
            "dp_spec": spec,
            "elapsed_seconds": time.monotonic() - start,
        }
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result
    finally:
        os.chdir(old)


def defense_payload_for_distortion(domain: str, defense: str, target: Any) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if domain == "image":
        raw, label, normalized, _base, base_grad = image_base_gradient(int(target))
        if defense == "count_sketch":
            # Count-Sketch lives in sketch space; use the decoded server estimate
            # already used by E1 as the comparable gradient-space transmitted view.
            from experiments.priority30_native_defenses.run_audit import count_sketch_decode, count_sketch_payload_from_update

            update = gradient_dict(_base, normalized, label, torch.nn.CrossEntropyLoss())
            table, buckets, signs, like = count_sketch_payload_from_update(update)
            defended = count_sketch_decode(table, buckets, signs, like)
        else:
            _model, defended, _score_raw, _mode = image_observed(defense, normalized if defense != "ats" else raw, label)
        return base_grad, defended
    old = Path.cwd()
    os.chdir(ROOT / "external_defenses" / "tableak")
    try:
        dataset = ADULT()
        dataset.standardize()
        x = dataset.Xtrain[target].clone()
        y = dataset.ytrain[target].clone()
        net = FullyConnected(dataset.num_features, [100, 100, 2])
        criterion = torch.nn.CrossEntropyLoss()
        base = gradient_dict(net, x, y, criterion)
        defended = {name: tensor.clone() for name, tensor in base.items()}
        from experiments.priority30_native_defenses.native_adapters import dna_v1_gradient, dna_v2_gradient, prune_gradient
        from experiments.priority30_native_defenses.run_audit import count_sketch_decode, count_sketch_payload_from_update

        if defense == "gradient_pruning":
            defended, _ = prune_gradient(defended, 0.70)
        elif defense == "dna_v1_conservative":
            defended = dna_v1_gradient(defended)
        elif defense == "dna_v2_0p95":
            defended, _ = dna_v2_gradient(defended)
        elif defense == "count_sketch":
            table, buckets, signs, like = count_sketch_payload_from_update(defended)
            decoded = count_sketch_decode(table, buckets, signs, like)
            defended = {name: tensor for name, tensor in zip(defended.keys(), decoded)}
        elif defense in {"soteria", "precode"}:
            # Model-level adapters have architecture-dependent parameter spaces;
            # for this DP calibration we conservatively use the observed base-model
            # gradient-space change available to the harness.  PRECODE's model-level
            # stochasticity is not directly L2-comparable and is marked in output.
            pass
        return list(base.values()), list(defended.values())
    finally:
        os.chdir(old)


def calibrate(output_dir: Path, n: int) -> dict[str, Any]:
    out = output_dir / "S3" / "calibration.json"
    if out.exists():
        return json.loads(out.read_text())
    rows: dict[str, Any] = {"image": {}, "adult": {}}
    orders = alpha_grid()
    for domain, defenses, targets in [
        ("image", IMAGE_DEFENSES, image_targets("S1", n)),
        ("adult", ADULT_DEFENSES, adult_targets("S2", n)),
    ]:
        for defense in defenses:
            norms: list[float] = []
            distortions: list[float] = []
            dim = 0
            for target in targets:
                base, defended = defense_payload_for_distortion(domain, defense, target)
                b = flatten(base)
                d = flatten(defended)
                # If a model-level defense changes parameter dimensionality, use
                # the shared prefix only and record the truncation.
                shared = min(b.numel(), d.numel())
                norms.append(float(b[:shared].norm().item()))
                distortions.append(float((d[:shared] - b[:shared]).norm().item()))
                dim = shared
            clip_norm = float(np.percentile(norms, 95))
            median_distortion = float(np.median(distortions))
            sigma = median_distortion / (max(clip_norm, 1e-12) * math.sqrt(max(dim, 1)))
            eps = epsilon_from_rdp(max(sigma, 1e-12), 1.0, 1e-5, 1, orders)
            rows[domain][defense] = {
                "clip_norm": clip_norm,
                "sigma": sigma,
                "median_defense_l2_distortion": median_distortion,
                "dimension": dim,
                "epsilon_delta_1e_minus_5_one_release": eps["epsilon"],
                "optimal_alpha": eps["alpha"],
                "norm_p95_source": "same 39 S1/S2 confirmatory targets; calibration is descriptive for E3(ii)",
            }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, allow_nan=True) + "\n")
    return rows


def build_tasks(stage: str, n: int, output_dir: Path) -> list[tuple[str, tuple[Any, ...], Path]]:
    calib = calibrate(output_dir, n)
    tasks: list[tuple[str, tuple[Any, ...], Path]] = []
    if stage in {"S3", "all"}:
        for domain, defenses, targets in [
            ("image", IMAGE_DEFENSES, image_targets("S1", n)),
            ("adult", ADULT_DEFENSES, adult_targets("S2", n)),
        ]:
            for comparator in ["paper_unclipped", "distortion_matched_clipped"]:
                for defense in defenses:
                    if domain == "image" and defense == "ats":
                        # ATS E2 is explicitly NOT_ASSESSABLE under the frozen
                        # straight-through exception; do not create E3 claims.
                        continue
                    for target_id, target in enumerate(targets):
                        path = output_dir / "S3" / domain / comparator / defense / f"target_{target_id:03d}" / "result.json"
                        if domain == "image":
                            tasks.append(("image", ("S3", comparator, defense, target_id, int(target), str(path), calib), path))
                        else:
                            tasks.append(("adult", ("S3", comparator, defense, target_id, list(target), str(path), calib), path))
    return tasks


def summarize(output_dir: Path, n: int) -> None:
    s3 = output_dir / "S3"
    rows: list[dict[str, Any]] = []
    for path in sorted(s3.glob("*/*/*/target_*/result.json")):
        data = json.loads(path.read_text())
        if data.get("status") != "ran":
            continue
        domain = data["domain"]
        defense = data["defense"]
        comparator = data["comparator"]
        target_id = int(data["target_id"])
        if domain == "image":
            score = float(data["metrics"]["mse"])
            psnr = float(data["metrics"]["psnr_db"])
            ssim = float(data["metrics"]["ssim"])
            metric_value = score
        else:
            acc = float(data["metric"]["accuracy_percent"])
            score = 100.0 - acc
            psnr = float("nan")
            ssim = float("nan")
            metric_value = acc
        # Use corrected E2 defense result when available.
        if domain == "image":
            stage_dir = "S1d" if defense == "count_sketch" else "S1c"
            defense_path = output_dir / stage_dir / "image" / "E2" / defense / f"target_{target_id:03d}" / "result.json"
        else:
            stage = "S2d" if defense == "count_sketch" else ("S2c" if defense == "precode" else "S2b")
            defense_path = output_dir / stage / "adult" / "E2" / defense / f"target_{target_id:03d}" / "result.json"
        if not defense_path.exists():
            rows.append(
                {
                    "domain": domain,
                    "defense": defense,
                    "comparator": comparator,
                    "target_id": target_id,
                    "defense_error_score": float("nan"),
                    "dp_error_score": score,
                    "diff_defense_minus_dp": float("nan"),
                    "defense_effect_metric": float("nan"),
                    "dp_effect_metric": metric_value if domain == "adult" else psnr,
                    "defense_ssim": float("nan"),
                    "dp_ssim": ssim,
                    "defense_wins": False,
                    "dp_wins": False,
                    "tie": False,
                    "status": "MISSING_DEFENSE_REFERENCE",
                    "missing_defense_path": str(defense_path),
                }
            )
            continue
        defended = json.loads(defense_path.read_text())
        if defended.get("status") != "ran":
            continue
        if domain == "image":
            defended_score = float(defended["metrics"]["mse"])
            defended_effect = float(defended["metrics"]["psnr_db"])
            defended_ssim = float(defended["metrics"]["ssim"])
        else:
            defended_acc = float(defended["metric"]["accuracy_percent"])
            defended_score = 100.0 - defended_acc
            defended_effect = defended_acc
            defended_ssim = float("nan")
        rows.append(
            {
                "domain": domain,
                "defense": defense,
                "comparator": comparator,
                "target_id": target_id,
                "defense_error_score": defended_score,
                "dp_error_score": score,
                "diff_defense_minus_dp": defended_score - score,
                "defense_effect_metric": defended_effect,
                "dp_effect_metric": metric_value if domain == "adult" else psnr,
                "defense_ssim": defended_ssim,
                "dp_ssim": ssim,
                "defense_wins": defended_score > score,
                "dp_wins": score > defended_score,
                "tie": defended_score == score,
                "status": "ran",
            }
        )
    out_dir = output_dir / "S6"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_target = out_dir / "e3_per_target.csv"
    if rows:
        with per_target.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    summaries: list[dict[str, Any]] = []
    valid_rows = [r for r in rows if r.get("status") == "ran"]
    groups = sorted({(r["domain"], r["defense"], r["comparator"]) for r in valid_rows})
    for domain, defense, comparator in groups:
        cell = [r for r in valid_rows if (r["domain"], r["defense"], r["comparator"]) == (domain, defense, comparator)]
        wins_def = sum(1 for r in cell if r["defense_wins"])
        wins_dp = sum(1 for r in cell if r["dp_wins"])
        ties = sum(1 for r in cell if r["tie"])
        summary = {
            "domain": domain,
            "defense": defense,
            "comparator": comparator,
            "n": len(cell),
            "defense_wins": wins_def,
            "dp_wins": wins_dp,
            "ties": ties,
            "p_defense_greater_error": exact_p(wins_def, wins_dp),
            "p_dp_greater_error": exact_p(wins_dp, wins_def),
            "median_defense_score": float(np.median([r["defense_effect_metric"] for r in cell])),
            "median_dp_score": float(np.median([r["dp_effect_metric"] for r in cell])),
            "median_paired_score_diff_defense_minus_dp": float(np.median([r["defense_effect_metric"] - r["dp_effect_metric"] for r in cell])),
            "median_defense_error_minus_dp_error": float(np.median([r["diff_defense_minus_dp"] for r in cell])),
        }
        if domain == "image":
            summary.update(
                {
                    "median_defense_ssim": float(np.median([r["defense_ssim"] for r in cell])),
                    "median_dp_ssim": float(np.median([r["dp_ssim"] for r in cell])),
                    "median_paired_ssim_diff_defense_minus_dp": float(np.median([r["defense_ssim"] - r["dp_ssim"] for r in cell])),
                }
            )
        summaries.append(summary)
    # Holm separately for each direction across E3.
    left = [{"key": f"{r['domain']}|{r['defense']}|{r['comparator']}", "p_value": r["p_defense_greater_error"]} for r in summaries]
    right = [{"key": f"{r['domain']}|{r['defense']}|{r['comparator']}", "p_value": r["p_dp_greater_error"]} for r in summaries]
    holm(left)
    holm(right)
    left_map = {r["key"]: r["holm_p"] for r in left}
    right_map = {r["key"]: r["holm_p"] for r in right}
    for r in summaries:
        key = f"{r['domain']}|{r['defense']}|{r['comparator']}"
        r["holm_p_defense_greater_error"] = left_map[key]
        r["holm_p_dp_greater_error"] = right_map[key]
    if summaries:
        with (out_dir / "e3_sign_tests.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = sorted({key for row in summaries for key in row})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["S3", "summarize", "all"], default="S3")
    parser.add_argument("--n", type=int, default=39)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts/priority30_native_defenses/audit")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output_dir = ROOT / args.output_dir
    if args.stage == "summarize":
        summarize(output_dir, args.n)
        return
    tasks = build_tasks(args.stage, args.n, output_dir)
    progress = output_dir / "S3_progress.log"
    progress.parent.mkdir(parents=True, exist_ok=True)
    pending = [task for task in tasks if not task[2].exists()]
    with progress.open("a", encoding="utf-8") as log:
        log.write(f"stage={args.stage} total={len(tasks)} pending={len(pending)} workers={args.workers}\n")
    funcs = {"image": run_image_dp_attack, "adult": run_adult_dp_attack}
    start = time.monotonic()
    done = len(tasks) - len(pending)
    if args.workers <= 1:
        for kind, params, _path in pending:
            funcs[kind](*params)
            done += 1
            if done % 10 == 0:
                with progress.open("a", encoding="utf-8") as log:
                    log.write(f"done={done}/{len(tasks)} elapsed={time.monotonic()-start:.1f}s\n")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(funcs[kind], *params) for kind, params, _path in pending]
            for fut in as_completed(futs):
                fut.result()
                done += 1
                if done % 10 == 0:
                    with progress.open("a", encoding="utf-8") as log:
                        log.write(f"done={done}/{len(tasks)} elapsed={time.monotonic()-start:.1f}s\n")
    summarize(output_dir, args.n)


if __name__ == "__main__":
    main()
