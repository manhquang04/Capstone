"""Priority 7 CIFAR-10 image-domain development gates.

This runner is development-only. It creates source-disjoint CIFAR-10 target
sets and runs the pre-authorized image-domain gate cells from
``protocols/config/rq1_image_domain_pre_pilot.yaml``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.image_metrics import compute_image_metrics
from dna_encoder.transform_defense import DNATransformConfig
from dna_encoder.transform_defense_v2 import DNATransformV2Config
from experiments.run_phase4_dna_level1_forward_attack import (
    _transmit_observed,
)
from experiments.analyze_phase4_dna_level1_surrogate_inversion import _surrogate_seed
from experiments.run_phase4_dna_v2_sketch_space_attack import (
    _candidate_sketches,
    _observed_sketches,
)
from privacy.seed_manager import derive_seed


V1_MEDIUM = dict(block_size=256, mix_ratio=0.10, keep_ratio=0.85, shrink_factor=0.40, seed=681958327)
V2 = dict(compression_ratio=0.95, quantization_eta=0.01, seed=20260916)
AUTHORIZED_CELLS = (
    ("v1_medium", "GEN_IDLG_STYLE"),
    ("v2_ratio0p95_eta0p01", "GEN_COSINE_TV"),
    ("v1_medium", "GEN_RAW_LIFT"),
    ("v2_ratio0p95_eta0p01", "GEN_RAW_LIFT"),
)


def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


class LeNetCIFAR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Minimal LeNet-5-style CIFAR-10 variant frozen in
        # protocols/amendments/2026-09-17_priority7_image_domain_minimal_lenet_timeout.md.
        # CIFAR images remain 3x32x32; only channel/FC widths are reduced.
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(1024, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def _make_model(seed: int) -> LeNetCIFAR:
    _seed_everything(seed)
    model = LeNetCIFAR()
    model.train()
    return model


def _named_update(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, lr: float) -> dict[str, torch.Tensor]:
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    grads = torch.autograd.grad(loss, [param for _, param in params], create_graph=images.requires_grad)
    return {name: -lr * grad for (name, _), grad in zip(params, grads)}


def _global_cosine(candidate: dict[str, torch.Tensor], signal: dict[str, torch.Tensor]) -> torch.Tensor:
    keys = [key for key in signal if key in candidate and signal[key].is_floating_point()]
    left = torch.cat([candidate[key].reshape(-1) for key in keys])
    right = torch.cat([signal[key].reshape(-1).to(dtype=left.dtype, device=left.device) for key in keys])
    return 1.0 - F.cosine_similarity(left, right, dim=0, eps=1e-12)


def _dict_l2(candidate: dict[str, torch.Tensor], signal: dict[str, torch.Tensor]) -> torch.Tensor:
    losses = []
    for key, target in signal.items():
        if key in candidate and target.is_floating_point():
            losses.append(F.mse_loss(candidate[key], target.to(dtype=candidate[key].dtype, device=candidate[key].device)))
    return torch.stack(losses).mean()


def _defense_payload(observed: dict[str, torch.Tensor], defense: str, group_id: int, restart: int):
    if defense == "v1_medium":
        config = DNATransformConfig(**V1_MEDIUM)
        transmitted = _transmit_observed(observed, config)
        plan = _lightweight_v1_surrogate_plan(observed, config, group_id, restart)
        return transmitted, ("v1", plan)
    if defense == "v2_ratio0p95_eta0p01":
        config = DNATransformV2Config(**V2)
        payload = _observed_sketches(observed, config, group_id)
        sketch = {key: item["sketch"].to(dtype=observed[key].dtype) for key, item in payload.items()}
        return sketch, ("v2", payload)
    raise ValueError(defense)


def _candidate_defended(delta: dict[str, torch.Tensor], payload) -> dict[str, torch.Tensor]:
    kind, plan = payload
    if kind == "v1":
        return _apply_lightweight_v1_surrogate(delta, plan)
    args = SimpleNamespace(
        compression_ratio=V2["compression_ratio"],
        quantization_eta=V2["quantization_eta"],
        v2_base_seed=V2["seed"],
        ste_quantization=True,
    )
    return _candidate_sketches(delta, plan, args)


def _lightweight_v1_surrogate_plan(
    state_delta: dict[str, torch.Tensor],
    config: DNATransformConfig,
    group_id: int,
    realization_id: int,
) -> dict[str, list[tuple[int, int, torch.Tensor, torch.Tensor]]]:
    """Sparse/blockwise equivalent of the dense v1 surrogate matrix plan.

    The historical tabular runner materializes a dense 256x256 matrix for every
    block. That is acceptable for small MLP updates but not for CIFAR CNN
    gradients. This plan stores only the permutation and shrink vector, applying
    the same linear map without changing the formula.
    """

    plan: dict[str, list[tuple[int, int, torch.Tensor, torch.Tensor]]] = {}
    for tensor_index, (name, value) in enumerate(state_delta.items()):
        if not value.is_floating_point():
            continue
        rows = []
        for block_index, start in enumerate(range(0, value.numel(), config.block_size)):
            stop = min(start + config.block_size, value.numel())
            block_size = stop - start
            seed = _surrogate_seed(config, group_id, realization_id, tensor_index, block_index)
            rng = np.random.default_rng(seed)
            permutation = torch.as_tensor(rng.permutation(block_size), dtype=torch.long, device=value.device)
            shrink = np.ones(block_size, dtype=np.float32)
            low_count = int(round((1.0 - min(max(config.keep_ratio, 0.0), 1.0)) * block_size))
            if low_count:
                low_positions = rng.choice(block_size, size=min(low_count, block_size), replace=False)
                shrink[low_positions] = float(config.shrink_factor)
            shrink_tensor = torch.as_tensor(shrink, dtype=value.dtype, device=value.device)
            rows.append((start, stop, permutation, shrink_tensor))
        plan[name] = rows
    return plan


def _apply_lightweight_v1_surrogate(
    state_delta: dict[str, torch.Tensor],
    plan: dict[str, list[tuple[int, int, torch.Tensor, torch.Tensor]]],
) -> dict[str, torch.Tensor]:
    transformed = {}
    for name, value in state_delta.items():
        if not value.is_floating_point():
            transformed[name] = value.clone()
            continue
        flat = value.reshape(-1)
        blocks = []
        for start, stop, permutation, shrink in plan[name]:
            block = flat[start:stop]
            blocks.append((1.0 - V1_MEDIUM["mix_ratio"]) * block + V1_MEDIUM["mix_ratio"] * shrink * block[permutation])
        transformed[name] = torch.cat(blocks).reshape_as(value)
    return transformed


def _objective(
    defended: dict[str, torch.Tensor],
    signal: dict[str, torch.Tensor],
    generation: str,
) -> torch.Tensor:
    if generation == "GEN_COSINE_TV":
        return _global_cosine(defended, signal)
    if generation in {"GEN_IDLG_STYLE", "GEN_RAW_LIFT"}:
        return _dict_l2(defended, signal)
    raise ValueError(generation)


def _load_targets(path: Path) -> list[dict]:
    payload = torch.load(path, weights_only=False)
    if not isinstance(payload, dict) or "groups" not in payload:
        raise ValueError(f"not a Priority 7 target bundle: {path}")
    return payload["groups"]


def _existing_image_indices() -> dict[str, set[int]]:
    historical = {}
    for manifest in sorted((ROOT / "artifacts/priority7_image_domain").rglob("target_manifest.json")):
        try:
            payload = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            continue
        indices = payload.get("target_indices", [])
        historical[str(manifest.relative_to(ROOT))] = {int(v) for v in indices}
    return historical


def prepare_targets(args) -> None:
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    _seed_everything(args.seed)
    dataset = CIFAR10(root=str(ROOT / "datasets/cifar10"), train=True, download=True, transform=ToTensor())
    historical = _existing_image_indices()
    excluded = set().union(*historical.values()) if historical else set()
    rng = np.random.default_rng(args.seed)
    available = np.asarray([idx for idx in range(len(dataset)) if idx not in excluded], dtype=np.int64)
    needed = args.groups * args.images_per_group
    if len(available) < needed:
        raise RuntimeError(f"not enough fresh CIFAR-10 indices: available={len(available)} needed={needed}")
    chosen = rng.choice(available, size=needed, replace=False).tolist()
    groups = []
    for group_id in range(args.groups):
        indices = chosen[group_id * args.images_per_group : (group_id + 1) * args.images_per_group]
        images = []
        labels = []
        for idx in indices:
            image, label = dataset[int(idx)]
            images.append(image)
            labels.append(int(label))
        groups.append(
            {
                "group_id": group_id,
                "source_indices": [int(v) for v in indices],
                "images": torch.stack(images).to(dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        )
    bundle = {
        "dataset": "CIFAR-10",
        "split": "train",
        "groups": groups,
        "seed": args.seed,
        "images_per_group": args.images_per_group,
    }
    target_path = output / "cifar10_priority7_targets.pt"
    torch.save(bundle, target_path)
    new_indices = set(int(v) for v in chosen)
    overlaps = {path: len(new_indices & values) for path, values in historical.items()}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "CIFAR-10",
        "split": "train",
        "seed": args.seed,
        "groups": args.groups,
        "images_per_group": args.images_per_group,
        "target_indices": sorted(new_indices),
        "historical_image_manifests_checked": len(historical),
        "overlaps": overlaps,
        "max_overlap": max(overlaps.values(), default=0),
        "disjointness_gate": "PASS" if max(overlaps.values(), default=0) == 0 else "FAIL",
        "target_sha256": _sha256(target_path),
    }
    if manifest["disjointness_gate"] != "PASS":
        raise RuntimeError("CIFAR image-domain source-disjointness gate failed")
    _dump(output / "target_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def _image_job(job: dict) -> dict:
    torch.set_num_threads(1)
    _seed_everything(derive_seed(job["seed"], "job", job["defense"], job["generation"], job["group"], job["restart"]))
    groups = _load_targets(Path(job["target"]))
    group = groups[job["group"]]
    images = group["images"].to(dtype=torch.float32)
    labels = group["labels"].to(dtype=torch.long)
    model = _make_model(derive_seed(job["seed"], "model", job["group"]))
    observed = {name: value.detach() for name, value in _named_update(model, images, labels, job["local_lr"]).items()}
    observed_defended, payload = _defense_payload(observed, job["defense"], job["group"], job["restart"])
    folder = Path(job["folder"])
    folder.mkdir(parents=True, exist_ok=True)

    init_seed = derive_seed(job["seed"], "init", job["defense"], job["generation"], job["group"], job["restart"])
    generator = torch.Generator().manual_seed(init_seed)
    initial_logits = torch.randn(images.shape, generator=generator, dtype=torch.float32)
    prior_reconstruction = torch.sigmoid(initial_logits).detach()
    prior_metrics = _score(images, prior_reconstruction)

    rows = []
    for method in ("baseline", "zero_update"):
        signal = observed_defended if method == "baseline" else {k: torch.zeros_like(v) for k, v in observed_defended.items()}
        latent = initial_logits.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=job["attack_lr"])
        best = float("inf")
        best_step = 0
        best_latent = latent.detach().clone()
        for step in range(job["iterations"] + 1):
            reconstruction = torch.sigmoid(latent)
            delta = _named_update(model, reconstruction, labels, job["local_lr"])
            defended = _candidate_defended(delta, payload)
            loss = _objective(defended, signal, job["generation"])
            value = float(loss.detach())
            if value < best:
                best = value
                best_step = step
                best_latent = latent.detach().clone()
            if step < job["iterations"]:
                (grad,) = torch.autograd.grad(loss, latent)
                optimizer.zero_grad()
                latent.grad = grad
                optimizer.step()
        reconstruction = torch.sigmoid(best_latent).detach()
        metrics = _score(images, reconstruction)
        artifact = {
            "dataset": "CIFAR-10",
            "defense": job["defense"],
            "generation": job["generation"],
            "group": job["group"],
            "restart": job["restart"],
            "method": method,
            "source_indices": group["source_indices"],
            "labels": labels,
            "original": images,
            "reconstruction": reconstruction,
            "prior_reconstruction": prior_reconstruction,
            "best_objective": best,
            "best_step": best_step,
            "objective": "global_cosine" if job["generation"] == "GEN_COSINE_TV" else "l2_defended_update",
            "labels_fixed_to_known_cifar10_labels": True,
        }
        torch.save(artifact, folder / f"{method}.pt")
        rows.append({"method": method, "objective": best, "best_step": best_step, "metrics": metrics, "prior": prior_metrics})
    _dump(folder / "results.json", rows)
    return {key: job[key] for key in ("defense", "generation", "group", "restart")} | {"status": "SUCCESS", "rows": rows}


def _score(original: torch.Tensor, reconstruction: torch.Tensor) -> dict:
    per_image = []
    for i in range(original.shape[0]):
        metrics = compute_image_metrics(original[i], reconstruction[i])
        per_image.append({"mse": metrics.mse, "psnr": metrics.psnr, "ssim": metrics.ssim})
    return {
        "image_mse": float(np.mean([row["mse"] for row in per_image])),
        "psnr": float(np.mean([row["psnr"] for row in per_image])),
        "ssim": float(np.mean([row["ssim"] for row in per_image])),
        "per_image": per_image,
    }


def _sign_tail(wins: int, total: int) -> float:
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / 2**total if total else 1.0


def _evaluate(records: list[dict]) -> dict:
    selected = []
    for group in sorted({int(row["group"]) for row in records}):
        candidates = [row for row in records if int(row["group"]) == group]
        baseline_pool = [(row, value) for row in candidates for value in row["rows"] if value["method"] == "baseline"]
        zero_pool = [(row, value) for row in candidates for value in row["rows"] if value["method"] == "zero_update"]
        baseline_job, baseline = min(baseline_pool, key=lambda pair: pair[1]["objective"])
        zero_job, zero = min(zero_pool, key=lambda pair: pair[1]["objective"])
        prior_values = [value["prior"]["image_mse"] for row in candidates for value in row["rows"] if value["method"] == "baseline"]
        prior_mse = float(np.mean(prior_values))
        baseline_mse = float(baseline["metrics"]["image_mse"])
        zero_mse = float(zero["metrics"]["image_mse"])
        selected.append(
            {
                "group": group,
                "baseline_restart": int(baseline_job["restart"]),
                "zero_restart": int(zero_job["restart"]),
                "baseline_mse": baseline_mse,
                "prior_mse": prior_mse,
                "zero_mse": zero_mse,
                "baseline_psnr": float(baseline["metrics"]["psnr"]),
                "baseline_ssim": float(baseline["metrics"]["ssim"]),
                "baseline_minus_prior": baseline_mse - prior_mse,
                "baseline_minus_zero": baseline_mse - zero_mse,
            }
        )
    gates = {}
    for control in ("prior", "zero"):
        values = np.asarray([row[f"baseline_minus_{control}"] for row in selected], dtype=float)
        non_ties = values[values != 0.0]
        wins = int((non_ties < 0).sum())
        p = _sign_tail(wins, len(non_ties))
        gates[control] = {
            "wins": wins,
            "non_ties": int(len(non_ties)),
            "mean_difference": float(values.mean()) if len(values) else float("nan"),
            "median_difference": float(np.median(values)) if len(values) else float("nan"),
            "one_sided_exact_sign_p": p,
            "pass": bool(
                len(non_ties) > 0
                and float(values.mean()) < 0
                and float(np.median(values)) < 0
                and p < 0.05
            ),
        }
    gates["overall"] = bool(gates["prior"]["pass"] and gates["zero"]["pass"])
    return {"gate": gates, "selected": selected}


def execute(args) -> None:
    torch.set_num_threads(1)
    output = args.output.resolve()
    if args.resume:
        if not output.is_dir():
            raise RuntimeError("--resume requires existing output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
    cells = []
    requested = AUTHORIZED_CELLS if not args.cells else tuple(tuple(item.split("__", 1)) for item in args.cells)
    for cell in requested:
        if cell not in AUTHORIZED_CELLS:
            raise ValueError(f"unauthorized cell: {cell}")
        cells.append(cell)
    jobs = []
    for defense, generation in cells:
        for group in range(args.groups):
            for restart in range(args.restarts):
                folder = output / defense / generation / f"group_{group}" / f"restart_{restart}"
                jobs.append(
                    {
                        "target": str(args.target.resolve()),
                        "seed": args.seed,
                        "defense": defense,
                        "generation": generation,
                        "group": group,
                        "restart": restart,
                        "folder": str(folder),
                        "iterations": args.iterations,
                        "attack_lr": args.attack_lr,
                        "local_lr": args.local_lr,
                    }
                )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": "protocols/config/rq1_image_domain_pre_pilot.yaml",
        "target": str(args.target),
        "target_sha256": _sha256(args.target.resolve()),
        "cells": [f"{a}__{b}" for a, b in cells],
        "jobs": len(jobs),
        "groups": args.groups,
        "restarts": args.restarts,
        "iterations": args.iterations,
        "attack_lr": args.attack_lr,
        "local_lr": args.local_lr,
        "torch_num_threads": 1,
        "development_only": True,
    }
    if not args.resume:
        _dump(output / "execution_manifest.json", manifest)
    records = []
    pending = []
    for job in jobs:
        result_path = Path(job["folder"]) / "results.json"
        if args.resume and result_path.exists():
            records.append(
                {key: job[key] for key in ("defense", "generation", "group", "restart")}
                | {"status": "SUCCESS_REUSED", "rows": json.loads(result_path.read_text())}
            )
        else:
            pending.append(job)
    if args.workers == 1:
        for job in pending:
            row = _image_job(job)
            records.append(row)
            print(json.dumps({key: row[key] for key in ("defense", "generation", "group", "restart", "status")}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
            futures = [pool.submit(_image_job, job) for job in pending]
            for future in as_completed(futures):
                row = future.result()
                records.append(row)
                print(json.dumps({key: row[key] for key in ("defense", "generation", "group", "restart", "status")}), flush=True)
    summaries = {}
    for defense, generation in cells:
        rows = [row for row in records if row["defense"] == defense and row["generation"] == generation]
        summaries[f"{defense}__{generation}"] = _evaluate(rows)
    report = manifest | {"completed_jobs": len(records), "failed_jobs": 0, "cells_summary": summaries}
    _dump(output / "priority7_image_gate_report.json", report)
    print(json.dumps({"completed_jobs": len(records), "cells": {key: value["gate"] for key, value in summaries.items()}}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-targets")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--seed", type=int, required=True)
    prep.add_argument("--groups", type=int, default=8)
    prep.add_argument("--images-per-group", type=int, required=True)
    run = sub.add_parser("execute")
    run.add_argument("--target", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--groups", type=int, default=8)
    run.add_argument("--restarts", type=int, default=4)
    run.add_argument("--iterations", type=int, default=300)
    run.add_argument("--attack-lr", type=float, default=0.05)
    run.add_argument("--local-lr", type=float, default=0.01)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--cells", nargs="+", help="Optional authorized cells as defense__generation")
    args = parser.parse_args()
    if args.command == "prepare-targets":
        prepare_targets(args)
    else:
        if not 1 <= args.workers <= 9:
            raise ValueError("--workers must be in [1, 9]")
        execute(args)


if __name__ == "__main__":
    main()
