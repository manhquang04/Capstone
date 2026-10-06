"""Priority 23 repaired CIFAR-10 attack audit.

Runs the pre-registered repaired image-domain audit on the frozen Priority-7
CIFAR target bundle.  This is development/audit evidence, not a new
confirmatory escalation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.stats import binomtest
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.image_metrics import compute_image_metrics  # noqa: E402
from experiments.run_priority7_image_domain_gate import (  # noqa: E402
    LeNetCIFAR,
    _apply_lightweight_v1_surrogate,
    _candidate_sketches,
    _defense_payload,
    _global_cosine,
    _load_targets,
    _named_update,
    _seed_everything,
)
from privacy.seed_manager import derive_seed  # noqa: E402


TARGET = ROOT / "artifacts/priority7_image_domain/confirmatory_v2_cosine_targets_20260917/cifar10_priority7_targets.pt"
AMENDMENT = ROOT / "protocols/amendments/2026-09-28_priority23_bn_channel_and_cifar_repair.md"
BRANCHES = ("none", "v2_ratio0p95_eta0p01", "v1_medium")
BUDGETS = {
    "original_25_cosine": {"iterations": 25, "attack_lr": 0.05, "tv_lambda": 0.0},
    "strong_250_cosine_tv": {"iterations": 250, "attack_lr": 0.05, "tv_lambda": 1e-4},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_model(seed: int) -> LeNetCIFAR:
    _seed_everything(seed)
    model = LeNetCIFAR()
    model.train()
    return model


def image_metric_dict(original: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, float]:
    vals = [compute_image_metrics(original[i], reconstruction[i]) for i in range(original.shape[0])]
    return {
        "mse": float(np.mean([v.mse for v in vals])),
        "psnr": float(np.mean([v.psnr for v in vals])),
        "ssim": float(np.mean([v.ssim for v in vals])),
    }


def tv_loss(image: torch.Tensor) -> torch.Tensor:
    return (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean() + (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean()


def defended_signal(branch: str, observed: dict[str, torch.Tensor], group: int, restart: int):
    if branch == "none":
        return observed, ("none", None)
    return _defense_payload(observed, branch, group, restart)


def candidate_for(branch: str, delta: dict[str, torch.Tensor], payload):
    if branch == "none":
        return delta
    # Reuse the exact Priority-7 candidate mapping for v1/v2.
    from experiments.run_priority7_image_domain_gate import _candidate_defended

    return _candidate_defended(delta, payload)


def cifar_mean_image() -> torch.Tensor:
    dataset = CIFAR10(root=str(ROOT / "datasets/cifar10"), train=True, download=True, transform=ToTensor())
    total = torch.zeros(3, 32, 32)
    for image, _ in dataset:
        total += image
    return (total / len(dataset)).unsqueeze(0)


def optimize_one(
    *,
    model: LeNetCIFAR,
    labels: torch.Tensor,
    target_images: torch.Tensor,
    signal: dict[str, torch.Tensor],
    payload,
    branch: str,
    budget: dict,
    seed: int,
    group: int,
    restart: int,
    local_lr: float,
) -> dict:
    generator = torch.Generator().manual_seed(derive_seed(seed, "priority23-cifar-init", branch, group, restart))
    initial_logits = torch.randn(target_images.shape, generator=generator, dtype=torch.float32)
    latent = initial_logits.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=float(budget["attack_lr"]))
    best_objective = float("inf")
    best_step = 0
    best_latent = latent.detach().clone()
    history = []
    for step in range(int(budget["iterations"]) + 1):
        reconstruction = torch.sigmoid(latent)
        delta = _named_update(model, reconstruction, labels, local_lr)
        defended = candidate_for(branch, delta, payload)
        loss = _global_cosine(defended, signal)
        if float(budget["tv_lambda"]) > 0:
            loss = loss + float(budget["tv_lambda"]) * tv_loss(reconstruction)
        objective = float(loss.detach())
        history.append(objective)
        if objective < best_objective:
            best_objective = objective
            best_step = step
            best_latent = latent.detach().clone()
        if step < int(budget["iterations"]):
            (grad,) = torch.autograd.grad(loss, latent)
            optimizer.zero_grad()
            latent.grad = grad
            optimizer.step()
    reconstruction = torch.sigmoid(best_latent).detach()
    prior = torch.sigmoid(initial_logits).detach()
    return {
        "reconstruction": reconstruction,
        "prior": prior,
        "best_objective": best_objective,
        "best_step": best_step,
        "history": history,
    }


def run(args) -> None:
    torch.set_num_threads(1)
    out = args.output.resolve()
    if out.exists():
        if not args.resume:
            raise FileExistsError(f"{out} exists; pass --resume for technical continuation")
    else:
        out.mkdir(parents=True, exist_ok=False)
    groups = _load_targets(TARGET)
    groups = groups[: args.groups]
    mean_image = cifar_mean_image()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": str(AMENDMENT.relative_to(ROOT)),
        "amendment_sha256": sha256(AMENDMENT),
        "target": str(TARGET.relative_to(ROOT)),
        "target_sha256": sha256(TARGET),
        "groups": len(groups),
        "restarts": args.restarts,
        "branches": list(BRANCHES),
        "budgets": BUDGETS,
        "controls": ["prior", "decoy", "gray_0p5", "cifar_train_mean"],
        "torch_num_threads": torch.get_num_threads(),
    }
    manifest_path = out / "execution_manifest.json"
    if not manifest_path.exists():
        dump(manifest_path, manifest)
    rows = []
    csv_path = out / "priority23_cifar_repaired_per_group.csv"
    completed_keys: set[tuple[str, str, int]] = set()
    if args.resume and csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                # Convert numeric fields back to float/int for downstream summaries.
                row["group"] = int(row["group"])
                for key, value in list(row.items()):
                    if key in {"budget", "branch", "source_indices"}:
                        continue
                    if key.endswith("restart") or key == "group":
                        row[key] = int(value)
                    else:
                        try:
                            row[key] = float(value)
                        except (TypeError, ValueError):
                            pass
                rows.append(row)
                completed_keys.add((str(row["budget"]), str(row["branch"]), int(row["group"])))
    fieldnames = [
        "budget", "branch", "group", "source_indices", "own_restart", "decoy_restart",
        "own_objective", "decoy_objective",
        "own_mse", "own_psnr", "own_ssim",
        "decoy_mse", "decoy_psnr", "decoy_ssim",
        "prior_mse", "prior_psnr", "prior_ssim",
        "gray_0p5_mse", "gray_0p5_psnr", "gray_0p5_ssim",
        "cifar_train_mean_mse", "cifar_train_mean_psnr", "cifar_train_mean_ssim",
    ]
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
    example_groups = {0, 1, 2, 3}
    examples: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for budget_name, budget in BUDGETS.items():
        for branch in BRANCHES:
            for group_id, group in enumerate(groups):
                if (budget_name, branch, group_id) in completed_keys:
                    print(json.dumps({"budget": budget_name, "branch": branch, "group": group_id, "status": "reused"}), flush=True)
                    continue
                images = group["images"].to(dtype=torch.float32)
                labels = group["labels"].to(dtype=torch.long)
                decoy_group = groups[(group_id + 1) % len(groups)]
                decoy_images = decoy_group["images"].to(dtype=torch.float32)
                decoy_labels = decoy_group["labels"].to(dtype=torch.long)
                model = make_model(derive_seed(args.seed, "model", group_id))
                observed = {name: value.detach() for name, value in _named_update(model, images, labels, args.local_lr).items()}
                decoy_model = make_model(derive_seed(args.seed, "model", (group_id + 1) % len(groups)))
                decoy_observed_raw = {name: value.detach() for name, value in _named_update(decoy_model, decoy_images, decoy_labels, args.local_lr).items()}
                best_own = None
                best_decoy = None
                for restart in range(args.restarts):
                    own_signal, own_payload = defended_signal(branch, observed, group_id, restart)
                    decoy_signal, decoy_payload = defended_signal(branch, decoy_observed_raw, (group_id + 1) % len(groups), restart)
                    own = optimize_one(model=model, labels=labels, target_images=images, signal=own_signal, payload=own_payload, branch=branch, budget=budget, seed=args.seed, group=group_id, restart=restart, local_lr=args.local_lr)
                    decoy = optimize_one(model=model, labels=labels, target_images=images, signal=decoy_signal, payload=decoy_payload, branch=branch, budget=budget, seed=args.seed, group=group_id, restart=restart, local_lr=args.local_lr)
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
                    "budget": budget_name,
                    "branch": branch,
                    "group": group_id,
                    "source_indices": json.dumps(group["source_indices"]),
                    "own_restart": best_own["restart"],
                    "decoy_restart": best_decoy["restart"],
                    "own_objective": best_own["best_objective"],
                    "decoy_objective": best_decoy["best_objective"],
                }
                for control, vals in metrics.items():
                    for metric, value in vals.items():
                        row[f"{control}_{metric}"] = value
                rows.append(row)
                with csv_path.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writerow(row)
                if group_id in example_groups:
                    key = f"{budget_name}__{branch}"
                    examples.setdefault(key, [])
                    examples[key].append(("original", images[0].detach().cpu()))
                    examples[key].append(("own", best_own["reconstruction"][0].detach().cpu()))
                    examples[key].append(("decoy", best_decoy["reconstruction"][0].detach().cpu()))
                    examples[key].append(("gray", gray[0].detach().cpu()))
                print(json.dumps({"budget": budget_name, "branch": branch, "group": group_id, "status": "done"}), flush=True)
    summary = summarize(rows)
    dump(out / "priority23_cifar_repaired_summary.json", summary)
    image_paths = write_example_grids(out / "examples", examples)
    final = manifest | {
        "per_group_csv_sha256": sha256(csv_path),
        "summary_json_sha256": sha256(out / "priority23_cifar_repaired_summary.json"),
        "example_pngs": image_paths,
    }
    dump(out / "manifest.json", final)
    print(json.dumps({"summary": summary, "hashes": {"csv": sha256(csv_path), "summary": sha256(out / "priority23_cifar_repaired_summary.json")}}, indent=2))


def summarize(rows: list[dict]) -> dict:
    out = {}
    for budget in sorted({row["budget"] for row in rows}):
        for branch in sorted({row["branch"] for row in rows}):
            subset = [row for row in rows if row["budget"] == budget and row["branch"] == branch]
            if not subset:
                continue
            key = f"{budget}__{branch}"
            own_mse = np.asarray([row["own_mse"] for row in subset], dtype=float)
            decoy_mse = np.asarray([row["decoy_mse"] for row in subset], dtype=float)
            gray_mse = np.asarray([row["gray_0p5_mse"] for row in subset], dtype=float)
            mean_mse = np.asarray([row["cifar_train_mean_mse"] for row in subset], dtype=float)
            prior_mse = np.asarray([row["prior_mse"] for row in subset], dtype=float)
            none_psnr = None
            if branch != "none":
                none_key = f"{budget}__none"
                none_rows = [row for row in rows if row["budget"] == budget and row["branch"] == "none"]
                if len(none_rows) == len(subset):
                    none_psnr = float(np.mean([row["own_psnr"] for row in none_rows]))
            wins_decoy = int(np.sum(own_mse < decoy_mse))
            wins_gray = int(np.sum(own_mse < gray_mse))
            wins_mean = int(np.sum(own_mse < mean_mse))
            wins_prior = int(np.sum(own_mse < prior_mse))
            out[key] = {
                "n": len(subset),
                "own_mse_mean": float(own_mse.mean()),
                "own_mse_median": float(np.median(own_mse)),
                "own_psnr_mean": float(np.mean([row["own_psnr"] for row in subset])),
                "own_ssim_mean": float(np.mean([row["own_ssim"] for row in subset])),
                "wins_vs_decoy": wins_decoy,
                "p_vs_decoy": float(binomtest(wins_decoy, len(subset), 0.5, alternative="greater").pvalue),
                "wins_vs_gray": wins_gray,
                "p_vs_gray": float(binomtest(wins_gray, len(subset), 0.5, alternative="greater").pvalue),
                "wins_vs_mean": wins_mean,
                "p_vs_mean": float(binomtest(wins_mean, len(subset), 0.5, alternative="greater").pvalue),
                "wins_vs_prior": wins_prior,
                "p_vs_prior": float(binomtest(wins_prior, len(subset), 0.5, alternative="greater").pvalue),
                "defense_efficacy_ratio_psnr_vs_none": (
                    float(np.mean([row["own_psnr"] for row in subset]) / none_psnr)
                    if none_psnr not in (None, 0.0)
                    else None
                ),
            }
    return out


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    arr = (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def write_example_grids(output: Path, examples: dict[str, list[tuple[str, torch.Tensor]]]) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, items in examples.items():
        cell = 64
        label_h = 14
        cols = 4
        rows = math.ceil(len(items) / cols)
        canvas = Image.new("RGB", (cols * cell, rows * (cell + label_h)), "white")
        draw = ImageDraw.Draw(canvas)
        for idx, (label, tensor) in enumerate(items):
            x = (idx % cols) * cell
            y = (idx // cols) * (cell + label_h)
            img = tensor_to_pil(tensor).resize((cell, cell), Image.Resampling.NEAREST)
            canvas.paste(img, (x, y))
            draw.text((x + 1, y + cell), label[:8], fill=(0, 0, 0))
        path = output / f"{key}.png"
        canvas.save(path)
        paths[key] = str(path.relative_to(ROOT))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/priority23_cifar_repaired_20260928")
    parser.add_argument("--seed", type=int, default=2026092801)
    parser.add_argument("--groups", type=int, default=39)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--local-lr", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
