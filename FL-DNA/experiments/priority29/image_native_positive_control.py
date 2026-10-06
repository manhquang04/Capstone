"""Priority 29 CIFAR-10 native positive-control gate.

This driver ports the validated Geiping et al. reference setup from
external_defenses/reference_image.py into the project artifact tree.  It imports
the external reference code read-only and writes outputs only under
artifacts/priority29_native_audit/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external_defenses" / "invertinggradients"))
sys.path.insert(0, str(ROOT))
import inversefed  # noqa: E402


CONFIG = {
    "signed": True,
    "boxed": True,
    "cost_fn": "sim",
    "indices": "def",
    "weights": "equal",
    "lr": 0.1,
    "optim": "adam",
    "restarts": 1,
    "max_iterations": 4800,
    "total_variation": 0.01,
    "init": "randn",
    "filter": "none",
    "lr_decay": True,
    "scoring_choice": "loss",
}


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


def image_metrics(raw: torch.Tensor, estimate: torch.Tensor) -> dict[str, float]:
    raw_np = raw[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    est_np = estimate[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    mse = float(np.mean((raw_np - est_np) ** 2))
    psnr = float("inf") if mse == 0.0 else float(-10.0 * np.log10(mse))
    ssim = float(structural_similarity(raw_np, est_np, data_range=1.0, channel_axis=2))
    return {"mse": mse, "psnr_db": psnr, "ssim": ssim}


def tensor_to_png(t: torch.Tensor, path: Path) -> None:
    arr = (
        t[0]
        .detach()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
        * 255.0
    )
    Image.fromarray(arr.round().astype("uint8")).save(path)


def target_indices(stage: str, n: int) -> list[int]:
    # Fresh Priority 29 image targets: CIFAR-10 test split. Previous project
    # CIFAR attack artifacts used train split; within Priority 29, each stage is
    # a disjoint slice of this deterministic shuffle.
    seed = 29_000_801
    population = list(range(10_000))
    rng = random.Random(seed)
    rng.shuffle(population)
    offsets = {"n8": 0, "n24": 100, "confirmatory": 500}
    if stage not in offsets:
        raise ValueError(f"unknown stage {stage!r}")
    return population[offsets[stage] : offsets[stage] + n]


def load_cifar_mean() -> torch.Tensor:
    train = CIFAR10(str(ROOT / "datasets" / "cifar10"), train=True, download=False)
    mean = torch.from_numpy(train.data.mean(axis=0).astype("float32")).permute(2, 0, 1)
    return (mean / 255.0).unsqueeze(0)


def reconstruct_one(raw: torch.Tensor, label: int, attack_seed: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(42)
    np.random.seed(42)

    model, model_seed = inversefed.construct_model("LeNetZhu", seed=42)
    model.eval()
    labels = torch.tensor([label], dtype=torch.long)
    dm = torch.tensor(inversefed.consts.cifar10_mean)[:, None, None]
    ds = torch.tensor(inversefed.consts.cifar10_std)[:, None, None]
    inputs = (raw - dm) / ds
    loss = F.cross_entropy(model(inputs), labels)
    observed = [v.detach() for v in torch.autograd.grad(loss, model.parameters())]

    attack = inversefed.GradientReconstructor(model, (dm, ds), CONFIG.copy(), num_images=1)

    torch.manual_seed(attack_seed)
    np.random.seed(attack_seed % (2**32 - 1))
    prior_normalized = attack._init_images((3, 32, 32))[0].detach()
    prior_raw = prior_normalized * ds + dm

    torch.manual_seed(attack_seed)
    np.random.seed(attack_seed % (2**32 - 1))
    start = time.monotonic()
    reconstructed_normalized, stats = attack.reconstruct(
        observed, labels, img_shape=(3, 32, 32), dryrun=False, eval=True
    )
    elapsed = time.monotonic() - start
    reconstructed_raw = reconstructed_normalized * ds + dm
    return {
        "reconstruction": reconstructed_raw.detach().cpu(),
        "prior": prior_raw.detach().cpu(),
        "stats": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in dict(stats).items()},
        "elapsed_seconds": elapsed,
        "model_seed": model_seed,
    }


def make_grid(rows: list[dict[str, Any]], out_png: Path, max_rows: int = 8) -> None:
    thumbs: list[tuple[str, Image.Image]] = []
    for row in rows[:max_rows]:
        for key in ["raw_png", "reconstruction_png", "prior_png", "decoy_png"]:
            img = Image.open(row[key]).resize((64, 64))
            thumbs.append((key.replace("_png", ""), img))
    cols = 4
    w, h = 64, 82
    grid = Image.new("RGB", (cols * w, min(max_rows, len(rows)) * h), "white")
    draw = ImageDraw.Draw(grid)
    for idx, (label, img) in enumerate(thumbs):
        r, c = divmod(idx, cols)
        grid.paste(img, (c * w, r * h + 14))
        draw.text((c * w + 2, r * h + 1), label, fill=(0, 0, 0))
    grid.save(out_png)


def run_stage(args: argparse.Namespace) -> None:
    out_dir = ROOT / args.output_dir / args.stage
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_existing:
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    data = CIFAR10(str(ROOT / "datasets" / "cifar10"), train=False, download=False, transform=ToTensor())
    cifar_mean = load_cifar_mean()
    gray = torch.full((1, 3, 32, 32), 0.5)
    indices = target_indices(args.stage, args.n)
    if len(set(indices)) != len(indices):
        raise RuntimeError("duplicate target indices in stage")

    rows: list[dict[str, Any]] = []
    reconstructions: list[torch.Tensor] = []
    raws: list[torch.Tensor] = []

    run_start = time.monotonic()
    for group, index in enumerate(indices):
        raw, label = data[index]
        raw = raw.unsqueeze(0)
        raws.append(raw)
        target_dir = out_dir / f"group_{group:03d}_idx_{index:05d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        attack_seed = 29_010_000 + group
        result = reconstruct_one(raw, int(label), attack_seed)
        recon = result["reconstruction"]
        prior = result["prior"]
        reconstructions.append(recon)

        raw_png = target_dir / "raw.png"
        recon_png = target_dir / "reconstruction.png"
        prior_png = target_dir / "prior.png"
        tensor_to_png(raw, raw_png)
        tensor_to_png(recon, recon_png)
        tensor_to_png(prior, prior_png)
        torch.save(
            {
                "index": index,
                "label": int(label),
                "raw": raw,
                "reconstruction": recon,
                "prior": prior,
                "attack_seed": attack_seed,
                "stats": result["stats"],
            },
            target_dir / "artifact.pt",
        )

        row = {
            "stage": args.stage,
            "group": group,
            "cifar10_split": "test",
            "cifar10_index": index,
            "label": int(label),
            "attack_seed": attack_seed,
            "raw_png": str(raw_png),
            "reconstruction_png": str(recon_png),
            "prior_png": str(prior_png),
            "elapsed_seconds": result["elapsed_seconds"],
            "objective": result["stats"].get("opt"),
        }
        row.update({f"attack_{k}": v for k, v in image_metrics(raw, recon).items()})
        row.update({f"prior_{k}": v for k, v in image_metrics(raw, prior).items()})
        row.update({f"gray_{k}": v for k, v in image_metrics(raw, gray).items()})
        row.update({f"cifar_mean_{k}": v for k, v in image_metrics(raw, cifar_mean).items()})
        rows.append(row)
        print(json.dumps({"completed_group": group, "index": index, "attack_mse": row["attack_mse"]}), flush=True)

    # Use the next target's own reconstruction as the decoy and score it against
    # the current raw image. This implements "another target's reconstruction,
    # scored against the current target" without running extra decoy attacks.
    for group, row in enumerate(rows):
        decoy_group = (group + 1) % len(rows)
        decoy = reconstructions[decoy_group]
        decoy_png = out_dir / f"group_{group:03d}_idx_{indices[group]:05d}" / "decoy_next_reconstruction.png"
        tensor_to_png(decoy, decoy_png)
        row["decoy_group"] = decoy_group
        row["decoy_cifar10_index"] = indices[decoy_group]
        row["decoy_png"] = str(decoy_png)
        row.update({f"decoy_{k}": v for k, v in image_metrics(raws[group], decoy).items()})
        for baseline in ["prior", "decoy", "gray", "cifar_mean"]:
            row[f"win_vs_{baseline}"] = int(row["attack_mse"] < row[f"{baseline}_mse"])
            row[f"tie_vs_{baseline}"] = int(row["attack_mse"] == row[f"{baseline}_mse"])

    fieldnames = list(rows[0].keys()) if rows else []
    per_target = out_dir / "per_target.csv"
    with per_target.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summaries = {}
    for baseline in ["prior", "decoy", "gray", "cifar_mean"]:
        wins = sum(int(r[f"win_vs_{baseline}"]) for r in rows)
        ties = sum(int(r[f"tie_vs_{baseline}"]) for r in rows)
        losses = len(rows) - wins - ties
        summaries[baseline] = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "one_sided_exact_p": exact_one_sided_p(wins, losses),
        }

    grid_png = out_dir / "reconstruction_grid.png"
    make_grid(rows, grid_png)

    manifest = {
        "priority": 29,
        "stage": args.stage,
        "n": args.n,
        "target_indices": indices,
        "source_disjoint_evidence": {
            "split": "CIFAR-10 test",
            "within_stage_unique": len(set(indices)) == len(indices),
            "priority29_stage_offsets": {"n8": 0, "n24": 100, "confirmatory": 500},
            "note": "Earlier project CIFAR attack artifacts used train split; Priority 29 native image gates use test split.",
        },
        "model": "external_defenses/invertinggradients construct_model('LeNetZhu', seed=42)",
        "attacker": {
            "class": "inversefed.GradientReconstructor",
            "config": CONFIG,
            "known_labels": True,
        },
        "baselines": ["prior", "decoy", "gray", "cifar_mean"],
        "summary": summaries,
        "elapsed_seconds": time.monotonic() - run_start,
        "threads": torch.get_num_threads(),
        "outputs": {
            "per_target_csv": str(per_target),
            "reconstruction_grid_png": str(grid_png),
        },
    }
    manifest_path = out_dir / "summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["n8", "n24", "confirmatory"], required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--output-dir", default="artifacts/priority29_native_audit/image_positive_control")
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()
    run_stage(args)


if __name__ == "__main__":
    main()
