"""Run a lightweight iDLG-style MNIST gradient inversion attack."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mnist_cnn import MNISTCNN

RESULTS_DIR = PROJECT_ROOT / "results" / "mnist"
MODEL_PATH = RESULTS_DIR / "base_model.pt"
RECONSTRUCTION_DIR = RESULTS_DIR / "reconstructions"
ATTACK_METRICS_PATH = RESULTS_DIR / "attack_metrics.json"
ORIGINAL_IMAGE_PATH = RECONSTRUCTION_DIR / "original.png"
COMPARISON_GRID_PATH = RECONSTRUCTION_DIR / "comparison_grid.png"
GRADIENT_PATHS = {
    "RAW": RESULTS_DIR / "raw_gradient.pt",
    "DNA": RESULTS_DIR / "dna_gradient.pt",
    "DP": RESULTS_DIR / "dp_gradient.pt",
}
QUICK_MODE = os.environ.get("QUICK") == "1"
NUM_ITERATIONS = 500 if QUICK_MODE else 2_000
LEARNING_RATE = 0.1


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def infer_label(gradients: list[torch.Tensor]) -> int:
    """Infer the single-sample label from the final classifier bias gradient."""
    return int(torch.argmin(gradients[-1]).item())


def total_variation(image: torch.Tensor) -> torch.Tensor:
    horizontal = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    vertical = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    return horizontal + vertical


def reconstruct_image(
    model: nn.Module,
    target_gradients: list[torch.Tensor],
    image_shape: list[int],
    method: str,
    true_label: int,
    iterations: int,
    learning_rate: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Optimize one dummy image so its gradients match the observed gradients."""
    inferred_label = infer_label(target_gradients)
    label = torch.tensor([true_label], dtype=torch.long)
    image_logits = torch.randn(image_shape, requires_grad=True)
    optimizer = torch.optim.Adam([image_logits], lr=learning_rate)
    final_gradient_loss = 0.0

    for iteration in range(1, iterations + 1):
        optimizer.zero_grad()
        dummy_image = torch.sigmoid(image_logits)
        loss = nn.CrossEntropyLoss()(model(dummy_image), label)
        dummy_gradients = torch.autograd.grad(
            loss,
            tuple(model.parameters()),
            create_graph=True,
        )
        gradient_loss = sum(
            torch.mean((dummy_gradient - target_gradient).square())
            for dummy_gradient, target_gradient in zip(
                dummy_gradients,
                target_gradients,
            )
        )
        objective = gradient_loss + 1e-5 * total_variation(dummy_image)
        objective.backward()
        optimizer.step()
        # Sigmoid parameterization keeps every optimized image pixel in [0, 1].
        with torch.no_grad():
            image_logits.clamp_(-12.0, 12.0)
        final_gradient_loss = float(gradient_loss.detach().item())

        if iteration == 1 or iteration % 100 == 0 or iteration == iterations:
            print(
                f"[{method}] Iter {iteration}/{iterations} | "
                f"grad_loss = {final_gradient_loss:.8f}"
            )

    return torch.sigmoid(image_logits).detach(), {
        "true_label": true_label,
        "inferred_label": inferred_label,
        "iterations": iterations,
        "learning_rate": learning_rate,
        "final_gradient_loss": final_gradient_loss,
    }


def create_comparison_grid() -> None:
    """Create a labeled Original | RAW | DNA | DP image grid."""
    labels = ("Original", "RAW", "DNA", "DP")
    paths = (
        ORIGINAL_IMAGE_PATH,
        RECONSTRUCTION_DIR / "reconstructed_raw.png",
        RECONSTRUCTION_DIR / "reconstructed_dna.png",
        RECONSTRUCTION_DIR / "reconstructed_dp.png",
    )
    images = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Grid source image not found: '{path}'")
        with Image.open(path) as image:
            images.append(image.convert("L").resize((112, 112)))

    label_height = 22
    grid = Image.new("L", (112 * len(images), 112 + label_height), color=255)
    draw = ImageDraw.Draw(grid)
    for index, (label, image) in enumerate(zip(labels, images)):
        offset_x = index * 112
        grid.paste(image, (offset_x, label_height))
        draw.text((offset_x + 8, 4), label, fill=0)
    grid.save(COMPARISON_GRID_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"MNIST model not found at '{MODEL_PATH}'. Run "
            "'python experiments/train_mnist_model.py' first."
        )
    missing = [str(path) for path in GRADIENT_PATHS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Gradient artifacts are missing. Generate raw, DNA, and DP gradients "
            f"before running inversion: {missing}"
        )

    RECONSTRUCTION_DIR.mkdir(parents=True, exist_ok=True)
    attack_metrics = {}
    for method, gradient_path in GRADIENT_PATHS.items():
        set_seed()
        bundle = torch.load(gradient_path, map_location="cpu", weights_only=True)
        model = MNISTCNN()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        model.eval()
        gradients = [gradient.detach() for gradient in bundle["gradients"]]

        print(f"{method} gradient inversion")
        reconstructed, metrics = reconstruct_image(
            model,
            gradients,
            bundle["image_shape"],
            method=method,
            true_label=bundle["label"],
            iterations=args.iterations,
            learning_rate=args.learning_rate,
        )
        output_path = RECONSTRUCTION_DIR / f"reconstructed_{method.lower()}.png"
        save_image(reconstructed, output_path)
        attack_metrics[method] = metrics
        print(f"Saved reconstruction: {output_path.relative_to(PROJECT_ROOT)}")

    create_comparison_grid()
    ATTACK_METRICS_PATH.write_text(
        json.dumps(attack_metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved comparison grid: {COMPARISON_GRID_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved attack metrics: {ATTACK_METRICS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
