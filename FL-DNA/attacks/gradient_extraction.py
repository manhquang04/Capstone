"""Extract one MNIST sample gradient for inversion experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_mnist import load_mnist_data
from models.mnist_cnn import MNISTCNN

RESULTS_DIR = PROJECT_ROOT / "results" / "mnist"
MODEL_PATH = RESULTS_DIR / "base_model.pt"
RAW_GRADIENT_PATH = RESULTS_DIR / "raw_gradient.pt"
RECONSTRUCTION_DIR = RESULTS_DIR / "reconstructions"
ORIGINAL_IMAGE_PATH = RECONSTRUCTION_DIR / "original.png"
ORIGINAL_LABEL_PATH = RESULTS_DIR / "original_label.json"
SEARCH_SAMPLES = 256


def select_informative_sample(model, test_loader):
    """Select one deterministic high-loss sample with a useful gradient signal."""
    criterion = nn.CrossEntropyLoss()
    best_sample = None
    best_loss = -1.0
    with torch.no_grad():
        for sample_index, (image, label) in enumerate(test_loader):
            if sample_index >= SEARCH_SAMPLES:
                break
            loss = float(criterion(model(image), label).item())
            if loss > best_loss:
                best_sample = (image, label)
                best_loss = loss

    if best_sample is None:
        raise ValueError("MNIST test loader is empty")
    return *best_sample, best_loss


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"MNIST model not found at '{MODEL_PATH}'. Run "
            "'python experiments/train_mnist_model.py' first."
        )

    torch.manual_seed(42)
    model = MNISTCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    _, test_loader = load_mnist_data(batch_size=1)
    image, label, selected_loss = select_informative_sample(model, test_loader)
    loss = nn.CrossEntropyLoss()(model(image), label)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    bundle = {
        "gradients": [gradient.detach().cpu() for gradient in gradients],
        "label": int(label.item()),
        "image_shape": list(image.shape),
        "selection_loss": selected_loss,
        "source": "raw",
    }

    RECONSTRUCTION_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, RAW_GRADIENT_PATH)
    save_image(image, ORIGINAL_IMAGE_PATH)
    ORIGINAL_LABEL_PATH.write_text(
        json.dumps({"label": bundle["label"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Extracted gradient for MNIST label: {bundle['label']}")
    print(f"Selected sample cross-entropy: {selected_loss:.6f}")
    print(f"Saved gradient: {RAW_GRADIENT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved original image: {ORIGINAL_IMAGE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved original label: {ORIGINAL_LABEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
