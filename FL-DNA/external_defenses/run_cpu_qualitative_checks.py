"""Small, CPU-only mechanism checks for the five pinned external defenses.

These checks deliberately do not claim to reproduce the papers' trained-model
privacy metrics. They make the protocol-level operations executable on the
current host when the original CUDA-era experiment stacks cannot run.
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
torch.set_num_threads(4)
torch.manual_seed(7)
np.random.seed(7)
random.seed(7)


def psnr(a, b):
    mse = float(torch.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def soteria_check():
    """Mirror reconstruct_image.py:88-110 on a small feature/classifier model."""
    x = torch.rand(1, 1, 4, 4, requires_grad=True)
    features = nn.Sequential(nn.Flatten(), nn.Linear(16, 8), nn.Tanh())
    classifier = nn.Linear(8, 3)
    representation = features(x)
    loss = F.cross_entropy(classifier(representation), torch.tensor([1]))
    gradients = list(torch.autograd.grad(loss, list(features.parameters()) + list(classifier.parameters()), retain_graph=True))

    scores = []
    for feature_id in range(representation.shape[1]):
        dx = torch.autograd.grad(representation[0, feature_id], x, retain_graph=True)[0]
        scores.append(float(dx.norm() / (representation[0, feature_id].detach().abs() + 0.1)))
    threshold = float(np.percentile(np.abs(scores), 60))
    mask = torch.tensor([float(abs(score) >= threshold) for score in scores])
    # The official implementation zeros rows of the final classifier weight gradient.
    classifier_weight_grad = gradients[-2].clone()
    masked = classifier_weight_grad * mask.unsqueeze(0)
    return {
        "pruning_rate_percent": 60,
        "representation_units": 8,
        "masked_units": int((mask == 0).sum()),
        "classifier_gradient_zero_rows": int((masked.abs().sum(dim=0) == 0).sum()),
        "assertion": bool(torch.all(masked[:, mask == 0] == 0)),
    }


def precode_check():
    """Exercise the official VariationalBottleneck source twice at fixed weights."""
    sys.path.insert(0, str(ROOT / "precode"))
    from src.VariationalBottleneck import VariationalBottleneck

    vb = VariationalBottleneck((8,), K=4, beta=1e-3)
    head = nn.Linear(8, 3)
    x = torch.randn(1, 8)
    y = torch.tensor([2])
    gradients = []
    outputs = []
    for _ in range(2):
        vb.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        output = head(vb(x))
        (F.cross_entropy(output, y) + vb.loss()).backward()
        gradients.append(torch.cat([p.grad.flatten() for p in list(vb.parameters()) + list(head.parameters())]))
        outputs.append(output.detach())
    cosine = F.cosine_similarity(gradients[0], gradients[1], dim=0).item()
    return {
        "K": 4,
        "beta": 0.001,
        "two_forward_outputs_equal": bool(torch.equal(outputs[0], outputs[1])),
        "two_gradient_vectors_equal": bool(torch.equal(gradients[0], gradients[1])),
        "gradient_cosine_similarity": cosine,
        "assertion": bool(cosine < 0.999999),
    }


def prune_tensor(gradient, fraction):
    k = int(gradient.numel() * fraction)
    idx = torch.topk(gradient.abs().flatten(), k, largest=False).indices
    result = gradient.clone()
    result.flatten()[idx] = 0
    return result


def pruning_check():
    """Match magnitude-based per-tensor pruning described in DLG Section 5.2."""
    gradient = torch.randn(11, 13)
    pruned = prune_tensor(gradient, 0.70)
    return {
        "pruning_rate_percent": 70,
        "tensor_elements": gradient.numel(),
        "zeroed_elements": int((pruned == 0).sum()),
        "retained_min_abs": float(pruned[pruned != 0].abs().min()),
        "removed_max_abs": float(gradient[pruned == 0].abs().max()),
        "assertion": bool((pruned == 0).sum() == int(gradient.numel() * 0.70)),
    }


def ats_check():
    """Run the official zero-based 3-1-7 policy operations on a synthetic image."""
    sys.path.insert(0, str(ROOT / "ats"))
    # ATS targets NumPy 1.18, where np.int existed; restore that removed alias
    # in-process rather than changing the pinned checkout.
    np.int = int
    import policy

    image = torch.zeros(3, 32, 32)
    image[:, 8:24, 8:24] = 1.0
    from PIL import Image

    pil = Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    # policy.py maps 3, 1, 7 to translateX, contrast, translateY.
    for index in (3, 1, 7):
        pil = policy.policies[index](pil)
    transformed = torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1).float() / 255.0
    return {
        "policy": "3-1-7",
        "raw_to_transformed_psnr_db": psnr(image, transformed),
        "pixels_changed": int((image != transformed).sum()),
        "assertion": bool(not torch.equal(image, transformed)),
    }


def count_sketch(vector, buckets, rows, seed):
    """Fixed-hash Count Sketch encoder; mirrors FetchSGD's required linear map."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(buckets, (rows, vector.numel()), generator=generator)
    signs = (torch.randint(0, 2, (rows, vector.numel()), generator=generator) * 2 - 1).to(vector.dtype)
    table = torch.zeros(rows, buckets, dtype=vector.dtype)
    for row in range(rows):
        table[row].scatter_add_(0, indices[row], signs[row] * vector)
    return table


def fetchsgd_check():
    x = torch.randn(1000)
    y = torch.randn(1000)
    same_seed_error = float((count_sketch(x + y, 41, 5, 21) - (count_sketch(x, 41, 5, 21) + count_sketch(y, 41, 5, 21))).abs().max())
    different_seed_error = float((count_sketch(x + y, 41, 5, 21) - (count_sketch(x, 41, 5, 21) + count_sketch(y, 41, 5, 22))).abs().max())
    return {
        "dimension": 1000,
        "buckets": 41,
        "rows": 5,
        "same_hash_max_abs_error": same_seed_error,
        "mismatched_hash_max_abs_error": different_seed_error,
        "assertion": bool(same_seed_error <= 1e-5 and different_seed_error > 0.0),
    }


def main():
    results = {
        "soteria": soteria_check(),
        "precode": precode_check(),
        "gradient_pruning": pruning_check(),
        "ats": ats_check(),
        "fetchsgd": fetchsgd_check(),
    }
    print(json.dumps(results, indent=2, sort_keys=True))
    if not all(item["assertion"] for item in results.values()):
        raise SystemExit("One or more qualitative mechanism checks failed")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
