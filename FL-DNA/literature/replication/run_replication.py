"""Independent five-target image replication using the official optimizer API."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

ROOT = Path(__file__).resolve().parents[2]
IG = ROOT / "external_defenses" / "invertinggradients"
sys.path.insert(0, str(IG))
import inversefed  # noqa: E402
from inversefed.reconstruction_algorithms import GradientReconstructor, TV  # noqa: E402

TARGETS = [(0, 2310, 6), (1, 2290, 9), (2, 241, 1), (3, 5402, 4), (4, 8641, 7)]
CONFIG = {"cost_fn": "sim", "optim": "adam", "lr": 0.1, "lr_decay": True,
          "restarts": 1, "max_iterations": 4800, "total_variation": 0.01,
          "boxed": True, "signed": True, "init": "randn", "filter": "none",
          "weights": "equal", "indices": "def", "scoring_choice": "loss"}


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform of a one-dimensional tensor."""
    out = x
    width = 1
    while width < out.numel():
        pairs = out.reshape(-1, 2 * width)
        left, right = pairs[:, :width], pairs[:, width:]
        out = torch.cat((left + right, left - right), dim=1).reshape(-1)
        width *= 2
    return out / math.sqrt(out.numel())


def mix_seed(*values: int) -> int:
    state = 0x9E3779B9
    for value in values:
        state ^= int(value) + 0x9E3779B9 + ((state << 6) & 0xFFFFFFFF) + (state >> 2)
        state &= 0xFFFFFFFF
    return state


def v2_metadata(size: int, tensor_index: int, seed: int = 30002) -> tuple[torch.Tensor, torch.Tensor, float]:
    padded = 1 << (size - 1).bit_length()
    kept = int(math.ceil(0.95 * padded))
    local_seed = mix_seed(seed, tensor_index)
    rng = np.random.default_rng(mix_seed(local_seed, 17))
    signs = torch.from_numpy(rng.choice(np.array([-1., 1.]), padded).astype(np.float32))
    rng = np.random.default_rng(mix_seed(local_seed, 29))
    sampled = torch.from_numpy(np.sort(rng.choice(padded, kept, replace=False)).astype(np.int64))
    return signs, sampled, math.sqrt(padded / kept)


def v2_quantized_sketch(gradient: torch.Tensor, tensor_index: int) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, float]]:
    flat = gradient.detach().reshape(-1).cpu().float()
    signs, sampled, scale = v2_metadata(flat.numel(), tensor_index)
    padded = torch.zeros(signs.numel())
    padded[:flat.numel()] = flat
    values = scale * fwht(padded * signs)[sampled]
    delta = 0.01 * float(torch.linalg.vector_norm(flat)) / math.sqrt(flat.numel())
    if delta:
        generator = np.random.default_rng(mix_seed(mix_seed(mix_seed(30002, tensor_index), tensor_index), 43))
        lower = torch.floor(values / delta)
        draws = torch.from_numpy(generator.random(values.numel()).astype(np.float32))
        values = (lower + (draws < (values / delta - lower))) * delta
    return values, (signs, sampled, scale)


class AdaptiveReconstructor(GradientReconstructor):
    """Official optimizer loop with independently defined observation losses."""
    def __init__(self, model, mean_std, mode, observation, aux):
        super().__init__(model, mean_std, CONFIG, num_images=1)
        self.mode, self.observation, self.aux = mode, observation, aux

    def _loss(self, gradient):
        if self.mode == "omit":
            selected = [(a, b) for i, (a, b) in enumerate(zip(gradient, self.observation)) if i != self.aux]
            dot = sum((a * b).sum() for a, b in selected)
            return 1.0 - dot / (sum(a.pow(2).sum() for a, _ in selected).sqrt() * sum(b.pow(2).sum() for _, b in selected).sqrt() + 1e-12)
        if self.mode == "mask":
            masks = self.aux
            selected = [(a[m], b[m]) for a, b, m in zip(gradient, self.observation, masks) if m.any()]
            dot = sum((a * b).sum() for a, b in selected)
            return 1.0 - dot / (sum(a.pow(2).sum() for a, _ in selected).sqrt() * sum(b.pow(2).sum() for _, b in selected).sqrt() + 1e-12)
        if self.mode == "sketch":
            total = gradient[0].new_zeros(())
            for tensor_index, (candidate, observed, meta) in enumerate(zip(gradient, self.observation, self.aux)):
                signs, sampled, scale = (v.to(candidate.device) if isinstance(v, torch.Tensor) else v for v in meta)
                padded = candidate.new_zeros(signs.numel())
                padded[:candidate.numel()] = candidate.reshape(-1)
                predicted = scale * fwht(padded * signs)[sampled]
                total = total + ((predicted - observed.to(candidate.device)) ** 2).mean()
            return total
        raise ValueError(self.mode)

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label):
        def closure():
            optimizer.zero_grad(); self.model.zero_grad()
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            rec_loss = self._loss(gradient) + self.config["total_variation"] * TV(x_trial)
            rec_loss.backward()
            if self.config["signed"]:
                x_trial.grad.sign_()
            return rec_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        self.model.zero_grad(); x_trial.grad = None
        gradient = torch.autograd.grad(self.loss_fn(self.model(x_trial), label), self.model.parameters())
        return self._loss(gradient)


def metrics(raw: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, float]:
    a = raw[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    b = reconstruction[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    mse = float(np.mean((a - b) ** 2))
    return {"mse": mse, "psnr_db": -10.0 * math.log10(mse),
            "ssim": float(structural_similarity(a, b, data_range=1.0, channel_axis=2))}


def soteria_gradient(model, normalized, label):
    """Official sensitivity ranking applied to LeNet-Zhu's final classifier."""
    source = normalized.detach().clone().requires_grad_(True)
    features = model.body(source).reshape(1, -1)
    scores = torch.zeros(features.shape[1])
    for feature in range(features.shape[1]):
        model.zero_grad(); source.grad = None
        features[0, feature].backward(retain_graph=True)
        scores[feature] = source.grad.reshape(-1).norm() / (features[0, feature].detach() + 0.1)
    mask = scores.abs() >= torch.quantile(scores.abs(), 0.80)
    loss = torch.nn.functional.cross_entropy(model(normalized), label)
    gradients = [g.detach() for g in torch.autograd.grad(loss, model.parameters())]
    gradients[-2] = gradients[-2] * mask.to(gradients[-2]).unsqueeze(0)
    return gradients


def run_one(target_id, cifar_index, expected_label):
    torch.manual_seed(42)
    model, _ = inversefed.construct_model("LeNetZhu", seed=42)
    model.eval()
    dataset = CIFAR10(str(ROOT / "datasets" / "cifar10"), train=False, download=False, transform=ToTensor())
    raw, label_value = dataset[cifar_index]
    assert label_value == expected_label
    raw, label = raw.unsqueeze(0), torch.tensor([label_value])
    mean = torch.tensor(inversefed.consts.cifar10_mean)[:, None, None]
    std = torch.tensor(inversefed.consts.cifar10_std)[:, None, None]
    normalized = (raw - mean) / std
    full = [g.detach() for g in torch.autograd.grad(torch.nn.functional.cross_entropy(model(normalized), label), model.parameters())]
    observations = {"undefended": ("official", full, None),
                    "soteria_omit": ("omit", soteria_gradient(model, normalized, label), len(full) - 2)}
    masks, pruned = [], []
    for g in full:
        count = int(0.70 * g.numel())
        retained = torch.ones(g.numel(), dtype=torch.bool)
        retained[torch.argsort(g.abs().reshape(-1), stable=True)[:count]] = False
        masks.append(retained.reshape_as(g)); pruned.append(g * masks[-1])
    observations["pruning_mask"] = ("mask", pruned, masks)
    sketch, meta = zip(*(v2_quantized_sketch(g, i) for i, g in enumerate(full)))
    observations["v2_sketch"] = ("sketch", list(sketch), list(meta))
    results = {}
    for condition, (mode, observed, aux) in observations.items():
        torch.manual_seed(30600 + target_id)
        if mode == "official":
            attack = GradientReconstructor(model, (mean, std), CONFIG, num_images=1)
        else:
            attack = AdaptiveReconstructor(model, (mean, std), mode, observed, aux)
        reconstructed, stats = attack.reconstruct(observed, label, img_shape=(3, 32, 32))
        reconstruction_raw = reconstructed * std + mean
        results[condition] = {"metrics": metrics(raw, reconstruction_raw), "objective": float(dict(stats).get("opt", float("nan")))}
    return {"target_id": target_id, "cifar10_index": cifar_index, "label": label_value, "results": results}


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output = {"implementation": "independent; no Priority 30 experiment import", "config": CONFIG, "targets": []}
    for target in TARGETS:
        output["targets"].append(run_one(*target))
    destination = Path(__file__).with_name("replication_results.json")
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
