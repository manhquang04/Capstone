"""Qualitative, untrained CPU smoke test for original/adaptive attack plumbing.

This is intentionally not an empirical privacy result. It uses four genuine
PaySim records for the tabular MLP and four genuine CIFAR-10 records for the
CNN, random model weights, and a bounded attack budget solely to exercise both
model families.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms

from experiments.harness.interfaces import DefenseContext, IdentityDefense, ServerKnowledge
from external_defenses.adaptive import (
    ATSDefense,
    CountSketchDefense,
    GradientMatchingAttack,
    GradientPruningDefense,
    KnownTransformAttack,
    MaskAwareGradientAttack,
    OmitGradientAttack,
    PRECODEDefense,
    SketchGradientAttack,
    SoteriaDefense,
)

THREADS = 4
STEPS = 20


class TabularMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(13, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 10))

    def forward(self, x):
        return self.net(x)


class LeNet137k(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 16, 5), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 5), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(800, 150), nn.ReLU(), nn.Linear(150, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


def gradients(model, x, labels):
    model.eval()
    loss = F.cross_entropy(model(x), labels)
    return {name: grad.detach() for (name, _), grad in zip(model.named_parameters(), torch.autograd.grad(loss, tuple(model.parameters())))}


def view(defense, update):
    context = DefenseContext(0, 0, tuple(update), ServerKnowledge(seeds=frozenset({"sketch_hashes"})))
    payload = defense.encode(update, context)
    return defense.server_view(payload, context.server_knowledge), payload


def mse(result, raw):
    return float(((result.estimate - raw) ** 2).mean())


def baseline_mses(raw, baselines):
    return {name: float(((candidate - raw) ** 2).mean()) for name, candidate in baselines.items()}


def qualitative_success(value, baselines):
    """An attack is a qualitative success only if it beats every baseline."""
    return value < min(baselines.values())


def positive_control(model, raw, labels, baselines):
    update = gradients(model, raw, labels)
    identity = IdentityDefense()
    raw_view = identity.server_view(identity.encode(update, DefenseContext(0, 0, tuple(update), ServerKnowledge())), ServerKnowledge())
    result = GradientMatchingAttack(tuple(raw.shape), labels, steps=STEPS, lr=0.1, seed=19).reconstruct(raw_view, model, ServerKnowledge())
    value = mse(result, raw)
    return {"raw_mse": value, "succeeds_vs_baselines": qualitative_success(value, baselines)}


def attack_pair(model, raw, labels, baselines, defense_name, defense, adaptive, transform=lambda x: x):
    protected_input = transform(raw)
    protected_update = gradients(model, protected_input, labels)
    server_view, payload = view(defense, protected_update)
    common = dict(input_shape=tuple(raw.shape), label=labels, steps=STEPS, lr=0.1, seed=19)
    if defense_name == "ats":
        original = GradientMatchingAttack(**common)
    elif defense_name == "count_sketch":
        # FetchSGD has no original reconstruction attacker; report unavailable.
        original = None
    else:
        original = GradientMatchingAttack(**common)
    if original is None:
        original_row = {"status": "not_published", "raw_mse": None, "blocked_vs_baselines": None}
    else:
        result = original.reconstruct(server_view, model, server_view.server_knowledge)
        value = mse(result, raw)
        original_row = {"status": "ran", "raw_mse": value, "blocked_vs_baselines": not qualitative_success(value, baselines)}
    result = adaptive(server_view, payload, common)
    adaptive_mse = mse(result, raw)
    return {"original": original_row, "adaptive": {"status": result.metadata.get("status", "ran"), "raw_mse": adaptive_mse, "succeeds_vs_baselines": qualitative_success(adaptive_mse, baselines)}}


def one_architecture(name, model, raw, labels, baselines):
    torch.manual_seed(23)
    final_weight = [key for key in gradients(model, raw, labels) if key.endswith("weight")][-1]
    representation = gradients(model, raw, labels)[final_weight].shape[1]
    scores = torch.linspace(0, 1, representation)
    transforms_known = lambda x: x * 0.5 + 0.25
    defenses = {
        "soteria": (SoteriaDefense(final_weight, scores, 0.5), lambda v, p, c: OmitGradientAttack(**c, omit_names=(final_weight,)).reconstruct(v, model, v.server_knowledge), lambda x: x),
        "precode": (PRECODEDefense((final_weight,)), lambda v, p, c: OmitGradientAttack(**c, omit_names=(final_weight,)).reconstruct(v, model, v.server_knowledge), lambda x: x),
        "gradient_pruning": (GradientPruningDefense(0.5), lambda v, p, c: MaskAwareGradientAttack(**c, masks=p.metadata["mask"]).reconstruct(v, model, v.server_knowledge), lambda x: x),
        "ats": (ATSDefense("affine_smoke", True), lambda v, p, c: KnownTransformAttack(**c, transform=transforms_known).reconstruct(v, model, v.server_knowledge), transforms_known),
        "count_sketch": (CountSketchDefense(rows=3, columns=257, seed=29, reveal_hashes=True), lambda v, p, c: SketchGradientAttack(**c).reconstruct(v, model, v.server_knowledge), lambda x: x),
    }
    result = {
        "baselines_raw_mse": baseline_mses(raw, baselines),
        "undefended_positive_control": positive_control(model, raw, labels, baseline_mses(raw, baselines)),
        "defenses": {
            defense_name: attack_pair(model, raw, labels, baseline_mses(raw, baselines), defense_name, defense, adaptive, transform)
            for defense_name, (defense, adaptive, transform) in defenses.items()
        },
    }
    unknown = CountSketchDefense(rows=3, columns=257, seed=29, reveal_hashes=False)
    unknown_view, _ = view(unknown, gradients(model, raw, labels))
    unavailable = SketchGradientAttack(tuple(raw.shape), labels, steps=STEPS, lr=0.1, seed=19).reconstruct(unknown_view, model, unknown_view.server_knowledge)
    result["defenses"]["count_sketch"]["adaptive_unknown_hashes"] = {
        "status": unavailable.metadata["status"],
        "raw_mse": mse(unavailable, raw),
        "succeeds_vs_baselines": False,
    }
    return result


def load_paysim_records():
    """Use the project loader read-only, with 0 data-loader workers."""
    os.environ["DATALOADER_NUM_WORKERS"] = "0"
    from data.load_creditcard import load_paysim_splits

    train_loader, _, test_loader, features, _, _ = load_paysim_splits(
        batch_size=4, seed=42, max_rows=5_000
    )
    if features != 13:
        raise ValueError(f"expected 13 PaySim features, got {features}")
    raw, labels = next(iter(test_loader))
    population_mean = train_loader.dataset.tensors[0].mean(dim=0, keepdim=True).expand_as(raw)
    return raw, labels.reshape(-1).long(), {"population_mean": population_mean}


def load_cifar_records():
    root = Path(__file__).resolve().parents[2] / "datasets" / "cifar10"
    cifar = datasets.CIFAR10(root=root, train=False, download=False, transform=transforms.ToTensor())
    images, labels = zip(*(cifar[index] for index in range(4)))
    raw = torch.stack(images)
    mean_image = torch.from_numpy(cifar.data).permute(0, 3, 1, 2).float().mean(dim=0, keepdim=True) / 255.0
    return raw, torch.tensor(labels), {"constant_gray": torch.full_like(raw, 0.5), "mean_image": mean_image.expand_as(raw)}


def main():
    torch.set_num_threads(THREADS)
    torch.manual_seed(17)
    tabular, tabular_labels, tabular_baselines = load_paysim_records()
    images, image_labels, image_baselines = load_cifar_records()
    report = {
        "scope": {"qualitative_only": True, "steps": STEPS, "criterion": "raw-input MSE must beat every named data-free baseline", "paysim_max_rows": 5000, "cifar10_test_indices": [0, 1, 2, 3]},
        "tabular_mlp_bn": one_architecture("tabular_mlp_bn", TabularMLP(), tabular, tabular_labels, tabular_baselines),
        "lenet_137k": one_architecture("lenet_137k", LeNet137k(), images, image_labels, image_baselines),
    }
    report["lenet_137k"]["parameter_count"] = sum(parameter.numel() for parameter in LeNet137k().parameters())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
    main()
