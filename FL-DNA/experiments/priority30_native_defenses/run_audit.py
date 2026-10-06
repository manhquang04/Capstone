"""Priority 30 staged native audit driver.

This is intentionally resumable: every (stage, domain, defense, evaluation,
target) writes one JSON result as soon as it finishes, and existing result files
are skipped on replay.
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
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToPILImage, ToTensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Tableak must precede the project root for `import attacks.*`; invertinggradients
# is also read-only and provides `inversefed`.
sys.path.insert(0, str(ROOT / "external_defenses" / "invertinggradients"))
sys.path.insert(0, str(ROOT / "external_defenses" / "tableak"))

import inversefed  # noqa: E402
from inversefed.reconstruction_algorithms import GradientReconstructor, reconstruction_costs  # noqa: E402
from inversefed.metrics import total_variation as TV  # noqa: E402
from attacks import invert_grad  # noqa: E402
import attacks.gradient_inversion_attack as tableak_gia  # noqa: E402
from datasets import ADULT  # noqa: E402
from models import FullyConnected  # noqa: E402
from utils import batch_feature_wise_accuracy_score, match_reconstruction_ground_truth, post_process_continuous  # noqa: E402
from dna_encoder.transform_defense_v2 import DNATransformV2Config, materialize_projection_matrix_v2, transform_update_array_v2, _signs  # noqa: E402

from experiments.priority29.tabular_native_positive_control import official_config  # noqa: E402
from experiments.priority30_native_defenses.native_adapters import (  # noqa: E402
    PrecodeAdultFC,
    PrecodeLeNetZhu,
    count_sketch,
    dna_v1_gradient,
    dna_v2_gradient,
    gradient_dict,
    prune_gradient,
)


IMAGE_DEFENSES = ["precode", "soteria", "gradient_pruning", "ats", "count_sketch", "dna_v1_conservative", "dna_v2_0p95"]
ADULT_DEFENSES = ["precode", "soteria", "gradient_pruning", "count_sketch", "dna_v1_conservative", "dna_v2_0p95"]
EVALS = ["E1", "E2"]
IMAGE_CONFIG = {
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


def exact_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n))


def image_metrics(raw: torch.Tensor, estimate: torch.Tensor) -> dict[str, float]:
    a = raw[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    b = estimate[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0.0 else float(-10.0 * np.log10(mse))
    return {
        "mse": mse,
        "psnr_db": psnr,
        "ssim": float(structural_similarity(a, b, data_range=1.0, channel_axis=2)),
    }


def cosine_loss_lists(left: list[torch.Tensor], right: list[torch.Tensor], masks: list[torch.Tensor] | None = None) -> torch.Tensor:
    numerator = None
    left_norm = None
    right_norm = None
    for idx, (a, b) in enumerate(zip(left, right)):
        aa = a.reshape(-1)
        bb = b.reshape(-1)
        if masks is not None:
            mm = masks[idx].to(device=aa.device, dtype=torch.bool).reshape(-1)
            aa = aa[mm]
            bb = bb[mm]
        if aa.numel() == 0:
            continue
        dot = (aa * bb).sum()
        an = aa.pow(2).sum()
        bn = bb.pow(2).sum()
        numerator = dot if numerator is None else numerator + dot
        left_norm = an if left_norm is None else left_norm + an
        right_norm = bn if right_norm is None else right_norm + bn
    if numerator is None:
        ref = left[0]
        return ref.new_tensor(1.0)
    return 1.0 - numerator / (left_norm.sqrt() * right_norm.sqrt() + 1e-10)


def flatten_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def unflatten_like(flat: torch.Tensor, like: list[torch.Tensor]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    offset = 0
    for tensor in like:
        num = tensor.numel()
        out.append(flat[offset : offset + num].reshape_as(tensor))
        offset += num
    return out


def dna_v1_debias(update: dict[str, torch.Tensor], mix: float = 0.08) -> dict[str, torch.Tensor]:
    # Priority 30 continuation amendment: structure-aware Level-1 debiasing
    # per transmitted block/tensor, (T - m*mean(T))/(1-m).
    return {name: (tensor - mix * tensor.mean()) / (1.0 - mix) for name, tensor in update.items()}


def fwht_normalized_torch(x: torch.Tensor) -> torch.Tensor:
    y = x
    n = y.numel()
    h = 1
    while h < n:
        y = y.reshape(-1, h * 2)
        left = y[:, :h]
        right = y[:, h:]
        y = torch.cat((left + right, left - right), dim=1).reshape(-1)
        h *= 2
    return y / math.sqrt(float(n))


def v2_project_with_plan(tensor: torch.Tensor, plan: dict[str, torch.Tensor | int | float]) -> torch.Tensor:
    flat = tensor.reshape(-1)
    padded_size = int(plan["padded_size"])
    padded = flat.new_zeros(padded_size)
    padded[: flat.numel()] = flat
    signed = padded * plan["signs"].to(device=flat.device, dtype=flat.dtype)
    projected_full = fwht_normalized_torch(signed)
    sampled = plan["sampled"].to(device=flat.device, dtype=torch.long)
    return float(plan["scale"]) * projected_full[sampled]


def v2_sketch_payload(update: dict[str, torch.Tensor], seed: int = 30_002) -> tuple[list[torch.Tensor], list[Any]]:
    """Return observed quantized sketches and differentiable SRHT projection plans."""
    config = DNATransformV2Config(compression_ratio=0.95, quantization_eta=0.01, seed=seed)
    sketches: list[torch.Tensor] = []
    plans: list[Any] = []
    for tensor_index, (_name, tensor) in enumerate(update.items()):
        array = tensor.detach().cpu().numpy().astype("float32", copy=False)
        sketch, meta = transform_update_array_v2(array, config, tensor_index=tensor_index, quantization_seed=tensor_index)
        sketches.append(torch.from_numpy(sketch.copy()).to(dtype=tensor.dtype, device=tensor.device).reshape(-1))
        sampled = torch.tensor(meta.sampled_indices, dtype=torch.long, device=tensor.device)
        signs = torch.from_numpy(_signs(meta.padded_size, meta.seed).copy()).to(dtype=tensor.dtype, device=tensor.device)
        plans.append(
            {
                "padded_size": meta.padded_size,
                "sampled": sampled,
                "signs": signs,
                "scale": float(math.sqrt(meta.padded_size / meta.sketch_size)),
            }
        )
    return sketches, plans


def v2_sketch_loss(candidate: list[torch.Tensor], sketches: list[torch.Tensor], matrices: list[Any]) -> torch.Tensor:
    projected = [
        v2_project_with_plan(tensor, plan) if isinstance(plan, dict) else plan @ tensor.reshape(-1)
        for tensor, plan in zip(candidate, matrices)
    ]
    return cosine_loss_lists(projected, sketches)


def count_sketch_payload_from_update(
    update: dict[str, torch.Tensor], rows: int = 5, columns: int | None = None, seed: int = 21
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    transmitted = count_sketch(update, rows=rows, columns=columns, seed=seed)
    table = transmitted.tensors["sketch"]
    flat_like = [update[name] for name in transmitted.metadata["tensor_order"]]
    dim = int(transmitted.metadata["dimension"])
    columns = int(transmitted.metadata["columns"])
    rows = int(transmitted.metadata["rows"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    buckets = torch.randint(columns, (rows, dim), generator=generator, dtype=torch.long).to(table.device)
    signs = (2 * torch.randint(2, (rows, dim), generator=generator) - 1).to(table.device, table.dtype)
    return table, buckets, signs, flat_like


def count_sketch_project(flat: torch.Tensor, buckets: torch.Tensor, signs: torch.Tensor, columns: int) -> torch.Tensor:
    table = flat.new_zeros((buckets.shape[0], columns))
    for row in range(buckets.shape[0]):
        table[row].scatter_add_(0, buckets[row].to(flat.device), signs[row].to(flat.device, flat.dtype) * flat)
    return table


def count_sketch_loss(candidate: list[torch.Tensor], table: torch.Tensor, buckets: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    pred = count_sketch_project(flatten_tensors(candidate), buckets, signs, table.shape[1])
    return cosine_loss_lists([pred], [table])


def count_sketch_compact(table: torch.Tensor, buckets: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    observed: list[torch.Tensor] = []
    inverse: list[torch.Tensor] = []
    for row in range(buckets.shape[0]):
        unique, inv = torch.unique(buckets[row], sorted=True, return_inverse=True)
        observed.append(table[row, unique])
        inverse.append(inv)
    return observed, inverse


def count_sketch_loss_compact(
    candidate: list[torch.Tensor],
    observed: list[torch.Tensor],
    inverse: list[torch.Tensor],
    signs: torch.Tensor,
) -> torch.Tensor:
    flat = flatten_tensors(candidate)
    pred_rows: list[torch.Tensor] = []
    for row in range(signs.shape[0]):
        pred = flat.new_zeros(observed[row].shape)
        pred.scatter_add_(0, inverse[row].to(flat.device), signs[row].to(flat.device, flat.dtype) * flat)
        pred_rows.append(pred)
    return cosine_loss_lists(pred_rows, observed)


def count_sketch_decode(table: torch.Tensor, buckets: torch.Tensor, signs: torch.Tensor, like: list[torch.Tensor]) -> list[torch.Tensor]:
    estimates = []
    for row in range(buckets.shape[0]):
        estimates.append(signs[row].to(table.dtype) * table[row, buckets[row]])
    flat = torch.stack(estimates, dim=0).median(dim=0).values
    return unflatten_like(flat, like)


class AdaptiveImageReconstructor(GradientReconstructor):
    """Official Geiping loop with a Priority-30 adaptive matching objective."""

    def __init__(self, *args: Any, adaptive_mode: str, adaptive_aux: dict[str, Any], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.adaptive_mode = adaptive_mode
        self.adaptive_aux = adaptive_aux

    def _adaptive_loss(self, gradient: list[torch.Tensor], input_gradient: list[torch.Tensor]) -> torch.Tensor:
        if self.adaptive_mode == "plain":
            return reconstruction_costs(
                [gradient],
                input_gradient,
                cost_fn=self.config["cost_fn"],
                indices=self.config["indices"],
                weights=self.config["weights"],
            )
        if self.adaptive_mode == "mask_aware":
            return cosine_loss_lists(gradient, input_gradient, self.adaptive_aux["masks"])
        if self.adaptive_mode == "v2_sketch":
            return v2_sketch_loss(gradient, self.adaptive_aux["sketches"], self.adaptive_aux["matrices"])
        if self.adaptive_mode == "count_sketch":
            if "compact_observed" in self.adaptive_aux:
                return count_sketch_loss_compact(
                    gradient,
                    self.adaptive_aux["compact_observed"],
                    self.adaptive_aux["compact_inverse"],
                    self.adaptive_aux["signs"],
                )
            return count_sketch_loss(gradient, self.adaptive_aux["table"], self.adaptive_aux["buckets"], self.adaptive_aux["signs"])
        raise ValueError(f"unknown adaptive image mode: {self.adaptive_mode}")

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label):
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = list(torch.autograd.grad(loss, self.model.parameters(), create_graph=True))
            rec_loss = self._adaptive_loss(gradient, input_gradient)
            if self.config["total_variation"] > 0:
                rec_loss = rec_loss + self.config["total_variation"] * TV(x_trial)
            rec_loss.backward()
            if self.config["signed"]:
                x_trial.grad.sign_()
            return rec_loss

        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        self.model.zero_grad()
        x_trial.grad = None
        loss = self.loss_fn(self.model(x_trial), label)
        gradient = list(torch.autograd.grad(loss, self.model.parameters(), create_graph=False))
        return self._adaptive_loss(gradient, input_gradient)


def image_targets(stage: str, n: int) -> list[int]:
    population = list(range(10_000))
    rng = random.Random(30_400)
    rng.shuffle(population)
    offsets = {"S0": 900, "S0u": 1_000, "S1": 1_000, "S1b": 1_000, "S1c": 1_000, "S1d": 1_000, "S3": 2_000}
    return population[offsets.get(stage, 3_000) : offsets.get(stage, 3_000) + n]


def adult_targets(stage: str, n: int) -> list[list[int]]:
    # Adult train set rows. Offsets are disjoint from Priority 29 positive
    # controls (0 and 2000) and the reserved confirmatory slice there (6000).
    dataset_size = 30_162
    population = list(range(dataset_size))
    rng = random.Random(30_500)
    rng.shuffle(population)
    offsets = {"S0": 8_000, "S0u": 10_000, "S2": 10_000, "S2b": 10_000, "S2c": 10_000, "S2d": 10_000, "S3": 15_000, "S4": 20_000}
    offset = offsets.get(stage, 24_000)
    return [population[offset + i * 8 : offset + (i + 1) * 8] for i in range(n)]


def tensor_to_png(t: torch.Tensor, path: Path) -> None:
    arr = t[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255
    Image.fromarray(arr.round().astype("uint8")).save(path)


def apply_ats(raw: torch.Tensor) -> torch.Tensor:
    ats_root = ROOT / "external_defenses" / "ats"
    sys.path.insert(0, str(ats_root))
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    import policy as ats_policy  # type: ignore

    pil = ToPILImage()(raw[0])
    random.seed(30_123)
    np.random.seed(30_123)
    transformed = pil
    for idx in [3, 1, 7]:
        transformed = ats_policy.policies[idx](transformed)
    return ToTensor()(transformed).unsqueeze(0)


def soteria_image_gradient(model: torch.nn.Module, raw: torch.Tensor, label: torch.Tensor) -> list[torch.Tensor]:
    undef = gradient_dict(model, raw, label, torch.nn.CrossEntropyLoss())
    x_req = raw.detach().clone().requires_grad_(True)
    rep = model.body(x_req).view(1, -1)
    scores = []
    for j in range(rep.shape[1]):
        if x_req.grad is not None:
            x_req.grad.zero_()
        model.zero_grad(set_to_none=True)
        rep[:, j].sum().backward(retain_graph=True)
        scores.append(x_req.grad.detach().reshape(1, -1).norm(dim=1) / (rep[:, j].detach().abs() + 0.1))
    score = torch.stack(scores).reshape(-1)
    threshold = torch.quantile(score.float(), 0.80)
    mask = (score >= threshold).to(undef["fc.0.weight"])
    defended = {k: v.clone() for k, v in undef.items()}
    defended["fc.0.weight"] = defended["fc.0.weight"] * mask.unsqueeze(0)
    return list(defended.values())


def image_observed(defense: str, raw: torch.Tensor, label: torch.Tensor) -> tuple[torch.nn.Module, list[torch.Tensor], torch.Tensor, str]:
    base, _key = inversefed.construct_model("LeNetZhu", seed=42)
    base.eval()
    labels = label
    if defense == "precode":
        model = PrecodeLeNetZhu(base).eval()
        return model, list(gradient_dict(model, raw, labels, torch.nn.CrossEntropyLoss()).values()), raw, "precode_model"
    if defense == "ats":
        ats_raw = apply_ats(raw)
        return base, list(gradient_dict(base, ats_raw, labels, torch.nn.CrossEntropyLoss()).values()), raw, "ats_policy_3_1_7_raw_scored"
    undef = gradient_dict(base, raw, labels, torch.nn.CrossEntropyLoss())
    if defense == "soteria":
        return base, soteria_image_gradient(base, raw, labels), raw, "soteria_mask"
    if defense == "gradient_pruning":
        pruned, _ = prune_gradient(undef, 0.70)
        return base, list(pruned.values()), raw, "per_tensor_prune_70"
    if defense == "dna_v1_conservative":
        return base, list(dna_v1_gradient(undef).values()), raw, "dna_v1_conservative"
    if defense == "dna_v2_0p95":
        v2, _ = dna_v2_gradient(undef)
        return base, list(v2.values()), raw, "dna_v2_0p95_decoded"
    raise NotImplementedError(defense)


def run_image_cell(stage: str, eval_name: str, defense: str, target_id: int, index: int, out_file: str) -> dict[str, Any]:
    torch.set_num_threads(1)
    start = time.monotonic()
    out = Path(out_file)
    if out.exists():
        return {"skipped": True, "path": str(out)}
    out.parent.mkdir(parents=True, exist_ok=True)
    data = CIFAR10(str(ROOT / "datasets" / "cifar10"), train=False, download=False, transform=ToTensor())
    raw, label_int = data[index]
    raw = raw.unsqueeze(0)
    label = torch.tensor([label_int], dtype=torch.long)
    dm = torch.tensor(inversefed.consts.cifar10_mean)[:, None, None]
    ds = torch.tensor(inversefed.consts.cifar10_std)[:, None, None]
    normalized = (raw - dm) / ds
    if defense != "count_sketch":
        model, observed, score_raw, mode = image_observed(defense, normalized if defense != "ats" else raw, label)
    else:
        model, observed, score_raw, mode = None, [], raw, "count_sketch"

    def run_attack(
        obs: list[torch.Tensor],
        attack_model: torch.nn.Module,
        variant: str,
        aux: dict[str, Any] | None = None,
        canonical_mode: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any], str]:
        config = IMAGE_CONFIG.copy()
        if aux is None:
            attack = inversefed.GradientReconstructor(attack_model, (dm, ds), config, num_images=1)
        else:
            if canonical_mode is None:
                raise ValueError(f"adaptive variant {variant} did not provide a canonical mode")
            attack = AdaptiveImageReconstructor(attack_model, (dm, ds), config, num_images=1, adaptive_mode=canonical_mode, adaptive_aux=aux)
        torch.manual_seed(30_600 + target_id)
        np.random.seed(30_600 + target_id)
        return (*attack.reconstruct(obs, label, img_shape=(3, 32, 32)), variant)

    stats: dict[str, Any]
    selected_variant = mode
    if defense == "count_sketch":
        base, _key = inversefed.construct_model("LeNetZhu", seed=42)
        base.eval()
        undef = gradient_dict(base, normalized, label, torch.nn.CrossEntropyLoss())
        table, buckets, signs, like = count_sketch_payload_from_update(undef)
        if eval_name == "E1":
            observed = count_sketch_decode(table, buckets, signs, like)
            recon_norm, stats, selected_variant = run_attack(observed, base, "count_sketch_decoded_e1")
        else:
            compact_observed, compact_inverse = count_sketch_compact(table, buckets)
            recon_norm, stats, selected_variant = run_attack(
                compact_observed,
                base,
                "count_sketch_sketch_space_e2",
                {
                    "table": table,
                    "buckets": buckets,
                    "signs": signs,
                    "compact_observed": compact_observed,
                    "compact_inverse": compact_inverse,
                },
                canonical_mode="count_sketch",
            )
    elif stage in {"S1b", "S1c"} and eval_name == "E2" and defense == "gradient_pruning":
        base, _key = inversefed.construct_model("LeNetZhu", seed=42)
        base.eval()
        undef = gradient_dict(base, normalized, label, torch.nn.CrossEntropyLoss())
        pruned, masks = prune_gradient(undef, 0.70)
        recon_norm, stats, selected_variant = run_attack(
            list(pruned.values()),
            base,
            "gradient_pruning_mask_aware_e2",
            {"masks": list(masks.values())},
            canonical_mode="mask_aware",
        )
    elif stage in {"S1b", "S1c"} and eval_name == "E2" and defense == "dna_v2_0p95":
        base, _key = inversefed.construct_model("LeNetZhu", seed=42)
        base.eval()
        undef = gradient_dict(base, normalized, label, torch.nn.CrossEntropyLoss())
        sketches, matrices = v2_sketch_payload(undef)
        recon_norm, stats, selected_variant = run_attack(
            sketches,
            base,
            "dna_v2_sketch_space_e2",
            {"sketches": sketches, "matrices": matrices},
            canonical_mode="v2_sketch",
        )
    elif stage in {"S1b", "S1c"} and eval_name == "E2" and defense == "dna_v1_conservative":
        base, _key = inversefed.construct_model("LeNetZhu", seed=42)
        base.eval()
        undef = gradient_dict(base, normalized, label, torch.nn.CrossEntropyLoss())
        transformed = dna_v1_gradient(undef)
        debiased = dna_v1_debias(transformed)
        recon_plain, stats_plain, _ = run_attack(list(transformed.values()), base, "dna_v1_plain_e2")
        recon_debias, stats_debias, _ = run_attack(list(debiased.values()), base, "dna_v1_debiased_e2")
        objective_plain = float(dict(stats_plain).get("opt", float("inf")))
        objective_debias = float(dict(stats_debias).get("opt", float("inf")))
        if objective_debias < objective_plain:
            recon_norm, stats, selected_variant = recon_debias, stats_debias, "dna_v1_debiased_e2"
        else:
            recon_norm, stats, selected_variant = recon_plain, stats_plain, "dna_v1_plain_e2"
        stats = dict(stats)
        stats["plain_opt"] = objective_plain
        stats["debiased_opt"] = objective_debias
    elif stage == "S1c" and eval_name == "E2" and defense == "precode":
        model, observed, score_raw, _mode = image_observed(defense, normalized, label)
        masks = [
            torch.zeros_like(parameter, dtype=torch.bool) if name.startswith("bottleneck.") else torch.ones_like(parameter, dtype=torch.bool)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        recon_norm, stats, selected_variant = run_attack(
            observed,
            model,
            "precode_omit_bottleneck_gradients_e2",
            {"masks": masks},
            canonical_mode="mask_aware",
        )
    elif stage == "S1c" and eval_name == "E2" and defense == "soteria":
        base, _key = inversefed.construct_model("LeNetZhu", seed=42)
        base.eval()
        observed = soteria_image_gradient(base, normalized, label)
        masks = []
        for name, parameter in base.named_parameters():
            if name.startswith("fc.0."):
                masks.append(torch.zeros_like(parameter, dtype=torch.bool))
            else:
                masks.append(torch.ones_like(parameter, dtype=torch.bool))
        recon_norm, stats, selected_variant = run_attack(
            observed,
            base,
            "soteria_omit_defended_layer_e2",
            {"masks": masks},
            canonical_mode="mask_aware",
        )
    elif stage == "S1c" and eval_name == "E2" and defense == "ats":
        # The official ATS policy used here is PIL-based and not differentiable.
        # The frozen best-available approximation is straight-through/identity
        # through the policy while still matching the server-observable defended
        # gradient and scoring against the raw image.
        model, observed, score_raw, _mode = image_observed(defense, raw, label)
        recon_norm, stats, selected_variant = run_attack(
            observed,
            model,
            "ats_identity_straight_through_approx_e2",
            {},
            canonical_mode="plain",
        )
        stats = dict(stats)
        stats["documented_e2_equals_e1_reason"] = "ATS policy path is PIL/non-differentiable; E2 uses the pre-registered identity straight-through approximation."
    elif eval_name == "E2" and defense in {"gradient_pruning", "count_sketch", "dna_v1_conservative", "dna_v2_0p95"}:
        result = {"status": "NOT_ASSESSABLE", "reason": f"native adaptive objective for {defense} is only run in S1b per Priority 30 continuation"}
        out.write_text(json.dumps(result, indent=2) + "\n")
        return result
    else:
        recon_norm, stats, selected_variant = run_attack(observed, model, mode)
    recon = recon_norm * ds + dm
    # All scoring is against raw image in [0,1].
    metrics = image_metrics(raw, recon)
    target_dir = out.parent
    tensor_to_png(raw, target_dir / "raw.png")
    tensor_to_png(recon, target_dir / "reconstruction.png")
    result = {
        "status": "ran",
        "stage": stage,
        "domain": "image",
        "eval": eval_name,
        "defense": defense,
        "target_id": target_id,
        "cifar10_index": index,
        "label": int(label_int),
        "mode": selected_variant,
        "metrics": metrics,
        "objective": float(dict(stats).get("opt", float("nan"))),
        "elapsed_seconds": time.monotonic() - start,
    }
    out.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    return result


def run_image_undefended(stage: str, target_id: int, index: int, out_file: str) -> dict[str, Any]:
    """Run the paired undefended comparator on the exact S1 image target."""
    torch.set_num_threads(1)
    start = time.monotonic()
    out = Path(out_file)
    if out.exists():
        try:
            existing = json.loads(out.read_text())
        except Exception:
            existing = {}
        if (
            stage == "S2d"
            and eval_name == "E2"
            and defense == "count_sketch"
            and existing.get("status") == "NOT_ASSESSABLE"
        ):
            preserved = out.with_suffix(".invalid_pre_fix.json")
            if not preserved.exists():
                preserved.write_text(out.read_text())
        else:
            return {"skipped": True, "path": str(out)}
    out.parent.mkdir(parents=True, exist_ok=True)
    data = CIFAR10(str(ROOT / "datasets" / "cifar10"), train=False, download=False, transform=ToTensor())
    raw, label_int = data[index]
    raw = raw.unsqueeze(0)
    label = torch.tensor([label_int], dtype=torch.long)
    dm = torch.tensor(inversefed.consts.cifar10_mean)[:, None, None]
    ds = torch.tensor(inversefed.consts.cifar10_std)[:, None, None]
    normalized = (raw - dm) / ds
    model, _key = inversefed.construct_model("LeNetZhu", seed=42)
    model.eval()
    observed = list(gradient_dict(model, normalized, label, torch.nn.CrossEntropyLoss()).values())
    attack = inversefed.GradientReconstructor(model, (dm, ds), IMAGE_CONFIG.copy(), num_images=1)
    torch.manual_seed(30_600 + target_id)
    np.random.seed(30_600 + target_id)
    recon_norm, stats = attack.reconstruct(observed, label, img_shape=(3, 32, 32))
    recon = recon_norm * ds + dm
    gray = torch.full_like(raw, 0.5)
    cifar_mean = dm.expand_as(raw).clone()
    target_dir = out.parent
    tensor_to_png(raw, target_dir / "raw.png")
    tensor_to_png(recon, target_dir / "reconstruction.png")
    tensor_to_png(gray, target_dir / "gray_baseline.png")
    tensor_to_png(cifar_mean, target_dir / "cifar_mean_baseline.png")
    result = {
        "status": "ran",
        "stage": stage,
        "domain": "image",
        "eval": "E1_E2_identity",
        "defense": "undefended",
        "target_id": target_id,
        "cifar10_index": index,
        "source_id": index,
        "label": int(label_int),
        "mode": "undefended_official_geiping_e1_e2_identity",
        "metrics": image_metrics(raw, recon),
        "baselines": {
            "gray": image_metrics(raw, gray),
            "cifar_mean": image_metrics(raw, cifar_mean),
        },
        "objective": float(dict(stats).get("opt", float("nan"))),
        "elapsed_seconds": time.monotonic() - start,
    }
    out.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    return result


def measure_adult(dataset: ADULT, x: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, Any]:
    rec = post_process_continuous(reconstruction.detach().clone(), dataset)
    truth = dataset.decode_batch(x, standardized=True)
    guess = dataset.decode_batch(rec, standardized=True)
    aligned, all_error, cat_error, cont_error = match_reconstruction_ground_truth(truth, guess, dataset.create_tolerance_map())
    return {
        "accuracy_percent": 100.0 * (1.0 - float(np.mean(all_error))),
        "categorical_accuracy_percent": 100.0 * (1.0 - float(np.mean(cat_error))),
        "continuous_accuracy_percent": 100.0 * (1.0 - float(np.mean(cont_error))),
        "per_feature_accuracy_percent": {
            k: 100.0 * (1.0 - float(v))
            for k, v in batch_feature_wise_accuracy_score(truth, aligned, dataset.create_tolerance_map(), dataset.train_features).items()
        },
    }


def adult_data_free_baselines(dataset: ADULT, batch_size: int) -> dict[str, torch.Tensor]:
    """Return deterministic mean/mode and empirical-marginal Adult guesses."""
    mean_mode = torch.zeros(batch_size, dataset.Xtrain.shape[1], dtype=dataset.Xtrain.dtype)
    empirical = torch.zeros_like(mean_mode)
    offset = 0
    for _feature, values in dataset.train_features.items():
        if values is None:
            value = dataset.Xtrain[:, offset].mean()
            mean_mode[:, offset] = value
            empirical[:, offset] = value
            offset += 1
        else:
            width = len(values)
            probs = dataset.Xtrain[:, offset : offset + width].float().mean(dim=0)
            mode_idx = int(torch.argmax(probs).item())
            mean_mode[:, offset + mode_idx] = 1.0
            empirical[:, offset : offset + width] = probs
            offset += width
    return {"mean_mode": mean_mode, "empirical_marginal": empirical}


_ADULT_ADAPTIVE_CONTEXT: dict[str, Any] = {}


def adult_adaptive_cosine_loss(
    reconstruct_gradient: list[torch.Tensor],
    true_grad: list[torch.Tensor],
    device: str,
    weights: torch.Tensor | None = None,
    alpha: float | None = None,
) -> torch.Tensor:
    mode = _ADULT_ADAPTIVE_CONTEXT.get("mode")
    if mode == "mask_aware":
        return cosine_loss_lists(list(reconstruct_gradient), list(true_grad), _ADULT_ADAPTIVE_CONTEXT["masks"])
    if mode == "v2_sketch":
        return v2_sketch_loss(list(reconstruct_gradient), _ADULT_ADAPTIVE_CONTEXT["sketches"], _ADULT_ADAPTIVE_CONTEXT["matrices"])
    if mode == "count_sketch":
        if "compact_observed" in _ADULT_ADAPTIVE_CONTEXT:
            return count_sketch_loss_compact(
                list(reconstruct_gradient),
                _ADULT_ADAPTIVE_CONTEXT["compact_observed"],
                _ADULT_ADAPTIVE_CONTEXT["compact_inverse"],
                _ADULT_ADAPTIVE_CONTEXT["signs"],
            )
        return count_sketch_loss(
            list(reconstruct_gradient),
            _ADULT_ADAPTIVE_CONTEXT["table"],
            _ADULT_ADAPTIVE_CONTEXT["buckets"],
            _ADULT_ADAPTIVE_CONTEXT["signs"],
        )
    if mode is not None:
        raise ValueError(f"unknown Adult adaptive mode: {mode}")
    return cosine_loss_lists(list(reconstruct_gradient), list(true_grad))


def run_tableak_with_optional_adaptive(
    *,
    net: torch.nn.Module,
    criterion: torch.nn.Module,
    observed: list[torch.Tensor],
    labels: torch.Tensor,
    x: torch.Tensor,
    dataset: ADULT,
    adaptive_context: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, Any, Any]:
    cfg = official_config()
    if adaptive_context is None:
        return invert_grad(
            net=net,
            training_criterion=criterion,
            true_grad=observed,
            true_label=labels,
            true_data=torch.empty_like(x),
            dataset=dataset,
            **cfg,
        )
    original = tableak_gia._cosine_similarity_loss
    _ADULT_ADAPTIVE_CONTEXT.clear()
    _ADULT_ADAPTIVE_CONTEXT.update(adaptive_context)
    tableak_gia._cosine_similarity_loss = adult_adaptive_cosine_loss
    try:
        cfg["reconstruction_loss"] = "cosine_sim"
        return invert_grad(
            net=net,
            training_criterion=criterion,
            true_grad=observed,
            true_label=labels,
            true_data=torch.empty_like(x),
            dataset=dataset,
            **cfg,
        )
    finally:
        tableak_gia._cosine_similarity_loss = original
        _ADULT_ADAPTIVE_CONTEXT.clear()


def run_adult_cell(stage: str, eval_name: str, defense: str, target_id: int, indices: list[int], out_file: str) -> dict[str, Any]:
    torch.set_num_threads(1)
    start = time.monotonic()
    out = Path(out_file)
    if out.exists():
        try:
            existing = json.loads(out.read_text())
        except Exception:
            existing = {}
        if (
            stage == "S2d"
            and eval_name == "E2"
            and defense == "count_sketch"
            and existing.get("status") == "NOT_ASSESSABLE"
        ):
            preserved = out.with_suffix(".invalid_pre_fix.json")
            if not preserved.exists():
                preserved.write_text(out.read_text())
        else:
            return {"skipped": True, "path": str(out)}
    out.parent.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    os.chdir(ROOT / "external_defenses" / "tableak")
    try:
        dataset = ADULT()
        dataset.standardize()
        x = dataset.Xtrain[indices].clone()
        y = dataset.ytrain[indices].clone()
        net = FullyConnected(dataset.num_features, [100, 100, 2])
        criterion = torch.nn.CrossEntropyLoss()
        if defense == "precode":
            net = PrecodeAdultFC(net)
        undef_grad = gradient_dict(net, x, y, criterion)
        true_grad = {name: value.clone() for name, value in undef_grad.items()}
        adaptive_context: dict[str, Any] | None = None
        mode = defense
        if defense == "precode" and stage == "S2c" and eval_name == "E2":
            masks = [
                torch.zeros_like(parameter, dtype=torch.bool) if name.startswith("bottleneck.") else torch.ones_like(parameter, dtype=torch.bool)
                for name, parameter in net.named_parameters()
                if parameter.requires_grad
            ]
            adaptive_context = {"mode": "mask_aware", "masks": masks}
            mode = "precode_omit_bottleneck_gradients_e2"
        if defense == "gradient_pruning":
            true_grad, masks = prune_gradient(true_grad, 0.70)
            if stage == "S2b" and eval_name == "E2":
                adaptive_context = {"mode": "mask_aware", "masks": list(masks.values())}
                mode = "gradient_pruning_mask_aware_e2"
        elif defense == "dna_v1_conservative":
            transformed = dna_v1_gradient(true_grad)
            if stage == "S2b" and eval_name == "E2":
                debiased = dna_v1_debias(transformed)
                candidates = []
                for variant_name, observed_dict in [("dna_v1_plain_e2", transformed), ("dna_v1_debiased_e2", debiased)]:
                    torch.manual_seed(30_700 + target_id)
                    np.random.seed(30_700 + target_id)
                    rec_variant, ensemble_variant, losses_variant = run_tableak_with_optional_adaptive(
                        net=net,
                        criterion=criterion,
                        observed=list(observed_dict.values()),
                        labels=y,
                        x=x,
                        dataset=dataset,
                        adaptive_context=None,
                    )
                    objective = float(np.nanmin(np.asarray(losses_variant, dtype=float))) if len(losses_variant) else float("inf")
                    candidates.append((objective, variant_name, rec_variant, ensemble_variant, losses_variant))
                objective, mode, rec, ensemble, losses = min(candidates, key=lambda item: item[0])
                metric = measure_adult(dataset, x, rec)
                np.savez(out.parent / "arrays.npz", truth=x.numpy(), labels=y.numpy(), reconstruction=rec.detach().numpy(), objective_losses=np.array(losses))
                result = {
                    "status": "ran",
                    "stage": stage,
                    "domain": "adult",
                    "eval": eval_name,
                    "defense": defense,
                    "target_id": target_id,
                    "adult_indices": indices,
                    "mode": mode,
                    "selected_objective": objective,
                    "metric": metric,
                    "elapsed_seconds": time.monotonic() - start,
                }
                out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                return result
            true_grad = transformed
        elif defense == "dna_v2_0p95":
            if stage == "S2b" and eval_name == "E2":
                sketches, matrices = v2_sketch_payload(true_grad)
                adaptive_context = {"mode": "v2_sketch", "sketches": sketches, "matrices": matrices}
                mode = "dna_v2_sketch_space_e2"
            else:
                true_grad, _ = dna_v2_gradient(true_grad)
        elif defense == "soteria":
            rep_x = x.detach().clone().requires_grad_(True)
            representation = net.layers[:-1](rep_x) if hasattr(net, "layers") else net.prefix(rep_x)
            scores = []
            for j in range(representation.shape[1]):
                if rep_x.grad is not None:
                    rep_x.grad.zero_()
                net.zero_grad(set_to_none=True)
                representation[:, j].sum().backward(retain_graph=True)
                scores.append(rep_x.grad.detach().reshape(rep_x.shape[0], -1).norm(dim=1).sum() / (representation[:, j].detach().abs().sum() + 0.1))
            score = torch.stack(scores)
            threshold = torch.quantile(score.float(), 0.40)
            mask = (score >= threshold).to(next(iter(true_grad.values())))
            key = "layers.3.weight"
            true_grad[key] = true_grad[key] * mask.unsqueeze(0)
            if stage == "S2b" and eval_name == "E2":
                masks = []
                for name, tensor in true_grad.items():
                    if name == key:
                        masks.append(mask.unsqueeze(0).expand_as(tensor).to(dtype=torch.bool))
                    else:
                        masks.append(torch.ones_like(tensor, dtype=torch.bool))
                adaptive_context = {"mode": "mask_aware", "masks": masks}
                mode = "soteria_mask_aware_e2"
        elif defense == "count_sketch":
            table, buckets, signs, like = count_sketch_payload_from_update(true_grad)
            if eval_name == "E1":
                decoded = count_sketch_decode(table, buckets, signs, like)
                true_grad = {name: tensor for name, tensor in zip(true_grad.keys(), decoded)}
                mode = "count_sketch_decoded_e1"
            elif stage in {"S2b", "S2d"} and eval_name == "E2":
                compact_observed, compact_inverse = count_sketch_compact(table, buckets)
                adaptive_context = {
                    "mode": "count_sketch",
                    "table": table,
                    "buckets": buckets,
                    "signs": signs,
                    "compact_observed": compact_observed,
                    "compact_inverse": compact_inverse,
                }
                mode = "count_sketch_sketch_space_e2"
            else:
                result = {"status": "NOT_ASSESSABLE", "reason": "Count-Sketch adaptive objective is only run in S2b per Priority 30 continuation"}
                out.write_text(json.dumps(result, indent=2) + "\n")
                return result
        if eval_name == "E2" and stage not in {"S2b", "S2c", "S2d"}:
            result = {"status": "NOT_ASSESSABLE", "reason": f"native Adult adaptive objective for {defense} is only run in S2b per Priority 30 continuation"}
            out.write_text(json.dumps(result, indent=2) + "\n")
            return result
        if stage == "S2b" and eval_name == "E2" and defense == "precode":
            mode = "precode_model_adaptive_e2"
        torch.manual_seed(30_700 + target_id)
        np.random.seed(30_700 + target_id)
        rec, ensemble, losses = run_tableak_with_optional_adaptive(
            net=net,
            criterion=criterion,
            observed=list(true_grad.values()),
            labels=y,
            x=x,
            dataset=dataset,
            adaptive_context=adaptive_context,
        )
        metric = measure_adult(dataset, x, rec)
        np.savez(out.parent / "arrays.npz", truth=x.numpy(), labels=y.numpy(), reconstruction=rec.detach().numpy(), objective_losses=np.array(losses))
        result = {
            "status": "ran",
            "stage": stage,
            "domain": "adult",
            "eval": eval_name,
            "defense": defense,
            "target_id": target_id,
            "adult_indices": indices,
            "mode": mode,
            "metric": metric,
            "elapsed_seconds": time.monotonic() - start,
        }
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result
    finally:
        os.chdir(old)


def run_adult_undefended(stage: str, target_id: int, indices: list[int], out_file: str) -> dict[str, Any]:
    """Run the paired undefended comparator on the exact S2 Adult target."""
    torch.set_num_threads(1)
    start = time.monotonic()
    out = Path(out_file)
    if out.exists():
        return {"skipped": True, "path": str(out)}
    out.parent.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    os.chdir(ROOT / "external_defenses" / "tableak")
    try:
        dataset = ADULT()
        dataset.standardize()
        x = dataset.Xtrain[indices].clone()
        y = dataset.ytrain[indices].clone()
        net = FullyConnected(dataset.num_features, [100, 100, 2])
        criterion = torch.nn.CrossEntropyLoss()
        true_grad = gradient_dict(net, x, y, criterion)
        torch.manual_seed(30_700 + target_id)
        np.random.seed(30_700 + target_id)
        rec, ensemble, losses = run_tableak_with_optional_adaptive(
            net=net,
            criterion=criterion,
            observed=list(true_grad.values()),
            labels=y,
            x=x,
            dataset=dataset,
            adaptive_context=None,
        )
        baselines = adult_data_free_baselines(dataset, x.shape[0])
        baseline_metrics = {name: measure_adult(dataset, x, guess) for name, guess in baselines.items()}
        np.savez(
            out.parent / "arrays.npz",
            truth=x.numpy(),
            labels=y.numpy(),
            reconstruction=rec.detach().numpy(),
            mean_mode=baselines["mean_mode"].numpy(),
            empirical_marginal=baselines["empirical_marginal"].numpy(),
            objective_losses=np.array(losses),
        )
        result = {
            "status": "ran",
            "stage": stage,
            "domain": "adult",
            "eval": "E1_E2_identity",
            "defense": "undefended",
            "target_id": target_id,
            "adult_indices": indices,
            "source_ids": indices,
            "mode": "undefended_official_tableak_e1_e2_identity",
            "metric": measure_adult(dataset, x, rec),
            "baselines": baseline_metrics,
            "elapsed_seconds": time.monotonic() - start,
        }
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result
    finally:
        os.chdir(old)


def build_tasks(stage: str, n: int, output_dir: Path) -> list[tuple[str, tuple[Any, ...], Path]]:
    tasks = []
    if stage == "S0u":
        for target_id, idx in enumerate(image_targets(stage, n)):
            path = output_dir / stage / "image" / f"target_{target_id:03d}" / "result.json"
            tasks.append(("image_undefended", (stage, target_id, idx, str(path)), path))
        for target_id, batch in enumerate(adult_targets(stage, n)):
            path = output_dir / stage / "adult" / f"target_{target_id:03d}" / "result.json"
            tasks.append(("adult_undefended", (stage, target_id, batch, str(path)), path))
    if stage in {"S0", "S1"}:
        indices = image_targets(stage, n)
        defenses = IMAGE_DEFENSES
        evals = EVALS
        for eval_name in evals:
            for defense in defenses:
                for target_id, idx in enumerate(indices):
                    path = output_dir / stage / "image" / eval_name / defense / f"target_{target_id:03d}" / "result.json"
                    tasks.append(("image", (stage, eval_name, defense, target_id, idx, str(path)), path))
    if stage == "S1b":
        indices = image_targets(stage, n)
        cells = [
            ("E2", "dna_v1_conservative"),
            ("E2", "dna_v2_0p95"),
            ("E2", "gradient_pruning"),
            ("E1", "count_sketch"),
            ("E2", "count_sketch"),
        ]
        for eval_name, defense in cells:
            for target_id, idx in enumerate(indices):
                path = output_dir / stage / "image" / eval_name / defense / f"target_{target_id:03d}" / "result.json"
                tasks.append(("image", (stage, eval_name, defense, target_id, idx, str(path)), path))
    if stage == "S1c":
        indices = image_targets(stage, n)
        for defense in IMAGE_DEFENSES:
            for target_id, idx in enumerate(indices):
                path = output_dir / stage / "image" / "E2" / defense / f"target_{target_id:03d}" / "result.json"
                tasks.append(("image", (stage, "E2", defense, target_id, idx, str(path)), path))
    if stage == "S1d":
        indices = image_targets(stage, n)
        for eval_name in ("E1", "E2"):
            for target_id, idx in enumerate(indices):
                path = output_dir / stage / "image" / eval_name / "count_sketch" / f"target_{target_id:03d}" / "result.json"
                tasks.append(("image", (stage, eval_name, "count_sketch", target_id, idx, str(path)), path))
    if stage in {"S0", "S2"}:
        batches = adult_targets(stage, n)
        for eval_name in EVALS:
            for defense in ADULT_DEFENSES:
                for target_id, batch in enumerate(batches):
                    path = output_dir / stage / "adult" / eval_name / defense / f"target_{target_id:03d}" / "result.json"
                    tasks.append(("adult", (stage, eval_name, defense, target_id, batch, str(path)), path))
    if stage == "S2b":
        batches = adult_targets(stage, n)
        cells = [("E1", "count_sketch")] + [("E2", defense) for defense in ADULT_DEFENSES]
        for eval_name, defense in cells:
            for target_id, batch in enumerate(batches):
                path = output_dir / stage / "adult" / eval_name / defense / f"target_{target_id:03d}" / "result.json"
                tasks.append(("adult", (stage, eval_name, defense, target_id, batch, str(path)), path))
    if stage == "S2c":
        batches = adult_targets(stage, n)
        for target_id, batch in enumerate(batches):
            path = output_dir / stage / "adult" / "E2" / "precode" / f"target_{target_id:03d}" / "result.json"
            tasks.append(("adult", (stage, "E2", "precode", target_id, batch, str(path)), path))
    if stage == "S2d":
        batches = adult_targets(stage, n)
        for eval_name in ("E1", "E2"):
            for target_id, batch in enumerate(batches):
                path = output_dir / stage / "adult" / eval_name / "count_sketch" / f"target_{target_id:03d}" / "result.json"
                tasks.append(("adult", (stage, eval_name, "count_sketch", target_id, batch, str(path)), path))
    return tasks


def write_s0u_target_match_check(output_dir: Path, n: int) -> None:
    image_s0u = image_targets("S0u", n)
    image_s1 = image_targets("S1", n)
    adult_s0u = adult_targets("S0u", n)
    adult_s2 = adult_targets("S2", n)
    check = {
        "stage": "S0u",
        "n": n,
        "image_matches_S1": image_s0u == image_s1,
        "adult_matches_S2": adult_s0u == adult_s2,
        "image_source_ids": image_s0u,
        "adult_source_ids": adult_s0u,
    }
    if not check["image_matches_S1"] or not check["adult_matches_S2"]:
        raise RuntimeError("S0u target/source IDs do not exactly match S1/S2")
    target = output_dir / "S0u" / "target_match_check.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(check, indent=2) + "\n")


def write_summary(stage_dir: Path) -> None:
    rows = []
    for path in stage_dir.rglob("result.json"):
        data = json.loads(path.read_text())
        row = {"path": str(path), **{k: data.get(k) for k in ["status", "stage", "domain", "eval", "defense", "target_id", "elapsed_seconds"]}}
        if data.get("domain") == "image" and data.get("metrics"):
            row.update({f"image_{k}": v for k, v in data["metrics"].items()})
        if data.get("domain") == "adult" and data.get("metric"):
            row["adult_accuracy_percent"] = data["metric"]["accuracy_percent"]
        rows.append(row)
    if not rows:
        return
    csv_path = stage_dir / "summary.csv"
    keys = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["S0", "S0u", "S1", "S1b", "S1c", "S1d", "S2", "S2b", "S2c", "S2d", "S3", "S4"], required=True)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts/priority30_native_defenses/audit")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    n = args.n if args.n is not None else (2 if args.stage == "S0" else 39)
    if args.stage == "S0u":
        write_s0u_target_match_check(output_dir, n)
    tasks = build_tasks(args.stage, n, output_dir)
    log = output_dir / f"{args.stage}_progress.log"
    start = time.monotonic()
    done = 0
    total = len(tasks)
    with log.open("a") as lf:
        lf.write(json.dumps({"event": "start", "stage": args.stage, "tasks": total, "workers": args.workers, "time": time.time()}) + "\n")
    if args.workers == 1:
        for kind, task_args, _path in tasks:
            if kind == "image":
                run_image_cell(*task_args)
            elif kind == "adult":
                run_adult_cell(*task_args)
            elif kind == "image_undefended":
                run_image_undefended(*task_args)
            elif kind == "adult_undefended":
                run_adult_undefended(*task_args)
            else:
                raise ValueError(f"unknown task kind {kind}")
            done += 1
            elapsed = time.monotonic() - start
            eta = (elapsed / done) * (total - done) if done else None
            with log.open("a") as lf:
                lf.write(json.dumps({"event": "progress", "done": done, "total": total, "elapsed": elapsed, "eta": eta}) + "\n")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = []
            for kind, task_args, _path in tasks:
                if kind == "image":
                    fn = run_image_cell
                elif kind == "adult":
                    fn = run_adult_cell
                elif kind == "image_undefended":
                    fn = run_image_undefended
                elif kind == "adult_undefended":
                    fn = run_adult_undefended
                else:
                    raise ValueError(f"unknown task kind {kind}")
                futs.append(pool.submit(fn, *task_args))
            for fut in as_completed(futs):
                fut.result()
                done += 1
                elapsed = time.monotonic() - start
                eta = (elapsed / done) * (total - done) if done else None
                with log.open("a") as lf:
                    lf.write(json.dumps({"event": "progress", "done": done, "total": total, "elapsed": elapsed, "eta": eta}) + "\n")
    write_summary(output_dir / args.stage)
    with log.open("a") as lf:
        lf.write(json.dumps({"event": "complete", "stage": args.stage, "tasks": total, "elapsed": time.monotonic() - start}) + "\n")


if __name__ == "__main__":
    main()
