"""Native defense adapter utilities for Priority 30.

The code in this file is a project-local port of the mechanisms described in
external_defenses/*/PORTING_NOTES.md.  No external_defenses/ file is modified.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from experiments.harness.interfaces import (
    DefenseContext,
    ServerKnowledge,
    TensorState,
    TransmittedUpdate,
)
from dna_encoder.transform_defense import DNATransformConfig, transform_update_array
from dna_encoder.transform_defense_v2 import (
    DNATransformV2Config,
    materialize_projection_matrix_v2,
    reconstruct_update_array_v2,
    transform_update_array_v2,
)


ROOT = Path(__file__).resolve().parents[2]
PRECODE_SRC = ROOT / "external_defenses" / "precode" / "src"
if str(PRECODE_SRC) not in sys.path:
    sys.path.insert(0, str(PRECODE_SRC))
from VariationalBottleneck import VariationalBottleneck  # noqa: E402


def gradient_dict(model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module) -> dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    params = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(loss, [parameter for _, parameter in params], create_graph=False)
    return {name: grad.detach() for (name, _), grad in zip(params, gradients)}


def l2_update_delta(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for name in a:
        if name not in b or not a[name].is_floating_point():
            continue
        diff = (a[name].detach() - b[name].detach()).double()
        total += float((diff * diff).sum().item())
    return total ** 0.5


def l2_update_delta_mapped(
    a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor], mapping: Mapping[str, str]
) -> float:
    total = 0.0
    for left, right in mapping.items():
        if left not in a or right not in b:
            continue
        diff = (a[left].detach() - b[right].detach()).double()
        total += float((diff * diff).sum().item())
    return total ** 0.5


class PrecodeLeNetZhu(nn.Module):
    """LeNet-Zhu with official PRECODE variational bottleneck before fc.

    Official source: external_defenses/precode/src/VariationalBottleneck.py,
    commit c66adc4cdd62993139eafebc1b57fc25b0694874.
    """

    def __init__(self, base: nn.Module, latent_k: int = 256, beta: float = 1e-3):
        super().__init__()
        self.body = base.body
        self.bottleneck = VariationalBottleneck((768,), K=latent_k, beta=beta)
        self.fc = base.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.bottleneck(out)
        return self.fc(out)


class PrecodeAdultFC(nn.Module):
    """Adult FullyConnected with official PRECODE bottleneck before classifier."""

    def __init__(self, base: nn.Module, hidden_dim: int = 100, latent_k: int = 256, beta: float = 1e-3):
        super().__init__()
        # Official FullyConnected for Adult config 46 is Flatten, LinReLU, Linear.
        self.prefix = nn.Sequential(*list(base.layers.children())[:-1])
        self.bottleneck = VariationalBottleneck((hidden_dim,), K=latent_k, beta=beta)
        self.classifier = list(base.layers.children())[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.prefix(x)
        out = self.bottleneck(out)
        return self.classifier(out)


def soteria_masked_gradient(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    final_weight_name: str,
    representation: torch.Tensor,
    pruning_percentile: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Apply Soteria's client-side representation sensitivity mask.

    This follows external_defenses/soteria/GS_attack/reconstruct_image.py:
    compute ||d feature_j / dx|| / (feature_j + 0.1), percentile-threshold the
    scores, and mask columns of the final classifier gradient.
    """

    if x.grad is not None:
        x.grad.zero_()
    x_req = x.detach().clone().requires_grad_(True)
    # Re-run forward to connect representation to x_req.
    modules = dict(model.named_modules())
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        captured["rep"] = out.reshape(out.shape[0], -1)

    # Pick the last non-final module by caller-provided representation shape if
    # no explicit hook is available; for Priority 30 tests we use direct helpers
    # where `representation` shape determines the final-gradient columns.
    hooks = []
    candidate_modules = [m for _, m in model.named_modules() if not isinstance(m, nn.Sequential)]
    if len(candidate_modules) >= 2:
        hooks.append(candidate_modules[-2].register_forward_hook(hook))
    try:
        loss = loss_fn(model(x_req), y)
        grads = {
            name: grad.detach()
            for (name, _), grad in zip(
                [(n, p) for n, p in model.named_parameters() if p.requires_grad],
                torch.autograd.grad(loss, [p for _, p in model.named_parameters() if p.requires_grad], retain_graph=True),
            )
        }
    finally:
        for h in hooks:
            h.remove()

    rep = captured.get("rep", representation.detach()).reshape(x.shape[0], -1)
    scores = []
    for j in range(rep.shape[1]):
        model.zero_grad(set_to_none=True)
        if x_req.grad is not None:
            x_req.grad.zero_()
        rep[:, j].sum().backward(retain_graph=True)
        norm = x_req.grad.detach().reshape(x_req.shape[0], -1).norm(dim=1)
        denom = rep[:, j].detach().abs() + 0.1
        scores.append((norm / denom).sum())
    score = torch.stack(scores)
    threshold = torch.quantile(score.detach().float(), pruning_percentile / 100.0)
    mask = (score >= threshold).to(grads[final_weight_name].device, grads[final_weight_name].dtype)
    defended = {name: value.clone() for name, value in grads.items()}
    defended[final_weight_name] = defended[final_weight_name] * mask.unsqueeze(0)
    return defended, mask.detach().cpu()


def prune_gradient(update: TensorState, fraction: float = 0.70) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    defended = {name: tensor.detach().clone() for name, tensor in update.items()}
    masks: dict[str, torch.Tensor] = {}
    for name, tensor in defended.items():
        mask = torch.ones_like(tensor, dtype=torch.bool)
        count = int(tensor.numel() * fraction)
        if count:
            idx = torch.argsort(tensor.abs().reshape(-1), stable=True)[:count]
            mask.reshape(-1)[idx] = False
            tensor.mul_(mask)
        masks[name] = mask
    return defended, masks


def dna_v1_gradient(update: TensorState) -> dict[str, torch.Tensor]:
    config = DNATransformConfig(mix_ratio=0.08, keep_ratio=0.88, shrink_factor=0.45, seed=30_001)
    defended: dict[str, torch.Tensor] = {}
    for tensor_index, (name, tensor) in enumerate(update.items()):
        array = tensor.detach().cpu().numpy().astype("float32", copy=False)
        transformed, _stats = transform_update_array(array, config, tensor_index=tensor_index)
        defended[name] = torch.from_numpy(transformed.copy()).to(dtype=tensor.dtype, device=tensor.device)
    return defended


def dna_v2_gradient(update: TensorState, seed: int = 30_002) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    config = DNATransformV2Config(compression_ratio=0.95, quantization_eta=0.01, seed=seed)
    defended: dict[str, torch.Tensor] = {}
    metadata: dict[str, object] = {}
    for tensor_index, (name, tensor) in enumerate(update.items()):
        array = tensor.detach().cpu().numpy().astype("float32", copy=False)
        sketch, meta = transform_update_array_v2(array, config, tensor_index=tensor_index, quantization_seed=tensor_index)
        reconstructed, _stats = reconstruct_update_array_v2(sketch, meta)
        defended[name] = torch.from_numpy(reconstructed.copy()).to(dtype=tensor.dtype, device=tensor.device)
        metadata[name] = meta
    return defended, metadata


def v2_least_squares_sketch_residual(vector_size: int = 17, seed: int = 30_002) -> float:
    config = DNATransformV2Config(compression_ratio=0.95, quantization_eta=0.0, seed=seed)
    matrix, _meta = materialize_projection_matrix_v2(vector_size, config, tensor_index=0)
    x = torch.linspace(-1, 1, vector_size).numpy()
    sketch = matrix @ x
    recovered, *_ = __import__("numpy").linalg.lstsq(matrix, sketch, rcond=None)
    residual = matrix @ recovered - sketch
    return float((residual * residual).mean())


FETCHSGD_ROWS = 5
FETCHSGD_PORTED_COMPRESSION = 10.0


def fetchsgd_columns_for_dimension(dimension: int, rows: int = FETCHSGD_ROWS) -> int:
    """Port FetchSGD's count-sketch to small audit models at ~10x compression.

    The official defaults in external_defenses/fetchsgd are 5 rows and 500,000
    columns for much larger models.  For the Priority 30 LeNet/Adult models,
    using those absolute defaults expands rather than compresses.  We keep the
    paper's five-row structure and scale columns so rows*columns is about one
    tenth of the current gradient dimension.
    """

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    return max(1, int((dimension / FETCHSGD_PORTED_COMPRESSION + rows - 1) // rows))


def count_sketch(update: TensorState, rows: int = FETCHSGD_ROWS, columns: int | None = None, seed: int = 21) -> TransmittedUpdate:
    order = tuple(update.keys())
    flat = torch.cat([update[name].detach().reshape(-1) for name in order])
    if columns is None:
        columns = fetchsgd_columns_for_dimension(int(flat.numel()), rows=rows)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    buckets = torch.randint(columns, (rows, flat.numel()), generator=generator, dtype=torch.long).to(flat.device)
    signs = (2 * torch.randint(2, (rows, flat.numel()), generator=generator) - 1).to(flat.device, flat.dtype)
    table = torch.zeros(rows, columns, device=flat.device, dtype=flat.dtype)
    for row in range(rows):
        table[row].scatter_add_(0, buckets[row], signs[row] * flat)
    return TransmittedUpdate(
        {"sketch": table},
        {
            "tensor_order": order,
            "shapes": {name: tuple(update[name].shape) for name in order},
            "dimension": flat.numel(),
            "sketch_seed": seed,
            "rows": rows,
            "columns": columns,
            "ported_compression_target": FETCHSGD_PORTED_COMPRESSION,
            "source": "external_defenses/fetchsgd/PORTING_NOTES.md: defaults k=50000, rows=5, columns=500000; columns scaled to keep sketch compressed for small audit models",
        },
        required_secrets_for_decode=frozenset({"sketch_hashes"}),
    )


def sketch_server_knowledge(known_hashes: bool) -> ServerKnowledge:
    if known_hashes:
        return ServerKnowledge(seeds=frozenset({"sketch_hashes"}), public_hyperparameters={"sketch_seed": 21})
    return ServerKnowledge()
