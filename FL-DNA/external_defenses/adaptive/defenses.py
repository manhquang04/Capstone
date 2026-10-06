"""Pure update-dictionary defenses implementing the harness Defense contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from experiments.harness.interfaces import (
    Defense,
    DefenseContext,
    DefenseView,
    ServerKnowledge,
    TensorState,
    TransmittedUpdate,
)


def _copy(update: TensorState) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in update.items()}


class _BaseDefense(Defense):
    """Common identity/server-view behavior for defenses with cleartext payloads."""

    name = "base"

    def server_view(self, transmitted: TransmittedUpdate, server_knowledge: ServerKnowledge) -> DefenseView:
        return DefenseView(transmitted.tensors, transmitted.metadata, server_knowledge, self.name)


@dataclass(frozen=True)
class GradientPruningDefense(_BaseDefense):
    """Per-tensor magnitude pruning.

    Zhu et al. (2019), §5.2: "we prune the gradients with small magnitude."
    `fraction=0` is the mandated identity path.  Ties use stable flattened index
    order, making this a deterministic pure function.
    """

    fraction: float = 0.0
    name: str = "gradient_pruning"

    def encode(self, update: TensorState, context: DefenseContext) -> TransmittedUpdate:
        if not 0.0 <= self.fraction < 1.0:
            raise ValueError("fraction must be in [0, 1)")
        tensors = _copy(update)
        if self.fraction == 0.0:
            return TransmittedUpdate(tensors, {"fraction": 0.0, "mask": {}})
        masks: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            count = int(tensor.numel() * self.fraction)
            mask = torch.ones_like(tensor, dtype=torch.bool)
            if count:
                indices = torch.argsort(tensor.abs().flatten(), stable=True)[:count]
                mask.flatten()[indices] = False
                tensor.mul_(mask)
            masks[name] = mask
        return TransmittedUpdate(tensors, {"fraction": self.fraction, "mask": masks})


@dataclass(frozen=True)
class SoteriaDefense(_BaseDefense):
    """Mask selected classifier-input columns in a named final-layer gradient.

    Sun et al. (2021), §4.2: "we perturb the representation"; the official
    implementation applies the selected mask as `input_gradient[8] =
    input_gradient[8] * mask` (reconstruct_image.py:107-110).  The sensitivity
    ranking must be computed at client-update time and is supplied as
    `representation_scores`; this keeps encode pure over the update dictionary.
    """

    defended_gradient: str
    representation_scores: torch.Tensor
    fraction: float = 0.0
    name: str = "soteria"

    def encode(self, update: TensorState, context: DefenseContext) -> TransmittedUpdate:
        tensors = _copy(update)
        if self.fraction == 0.0:
            return TransmittedUpdate(tensors, {"fraction": 0.0, "defended_gradient": self.defended_gradient})
        if self.defended_gradient not in tensors:
            raise KeyError(self.defended_gradient)
        gradient = tensors[self.defended_gradient]
        if gradient.ndim != 2 or gradient.shape[1] != self.representation_scores.numel():
            raise ValueError("Soteria requires a [classes, representation] weight gradient")
        count = int(gradient.shape[1] * self.fraction)
        mask = torch.ones(gradient.shape[1], dtype=torch.bool, device=gradient.device)
        if count:
            chosen = torch.argsort(self.representation_scores.to(gradient.device).abs(), stable=True)[:count]
            mask[chosen] = False
            gradient.mul_(mask.unsqueeze(0))
        return TransmittedUpdate(
            tensors,
            {"fraction": self.fraction, "defended_gradient": self.defended_gradient, "representation_mask": mask},
        )


@dataclass(frozen=True)
class PRECODEDefense(_BaseDefense):
    """Transport an update from a PRECODE model without further update distortion.

    Scheliga et al. (2022), §3: "The module maps a representation to a latent
    distribution, samples from it, and reconstructs the representation."  The
    stochastic variational bottleneck is necessarily in the model forward pass,
    not a transform of an already-computed update dict; this adapter records the
    stochastic-gradient names so attacks can explicitly include or omit them.
    """

    stochastic_gradient_names: tuple[str, ...] = ()
    name: str = "precode"

    def encode(self, update: TensorState, context: DefenseContext) -> TransmittedUpdate:
        return TransmittedUpdate(_copy(update), {"stochastic_gradient_names": self.stochastic_gradient_names})


@dataclass(frozen=True)
class ATSDefense(_BaseDefense):
    """Metadata adapter for updates computed on an ATS-transformed input.

    Gao et al. (2021), §4: "clients locally transform their raw data and then
    train the model using the transformed data."  Since an update dict contains
    no raw input, the client-side transform must be applied before gradient
    computation; this pure adapter transmits only the resulting update and an
    optional policy identifier, as in ordinary FedAvg.
    """

    policy_id: str = "identity"
    policy_known_to_server: bool = False
    name: str = "ats"

    def encode(self, update: TensorState, context: DefenseContext) -> TransmittedUpdate:
        metadata = {"policy_id": self.policy_id} if self.policy_known_to_server else {}
        return TransmittedUpdate(_copy(update), metadata)


@dataclass(frozen=True)
class CountSketchDefense(Defense):
    """Fixed-hash Count-Sketch encoder over a registered tensor order.

    Rothchild et al. (2020), §3.2: "S(g1 + g2) = S(g1) + S(g2)."  Song et al.
    (2023), §7.2 assumes an attacker can observe the sketching matrix.  Hashes
    are generated solely from `seed`; `rows=0` is an explicit test-only
    undefended path that returns the raw update unchanged.
    """

    rows: int = 5
    columns: int = 500_000
    seed: int = 21
    reveal_hashes: bool = True
    name: str = "count_sketch"

    def _map(self, total: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        buckets = torch.randint(self.columns, (self.rows, total), generator=generator, dtype=torch.long)
        signs = (2 * torch.randint(2, (self.rows, total), generator=generator) - 1).to(torch.float32)
        return buckets.to(device), signs.to(device)

    def encode(self, update: TensorState, context: DefenseContext) -> TransmittedUpdate:
        if self.rows == 0:
            return TransmittedUpdate(_copy(update), {"identity": True})
        order = context.tensor_order or tuple(update.keys())
        flat = torch.cat([update[name].detach().reshape(-1) for name in order])
        buckets, signs = self._map(flat.numel(), flat.device)
        table = torch.zeros(self.rows, self.columns, device=flat.device, dtype=flat.dtype)
        for row in range(self.rows):
            table[row].scatter_add_(0, buckets[row], signs[row].to(flat.dtype) * flat)
        shapes = {name: tuple(update[name].shape) for name in order}
        metadata = {"tensor_order": order, "shapes": shapes, "dimension": flat.numel()}
        secrets = frozenset()
        if self.reveal_hashes:
            metadata.update({"sketch_seed": self.seed, "rows": self.rows, "columns": self.columns})
        else:
            secrets = frozenset({"sketch_hashes"})
        return TransmittedUpdate({"sketch": table}, metadata, secrets)

    def server_view(self, transmitted: TransmittedUpdate, server_knowledge: ServerKnowledge) -> DefenseView:
        return DefenseView(transmitted.tensors, transmitted.metadata, server_knowledge, self.name)
