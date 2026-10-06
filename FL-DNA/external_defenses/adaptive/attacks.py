"""Original-style and defense-aware gradient reconstruction attacks.

All attack classes implement `Attack.reconstruct` from experiments/harness/
interfaces.py.  Labels, input geometry, transforms, and stochastic-model hooks
are fixed at construction because that intentionally minimal harness interface
does not include them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import torch
from torch.nn import functional as F

from experiments.harness.interfaces import AttackResult, DefenseView, ServerKnowledge


def _named_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def _gradient_dict(
    model: torch.nn.Module, x: torch.Tensor, label: torch.Tensor, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> dict[str, torch.Tensor]:
    output = model(x)
    loss = loss_fn(output, label)
    parameters = _named_parameters(model)
    gradients = torch.autograd.grad(loss, [parameter for _, parameter in parameters], create_graph=True)
    return {name: gradient for (name, _), gradient in zip(parameters, gradients)}


def _matching_loss(
    produced: Mapping[str, torch.Tensor], observed: Mapping[str, torch.Tensor], names: Sequence[str], masks: Mapping[str, torch.Tensor] | None = None
) -> torch.Tensor:
    terms = []
    for name in names:
        if name not in produced or name not in observed:
            continue
        difference = produced[name] - observed[name].to(produced[name])
        if masks is not None and name in masks:
            difference = difference[masks[name].to(difference.device)]
        terms.append((difference * difference).mean())
    if not terms:
        raise ValueError("no shared gradient names to match")
    return torch.stack(terms).mean()


@dataclass
class GradientMatchingAttack:
    """Non-adaptive DLG/IGA-style attack using the complete received update.

    Zhu et al. (2019), Eq. (4): "min_{x',y'} ||∇_W L(F(x',W),y') − ∇_W
    L(F(x,W),y)||²."  This implementation fixes the true label at construction
    for controlled tests; it does not claim the paper's joint dummy-label search.
    """

    input_shape: tuple[int, ...]
    label: torch.Tensor
    steps: int = 300
    lr: float = 0.1
    seed: int = 0
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.cross_entropy
    name: str = "original_gradient_matching"

    def _initial_input(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        generator = torch.Generator(device=device).manual_seed(self.seed)
        return torch.randn(self.input_shape, generator=generator, device=device, dtype=dtype, requires_grad=True)

    def _names(self, view: DefenseView, model: torch.nn.Module) -> list[str]:
        return [name for name, _ in _named_parameters(model) if name in view.tensors]

    def reconstruct(self, server_view: DefenseView, model: torch.nn.Module, knowledge: ServerKnowledge) -> AttackResult:
        observed = server_view.tensors
        parameter = next(model.parameters())
        x = self._initial_input(parameter.device, parameter.dtype)
        label = self.label.to(parameter.device)
        optimizer = torch.optim.Adam([x], lr=self.lr)
        names = self._names(server_view, model)
        was_training = model.training
        model.eval()  # deterministic BatchNorm/dropout for the original-style control.
        try:
            for _ in range(self.steps):
                optimizer.zero_grad()
                loss = _matching_loss(_gradient_dict(model, x, label, self.loss_fn), observed, names)
                loss.backward()
                optimizer.step()
            final = _matching_loss(_gradient_dict(model, x, label, self.loss_fn), observed, names)
        finally:
            model.train(was_training)
        return AttackResult(x.detach(), float(final.detach()), {"matched_names": tuple(names), "seed": self.seed})


@dataclass
class OmitGradientAttack(GradientMatchingAttack):
    """Soteria/PRECODE adaptive attack that excludes specified gradients.

    Balunovic et al. (2022), §5.2: "we simply remove the gradients of the
    defended layer from the set of observed gradients."  Scheliga et al. (2023),
    §4.2 similarly states that "omitting the gradients of the stochastic layers"
    bypasses PRECODE.  If `omit_names` is empty, every candidate parameter name
    is tried as a defended-layer hypothesis and the lowest objective is returned.
    """

    omit_names: tuple[str, ...] = ()
    name: str = "omit_gradient_adaptive"

    def _run(self, server_view: DefenseView, model: torch.nn.Module, excluded: set[str]) -> AttackResult:
        observed = {name: tensor for name, tensor in server_view.tensors.items() if name not in excluded}
        if not observed:
            raise ValueError("cannot omit every observed gradient")
        view = DefenseView(observed, server_view.metadata, server_view.server_knowledge, server_view.defense_name)
        result = super().reconstruct(view, model, server_view.server_knowledge)
        return AttackResult(result.estimate, result.objective_value, {**result.metadata, "omitted": tuple(sorted(excluded))})

    def reconstruct(self, server_view: DefenseView, model: torch.nn.Module, knowledge: ServerKnowledge) -> AttackResult:
        if self.omit_names:
            return self._run(server_view, model, set(self.omit_names))
        candidates = [name for name in server_view.tensors if name.endswith("weight")]
        if not candidates:
            return super().reconstruct(server_view, model, knowledge)
        results = [self._run(server_view, model, {candidate}) for candidate in candidates]
        return min(results, key=lambda result: result.objective_value)


@dataclass
class MaskAwareGradientAttack(GradientMatchingAttack):
    """Pruning-aware gradient matching on coordinates retained by the client.

    Yue et al. (2023), §3.2: "the attacker can incorporate the gradient
    post-processing function into the reconstruction optimization."  For a
    hard pruning mask, matching only entries where the observed update is
    unpruned is the exact differentiable objective away from ties.
    """

    masks: Mapping[str, torch.Tensor] | None = None
    name: str = "mask_aware_gradient_matching"

    def reconstruct(self, server_view: DefenseView, model: torch.nn.Module, knowledge: ServerKnowledge) -> AttackResult:
        parameter = next(model.parameters())
        x = self._initial_input(parameter.device, parameter.dtype)
        label = self.label.to(parameter.device)
        masks = self.masks or {
            name: tensor.ne(0) for name, tensor in server_view.tensors.items()
        }
        optimizer = torch.optim.Adam([x], lr=self.lr)
        names = self._names(server_view, model)
        was_training = model.training
        model.eval()
        try:
            for _ in range(self.steps):
                optimizer.zero_grad()
                loss = _matching_loss(_gradient_dict(model, x, label, self.loss_fn), server_view.tensors, names, masks)
                loss.backward()
                optimizer.step()
            final = _matching_loss(_gradient_dict(model, x, label, self.loss_fn), server_view.tensors, names, masks)
        finally:
            model.train(was_training)
        return AttackResult(x.detach(), float(final.detach()), {"matched_names": tuple(names), "mask_aware": True, "seed": self.seed})


@dataclass
class KnownTransformAttack(GradientMatchingAttack):
    """Recover a raw input by differentiating through the known ATS policy.

    Gao et al. (2021), §6 admits that "a more sophisticated adversary may try
    to bypass our defense."  This attack directly implements that threat: it
    solves the DLG objective over raw `x` while generating dummy gradients from
    `transform(x)`.  The transform must be differentiable; non-invertible PIL
    policies cannot be faithfully inverted from a gradient alone.
    """

    transform: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
    name: str = "known_transform_raw_input_attack"

    def reconstruct(self, server_view: DefenseView, model: torch.nn.Module, knowledge: ServerKnowledge) -> AttackResult:
        parameter = next(model.parameters())
        x = self._initial_input(parameter.device, parameter.dtype)
        label = self.label.to(parameter.device)
        optimizer = torch.optim.Adam([x], lr=self.lr)
        names = self._names(server_view, model)
        was_training = model.training
        model.eval()
        try:
            for _ in range(self.steps):
                optimizer.zero_grad()
                loss = _matching_loss(_gradient_dict(model, self.transform(x), label, self.loss_fn), server_view.tensors, names)
                loss.backward()
                optimizer.step()
            final = _matching_loss(_gradient_dict(model, self.transform(x), label, self.loss_fn), server_view.tensors, names)
        finally:
            model.train(was_training)
        return AttackResult(x.detach(), float(final.detach()), {"target_space": "raw_input", "seed": self.seed})


@dataclass
class StochasticNoiseAttack(GradientMatchingAttack):
    """Jointly optimize the dummy input and a stochastic-forward noise vector.

    Scheliga et al. (2022), abstract: "DIA jointly optimizes for client data and
    dropout masks to approximate the stochastic client model."  The caller must
    provide `noise_setter`, which injects the supplied tensor into a compatible
    stochastic model before each forward pass; generic PyTorch modules expose no
    standard hook for overriding reparameterization noise.
    """

    noise_shape: tuple[int, ...] = ()
    noise_setter: Callable[[torch.Tensor], None] | None = None
    name: str = "joint_stochastic_noise_attack"

    def reconstruct(self, server_view: DefenseView, model: torch.nn.Module, knowledge: ServerKnowledge) -> AttackResult:
        if self.noise_setter is None or not self.noise_shape:
            raise NotImplementedError("faithful joint-noise attack requires a model-specific noise_setter")
        parameter = next(model.parameters())
        x = self._initial_input(parameter.device, parameter.dtype)
        generator = torch.Generator(device=parameter.device).manual_seed(self.seed + 1)
        noise = torch.randn(self.noise_shape, generator=generator, device=parameter.device, dtype=parameter.dtype, requires_grad=True)
        label = self.label.to(parameter.device)
        optimizer = torch.optim.Adam([x, noise], lr=self.lr)
        names = self._names(server_view, model)
        was_training = model.training
        model.eval()
        try:
            for _ in range(self.steps):
                optimizer.zero_grad()
                self.noise_setter(noise)
                loss = _matching_loss(_gradient_dict(model, x, label, self.loss_fn), server_view.tensors, names)
                loss.backward()
                optimizer.step()
            self.noise_setter(noise)
            final = _matching_loss(_gradient_dict(model, x, label, self.loss_fn), server_view.tensors, names)
        finally:
            model.train(was_training)
        return AttackResult(x.detach(), float(final.detach()), {"noise": noise.detach(), "seed": self.seed})


@dataclass
class SketchGradientAttack(GradientMatchingAttack):
    """Song et al.'s attack objective for a known sketch map.

    Song et al. (2023), §7.2, Eq. (7.2): "L_R(x) := ||R(∇_w F(w,x)) − R(g)||²."
    The paper states: "It is reasonable to assume the attacker has access to R"
    because sketching frameworks share it.  Without hashes, this method returns
    an explicitly unavailable result instead of inventing a sketch map.
    """

    name: str = "known_sketch_gradient_attack"

    @staticmethod
    def _sketch(flat: torch.Tensor, rows: int, columns: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        buckets = torch.randint(columns, (rows, flat.numel()), generator=generator).to(flat.device)
        signs = (2 * torch.randint(2, (rows, flat.numel()), generator=generator) - 1).to(flat.device, flat.dtype)
        table = torch.zeros(rows, columns, device=flat.device, dtype=flat.dtype)
        for row in range(rows):
            table[row].scatter_add_(0, buckets[row], signs[row] * flat)
        return table

    def reconstruct(self, server_view: DefenseView, model: torch.nn.Module, knowledge: ServerKnowledge) -> AttackResult:
        metadata = server_view.metadata
        if "sketch_seed" not in metadata:
            parameter = next(model.parameters())
            estimate = torch.zeros(self.input_shape, device=parameter.device, dtype=parameter.dtype)
            return AttackResult(estimate, float("inf"), {"status": "unavailable_without_sketch_hashes"})
        parameter = next(model.parameters())
        x = self._initial_input(parameter.device, parameter.dtype)
        label = self.label.to(parameter.device)
        order = tuple(metadata["tensor_order"])
        observed = server_view.tensors["sketch"].to(parameter.device)
        optimizer = torch.optim.Adam([x], lr=self.lr)
        was_training = model.training
        model.eval()
        try:
            for _ in range(self.steps):
                optimizer.zero_grad()
                gradients = _gradient_dict(model, x, label, self.loss_fn)
                flat = torch.cat([gradients[name].reshape(-1) for name in order])
                loss = ((self._sketch(flat, int(metadata["rows"]), int(metadata["columns"]), int(metadata["sketch_seed"])) - observed) ** 2).mean()
                loss.backward()
                optimizer.step()
            gradients = _gradient_dict(model, x, label, self.loss_fn)
            flat = torch.cat([gradients[name].reshape(-1) for name in order])
            final = ((self._sketch(flat, int(metadata["rows"]), int(metadata["columns"]), int(metadata["sketch_seed"])) - observed) ** 2).mean()
        finally:
            model.train(was_training)
        return AttackResult(x.detach(), float(final.detach()), {"status": "known_hashes", "seed": self.seed})
