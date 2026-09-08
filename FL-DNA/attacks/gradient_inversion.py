"""Gradient matching attack for sample-level PaySim reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass(frozen=True)
class GradientInversionConfig:
    """Optimization settings for gradient inversion."""

    iterations: int = 300
    learning_rate: float = 0.05
    l2_weight: float = 1e-4
    optimizer: str = "adam"
    seed: int = 42


@dataclass
class GradientInversionResult:
    """Reconstruction output and attack diagnostics."""

    reconstructed_input: torch.Tensor
    initial_dummy: torch.Tensor
    best_loss: float
    loss_history: list[float]
    best_gradient_match_loss: float
    best_regularization_loss: float
    initial_total_objective: float
    best_iteration: int
    component_history: list[dict[str, float]] = field(default_factory=list)


def parameter_gradients(
    model: nn.Module,
    criterion: nn.Module,
    features: torch.Tensor,
    label: torch.Tensor,
    create_graph: bool = False,
) -> list[torch.Tensor]:
    """Compute per-parameter gradients for one known-label sample."""
    model.eval()
    logits = model(features)
    loss = criterion(logits, label)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(
        loss,
        params,
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=False,
    )
    return [gradient for gradient in gradients]


def gradient_inversion_attack(
    model: nn.Module,
    criterion: nn.Module,
    observed_gradients: list[torch.Tensor],
    label: torch.Tensor,
    input_dim: int,
    config: GradientInversionConfig,
) -> GradientInversionResult:
    """
    Reconstruct a tabular input by matching dummy gradients to observed gradients.

    This is a real gradient matching attack under a known-label threat model.
    """
    torch.manual_seed(config.seed)
    model.eval()
    observed = [gradient.detach() for gradient in observed_gradients]
    if config.iterations < 0 or config.optimizer.lower() not in {'adam', 'lbfgs'}:
        raise ValueError('Invalid iterations or optimizer')
    if not observed or not all(torch.isfinite(g).all() for g in observed):
        raise ValueError('Observed gradients must be finite and nonempty')
    dummy = torch.randn(1, input_dim, dtype=observed[0].dtype, requires_grad=True)
    initial_dummy = dummy.detach().clone()
    initial_match, initial_regularization, initial_total = _score_components(
        model, criterion, initial_dummy, observed, label, config.l2_weight
    )

    if config.optimizer.lower() == "lbfgs":
        optimizer = torch.optim.LBFGS([dummy], lr=config.learning_rate, max_iter=1)
    else:
        optimizer = torch.optim.Adam([dummy], lr=config.learning_rate)

    best_loss = initial_total
    best_dummy = dummy.detach().clone()
    best_match = initial_match
    best_regularization = initial_regularization
    best_iteration = 0
    loss_history: list[float] = []
    component_history = [dict(iteration=0, gradient_match_loss=initial_match,
                              regularization_loss=initial_regularization, total_objective=initial_total)]

    last_match = initial_match
    last_regularization = initial_regularization

    def closure() -> torch.Tensor:
        nonlocal last_match, last_regularization
        optimizer.zero_grad()
        dummy_gradients = parameter_gradients(
            model,
            criterion,
            dummy,
            label,
            create_graph=True,
        )
        match_loss = _gradient_match_loss(dummy_gradients, observed)
        regularization = config.l2_weight * dummy.square().mean()
        last_match = float(match_loss.detach().item())
        last_regularization = float(regularization.detach().item())
        loss = match_loss + regularization
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite attack objective')
        loss.backward()
        return loss

    for _ in range(config.iterations):
        if config.optimizer.lower() == "lbfgs":
            optimizer.step(closure)
        else:
            closure()
            optimizer.step()

        # Score and snapshot the post-update candidate. The previous code
        # recorded the pre-step loss but copied the post-step dummy, so the
        # saved candidate did not necessarily correspond to best_loss.
        loss = closure()
        loss_value = float(loss.detach().item())

        loss_history.append(loss_value)
        component_history.append(dict(iteration=len(loss_history), gradient_match_loss=last_match,
                                      regularization_loss=last_regularization, total_objective=loss_value))
        if loss_value < best_loss:
            best_loss = loss_value
            best_dummy = dummy.detach().clone()
            best_match = last_match
            best_regularization = last_regularization
            best_iteration = len(loss_history)

    return GradientInversionResult(
        reconstructed_input=best_dummy.detach(),
        initial_dummy=initial_dummy,
        best_loss=best_loss,
        loss_history=loss_history,
        best_gradient_match_loss=best_match,
        best_regularization_loss=best_regularization,
        initial_total_objective=initial_total,
        best_iteration=best_iteration,
        component_history=component_history,
    )


def _score_components(model, criterion, candidate, observed, label, l2_weight):
    gradients = parameter_gradients(model, criterion, candidate, label, create_graph=False)
    match = _gradient_match_loss(gradients, observed)
    regularization = l2_weight * candidate.square().mean()
    return (
        float(match.item()),
        float(regularization.item()),
        float((match + regularization).item()),
    )


def _gradient_match_loss(
    candidate_gradients: list[torch.Tensor],
    observed_gradients: list[torch.Tensor],
) -> torch.Tensor:
    if len(candidate_gradients) != len(observed_gradients) or any(
        a.shape != b.shape for a, b in zip(candidate_gradients, observed_gradients)
    ):
        raise ValueError('Gradient structures differ')
    losses = [
        torch.mean((candidate - observed) ** 2)
        for candidate, observed in zip(candidate_gradients, observed_gradients)
    ]
    return torch.stack(losses).mean()
