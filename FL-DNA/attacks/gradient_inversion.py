"""Gradient matching attack for sample-level PaySim reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

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
    best_loss: float
    loss_history: list[float]


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
    dummy = torch.randn(1, input_dim, dtype=observed[0].dtype, requires_grad=True)

    if config.optimizer.lower() == "lbfgs":
        optimizer = torch.optim.LBFGS([dummy], lr=config.learning_rate, max_iter=1)
    else:
        optimizer = torch.optim.Adam([dummy], lr=config.learning_rate)

    best_loss = float("inf")
    best_dummy = dummy.detach().clone()
    loss_history: list[float] = []

    def closure() -> torch.Tensor:
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
        loss = match_loss + regularization
        loss.backward()
        return loss

    for _ in range(config.iterations):
        if config.optimizer.lower() == "lbfgs":
            loss = optimizer.step(closure)
            loss_value = float(loss.detach().item())
        else:
            loss = closure()
            optimizer.step()
            loss_value = float(loss.detach().item())

        loss_history.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_dummy = dummy.detach().clone()

    return GradientInversionResult(
        reconstructed_input=best_dummy.detach(),
        best_loss=best_loss,
        loss_history=loss_history,
    )


def _gradient_match_loss(
    candidate_gradients: list[torch.Tensor],
    observed_gradients: list[torch.Tensor],
) -> torch.Tensor:
    losses = [
        torch.mean((candidate - observed) ** 2)
        for candidate, observed in zip(candidate_gradients, observed_gradients)
    ]
    return torch.stack(losses).mean()

