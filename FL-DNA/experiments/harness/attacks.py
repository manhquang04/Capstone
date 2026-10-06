"""Standard attack wrappers for the Priority 27 harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .interfaces import AttackResult, DefenseView, ServerKnowledge


@dataclass
class BNClosedFormAttack:
    """Closed-form recovery hook for BN-statistics channels."""

    recover: Callable[[DefenseView, torch.nn.Module], torch.Tensor]
    name: str = "BNClosedFormAttack"

    def reconstruct(self, server_view: DefenseView, model: torch.nn.Module, knowledge: ServerKnowledge) -> AttackResult:
        estimate = self.recover(server_view, model)
        return AttackResult(estimate=estimate, objective_value=0.0, metadata={"attack": self.name})


@dataclass
class TabLeakGradientAttack:
    """Configuration holder for a TabLeak-style tabular gradient attack."""

    optimizer: Callable[[DefenseView, torch.nn.Module, ServerKnowledge], AttackResult]
    steps: int = 2000
    restarts: int = 8
    attacker_lr: float = 0.05
    name: str = "TabLeakGradientAttack"

    def reconstruct(self, server_view: DefenseView, model: torch.nn.Module, knowledge: ServerKnowledge) -> AttackResult:
        result = self.optimizer(server_view, model, knowledge)
        metadata = dict(result.metadata)
        metadata.update({"steps": self.steps, "restarts": self.restarts, "attacker_lr": self.attacker_lr})
        return AttackResult(result.estimate, result.objective_value, metadata)


@dataclass
class ImageCosineTVAttack:
    """Configuration holder for Geiping-style image inversion."""

    optimizer: Callable[[DefenseView, torch.nn.Module, ServerKnowledge], AttackResult]
    steps: int = 250
    restarts: int = 4
    tv_weight: float = 1e-4
    name: str = "ImageCosineTVAttack"

    def reconstruct(self, server_view: DefenseView, model: torch.nn.Module, knowledge: ServerKnowledge) -> AttackResult:
        result = self.optimizer(server_view, model, knowledge)
        metadata = dict(result.metadata)
        metadata.update({"steps": self.steps, "restarts": self.restarts, "tv_weight": self.tv_weight})
        return AttackResult(result.estimate, result.objective_value, metadata)


@dataclass
class AdaptiveAttackHook:
    """Defense-supplied best attack under declared server knowledge."""

    hook: Callable[[DefenseView, torch.nn.Module, ServerKnowledge], AttackResult]
    name: str = "AdaptiveAttackHook"

    def reconstruct(self, server_view: DefenseView, model: torch.nn.Module, knowledge: ServerKnowledge) -> AttackResult:
        return self.hook(server_view, model, knowledge)

