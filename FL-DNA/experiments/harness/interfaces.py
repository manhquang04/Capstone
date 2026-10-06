"""Interfaces for Priority 27 defense-agnostic evaluation.

The harness is intentionally lightweight: it records what the server is
allowed to know and lets experiments plug in concrete defenses/attacks while
the validator enforces common checks before a report is emitted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import torch


TensorState = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class ServerKnowledge:
    """Explicit declaration of server-side knowledge for one cell."""

    keys: frozenset[str] = frozenset()
    seeds: frozenset[str] = frozenset()
    metadata: frozenset[str] = frozenset()
    public_hyperparameters: Mapping[str, Any] = field(default_factory=dict)

    def has(self, secret_name: str) -> bool:
        return (
            secret_name in self.keys
            or secret_name in self.seeds
            or secret_name in self.metadata
            or secret_name in self.public_hyperparameters
        )


@dataclass(frozen=True)
class DefenseContext:
    """Context available when a client encodes a local update."""

    round_id: int
    client_id: int
    tensor_order: tuple[str, ...]
    server_knowledge: ServerKnowledge


@dataclass(frozen=True)
class TransmittedUpdate:
    """Payload sent by a client."""

    tensors: TensorState
    metadata: Mapping[str, Any] = field(default_factory=dict)
    required_secrets_for_decode: frozenset[str] = frozenset()


@dataclass(frozen=True)
class DefenseView:
    """What the honest-but-curious server can observe."""

    tensors: TensorState
    metadata: Mapping[str, Any]
    server_knowledge: ServerKnowledge
    defense_name: str


@dataclass(frozen=True)
class DPComparatorSpec:
    """Auditable DP comparator metadata."""

    clip_scope: str
    clip_norm: float | Mapping[str, float]
    noise_multiplier: float
    epsilon_one_release: float
    epsilon_50_releases: float
    delta: float
    accounting_method: str
    sensitivity_convention: str = "update_level_add_remove"


class Defense(ABC):
    """Base interface for defenses."""

    name: str

    @abstractmethod
    def encode(self, update: TensorState, context: DefenseContext) -> TransmittedUpdate:
        """Encode a client update into the transmitted representation."""

    @abstractmethod
    def server_view(self, transmitted: TransmittedUpdate, server_knowledge: ServerKnowledge) -> DefenseView:
        """Return exactly what the server can observe."""


@dataclass(frozen=True)
class AttackResult:
    """One attack reconstruction output."""

    estimate: torch.Tensor
    objective_value: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Attack(Protocol):
    """Attack interface."""

    name: str

    def reconstruct(self, server_view: DefenseView, model: torch.nn.Module, knowledge: ServerKnowledge) -> AttackResult:
        """Reconstruct an input estimate from the server view."""


class IdentityDefense(Defense):
    """Undefended baseline defense used by positive controls."""

    name = "none"

    def encode(self, update: TensorState, context: DefenseContext) -> TransmittedUpdate:
        return TransmittedUpdate(
            tensors={name: tensor.detach().clone() for name, tensor in update.items()},
            metadata={"round_id": context.round_id, "client_id": context.client_id},
        )

    def server_view(self, transmitted: TransmittedUpdate, server_knowledge: ServerKnowledge) -> DefenseView:
        return DefenseView(
            tensors=transmitted.tensors,
            metadata=transmitted.metadata,
            server_knowledge=server_knowledge,
            defense_name=self.name,
        )

