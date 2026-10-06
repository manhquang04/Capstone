"""Reusable defense-agnostic evaluation harness for FL-DNA experiments."""

from .interfaces import (
    Attack,
    AttackResult,
    Defense,
    DefenseContext,
    DefenseView,
    DPComparatorSpec,
    IdentityDefense,
    ServerKnowledge,
    TransmittedUpdate,
)
from .metrics import (
    DecoyScoreRecord,
    ScoreRecord,
    exact_one_sided_sign_p,
    image_metrics,
    standardized_mse,
)
from .validators import (
    DefenseEffectRecord,
    HarnessResultValidator,
    KnowledgeReceiptRecord,
    LosslessSanityRecord,
    ValidationError,
)

__all__ = [
    "Attack",
    "AttackResult",
    "Defense",
    "DefenseContext",
    "DefenseView",
    "DPComparatorSpec",
    "IdentityDefense",
    "ServerKnowledge",
    "TransmittedUpdate",
    "DecoyScoreRecord",
    "ScoreRecord",
    "exact_one_sided_sign_p",
    "image_metrics",
    "standardized_mse",
    "HarnessResultValidator",
    "LosslessSanityRecord",
    "DefenseEffectRecord",
    "KnowledgeReceiptRecord",
    "ValidationError",
]
