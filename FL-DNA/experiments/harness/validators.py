"""Validation and report gating for Priority 27 harness outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .interfaces import DPComparatorSpec, ServerKnowledge, TransmittedUpdate
from .metrics import DecoyScoreRecord


class ValidationError(RuntimeError):
    """Raised when a harness result is not reportable."""


@dataclass
class PositiveControlGate:
    n8_wins_prior: int
    n8_p_prior: float
    n8_wins_decoy: int
    n8_p_decoy: float
    n24_wins_prior: int
    n24_p_prior: float
    n24_wins_decoy: int
    n24_p_decoy: float
    n8_wins_data_free: int | None = None
    n8_p_data_free: float | None = None
    n24_wins_data_free: int | None = None
    n24_p_data_free: float | None = None
    alpha: float = 0.05

    def passed(self) -> bool:
        base = (
            self.n8_p_prior < self.alpha
            and self.n8_p_decoy < self.alpha
            and self.n24_p_prior < self.alpha
            and self.n24_p_decoy < self.alpha
        )
        if not base:
            return False
        data_free_values = (
            self.n8_p_data_free,
            self.n24_p_data_free,
        )
        if any(value is not None for value in data_free_values):
            if self.n8_p_data_free is None or self.n24_p_data_free is None:
                return False
            return self.n8_p_data_free < self.alpha and self.n24_p_data_free < self.alpha
        return True


@dataclass(frozen=True)
class LosslessSanityRecord:
    """Identity/lossless recovery check for one adaptive attacker."""

    attack_name: str
    max_abs_error: float | None = None
    mse: float | None = None
    tolerance: float = 1e-5
    metric: str = "max_abs_error"

    def passed(self) -> bool:
        value = self.mse if self.metric == "mse" else self.max_abs_error
        return value is not None and value <= self.tolerance


@dataclass(frozen=True)
class DefenseEffectRecord:
    """Defense-not-identity check for one published defense setting."""

    defense_name: str
    changed_update: bool = False
    changed_forward: bool = False
    effect_size: float = 0.0
    tolerance: float = 1e-12

    def passed(self) -> bool:
        return (self.changed_update or self.changed_forward) and self.effect_size > self.tolerance


@dataclass(frozen=True)
class KnowledgeReceiptRecord:
    """Checks that the adaptive attacker received the declared server knowledge."""

    attack_name: str
    required_items: frozenset[str] = frozenset()
    received_items: frozenset[str] = frozenset()

    def missing(self) -> list[str]:
        return sorted(item for item in self.required_items if item not in self.received_items)


@dataclass
class HarnessResultValidator:
    """Checks that must pass before a result can be emitted."""

    positive_controls: Mapping[str, PositiveControlGate] = field(default_factory=dict)
    lossless_sanity: Mapping[str, LosslessSanityRecord] = field(default_factory=dict)
    defense_effects: Mapping[str, DefenseEffectRecord] = field(default_factory=dict)
    knowledge_receipts: Mapping[str, KnowledgeReceiptRecord] = field(default_factory=dict)
    scoring_space: str = ""
    dp_spec: DPComparatorSpec | None = None
    decoy_records: list[DecoyScoreRecord] = field(default_factory=list)
    require_data_free_baseline: bool = False

    def require_positive_control(self, attack_name: str) -> None:
        gate = self.positive_controls.get(attack_name)
        if gate is None:
            raise ValidationError(f"missing positive-control gate for {attack_name}")
        if self.require_data_free_baseline and (
            gate.n8_p_data_free is None or gate.n24_p_data_free is None
        ):
            raise ValidationError(f"missing data-free positive-control gate for {attack_name}")
        if not gate.passed():
            raise ValidationError(f"positive-control gate failed for {attack_name}: {gate}")

    def require_lossless_sanity(self, attack_name: str) -> None:
        record = self.lossless_sanity.get(attack_name)
        if record is None:
            raise ValidationError(f"missing lossless sanity check for {attack_name}")
        if not record.passed():
            raise ValidationError(f"lossless sanity failed for {attack_name}: {record}")

    def require_defense_not_identity(self, defense_name: str) -> None:
        record = self.defense_effects.get(defense_name)
        if record is None:
            raise ValidationError(f"missing defense-not-identity check for {defense_name}")
        if not record.passed():
            raise ValidationError(f"defense appears identity for {defense_name}: {record}")

    def require_attack_received_knowledge(self, attack_name: str) -> None:
        record = self.knowledge_receipts.get(attack_name)
        if record is None:
            raise ValidationError(f"missing attacker knowledge receipt for {attack_name}")
        missing = record.missing()
        if missing:
            raise ValidationError(f"attacker did not receive required knowledge for {attack_name}: {missing}")

    def require_input_space_scoring(self, domain: str) -> None:
        expected = "tabular_standardized_mse" if domain == "tabular" else "image_mse_psnr_ssim"
        if self.scoring_space != expected:
            raise ValidationError(f"invalid scoring space {self.scoring_space!r}; expected {expected!r}")

    def require_decode_knowledge(self, transmitted: TransmittedUpdate, server_knowledge: ServerKnowledge) -> None:
        missing = sorted(secret for secret in transmitted.required_secrets_for_decode if not server_knowledge.has(secret))
        if missing:
            raise ValidationError(f"server knowledge missing decode secret(s): {missing}")

    def require_dp_spec(self) -> None:
        if self.dp_spec is None:
            raise ValidationError("missing DP comparator specification")
        if self.dp_spec.clip_scope not in {"full_state", "per_tensor", "trainable_only", "bn_statistics"}:
            raise ValidationError(f"unexpected DP clip scope: {self.dp_spec.clip_scope}")
        if self.dp_spec.noise_multiplier < 0:
            raise ValidationError("negative DP noise multiplier")
        if self.dp_spec.delta <= 0:
            raise ValidationError("DP delta must be positive")
        if not self.dp_spec.accounting_method:
            raise ValidationError("missing DP accounting method")

    def require_decoy_scored_against_current_target(self) -> None:
        bad = [
            record
            for record in self.decoy_records
            if record.scored_against_target_id != record.target_id
        ]
        if bad:
            raise ValidationError(f"decoy scored against wrong target for {len(bad)} record(s)")

    def validate_for_report(self, attack_name: str, domain: str, transmitted: TransmittedUpdate, server_knowledge: ServerKnowledge) -> None:
        self.require_positive_control(attack_name)
        self.require_input_space_scoring(domain)
        self.require_decode_knowledge(transmitted, server_knowledge)
        self.require_dp_spec()
        self.require_decoy_scored_against_current_target()

    def validate_priority28_for_report(
        self,
        attack_name: str,
        defense_name: str,
        domain: str,
        transmitted: TransmittedUpdate,
        server_knowledge: ServerKnowledge,
    ) -> None:
        """Strict Priority 28 report gate with all four-check prerequisites."""

        self.validate_for_report(attack_name, domain, transmitted, server_knowledge)
        self.require_lossless_sanity(attack_name)
        self.require_defense_not_identity(defense_name)
        self.require_attack_received_knowledge(attack_name)
