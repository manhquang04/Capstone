from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.harness import (
    DPComparatorSpec,
    DecoyScoreRecord,
    DefenseEffectRecord,
    DefenseContext,
    HarnessResultValidator,
    IdentityDefense,
    KnowledgeReceiptRecord,
    LosslessSanityRecord,
    ServerKnowledge,
    TransmittedUpdate,
    ValidationError,
    exact_one_sided_sign_p,
    standardized_mse,
)
from experiments.harness.reporting import emit_validated_json_report
from experiments.harness.validators import PositiveControlGate


def _passing_gate() -> PositiveControlGate:
    return PositiveControlGate(
        n8_wins_prior=8,
        n8_p_prior=0.00390625,
        n8_wins_decoy=8,
        n8_p_decoy=0.00390625,
        n24_wins_prior=24,
        n24_p_prior=5.960464477539063e-08,
        n24_wins_decoy=24,
        n24_p_decoy=5.960464477539063e-08,
    )


def _dp_spec() -> DPComparatorSpec:
    return DPComparatorSpec(
        clip_scope="full_state",
        clip_norm=1.0,
        noise_multiplier=0.1,
        epsilon_one_release=10.0,
        epsilon_50_releases=500.0,
        delta=1e-5,
        accounting_method="RDP Gaussian accountant",
    )


def test_identity_defense_roundtrip_and_knowledge_declaration() -> None:
    defense = IdentityDefense()
    update = {"w": torch.tensor([1.0, 2.0])}
    knowledge = ServerKnowledge(metadata=frozenset({"shape"}))
    context = DefenseContext(1, 2, ("w",), knowledge)
    transmitted = defense.encode(update, context)
    view = defense.server_view(transmitted, knowledge)
    assert view.defense_name == "none"
    assert torch.equal(view.tensors["w"], update["w"])


def test_decode_secret_must_be_in_server_knowledge() -> None:
    validator = HarnessResultValidator()
    transmitted = TransmittedUpdate(tensors={}, required_secrets_for_decode=frozenset({"seed"}))
    with pytest.raises(ValidationError, match="missing decode"):
        validator.require_decode_knowledge(transmitted, ServerKnowledge())
    validator.require_decode_knowledge(transmitted, ServerKnowledge(seeds=frozenset({"seed"})))


def test_decoy_must_be_scored_against_current_target() -> None:
    good = DecoyScoreRecord("target-a", "target-b", "target-a", 1.0, 2.0)
    bad = DecoyScoreRecord("target-a", "target-b", "target-b", 1.0, 2.0)
    HarnessResultValidator(decoy_records=[good]).require_decoy_scored_against_current_target()
    with pytest.raises(ValidationError, match="wrong target"):
        HarnessResultValidator(decoy_records=[bad]).require_decoy_scored_against_current_target()


def test_report_refuses_failed_positive_control(tmp_path: Path) -> None:
    validator = HarnessResultValidator(
        positive_controls={
            "attack": PositiveControlGate(
                n8_wins_prior=4,
                n8_p_prior=0.63671875,
                n8_wins_decoy=8,
                n8_p_decoy=0.00390625,
                n24_wins_prior=24,
                n24_p_prior=5.960464477539063e-08,
                n24_wins_decoy=24,
                n24_p_decoy=5.960464477539063e-08,
            )
        },
        scoring_space="tabular_standardized_mse",
        dp_spec=_dp_spec(),
        decoy_records=[DecoyScoreRecord("a", "b", "a", 1.0, 2.0)],
    )
    with pytest.raises(ValidationError, match="positive-control"):
        emit_validated_json_report(
            tmp_path / "report.json",
            {"ok": True},
            validator=validator,
            attack_name="attack",
            domain="tabular",
            transmitted=TransmittedUpdate(tensors={}),
            server_knowledge=ServerKnowledge(),
        )


def test_report_emits_when_all_mandatory_checks_pass(tmp_path: Path) -> None:
    validator = HarnessResultValidator(
        positive_controls={"attack": _passing_gate()},
        scoring_space="tabular_standardized_mse",
        dp_spec=_dp_spec(),
        decoy_records=[DecoyScoreRecord("a", "b", "a", 1.0, 2.0)],
    )
    out = tmp_path / "report.json"
    emit_validated_json_report(
        out,
        {"ok": True},
        validator=validator,
        attack_name="attack",
        domain="tabular",
        transmitted=TransmittedUpdate(tensors={}, required_secrets_for_decode=frozenset({"seed"})),
        server_knowledge=ServerKnowledge(seeds=frozenset({"seed"})),
    )
    assert out.read_text().strip().endswith("}")


def test_priority28_report_gate_requires_data_free_positive_control() -> None:
    validator = HarnessResultValidator(
        positive_controls={"attack": _passing_gate()},
        scoring_space="tabular_standardized_mse",
        dp_spec=_dp_spec(),
        decoy_records=[DecoyScoreRecord("a", "b", "a", 1.0, 2.0)],
        require_data_free_baseline=True,
    )
    with pytest.raises(ValidationError, match="data-free"):
        validator.require_positive_control("attack")


def test_priority28_strict_gate_requires_lossless_defense_effect_and_knowledge() -> None:
    gate = PositiveControlGate(
        n8_wins_prior=8,
        n8_p_prior=0.00390625,
        n8_wins_decoy=8,
        n8_p_decoy=0.00390625,
        n24_wins_prior=24,
        n24_p_prior=5.960464477539063e-08,
        n24_wins_decoy=24,
        n24_p_decoy=5.960464477539063e-08,
        n8_wins_data_free=8,
        n8_p_data_free=0.00390625,
        n24_wins_data_free=24,
        n24_p_data_free=5.960464477539063e-08,
    )
    transmitted = TransmittedUpdate(tensors={}, required_secrets_for_decode=frozenset({"seed"}))
    common = dict(
        positive_controls={"attack": gate},
        scoring_space="tabular_standardized_mse",
        dp_spec=_dp_spec(),
        decoy_records=[DecoyScoreRecord("a", "b", "a", 1.0, 2.0)],
        require_data_free_baseline=True,
    )

    missing = HarnessResultValidator(**common)
    with pytest.raises(ValidationError, match="lossless sanity"):
        missing.validate_priority28_for_report(
            "attack",
            "defense",
            "tabular",
            transmitted,
            ServerKnowledge(seeds=frozenset({"seed"})),
        )

    identity = HarnessResultValidator(
        **common,
        lossless_sanity={"attack": LosslessSanityRecord("attack", max_abs_error=0.0)},
        defense_effects={"defense": DefenseEffectRecord("defense", changed_update=False, changed_forward=False)},
        knowledge_receipts={"attack": KnowledgeReceiptRecord("attack", frozenset({"seed"}), frozenset({"seed"}))},
    )
    with pytest.raises(ValidationError, match="identity"):
        identity.validate_priority28_for_report(
            "attack",
            "defense",
            "tabular",
            transmitted,
            ServerKnowledge(seeds=frozenset({"seed"})),
        )

    no_knowledge = HarnessResultValidator(
        **common,
        lossless_sanity={"attack": LosslessSanityRecord("attack", max_abs_error=0.0)},
        defense_effects={"defense": DefenseEffectRecord("defense", changed_update=True, effect_size=1.0)},
        knowledge_receipts={"attack": KnowledgeReceiptRecord("attack", frozenset({"seed"}), frozenset())},
    )
    with pytest.raises(ValidationError, match="did not receive"):
        no_knowledge.validate_priority28_for_report(
            "attack",
            "defense",
            "tabular",
            transmitted,
            ServerKnowledge(seeds=frozenset({"seed"})),
        )

    valid = HarnessResultValidator(
        **common,
        lossless_sanity={"attack": LosslessSanityRecord("attack", max_abs_error=0.0)},
        defense_effects={"defense": DefenseEffectRecord("defense", changed_update=True, effect_size=1.0)},
        knowledge_receipts={"attack": KnowledgeReceiptRecord("attack", frozenset({"seed"}), frozenset({"seed"}))},
    )
    valid.validate_priority28_for_report(
        "attack",
        "defense",
        "tabular",
        transmitted,
        ServerKnowledge(seeds=frozenset({"seed"})),
    )


def test_metrics_helpers() -> None:
    assert exact_one_sided_sign_p(8, 8) == pytest.approx(0.00390625)
    mse = standardized_mse(torch.tensor([2.0, 4.0]), torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
    assert mse == pytest.approx(1.0)
