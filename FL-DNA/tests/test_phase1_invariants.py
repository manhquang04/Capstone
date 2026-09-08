"""Small implementation invariants for the Phase 1 FL-DNA audit."""

from collections import OrderedDict

import numpy as np
import pytest
import torch

from attacks.inversion_metrics import reconstruction_metrics
from dna_encoder.encoder import DNAEncoder
from dna_encoder.transform_defense import DNATransformConfig, transform_update_array
from privacy.secure_agg import secure_aggregate_states


def test_dna_round_trip_preserves_float32_bits() -> None:
    values = np.array([0.0, -0.0, 1.25, -7.5, np.float32("inf")], dtype=np.float32)
    encoder = DNAEncoder(key=b"1" * 32)
    decoded = encoder.decode_array(encoder.encode_array(values), values.shape)
    assert np.array_equal(values.view(np.uint32), decoded.view(np.uint32))


def test_transform_identity_when_mix_is_zero() -> None:
    values = np.linspace(-2.0, 2.0, 17, dtype=np.float32)
    transformed, stats = transform_update_array(
        values,
        DNATransformConfig(block_size=8, mix_ratio=0.0, keep_ratio=0.5, shrink_factor=0.2, seed=123),
    )
    assert np.array_equal(values, transformed)
    assert stats.relative_l2_delta == 0.0


def test_fedavg_and_secureagg_match_within_float_tolerance() -> None:
    global_state = OrderedDict(weight=torch.zeros(4), counter=torch.tensor(0, dtype=torch.long))
    states = [
        OrderedDict(weight=torch.ones(4), counter=torch.tensor(1, dtype=torch.long)),
        OrderedDict(weight=torch.full((4,), 3.0), counter=torch.tensor(2, dtype=torch.long)),
    ]
    counts = [1, 3]
    expected = torch.full((4,), 2.5)
    direct = states[0]["weight"] * 0.25 + states[1]["weight"] * 0.75
    secure, metadata = secure_aggregate_states(states, global_state, counts, seed=123)
    assert torch.allclose(direct, expected)
    assert torch.allclose(secure["weight"], expected, atol=1e-6)
    assert metadata["server_sees_individual_raw_updates"] is False


def test_reconstruction_metrics_identity_is_well_defined() -> None:
    values = np.array([-1.0, 0.0, 2.0, 4.0], dtype=np.float32)
    metrics = reconstruction_metrics(values, values.copy())
    assert metrics["mse"] == 0.0
    assert metrics["cosine_similarity"] == 1.0
    assert metrics["sign_match_ratio"] == 1.0
    assert metrics["ssim"] > 0.99


def test_reconstruction_metrics_do_not_hide_constant_vector_error() -> None:
    metrics = reconstruction_metrics(np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32))
    assert metrics["feature_mse"] == 1.0
    assert metrics["feature_mae"] == 1.0
    assert metrics["psnr"] < 99.0
    assert metrics["ssim"] < 0.99


def test_reconstruction_metrics_reject_nonfinite_or_wrong_shape() -> None:
    with pytest.raises(ValueError):
        reconstruction_metrics(np.array([0.0, np.nan], dtype=np.float32), np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError):
        reconstruction_metrics(np.zeros(2, dtype=np.float32), np.zeros(3, dtype=np.float32))
