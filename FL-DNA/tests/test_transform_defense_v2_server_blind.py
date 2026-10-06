import inspect

import numpy as np
import pytest

from dna_encoder.transform_defense_v2_server_blind import (
    ServerBlindAggregator,
    ServerBlindV2Config,
    ServerBlindV2Sketch,
    generate_round_key,
    lift_raw_sketch_v2sb,
    lift_sketch_array_v2sb,
    materialize_projection_matrix_v2sb,
    sketch_update_array_v2sb,
    validate_round_key,
)
import dna_encoder.transform_defense_v2_server_blind as v2sb


KEY = bytes(range(32))


def test_key_is_256_bits_and_validated():
    key = generate_round_key()
    assert isinstance(key, bytes)
    assert len(key) == 32
    validate_round_key(KEY)
    with pytest.raises(ValueError):
        validate_round_key(b"short")


def test_no_modulo_seed_truncation_in_v2sb_path():
    source = inspect.getsource(v2sb)
    assert "% (2**32)" not in source
    assert "% (2**31)" not in source
    assert "0xFFFFFFFF" not in source


def test_server_aggregator_never_touches_key():
    agg = ServerBlindAggregator()
    assert getattr(agg, "key") is None
    assert getattr(agg, "round_key") is None
    cfg = ServerBlindV2Config(compression_ratio=0.95, quantization_eta=0.0)
    a = sketch_update_array_v2sb(np.arange(8, dtype=np.float32), cfg, round_key=KEY, round_number=1, tensor_index=0, client_id=0)
    b = sketch_update_array_v2sb(np.arange(8, dtype=np.float32) + 1, cfg, round_key=KEY, round_number=1, tensor_index=0, client_id=1)
    out = agg.aggregate([a, b], [0.25, 0.75])
    assert isinstance(out, ServerBlindV2Sketch)
    assert getattr(agg, "key") is None
    assert getattr(agg, "round_key") is None


def test_linearity_of_lift_over_aggregated_sketches_without_quantization():
    cfg = ServerBlindV2Config(compression_ratio=0.95, quantization_eta=0.0)
    x = np.linspace(-1, 1, 17, dtype=np.float32)
    y = np.linspace(0.5, -0.25, 17, dtype=np.float32)
    sx = sketch_update_array_v2sb(x, cfg, round_key=KEY, round_number=3, tensor_index=2, client_id=0)
    sy = sketch_update_array_v2sb(y, cfg, round_key=KEY, round_number=3, tensor_index=2, client_id=1)
    agg = ServerBlindAggregator().aggregate([sx, sy], [0.4, 0.6])
    lifted_agg = lift_sketch_array_v2sb(agg, round_key=KEY)
    lifted_separate = 0.4 * lift_sketch_array_v2sb(sx, round_key=KEY) + 0.6 * lift_sketch_array_v2sb(sy, round_key=KEY)
    np.testing.assert_allclose(lifted_agg, lifted_separate, rtol=1e-6, atol=1e-6)


def test_server_aggregate_allows_client_specific_quantization_delta():
    cfg = ServerBlindV2Config(compression_ratio=0.95, quantization_eta=0.01)
    small = np.linspace(-0.01, 0.01, 16, dtype=np.float32)
    large = np.linspace(-10.0, 10.0, 16, dtype=np.float32)
    sx = sketch_update_array_v2sb(small, cfg, round_key=KEY, round_number=6, tensor_index=3, client_id=0)
    sy = sketch_update_array_v2sb(large, cfg, round_key=KEY, round_number=6, tensor_index=3, client_id=1)
    assert sx.metadata.quantization_delta != sy.metadata.quantization_delta
    out = ServerBlindAggregator().aggregate([sx, sy], [0.5, 0.5])
    assert out.metadata.original_size == sx.metadata.original_size


def test_one_client_lift_matches_materialized_projection_pseudoinverse_style_lift():
    cfg = ServerBlindV2Config(compression_ratio=0.95, quantization_eta=0.0)
    x = np.linspace(-2, 2, 13, dtype=np.float32)
    sketch = sketch_update_array_v2sb(x, cfg, round_key=KEY, round_number=4, tensor_index=5, client_id=7)
    matrix = materialize_projection_matrix_v2sb(13, cfg, round_key=KEY, round_number=4, tensor_index=5)
    expected_q = matrix @ x.astype(np.float64)
    np.testing.assert_allclose(sketch.sketch, expected_q, rtol=1e-6, atol=1e-6)
    lifted = lift_sketch_array_v2sb(sketch, round_key=KEY)
    lifted_raw = lift_raw_sketch_v2sb(
        expected_q,
        original_shape=x.shape,
        config=cfg,
        round_key=KEY,
        round_number=4,
        tensor_index=5,
    )
    np.testing.assert_allclose(lifted, lifted_raw, rtol=1e-6, atol=1e-6)
