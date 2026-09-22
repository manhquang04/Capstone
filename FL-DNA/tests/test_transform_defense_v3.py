import numpy as np

from dna_encoder.transform_defense_v3 import (
    DNATransformV3Config,
    V3_SENSITIVE_PARAMETER_NAMES,
    reconstruct_update_state_v3,
    split_update_blocks_v3,
    transform_update_state_v3,
)


def _dummy_update() -> dict[str, np.ndarray]:
    return {
        "network.0.weight": np.arange(20, dtype=np.float32).reshape(4, 5),
        "network.0.bias": np.arange(4, dtype=np.float32),
        "network.9.weight": np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(1, 32),
        "network.9.bias": np.asarray([0.25], dtype=np.float32),
    }


def test_v3_splits_final_classifier_as_33_parameter_sensitive_block() -> None:
    blocks, specs = split_update_blocks_v3(_dummy_update())

    assert blocks["sensitive"].shape == (33,)
    assert blocks["rest"].shape == (24,)
    assert {spec.name for spec in specs if spec.block == "sensitive"} == set(
        V3_SENSITIVE_PARAMETER_NAMES
    )


def test_v3_is_deterministic_and_uses_block_specific_ratios() -> None:
    config = DNATransformV3Config(
        rest_compression_ratio=0.95,
        sensitive_compression_ratio=0.50,
        quantization_eta=0.01,
        seed=20260916,
    )
    update = _dummy_update()

    sketch_a, meta_a = transform_update_state_v3(update, config, quantization_seed=7)
    sketch_b, meta_b = transform_update_state_v3(update, config, quantization_seed=7)

    np.testing.assert_array_equal(sketch_a["rest"], sketch_b["rest"])
    np.testing.assert_array_equal(sketch_a["sensitive"], sketch_b["sensitive"])
    by_block = {item.block: item for item in meta_a.block_metadata}
    assert by_block["rest"].metadata.compression_ratio == 0.95
    assert by_block["sensitive"].metadata.compression_ratio == 0.50
    assert meta_a == meta_b


def test_v3_reconstructs_original_tensor_shapes() -> None:
    config = DNATransformV3Config(
        rest_compression_ratio=0.95,
        sensitive_compression_ratio=0.50,
        quantization_eta=0.01,
        seed=20260916,
    )
    update = _dummy_update()
    sketches, metadata = transform_update_state_v3(update, config, quantization_seed=11)

    reconstructed, stats = reconstruct_update_state_v3(sketches, metadata)

    assert set(reconstructed) == set(update)
    for key, value in update.items():
        assert reconstructed[key].shape == value.shape
    assert stats["sensitive"].sketch_rank_upper_bound < stats["sensitive"].padded_size
