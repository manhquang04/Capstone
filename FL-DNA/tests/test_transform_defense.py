import numpy as np

from dna_encoder.transform_defense import DNATransformConfig, transform_update_array


def _configs():
    return {
        "conservative": DNATransformConfig(
            block_size=7,
            mix_ratio=0.08,
            keep_ratio=0.88,
            shrink_factor=0.45,
            seed=12345,
        ),
        "medium": DNATransformConfig(
            block_size=11,
            mix_ratio=0.10,
            keep_ratio=0.85,
            shrink_factor=0.40,
            seed=23456,
        ),
        "stronger": DNATransformConfig(
            block_size=13,
            mix_ratio=0.12,
            keep_ratio=0.82,
            shrink_factor=0.35,
            seed=34567,
        ),
    }


def _arrays():
    rng = np.random.default_rng(20260921)
    return {
        "vector_37": rng.normal(0, 1, size=37).astype(np.float32),
        "matrix_5x9": rng.normal(0, 0.2, size=(5, 9)).astype(np.float32),
        "zeros_plus": np.linspace(-1.5, 1.5, 64, dtype=np.float32),
    }


def test_transform_update_array_is_exactly_reproducible_for_fixed_configs():
    for config in _configs().values():
        for array in _arrays().values():
            first, first_stats = transform_update_array(array, config, tensor_index=3)
            second, second_stats = transform_update_array(array, config, tensor_index=3)

            np.testing.assert_array_equal(first, second)
            assert first_stats == second_stats


def test_transform_update_array_preserves_shape_and_float32_dtype():
    for config in _configs().values():
        for array in _arrays().values():
            transformed, stats = transform_update_array(array, config, tensor_index=3)

            assert transformed.shape == array.shape
            assert transformed.dtype == np.float32
            assert stats.transformed_elements == array.size
            assert stats.blocks == int(np.ceil(array.size / config.block_size))
