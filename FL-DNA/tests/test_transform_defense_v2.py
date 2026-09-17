import numpy as np

from dna_encoder.transform_defense_v2 import (
    DNATransformV2Config,
    materialize_projection_matrix_v2,
    reconstruct_update_array_v2,
    transform_and_reconstruct_array_v2,
    transform_update_array_v2,
)


def test_v2_is_deterministic_for_same_seed() -> None:
    array = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    config = DNATransformV2Config(compression_ratio=0.75, quantization_eta=0.01, seed=123)

    sketch_a, meta_a = transform_update_array_v2(array, config, tensor_index=2, quantization_seed=7)
    sketch_b, meta_b = transform_update_array_v2(array, config, tensor_index=2, quantization_seed=7)

    np.testing.assert_array_equal(sketch_a, sketch_b)
    assert meta_a == meta_b


def test_v2_shape_and_lossy_rank() -> None:
    config = DNATransformV2Config(compression_ratio=0.5, quantization_eta=0.0, seed=5)
    matrix, metadata = materialize_projection_matrix_v2(8, config)

    assert matrix.shape == (4, 8)
    assert metadata.sketch_size == 4
    assert np.linalg.matrix_rank(matrix) <= 4
    assert np.linalg.matrix_rank(matrix) < 8


def test_v2_reconstructs_to_original_shape_and_has_nonzero_loss() -> None:
    array = np.arange(12, dtype=np.float32).reshape(3, 4) / 10.0
    config = DNATransformV2Config(compression_ratio=0.5, quantization_eta=0.0, seed=99)

    sketch, reconstructed, metadata, stats = transform_and_reconstruct_array_v2(array, config)

    assert sketch.shape == (metadata.sketch_size,)
    assert reconstructed.shape == array.shape
    assert stats.sketch_rank_upper_bound < metadata.padded_size
    assert stats.relative_l2_error > 1e-3


def test_v2_unquantized_projection_is_unbiased_empirically() -> None:
    # Average many independent sketches; the lifted estimator should approach u.
    array = np.linspace(-0.75, 0.9, 8, dtype=np.float32)
    reconstructions = []
    for seed in range(400):
        config = DNATransformV2Config(compression_ratio=0.5, quantization_eta=0.0, seed=seed)
        sketch, metadata = transform_update_array_v2(array, config)
        reconstruction, _ = reconstruct_update_array_v2(sketch, metadata)
        reconstructions.append(reconstruction)

    mean_reconstruction = np.mean(np.stack(reconstructions), axis=0)
    relative_error = np.linalg.norm(mean_reconstruction - array) / np.linalg.norm(array)
    assert relative_error < 0.12


def test_v2_stochastic_quantization_is_unbiased_empirically() -> None:
    array = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    config = DNATransformV2Config(compression_ratio=1.0, quantization_eta=0.25, seed=321)
    unquantized_config = DNATransformV2Config(compression_ratio=1.0, quantization_eta=0.0, seed=321)
    unquantized, _ = transform_update_array_v2(array, unquantized_config)

    quantized = []
    for q_seed in range(1000):
        sketch, _ = transform_update_array_v2(array, config, quantization_seed=q_seed)
        quantized.append(sketch)

    mean_quantized = np.mean(np.stack(quantized), axis=0)
    relative_error = np.linalg.norm(mean_quantized - unquantized) / np.linalg.norm(unquantized)
    assert relative_error < 0.03


def test_v2_rejects_invalid_config() -> None:
    array = np.ones(4, dtype=np.float32)
    bad = DNATransformV2Config(compression_ratio=0.0, quantization_eta=0.01, seed=1)

    try:
        transform_update_array_v2(array, bad)
    except ValueError as exc:
        assert "compression_ratio" in str(exc)
    else:
        raise AssertionError("invalid compression_ratio should fail")
