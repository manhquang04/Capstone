"""Synthetic-only tests: no datasets, source IDs or private noise keys."""
import unittest
import numpy as np
from dna_encoder.transform_defense_v2 import (
    DNATransformV2Config, materialize_projection_matrix_v2, transform_update_array_v2)
from experiments.priority34c_solvers import projection_operator, known_key_least_squares, ratio_record


class SolverTests(unittest.TestCase):
    def test_operator_and_adjoint(self):
        for d, ratio in ((13, .95), (17, .4), (32, .95), (58, .95)):
            matrix, meta = materialize_projection_matrix_v2(d, DNATransformV2Config(ratio, .01, 340042))
            op = projection_operator(meta)
            x = np.linspace(-1., 1., d)
            q = np.linspace(.2, 1., meta.sketch_size)
            np.testing.assert_allclose(op @ x, matrix @ x, atol=1e-12)
            np.testing.assert_allclose(op.rmatvec(q), matrix.T @ q, atol=1e-12)

    def test_least_squares_quantized_and_unquantized(self):
        for d, ratio in ((13, .95), (17, .4), (32, .95), (58, .95)):
            config = DNATransformV2Config(ratio, .01, 340042)
            matrix, meta = materialize_projection_matrix_v2(d, config)
            x = np.linspace(-1., 1., d).astype(np.float32)
            quantized, quant_meta = transform_update_array_v2(x, config, quantization_seed=340043)
            for q, metadata in ((matrix @ x, meta), (quantized, quant_meta)):
                solved, receipt = known_key_least_squares(q, metadata)
                expected = np.linalg.lstsq(matrix, q, rcond=None)[0]
                np.testing.assert_allclose(solved, expected, atol=1e-8, rtol=1e-8)
                self.assertIn(receipt["stop_code"], (0, 1, 2, 4, 5))

    def test_ratio_identity_and_tie(self):
        x = np.array([.25, -2., 4.])
        bias = np.array([2., -2., 1.])
        recovered, receipt = ratio_record(bias[:, None] * x, bias)
        np.testing.assert_array_equal(recovered, x)
        self.assertEqual(receipt["selected_row"], 0)

    def test_fail_closed(self):
        with self.assertRaises(ArithmeticError):
            ratio_record(np.zeros((2, 3)), np.zeros(2))
        with self.assertRaises(FloatingPointError):
            ratio_record(np.full((2, 3), np.nan), np.ones(2))
        _, meta = materialize_projection_matrix_v2(13, DNATransformV2Config(seed=340042))
        with self.assertRaises(FloatingPointError):
            known_key_least_squares(np.full(meta.sketch_size, np.nan), meta)


if __name__ == "__main__":
    unittest.main()
