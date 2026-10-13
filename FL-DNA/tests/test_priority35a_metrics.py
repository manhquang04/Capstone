"""Synthetic comparison against the read-only official TabLeak metric."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
import numpy as np
from experiments.priority35a_metrics import (
    aligned_feature_accuracy, tolerance_map, standardized_mse, bounded_score,
)


def reference():
    # Isolated package avoids importing official models/datasets or modifying
    # an earlier experiment's patched utility namespace.
    name = "priority35a_readonly_tableak_reference"
    folder = Path(__file__).resolve().parents[1] / "external_defenses/tableak/utils"
    package = types.ModuleType(name)
    package.__path__ = [str(folder)]
    sys.modules[name] = package
    for child in ("eval_metrics", "matching"):
        spec = importlib.util.spec_from_file_location(name + "." + child, folder / (child + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[name + ".matching"].match_reconstruction_ground_truth


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = staticmethod(reference())

    def check_native(self, truth, guess, tol):
        actual = aligned_feature_accuracy(truth, guess, tol)
        aligned, errors, _, _, rows, cols = self.native(truth, guess, tol, return_indices=True)
        self.assertEqual(actual["feature_accuracy"], 1-float(errors.mean()))
        np.testing.assert_array_equal(actual["aligned_guess"], aligned)
        np.testing.assert_array_equal(actual["per_row_error"], errors)
        np.testing.assert_array_equal(actual["guess_indices"], cols)

    def test_boundary_categories_and_batch_permutation(self):
        truth = np.array([[0., "A"], [2., "B"], [-2., "A"]], dtype=object)
        guess = np.array([[-2.5, "A"], [0.319, "A"], [2.1, "A"]], dtype=object)
        self.check_native(truth, guess, tolerance_map([1., 0.], [False, True]))
        actual = aligned_feature_accuracy([[0., "A"]], [[0.319, "A"]], [0.319, "cat"])
        self.assertEqual(actual["feature_accuracy"], 1.)
        actual = aligned_feature_accuracy([[0., "A"]], [[np.nextafter(0.319, np.inf), "A"]], [0.319, "cat"])
        self.assertEqual(actual["feature_accuracy"], 0.5)

    def test_absent_feature_type_matches_reference(self):
        self.check_native(np.array([[0., 2.], [3., 1.]], dtype=object),
                          np.array([[3., 2.], [0., 4.]], dtype=object), [0., 0.])
        self.check_native(np.array([["A"], ["B"]], dtype=object),
                          np.array([["C"], ["A"]], dtype=object), ["cat"])

    def test_zero_sd_and_mse_bounding(self):
        self.assertEqual(tolerance_map([0.], [False]), [0.])
        self.assertEqual(standardized_mse([[1., 1.]], [[3., 4.]], [2., 0.]), 5.)
        self.assertEqual(bounded_score(5., 2.), 2.)
        self.assertEqual(bounded_score(1., 2.), 1.)

    def test_nonfinite_and_missing_fail_closed(self):
        with self.assertRaises(ValueError):
            aligned_feature_accuracy([[float("nan")]], [[0.]], [0.319])
        with self.assertRaises(ValueError):
            aligned_feature_accuracy([], [], [])
        with self.assertRaises(ValueError):
            standardized_mse([[0.]], [[0.]], [-1.])
        with self.assertRaises(ValueError):
            bounded_score(float("inf"), 0.)


if __name__ == "__main__":
    unittest.main()
