"""Pure synthetic final reporting checks; no scientific measurement."""
import unittest
import numpy as np
from experiments.priority35a_report import interpretation, holm_numpy, exact_summary


class ReportTests(unittest.TestCase):
    def test_fixed_interpretation(self):
        self.assertIn("nothing about privacy", interpretation(False, False))
        self.assertIn("nothing about privacy", interpretation(False, True))
        self.assertIn("ADAM_STEP", interpretation(True, False))
        self.assertIn("UNKNOWN_LOCAL_BN", interpretation(True, True))

    def test_holm_full_family_and_general_algorithm(self):
        self.assertTrue(np.array_equal(holm_numpy(np.ones(6048)), np.ones(6048)))
        self.assertTrue(np.allclose(holm_numpy([.02, .001, .5]), [.04, .003, .5]))
        with self.assertRaises(ValueError):
            holm_numpy([float("nan")])

    def test_exact_ties_and_missing_fail_closed(self):
        result = exact_summary([1., 1., 1., 1., 1., 1., 1., 0.])
        self.assertEqual((result["wins"], result["losses"], result["ties"]), (7, 0, 1))
        self.assertEqual(result["p_one_sided"], 1/128)
        with self.assertRaises(ValueError):
            exact_summary([1.]*7)
        with self.assertRaises(ValueError):
            exact_summary([float("inf")]*8)


if __name__ == "__main__":
    unittest.main()
