"""Gate tests use synthetic score rows, never fraud targets."""
import unittest
from experiments.priority34c_qualify import gates
from experiments.priority34c_core import native
from scipy.stats import binomtest


def rows(values, control=50.):
    return [dict(config=dict(dataset="paysim", cell="ratio_batch1", stage="n8", index=i),
                 accuracy=dict(accuracy_percent=value),
                 baselines=dict(mean_mode=control, empirical_mean=control))
            for i, value in enumerate(values)]


class GateTests(unittest.TestCase):
    def test_pass_both_controls(self):
        gate = gates(rows([100.] * 8))["paysim/ratio_batch1"]
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["tests"]["mean_mode"]["p"], 1/256)

    def test_ties_not_successes(self):
        gate = gates(rows([50.] * 8))["paysim/ratio_batch1"]
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["tests"]["mean_mode"]["ties"], 8)
        self.assertEqual(gate["tests"]["mean_mode"]["p"], 1.)

    def test_must_beat_both(self):
        observations = rows([100.] * 8)
        for row in observations:
            row["baselines"]["empirical_mean"] = 100.
        self.assertFalse(gates(observations)["paysim/ratio_batch1"]["passed"])

    def test_independent_binomial_tail(self):
        for n in (8, 24, 39):
            for wins in range(n+1):
                self.assertAlmostEqual(native.exact_p(wins, n-wins),
                    binomtest(wins, n, .5, alternative="greater").pvalue, places=14)


if __name__ == "__main__":
    unittest.main()
