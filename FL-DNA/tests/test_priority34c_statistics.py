import unittest
import numpy as np
from scipy.stats import binomtest
from experiments.priority34c_statistics import family, exact_tail, paired_summary, fixed_holm, evaluate_family


class StatisticsTests(unittest.TestCase):
    def test_integer_tails(self):
        for n in (8, 24, 39):
            for wins in range(n+1):
                self.assertAlmostEqual(exact_tail(wins, n-wins),
                    binomtest(wins, n, .5, alternative="greater").pvalue, places=14)

    def test_fixed72_reservations(self):
        self.assertEqual(len(family()), 72)
        reservations = {key[:4]: "gate failed" for key in family()}
        rows = evaluate_family({}, reservations)
        self.assertEqual(len(rows), 72)
        self.assertTrue(all(row["raw_p"] == row["holm_p"] == 1 for row in rows))
        with self.assertRaises(ValueError):
            fixed_holm([1.] * 71)

    def test_ranks_and_missing_fail_closed(self):
        summary = paired_summary(np.arange(39), np.zeros(39))
        self.assertEqual(summary["paired_effect_order_interval"], [12., 26.])
        self.assertEqual(summary["ties"], 1)
        with self.assertRaises(ValueError):
            paired_summary(np.zeros(38), np.zeros(39))
        with self.assertRaises(ValueError):
            paired_summary(np.full(39, np.nan), np.zeros(39))

    def test_holm_separate_reference(self):
        raw = np.linspace(0., 1., 72).tolist()
        order = np.argsort(raw, kind="stable")
        scaled = np.minimum(1., np.asarray(raw)[order] * np.arange(72, 0, -1))
        expected = np.empty(72)
        expected[order] = np.maximum.accumulate(scaled)
        np.testing.assert_array_equal(fixed_holm(raw), expected)


if __name__ == "__main__":
    unittest.main()
