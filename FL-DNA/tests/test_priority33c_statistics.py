import unittest
from experiments.analyze_priority33c import sign, holm


class StatisticsTests(unittest.TestCase):
    def test_exact_sign_and_ties(self):
        self.assertEqual(sign(0, 0), 1.)
        self.assertEqual(sign(39, 0), 2.**-39)
        self.assertAlmostEqual(sign(30, 9), .00053250981727615)

    def test_fixed_family_includes_unassessable(self):
        rows = [dict(p_raw=.001), dict(p_raw=.01)] + [dict(p_raw=1.) for _ in range(18)]
        holm(rows, 'adjusted')
        self.assertEqual(rows[0]['adjusted'], .02)
        self.assertEqual(rows[1]['adjusted'], .19)
        self.assertTrue(all(r['adjusted'] == 1. for r in rows[2:]))


if __name__ == '__main__':
    unittest.main()
