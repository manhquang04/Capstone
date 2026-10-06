import unittest
from unittest.mock import patch
from experiments import analyze_priority34c as a


class AnalysisTests(unittest.TestCase):
    def test_independent_holm(self):
        values = [0., 1e-5, .03]+[1.]*69
        self.assertEqual(a.independent_holm(values), a.stats.fixed_holm(values))
        with self.assertRaises(ValueError):
            a.independent_holm(values[:-1])

    def test_independent_sign(self):
        for n in (8, 24, 39):
            for wins in range(n+1):
                self.assertAlmostEqual(a.independent_p(wins, n-wins), a.stats.exact_tail(wins, n-wins), places=14)

    def test_local_accounting(self):
        self.assertAlmostEqual(a.independent_accounting(a.c.LOCAL_SD)["epsilon"], 10., places=10)

    def test_no_audit_while_workers_active(self):
        with patch.object(a.q, "live_processes", return_value=["worker"]):
            with self.assertRaisesRegex(RuntimeError, "live"):
                a.run()


if __name__ == "__main__":
    unittest.main()
