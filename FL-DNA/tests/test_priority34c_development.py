import unittest
from experiments.priority34c_development import calibration


def fixture(n=24):
    methods = ("dna_v1_conservative", "dna_v2_0p95")
    return [dict(config=dict(dataset="paysim", cell="ratio_batch1", index=i),
                 raw_norm=10., measurements={m: dict(dna_l2_distortion=2., unit_gaussian_norm=100.) for m in methods})
            for i in range(n)]


class DevelopmentTests(unittest.TestCase):
    def test_registered_matching(self):
        result = calibration(fixture())["paysim/ratio_batch1"]
        for row in result.values():
            self.assertEqual(row["status"], "MATCHED")
            self.assertEqual(row["clip"], 10.1)
            self.assertAlmostEqual(row["noise_sd"], .02)
            self.assertAlmostEqual(row["median_relative_error"], 0.)

    def test_no_analysis_missing(self):
        with self.assertRaises(ValueError):
            calibration(fixture(23))

    def test_nonmatching_not_retuned(self):
        values = fixture()
        for i, row in enumerate(values):
            for measured in row["measurements"].values():
                measured["dna_l2_distortion"] = 1. if i < 12 else 10.
        results = calibration(values)["paysim/ratio_batch1"]
        self.assertTrue(all(row["status"] == "NOT_ASSESSABLE" for row in results.values()))


if __name__ == "__main__":
    unittest.main()
