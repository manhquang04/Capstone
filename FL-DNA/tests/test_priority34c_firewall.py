"""Source allocation fixtures only; no real IDs or target records."""
import copy
import unittest
from experiments.priority34c_firewall import reserve, validate


class FirewallTests(unittest.TestCase):
    def test_disjoint_and_deterministic(self):
        pool, excluded = list(range(1000)), list(range(100))
        for dataset in ("paysim", "ieee_cis", "baf"):
            doc = reserve(dataset, pool, excluded)
            self.assertEqual(doc, reserve(dataset, list(reversed(pool)), excluded))
            self.assertEqual(doc["total_reserved"], 380)
            self.assertTrue(validate(doc, pool, excluded))

    def test_pool_and_budget_fail_closed(self):
        for pool in ([1, 1], range(379), [1.5]):
            with self.assertRaises(ValueError):
                reserve("paysim", pool, [])
        with self.assertRaises(ValueError):
            reserve("unknown", range(1000), [])

    def test_overlap_and_drift_fail_closed(self):
        doc = reserve("paysim", range(1000), range(100))
        altered = copy.deepcopy(doc)
        altered["source_ids"]["ratio_batch1"]["n39"][0] = altered["source_ids"]["ratio_batch1"]["n8"][0]
        with self.assertRaises(ValueError):
            validate(altered, range(1000), range(100))
        with self.assertRaises(ValueError):
            validate(doc, range(1001), range(100))


if __name__ == "__main__":
    unittest.main()
