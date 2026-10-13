"""Synthetic source-ID fixtures only."""
import copy
import unittest
from experiments.priority35a_firewall import reserve, validate


class FirewallTests(unittest.TestCase):
    def setUp(self):
        self.parts = [list(range(i * 3000, (i + 1) * 3000)) for i in range(3)]
        self.excluded = list(range(100))

    def test_disjoint_complete_deterministic(self):
        doc = reserve("baf", self.parts, self.excluded, 351000)
        self.assertEqual(doc, reserve("baf", [list(reversed(p)) for p in self.parts], self.excluded, 351000))
        self.assertEqual(len(doc["targets"]), 426)
        self.assertEqual(sum(len(t["source_ids"]) for t in doc["targets"]), 3976)
        self.assertTrue(validate(doc, self.parts, self.excluded))

    def test_overlap_missing_and_drift_fail(self):
        original = reserve("baf", self.parts, self.excluded, 351000)
        altered = copy.deepcopy(original)
        altered["targets"][0]["source_ids"][0] = 0
        with self.assertRaises(ValueError):
            validate(altered, self.parts, self.excluded)
        altered = copy.deepcopy(original)
        altered["targets"].pop()
        with self.assertRaises(ValueError):
            validate(altered, self.parts, self.excluded)
        with self.assertRaises(ValueError):
            validate(original, self.parts, self.excluded + [2999])

    def test_invalid_pool_fail_closed(self):
        with self.assertRaises(ValueError):
            reserve("baf", [[1.5], [], []], [], 1)
        with self.assertRaises(ValueError):
            reserve("baf", [[1], [1], []], [], 1)
        with self.assertRaises(ValueError):
            reserve("baf", [[], [], []], [], 1)


if __name__ == "__main__":
    unittest.main()
