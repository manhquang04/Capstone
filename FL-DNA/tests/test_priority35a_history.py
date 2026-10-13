"""Read-only real-manifest integration test; allocations are synthetic IDs."""
import unittest
from experiments.priority35a_history import audit, classify, extract_ids
from experiments.priority35a_firewall import reserve


class HistoricalFirewallTests(unittest.TestCase):
    def test_recursive_ids_and_ambiguous_names_fail_closed(self):
        self.assertEqual(extract_ids({"groups": [{"source_ids": [[4, 5], [6]]}],
                                      "indices": [999], "seed": 12}), [4, 5, 6])
        with self.assertRaises(ValueError):
            extract_ids({"source_ids": [True]})
        with self.assertRaises(ValueError):
            classify("artifacts/priority27/targets.json", {"source_ids": [1]})

    def test_every_historical_target_candidate_is_inventoried_and_excluded(self):
        doc = audit()
        self.assertEqual(len(doc["inventory"]), doc["inventory_count"])
        self.assertEqual(len({r["path"] for r in doc["inventory"]}), doc["inventory_count"])
        self.assertEqual(doc["new_real_target_allocations"], 0)
        for dataset, historical in doc["exclusions"].items():
            self.assertGreater(len(historical), 0)
            # Synthetic namespaces above every historical ID. Insert one actual
            # old ID in the mock pool to prove it cannot be selected. No real
            # prepared-data pool or record is read or allocated here.
            start = max(historical) + 1
            parts = [list(range(start + i*3000, start + (i+1)*3000)) for i in range(3)]
            parts[0].append(historical[0])
            result = reserve(dataset, parts, historical, 351000)
            selected = {i for t in result["targets"] for i in t["source_ids"]}
            self.assertFalse(selected & set(historical))
        for row in doc["inventory"]:
            self.assertEqual(len(row["sha256"]), 64)
            if row["dataset"] in doc["exclusions"]:
                self.assertGreater(row["source_count"], 0)


if __name__ == "__main__":
    unittest.main()
