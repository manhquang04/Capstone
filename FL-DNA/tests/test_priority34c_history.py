import unittest
import numpy as np
from experiments.priority34c_history import extract


class HistoryTests(unittest.TestCase):
    def test_historical_schema_union(self):
        doc = dict(source_rows=np.array([1, 2]), targets=[dict(source_ids=[2, 3])],
                   sources=[4], earlier=[dict(transaction_ids=[5])])
        self.assertEqual(extract(doc), [1, 2, 3, 4, 5])

    def test_code_provenance_not_record_ids(self):
        self.assertEqual(extract(dict(sources=["source.py"], source_ids=[9])), [9])

    def test_explicit_noninteger_fails(self):
        with self.assertRaises(ValueError):
            extract(dict(source_ids=[1.25]))


if __name__ == "__main__":
    unittest.main()
