"""Synthetic contract tests; never touches benchmark data or previous artifacts."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from experiments.priority32_multidataset import capped_parts, preprocess, canonical_sha
from data.load_creditcard import _mild_non_iid_client_indices
from experiments.priority32_cost import clipped_noisy
from collections import OrderedDict


class Contracts(unittest.TestCase):
    def test_train_only_impute_scale_encoding(self):
        frame = pd.DataFrame({"numeric": [0., 2., np.nan, 1000., np.nan, -1000.],
                              "category": ["a", "b", "a", "unknown", "b", "a"]})
        arrays, meta = preprocess(frame, [np.array([0,1,2]), np.array([3,4]), np.array([5])], ["category"])
        self.assertEqual(meta["medians"]["numeric"], 1.)
        self.assertEqual(meta["one_hot_levels"]["category"], ["a", "b"])
        self.assertEqual(arrays[1][0,1:].sum(), 0.)
        self.assertEqual(arrays[1][1,0], 0.)
        self.assertTrue(all(np.isfinite(a).all() for a in arrays))

    def test_cap_keeps_partition_membership_order_and_labels(self):
        labels = np.tile([0, 0, 0, 1], 100)
        parts = [np.arange(200)[::-1], np.arange(200, 300), np.arange(300, 400)[::-1]]
        selected = capped_parts(parts, labels, cap=200)
        self.assertEqual(sum(map(len, selected)), 200)
        for original, indices in zip(parts, selected):
            keep = set(indices)
            np.testing.assert_array_equal(indices, [i for i in original if i in keep])
            self.assertEqual(set(labels[indices]), {0, 1})
        self.assertFalse(set(selected[0]) & set(selected[1]))

    def test_non_iid_covers_each_row_once(self):
        labels = np.array([1]*12+[0]*288, dtype=np.float32)
        frame = pd.DataFrame({"type": np.tile(["a", "b", "c"], 100)})
        parts = _mild_non_iid_client_indices(frame, labels, 3, 32)
        np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(300))
        self.assertEqual([int(labels[p].sum()) for p in parts], [4, 4, 4])

    def test_cost_dp_determinism_and_input_immutability(self):
        state = OrderedDict(a=np.array([300., 400.], dtype=np.float32))
        first = clipped_noisy(state, 12, sigma=0.)
        np.testing.assert_array_equal(first["a"], [60., 80.])
        np.testing.assert_array_equal(state["a"], [300., 400.])
        np.testing.assert_array_equal(clipped_noisy(state, 12)["a"], clipped_noisy(state, 12)["a"])

    def test_config_digest_sensitive(self):
        self.assertEqual(canonical_sha({"b": 1, "a": 2}), canonical_sha({"a": 2, "b": 1}))
        self.assertNotEqual(canonical_sha({"seed": 1}), canonical_sha({"seed": 2}))


if __name__ == "__main__":
    unittest.main()
