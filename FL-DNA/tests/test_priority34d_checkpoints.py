"""Synthetic preflight only; no benchmark data or target loading."""
import unittest
from unittest.mock import patch
import torch
from experiments import priority34d_checkpoints as d


class Gates(unittest.TestCase):
    def test_new_fixed_seeds(self):
        self.assertEqual(d.SEEDS, (342000, 342001, 342002))
        with self.assertRaises(ValueError):
            d.folder(321000)

    def test_write_firewall(self):
        with self.assertRaises(ValueError):
            d.write(d.ROOT / "artifacts/priority33a/forbidden.json", {})

    def test_bn_negative_fail_closed(self):
        with self.assertRaises(ValueError):
            d.b.strict_state({"network.1.running_var":torch.tensor([-1.])}, "synthetic")

    def test_nan_fail_closed(self):
        with self.assertRaises(ValueError):
            d.b.strict_state({"weight":torch.tensor([float("nan")])}, "synthetic")

    def test_no_science_on_import(self):
        self.assertEqual(d.b.OUT.name, "priority33a")
        self.assertEqual(d.b.p.OUT.name, "priority32_multidataset")

    def test_process_case_insensitive_path(self):
        with patch.object(d.subprocess, "check_output", return_value="123 /MacOS/Python priority34d_checkpoints.py --job 342000\n"):
            self.assertTrue(d.live())


if __name__ == "__main__":
    unittest.main()
