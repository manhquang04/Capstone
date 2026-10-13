"""Synthetic integration only; no benchmark record or outcome access."""
import unittest
from collections import OrderedDict
import torch
from experiments.priority35a_nuisance import estimator_model
from experiments.priority35a_estimator import reconstruct, objective, schema_blocks, candidate_rows
from experiments.priority34a_local_bn import p, domains


class EstimatorTest(unittest.TestCase):
    def test_schema_and_decode(self):
        blocks = schema_blocks(dict(feature_names=["x", "c=a", "c=b"], one_hot_levels={"c": ["a", "b"]}))
        raw = torch.tensor([[2., -3., 1.]])
        self.assertTrue(torch.equal(candidate_rows(raw, blocks, True), torch.tensor([[2., 0., 1.]])))
        self.assertAlmostEqual(float(candidate_rows(raw, blocks)[0, 1:].sum()), 1.)

    def test_synthetic_higher_order_both_objectives_and_labels(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        model = p.FraudMLP(3)
        _, allowed = domains(model)
        public = OrderedDict((name, model.state_dict()[name].clone()) for name in allowed)
        observed = OrderedDict((name, torch.full_like(value, .0001)) for name, value in public.items())
        meta = dict(feature_names=["x", "c=a", "c=b"], one_hot_levels={"c": ["a", "b"]})
        for instrument in ("tableak", "wan"):
            for setting in ("given", "estimated"):
                result = reconstruct(public, observed, meta, 4, setting, instrument, "synthetic-fixture",
                                     known_count=2 if setting == "given" else None, synthetic=True, test_iterations=2)
                self.assertTrue(torch.isfinite(result["rows"]).all())
                self.assertEqual(result["rows"].shape, (4, 3))
                self.assertEqual(result["torch_threads"], 1)
                self.assertEqual(len(result["nuisance"]), 6)
                if setting == "estimated":
                    self.assertIsNotNone(result["label_receipt"])
        with self.assertRaises(ValueError):
            reconstruct(public, observed, meta, 4, "given", "wan", "fixture", known_count=2, test_iterations=2)

    def test_objectives_fail_closed(self):
        for instrument in ("tableak", "wan"):
            # Native float32 cosine of identical vectors need not be exactly 1.
            self.assertAlmostEqual(float(objective(torch.ones(3), torch.ones(3), instrument)), 0., places=6)
            with self.assertRaises(ValueError):
                objective(torch.tensor([float("nan")]), torch.ones(1), instrument)


if __name__ == "__main__":
    unittest.main()
