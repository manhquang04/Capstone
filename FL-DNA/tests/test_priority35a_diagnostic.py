"""Synthetic-only privileged-BN diagnostic contracts."""
import unittest
from collections import OrderedDict
import torch
from experiments import priority34a_local_bn as old
from experiments.priority35a_diagnostic_estimator import oracle_model, operator, reconstruct, sign_cosines
from experiments.priority35a_estimator import observable, flatten


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(771)
        model = old.p.FraudMLP(3).cpu()
        bn, allowed = old.domains(model)
        self.shared = OrderedDict((k, model.state_dict()[k].clone()) for k in allowed)
        self.bn = OrderedDict((k, model.state_dict()[k].clone()) for k in sorted(bn))
        for k in self.bn:
            if k.endswith("weight"):
                self.bn[k].fill_(.7)
        self.meta = dict(feature_names=["a", "b", "c"], one_hot_levels={})
        self.rows = torch.randn(4, 3, requires_grad=True)
        self.labels = torch.tensor([[0.], [0.], [0.], [1.]])

    def test_oracle_fixed_affine_train_bn(self):
        model = oracle_model(self.shared, self.bn, 3, 1)
        for k, p in model.named_parameters():
            if k in self.bn:
                self.assertFalse(p.requires_grad)
                self.assertTrue(torch.equal(p, self.bn[k]))
        self.assertTrue(all(m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)))
        self.assertFalse(any(isinstance(m, torch.nn.Dropout) for m in model.modules()))

    def test_gradient_native_and_adam_frozen_operator(self):
        model = oracle_model(self.shared, self.bn, 3, 1)
        expected = torch.autograd.grad(old.p.BinaryFocalLoss(alpha=.95, gamma=2.)(model(self.rows), self.labels),
                                     tuple(p for k, p in model.named_parameters() if k in self.shared), create_graph=True)
        actual = operator(model, self.rows, self.labels, self.shared, "D1")
        self.assertTrue(torch.equal(actual, torch.cat([g.reshape(-1) for g in expected])))
        self.assertTrue(torch.equal(operator(model, self.rows, self.labels, self.shared, "D2"),
                                    observable(model, self.rows, self.labels, self.shared)))

    def test_budgets_and_finite_fail_closed(self):
        with self.assertRaises(ValueError):
            reconstruct(self.shared, self.shared, self.bn, self.meta, 4, "wan", "synthetic", 1, "D1", test_iterations=1)
        bad = OrderedDict((k, v.clone()) for k, v in self.bn.items())
        key = next(k for k in bad if k.endswith("running_var"))
        bad[key][0] = -1
        with self.assertRaises(ValueError):
            oracle_model(self.shared, bad, 3, 1)
        model = oracle_model(self.shared, self.bn, 3, 1)
        observed = operator(model, self.rows, self.labels, self.shared, "D1").detach()
        tensors = OrderedDict()
        start = 0
        for k, v in self.shared.items():
            tensors[k] = observed[start:start+v.numel()].reshape(v.shape)
            start += v.numel()
        result = reconstruct(self.shared, tensors, self.bn, self.meta, 4, "tableak", "synthetic", 1, "D1",
                             synthetic=True, test_iterations=2)
        self.assertTrue(result["synthetic"])
        self.assertEqual(len(result["start_objectives"]), 2)
        self.assertEqual(result["torch_threads"], 1)
        self.assertFalse(result["dropout_masks_supplied"])

    def test_signed_cosine_not_absolute(self):
        g = OrderedDict(x=torch.tensor([2., -2., 0.]))
        delta = OrderedDict(x=torch.tensor([-.001, .001, 0.]))
        c = sign_cosines(g, delta)
        self.assertAlmostEqual(c["adam_vs_sign"], -1., places=6)
        self.assertAlmostEqual(c["descent_aligned_adam_vs_sign"], 1., places=6)


if __name__ == "__main__":
    unittest.main()
