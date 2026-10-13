"""Synthetic knowledge-boundary checks; no private client state is supplied."""
from collections import OrderedDict
import unittest
import torch
from experiments.priority34a_local_bn import p, domains
from experiments.priority35a_nuisance import estimator_model


class NuisanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(350101)
            initial = p.FraudMLP(7).cpu()
        _, allowed = domains(initial)
        self.public = OrderedDict((n, initial.state_dict()[n].clone()) for n in allowed)

    def test_bn_defaults_train_mode_free_affine_and_no_dropout(self):
        model, nuisance = estimator_model(self.public, 7, 350102)
        self.assertTrue(model.training)
        self.assertEqual(len(nuisance), 6)
        self.assertTrue(all(v.requires_grad for v in nuisance.values()))
        bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        self.assertEqual(len(bn_layers), 3)
        for bn in bn_layers:
            self.assertTrue(bn.training)
            self.assertTrue(torch.equal(bn.weight, torch.ones_like(bn.weight)))
            self.assertTrue(torch.equal(bn.bias, torch.zeros_like(bn.bias)))
            self.assertTrue(torch.equal(bn.running_mean, torch.zeros_like(bn.running_mean)))
            self.assertTrue(torch.equal(bn.running_var, torch.ones_like(bn.running_var)))
        self.assertFalse(any(isinstance(m, torch.nn.Dropout) for m in model.modules()))

    def test_private_bn_tensor_is_rejected(self):
        extra = OrderedDict(self.public)
        extra["network.1.running_mean"] = torch.zeros(128)
        with self.assertRaises(ValueError):
            estimator_model(extra, 7, 350102)

    def test_forward_keeps_bn_training_and_parameters_are_public(self):
        model, _ = estimator_model(self.public, 7, 350102)
        for name, parameter in model.named_parameters():
            if name in self.public:
                self.assertTrue(torch.equal(parameter, self.public[name]))
        model(torch.arange(28).float().reshape(4, 7))
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                self.assertEqual(module.num_batches_tracked.item(), 1)
                self.assertTrue(torch.isfinite(module.running_var).all())
                self.assertTrue((module.running_var >= 0).all())


if __name__ == "__main__":
    unittest.main()
