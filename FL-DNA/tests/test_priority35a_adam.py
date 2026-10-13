"""Synthetic-only tests: synchronized randomness is not server knowledge."""
import copy
import unittest
from collections import OrderedDict

import torch

from experiments.priority34a_local_bn import p, domains
from experiments.priority35a_adam import (
    fresh_adam_parameters, parameter_difference, require_finite_cpu,
)


class FreshAdamTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_bit_exact_against_p34a_training_step(self):
        for batch_size in (4, 8, 16):
            with self.subTest(batch_size=batch_size):
                torch.manual_seed(350001)
                reference = p.FraudMLP(7).cpu().train()
                candidate = copy.deepcopy(reference).train()
                features = torch.randn(batch_size, 7)
                labels = (torch.arange(batch_size) % 2).float()[:, None]
                loss_function = p.BinaryFocalLoss(alpha=0.95, gamma=2.)
                optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3)
                optimizer.zero_grad()
                # Equal RNG here tests the same local step. Real instruments
                # must not get the client's dropout seed or realized masks.
                torch.manual_seed(350002)
                loss_function(reference(features), labels).backward()
                optimizer.step()
                torch.manual_seed(350002)
                loss_function(candidate(features), labels).backward()
                before = OrderedDict(candidate.named_parameters())
                gradients = OrderedDict((n, v.grad) for n, v in before.items())
                simulated = fresh_adam_parameters(before, gradients)
                for name, value in reference.named_parameters():
                    self.assertTrue(torch.equal(value, simulated[name]), name)
                _, uploaded = domains(reference)
                observed_before = OrderedDict((n, before[n]) for n in uploaded)
                observed_after = OrderedDict((n, simulated[n]) for n in uploaded)
                delta = parameter_difference(observed_before, observed_after)
                self.assertEqual(tuple(delta), uploaded)
                self.assertFalse(any("running_" in name for name in delta))

    def test_zero_gradient_matches_native_adam(self):
        parameter = torch.nn.Parameter(torch.tensor([0., 0.125, -0.25]))
        before = parameter.detach().clone()
        parameter.grad = torch.zeros_like(parameter)
        optimizer = torch.optim.Adam([parameter], lr=1e-3)
        optimizer.step()
        simulated = fresh_adam_parameters({"p": before}, {"p": parameter.grad})
        self.assertTrue(torch.equal(parameter, simulated["p"]))

    def test_fail_closed(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                fresh_adam_parameters({"p": torch.ones(1)}, {"p": torch.tensor([value])})
        with self.assertRaises(ValueError):
            require_finite_cpu({"bn.running_var": torch.tensor([-0.01])}, "fixture")
        with self.assertRaises(ValueError):
            fresh_adam_parameters({"p": torch.ones(1)}, {"other": torch.ones(1)})
        with self.assertRaises(ValueError):
            parameter_difference({"p": torch.ones(1)}, {})

    def test_zero_gradient_backward_is_finite_without_forward_clamp(self):
        gradient = torch.tensor([0., -0.125, 0.25], requires_grad=True)
        parameter = torch.tensor([0., 0.125, -0.25])
        simulated = fresh_adam_parameters({"p": parameter}, {"p": gradient})["p"]
        derivative, = torch.autograd.grad(simulated.sum(), (gradient,))
        self.assertTrue(torch.isfinite(derivative).all())
        self.assertEqual(simulated[0].item(), parameter[0].item())
        reference = torch.nn.Parameter(parameter.clone())
        reference.grad = gradient.detach().clone()
        torch.optim.Adam([reference], lr=1e-3).step()
        self.assertTrue(torch.equal(reference, simulated))

    def test_singleton_training_bn_is_not_a_config_c_upload(self):
        model = p.FraudMLP(7).cpu().train()
        with self.assertRaises(ValueError):
            model(torch.ones(1, 7))


if __name__ == "__main__":
    unittest.main()
