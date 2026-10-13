"""Correctness check ONLY: synthetic model without BN, not Config-C evidence."""
import unittest
import torch
from experiments.priority35a_ratio import synthetic_single_record_ratio


class SyntheticRatioTests(unittest.TestCase):
    def test_no_bn_linear_gradient_ratio(self):
        torch.set_num_threads(1)
        layer = torch.nn.Linear(3, 2, dtype=torch.float64)
        self.assertFalse(any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                             for m in layer.modules()))
        record = torch.tensor([[0.25, -2., 3.]], dtype=torch.float64)
        layer(record).sum().backward()
        recovered = synthetic_single_record_ratio(layer.weight.grad, layer.bias.grad)
        self.assertTrue(torch.equal(recovered, record.expand(2, -1)))

    def test_undefined_and_nonfinite_fail(self):
        with self.assertRaises(ValueError):
            synthetic_single_record_ratio(torch.ones(1, 3), torch.zeros(1))
        with self.assertRaises(ValueError):
            synthetic_single_record_ratio(torch.full((1, 3), float("nan")), torch.ones(1))


if __name__ == "__main__":
    unittest.main()
