import unittest
import numpy as np
import torch
from experiments.priority33c_image_recovery_contiguous import recovery, original_tensors, contiguous_tensors


class LayoutTests(unittest.TestCase):
    def test_real_frozen_target_values_rng_and_gradient(self):
        data = recovery.utility.p31.dp.CIFAR10(str(recovery.ROOT / 'datasets/cifar10'), train=False, download=False)
        before = torch.get_rng_state().clone()
        old, labels = original_tensors(data, [7728])
        fixed, fixed_labels = contiguous_tensors(data, [7728])
        self.assertTrue(fixed.is_contiguous())
        torch.testing.assert_close(old, fixed, atol=0, rtol=0)
        torch.testing.assert_close(labels, fixed_labels, atol=0, rtol=0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        model, _ = recovery.native.inversefed.construct_model('LeNetZhu', seed=42)
        model.eval()
        gradients = recovery.native.gradient_dict(model, fixed, labels, torch.nn.CrossEntropyLoss())
        reference = recovery.native.gradient_dict(model, old.clone(memory_format=torch.contiguous_format), labels, torch.nn.CrossEntropyLoss())
        for key in gradients:
            torch.testing.assert_close(gradients[key], reference[key], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
