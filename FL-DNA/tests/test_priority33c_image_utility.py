import unittest
import numpy as np
import torch
from experiments.priority33c_image_utility import per_tensor_noise, transform, V1
from dna_encoder.transform_defense import DNATransformConfig, transform_update_array


class ImageUtilityTests(unittest.TestCase):
    def test_identity_transform(self):
        source = np.arange(300, dtype=np.float32)
        result, _ = transform_update_array(source, DNATransformConfig(mix_ratio=0, seed=30001))
        np.testing.assert_array_equal(source, result)

    def test_tensor_clipping(self):
        values = [torch.tensor([3., 4.]), torch.tensor([0., 2.])]
        result = per_tensor_noise(values, [1., 3.], 0, 12)
        torch.testing.assert_close(result[0], values[0]/5)
        torch.testing.assert_close(result[1], values[1])

    def test_lossless_noise(self):
        values = [torch.tensor([3., 4.])]
        torch.testing.assert_close(per_tensor_noise(values, [10.], 0, 12)[0], values[0])

    def test_fail_closed(self):
        with self.assertRaises(ValueError):
            transform({'x': torch.ones(3)}, 'unknown')
        with self.assertRaises(FloatingPointError):
            per_tensor_noise([torch.tensor([float('nan')])], [1.], .1, 1)

    def test_registered_settings(self):
        self.assertEqual(V1['dna_v1_medium'], (.10, .85, .40))
        self.assertEqual(V1['dna_v1_stronger'], (.12, .82, .35))


if __name__ == '__main__':
    unittest.main()
