import unittest
import numpy as np
import torch
from experiments.priority33c_image_recovery import ObservableReconstructor, block_debias, native


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.model, _ = native.inversefed.construct_model('LeNetZhu', seed=42)
        self.model.eval()
        self.x = torch.linspace(-1, 1, 3072).reshape(1, 3, 32, 32)
        self.label = torch.tensor([2])
        self.gradient = list(native.gradient_dict(self.model, self.x, self.label, torch.nn.CrossEntropyLoss()).values())
        self.config = native.IMAGE_CONFIG.copy()
        self.config['max_iterations'] = 8
        self.normal = (torch.zeros(3, 1, 1), torch.ones(3, 1, 1))

    def test_unknown_mode_and_payload_substitution_fail(self):
        with self.assertRaises(ValueError):
            ObservableReconstructor(self.model, self.normal, self.config, self.gradient, 'unknown')
        attack = ObservableReconstructor(self.model, self.normal, self.config, self.gradient, 'plain')
        with self.assertRaises(ValueError):
            attack.reconstruct(list(self.gradient), self.label)

    def test_identity_losses(self):
        plain = ObservableReconstructor(self.model, self.normal, self.config, self.gradient, 'plain')
        debias = ObservableReconstructor(self.model, self.normal, self.config, self.gradient, 'v1_debias', mix=0)
        sketch = [v.flatten() for v in self.gradient]
        v2 = ObservableReconstructor(self.model, self.normal, self.config, sketch, 'v2_sketch',
                                     plans=[{'identity': True} for _ in sketch])
        expected = plain._adaptive_loss(self.gradient, self.gradient)
        torch.testing.assert_close(debias._adaptive_loss(self.gradient, self.gradient), expected, atol=0, rtol=0)
        torch.testing.assert_close(v2._adaptive_loss(self.gradient, sketch), expected, atol=0, rtol=0)

    def test_identity_recovery_same_native_result(self):
        outputs = []
        for mode in ['plain', 'v1_debias', 'v2_sketch']:
            receipt = [v.flatten() for v in self.gradient] if mode == 'v2_sketch' else self.gradient
            plans = [{'identity': True} for _ in receipt] if mode == 'v2_sketch' else None
            torch.manual_seed(333999); np.random.seed(333999)
            attack = ObservableReconstructor(self.model, self.normal, self.config, receipt, mode, plans=plans)
            outputs.append(attack.reconstruct(receipt, self.label)[0])
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
        torch.testing.assert_close(outputs[0], outputs[2], atol=0, rtol=0)

    def test_debias_uses_blocks_not_whole_tensor(self):
        value = torch.cat([torch.ones(256), torch.ones(5)*3])
        result = block_debias([value], .1)[0]
        torch.testing.assert_close(result, value)


if __name__ == '__main__':
    unittest.main()
