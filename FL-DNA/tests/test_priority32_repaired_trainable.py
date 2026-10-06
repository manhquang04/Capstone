"""Regression checks for explicitly approved raw-BN protocol variant."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import priority32_repaired_trainable as r


class RepairTests(unittest.TestCase):
    def test_original_parameter_math_and_raw_buffers(self):
        p = r.p
        for dimension in (13, 58, 476):
            p.torch.manual_seed(321001)
            model = p.FraudMLP(dimension)
            global_state = copy.deepcopy(model.state_dict())
            local = copy.deepcopy(global_state)
            for key, value in local.items():
                if value.is_floating_point():
                    value.add_(p.torch.randn_like(value)*.03)
                    if key.endswith("running_var"):
                        value.abs_()
            cases = [(r.V1, r.v1, (p.DNATransformConfig(seed=77,mix_ratio=.08,keep_ratio=.88,shrink_factor=.45),)),
                     (r.V2, r.v2, (p.DNATransformV2Config(seed=77,compression_ratio=.95,quantization_eta=.01), 99))]
            for original, repaired, args in cases:
                expected, _ = original(local, global_state, *args)
                actual, _ = repaired(local, global_state, *args)
                for key in local:
                    target = expected[key] if key in r.TRAINABLE else local[key]
                    self.assertTrue(p.torch.equal(actual[key], target), key)
                averaged = p.fed_avg([actual, actual], [1, 1])
                for key in averaged:
                    if key.endswith("running_var"):
                        self.assertTrue(bool((averaged[key] >= 0).all()))
                self.assertTrue(all(p.torch.equal(local[key], actual[key]) for key in local if key not in r.TRAINABLE))


if __name__ == "__main__":
    unittest.main()
