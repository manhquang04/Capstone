"""Pre-run synthetic algebra, receipt and invalid-state gates, no benchmark targets."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments import priority33a_bn_mean as a


class Gates(unittest.TestCase):
    def test_negative_variance_and_nonfinite_fail_closed(self):
        for state in ({"network.1.running_var": torch.tensor([-1.])}, {"weight": torch.tensor([float("nan")])}):
            with self.assertRaises(ValueError):
                a.strict_state(state, "unit")

    def test_lossless_sketch_equals_plain_recovery(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(33)
            model = a.p.FraudMLP(58)
        rng = np.random.default_rng(33)
        raw = torch.from_numpy(rng.normal(size=128).astype(np.float32))
        config = a.v2.DNATransformV2Config(compression_ratio=1., quantization_eta=0., seed=20260916)
        q, metadata = a.v2.transform_update_array_v2(raw.numpy(), config, tensor_index=list(model.state_dict()).index(a.BN_KEY))
        receipt = dict(kind="v2", q=q, metadata=metadata)
        reconstructed = a.recover_payload(model, receipt)
        np.testing.assert_allclose(reconstructed, a.recover_mean(model, raw), rtol=1e-5, atol=1e-5)
        # Attack receipt contains the transmitted sketch and server metadata only.
        self.assertIs(receipt["q"], q)
        self.assertEqual(set(receipt), {"kind", "q", "metadata"})
        self.assertNotIn("raw", receipt)

    def test_known_R_matches_official_lossless_projection(self):
        raw = np.arange(128, dtype=np.float32)/128
        config = a.v2.DNATransformV2Config(compression_ratio=.95, quantization_eta=0., seed=20260916)
        q, metadata = a.v2.transform_update_array_v2(raw, config, tensor_index=4)
        np.testing.assert_allclose(a.projection(metadata)@raw, q, atol=1e-6)

    def test_unknown_payload_no_fallback(self):
        with self.assertRaises(ValueError):
            a.recover_payload(None, dict(kind="unknown"))

    def test_v1_identity_and_real_defenses_not_identity(self):
        from dna_encoder.transform_defense import DNATransformConfig
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(33)
            model = a.p.FraudMLP(58)
        raw = torch.linspace(-2., 2., 128)
        identity = DNATransformConfig(block_size=256, mix_ratio=0., keep_ratio=1., shrink_factor=1., seed=681958327)
        q, _ = a.transform_update_array(raw.numpy(), identity, tensor_index=4)
        np.testing.assert_allclose(q, raw.numpy(), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(a.recover_mean(model, torch.from_numpy(q)), a.recover_mean(model, raw), rtol=1e-5, atol=1e-5)
        for method in a.METHODS:
            receipt = a.payload(model, raw, method, 0)
            if receipt["kind"] == "v1":
                decoded = receipt["q"].numpy()
            else:
                decoded, _ = a.v2.reconstruct_update_array_v2(receipt["q"], receipt["metadata"])
            self.assertGreater(np.linalg.norm(decoded-raw.numpy()), 1e-6)

    def test_distortion_calibration_uses_real_payload(self):
        with torch.random.fork_rng(devices=[]):
            model = a.p.FraudMLP(58)
        rng = np.random.default_rng(33)
        cap = [dict(target=i, raw=torch.from_numpy(rng.normal(size=128).astype(np.float32))) for i in range(24)]
        with patch.object(a, "write"):
            result = a.distortion("synthetic", model, cap)
        self.assertTrue(all(r["sigma"] > 0 and r["median_distortion"] > 0 for r in result.values()))


if __name__ == "__main__":
    unittest.main()
