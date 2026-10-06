"""Synthetic public models only; no dataset preparation or target access."""
import unittest
import numpy as np
import torch
from experiments.priority34c_core import ratio_model, gradient, defend, ratio_decode, strict_model
from experiments.priority34c_core import native


class CoreTests(unittest.TestCase):
    def test_identity_and_bn_firewall(self):
        for dim in (13, 58, 476):
            net = ratio_model(dim)
            x = torch.linspace(-.5, .5, dim).reshape(1, dim)
            raw, receipt = gradient(net, x, torch.tensor([1]), ratio=True)
            rec, _ = ratio_decode(list(raw), list(raw.values()))
            np.testing.assert_allclose(rec.numpy(), x.numpy(), atol=2e-6)
            self.assertTrue(receipt["no_bn_payload"])
            self.assertEqual(len(receipt["omitted_bn_parameters"]), 6)
            self.assertEqual(len(raw), 8)

    def test_noise_rng_isolation_and_clipping(self):
        raw = {"weight": torch.tensor([3., 4.])}
        state = torch.random.get_rng_state().clone()
        payload, _, _ = defend(raw, "dp_eps10", noise_seed=99, clip=.01, noise_sd=0.)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertAlmostEqual(float(payload[0].norm()), .01, places=7)

    def test_fail_closed_bn(self):
        net = ratio_model(13)
        net.network[1].running_var[0] = -1
        with self.assertRaises(FloatingPointError):
            strict_model(net)

    def test_v2_public_sketch(self):
        raw = {"weight": torch.arange(13, dtype=torch.float32).reshape(1, 13)}
        payload, plans, meta = defend(raw, "dna_v2_0p95")
        self.assertEqual(payload[0].numel(), 16)
        self.assertEqual(meta[0].original_size, 13)
        self.assertEqual(plans[0]["padded_size"], 16)

    def test_native_sketch_plan_matches_observation(self):
        raw = {"weight": torch.linspace(-1., 1., 1300).reshape(100, 13),
               "bias": torch.linspace(-.2, .2, 100)}
        payload, plans, metadata = defend(raw, "dna_v2_0p95")
        from experiments.priority30_native_defenses.run_audit import fwht_normalized_torch
        from dna_encoder.transform_defense_v2 import _fwht_normalized, _signs
        for g, plan, meta in zip(raw.values(), plans, metadata):
            padded = g.new_zeros(plan["padded_size"])
            padded[:g.numel()] = g.flatten()
            candidate = plan["scale"] * fwht_normalized_torch(padded * plan["signs"])[plan["sampled"]]
            reference = np.zeros(meta.padded_size)
            reference[:meta.original_size] = g.numpy().reshape(-1)
            reference = plan["scale"] * _fwht_normalized(reference * _signs(meta.padded_size, meta.seed))[list(meta.sampled_indices)]
            np.testing.assert_allclose(candidate.numpy(), reference, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
