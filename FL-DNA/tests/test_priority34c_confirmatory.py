import unittest
from unittest.mock import patch
from types import SimpleNamespace
from experiments import priority34c_confirmatory as c


class ConfirmatoryTests(unittest.TestCase):
    def test_local_matches_p34b_clip_noise_precision(self):
        raw = {"weight": c.torch.tensor([.1, -.2]), "bias": c.torch.tensor([.3])}
        actual = c.local_payload(raw, 42)
        norm = float(c.torch.sqrt(sum(g.double().square().sum() for g in raw.values())))
        generator = c.torch.Generator().manual_seed(42)
        for g, observed in zip(raw.values(), actual):
            clipped = (g.double()*min(1., .01/norm)*(1-4*c.np.finfo(c.np.float32).eps)).to(g)
            expected = (clipped.double()+c.torch.randn(g.shape, dtype=c.torch.float64, generator=generator)*c.LOCAL_SD).to(g)
            c.torch.testing.assert_close(observed, expected, rtol=0, atol=0)

    def test_ratio_observable_identity(self):
        net = c.ratio_model(13)
        x = c.torch.linspace(-.2, .2, 13).reshape(1, 13)
        labels = c.torch.tensor([1])
        raw, _ = c.gradient(net, x, labels, ratio=True)
        data = SimpleNamespace(mean=c.torch.zeros(13), std=c.torch.ones(13))
        candidates, selected = c.recover(net, data, list(raw), list(raw.values()), labels, 42,
                                        "unprotected", None, None, True)
        self.assertEqual(selected, 0)
        self.assertLess(abs(candidates[0]["objective"]), 1e-9)
        c.torch.testing.assert_close(candidates[0]["reconstruction"], x)

    def test_v2_uses_lsmr_residual(self):
        net = c.ratio_model(13)
        labels = c.torch.tensor([0])
        raw, _ = c.gradient(net, c.torch.ones(1, 13)*.1, labels, ratio=True)
        payload, plans, metadata = c.defend(raw, "dna_v2_0p95")
        data = SimpleNamespace(mean=c.torch.zeros(13), std=c.torch.ones(13))
        candidates, selected = c.recover(net, data, list(raw), payload, labels, 42,
                                        "dna_v2_0p95", plans, metadata, True)
        self.assertEqual(selected, 0)
        self.assertEqual(len(candidates[0]["solver"]["least_squares"]), len(raw))
        self.assertTrue(c.np.isfinite(candidates[0]["objective"]))

    def test_no_truth_in_recover_interface(self):
        import inspect
        params = inspect.signature(c.recover).parameters
        self.assertNotIn("truth", params)
        self.assertNotIn("noise_seed", params)
        self.assertNotIn("raw", params)

    def test_native_v1_selection_is_objective_only(self):
        data = SimpleNamespace()
        one, two = c.torch.ones(1, 2), c.torch.zeros(1, 2)
        outputs = [(one, [one]*30, [2.]*30), (two, [two]*30, [1.]*30)]
        with patch.object(c.native, "native_recover", side_effect=outputs):
            candidates, selected = c.recover(None, data, ["w"], [c.torch.ones(2)],
                    c.torch.tensor([0]), 42, "dna_v1_conservative", None, None, False)
        self.assertEqual(selected, 1)
        self.assertEqual(candidates[selected]["name"], "structure_debias")

    def test_prepare_blocks_active_workers(self):
        with patch.object(c, "live_processes", return_value=["worker"]):
            with self.assertRaisesRegex(RuntimeError, "active"):
                c.prepare()

    def test_zero_denominator_fails(self):
        net = c.ratio_model(13)
        with self.assertRaises(ValueError):
            c.ratio_objective(net, c.torch.zeros(1, 13), c.torch.tensor([1]),
                             ["wrong"], [c.torch.zeros(1)])


if __name__ == "__main__":
    unittest.main()
