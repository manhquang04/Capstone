"""Synthetic-only firewall and local BN persistence regression tests."""
import copy
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch
from experiments import priority34a_local_bn as r
from experiments.priority34a_analysis import paired_summary, independent_summary


class LocalBNTests(unittest.TestCase):
    def setUp(self):
        r.torch.manual_seed(77)
        self.model = r.p.FraudMLP(13)
        self.bn, self.allowed = r.domains(self.model)
        self.global_parameters = OrderedDict((name, self.model.state_dict()[name].clone()) for name in self.allowed)

    def test_domain(self):
        self.assertEqual(len(self.bn), 15)
        self.assertEqual(len(self.allowed), 8)
        self.assertFalse(set(self.allowed) & self.bn)

    def test_broadcast_preserves_all_bn(self):
        with r.torch.no_grad():
            for name, parameter in self.model.named_parameters():
                parameter.add_(1)
        old = copy.deepcopy(self.model.state_dict())
        r.broadcast(self.model, self.global_parameters, self.bn, self.allowed)
        for name in self.bn:
            self.assertTrue(r.torch.equal(self.model.state_dict()[name], old[name]))
        for name in self.allowed:
            self.assertTrue(r.torch.equal(self.model.state_dict()[name], self.global_parameters[name]))

    def test_firewall_rejects_bn_and_missing(self):
        bad = copy.deepcopy(self.global_parameters)
        bad[next(iter(self.bn))] = r.torch.zeros(1)
        with self.assertRaises(ValueError):
            r.assert_payload(bad, self.bn, self.allowed)
        bad = copy.deepcopy(self.global_parameters)
        bad.pop(next(iter(bad)))
        with self.assertRaises(ValueError):
            r.assert_payload(bad, self.bn, self.allowed)

    def test_all_methods_upload_only_allowlist_bn_unchanged(self):
        self.model.train()
        optimizer = r.torch.optim.Adam(self.model.parameters(), lr=.001)
        loss = self.model(r.torch.randn(8, 13)).square().mean()
        loss.backward(); optimizer.step()
        bn_state = {name: self.model.state_dict()[name].clone() for name in self.bn}
        for method in r.p.METHODS:
            actual = r.upload(self.model, self.global_parameters, method, 321000, 1, 0, self.bn, self.allowed)
            r.assert_payload(actual, self.bn, self.allowed)
            for name in self.bn:
                self.assertTrue(r.torch.equal(bn_state[name], self.model.state_dict()[name]))
            repeated = r.upload(self.model, self.global_parameters, method, 321000, 1, 0, self.bn, self.allowed)
            for name in actual:
                self.assertTrue(r.torch.equal(actual[name], repeated[name]))

    def test_negative_variance_fails_without_clamp(self):
        self.model.network[1].running_var[0] = -1
        with self.assertRaises(ValueError):
            r.guard(self.model, "fixture")
        self.assertEqual(float(self.model.network[1].running_var[0]), -1)

    def test_nonfinite_and_cpu_guard(self):
        self.model.network[0].weight.data[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            r.guard(self.model, "fixture")
        self.global_parameters[next(iter(self.global_parameters))].flatten()[0] = float("inf")
        with self.assertRaises(ValueError):
            r.assert_payload(self.global_parameters, self.bn, self.allowed)

    def test_client_bn_persists_two_rounds(self):
        self.model.train()
        self.model(r.torch.randn(8, 13))
        count = self.model.network[1].num_batches_tracked.clone()
        r.broadcast(self.model, self.global_parameters, self.bn, self.allowed)
        self.model(r.torch.randn(8, 13))
        self.assertEqual(int(self.model.network[1].num_batches_tracked), int(count)+1)

    def test_weighted_fedavg_parameters_only(self):
        first = OrderedDict((name, value+1) for name, value in self.global_parameters.items())
        second = OrderedDict((name, value+2) for name, value in self.global_parameters.items())
        actual = r.p.fed_avg([first, second], [1, 3])
        r.assert_payload(actual, self.bn, self.allowed)
        for name in actual:
            self.assertTrue(r.torch.allclose(actual[name], self.global_parameters[name]+1.75))

    def test_independent_ci_and_count(self):
        values = [.001*i for i in range(21)]
        primary, other = paired_summary(values), independent_summary(values)
        for field in other:
            self.assertAlmostEqual(primary[field], other[field], places=14)
        with self.assertRaises(ValueError):
            paired_summary(values[:20])

    def test_guarded_eval(self):
        result = r.checked_probabilities(self.model, r.np.zeros((8, 13), dtype=r.np.float32))
        self.assertTrue(r.np.isfinite(result).all())
        self.model.network[1].running_var[0] = -1
        with self.assertRaises(ValueError):
            r.checked_probabilities(self.model, r.np.zeros((8, 13), dtype=r.np.float32))

    def test_synthetic_job_checkpoint_and_validated_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "original/prepared/fixture"
            prepared.mkdir(parents=True)
            rng = r.np.random.default_rng(123)
            for split, rows in (("train", 96), ("validation", 16), ("test", 16)):
                r.np.save(prepared / (split+"_x.npy"), rng.normal(size=(rows, 13)).astype(r.np.float32))
                r.np.save(prepared / (split+"_y.npy"), r.np.tile([0., 1.], rows//2).astype(r.np.float32))
            r.np.save(prepared / "train_categories.npy", r.np.array(["a"]*96))
            out = root / "new"
            config = {"dataset": "fixture", "method": "baseline", "seed": 321000,
                      "training": {"rounds": 2, "lr": .001, "alpha": .95}}
            with patch.object(r, "OUT", out), patch.object(r.p, "OUT", root / "original"), patch.object(
                    r.p, "_mild_non_iid_client_indices", return_value=[r.np.arange(i*32, (i+1)*32) for i in range(3)]):
                path = out / "jobs/fixture.json"
                result = r.train_job(config, str(path))
                self.assertFalse(result["skipped"])
                self.assertTrue(r.valid_result(path, config))
                doc = json.loads(path.read_text())
                self.assertEqual(doc["payload_assertions"], 6)
                checkpoint_path = next(relative for relative in doc["artifact_sha256"] if relative.endswith("final_checkpoint.pt"))
                checkpoint = r.torch.load(out / checkpoint_path, map_location="cpu")
                for bn in checkpoint["client_bn"]:
                    self.assertEqual(int(bn["network.1.num_batches_tracked"]), 2)
                self.assertFalse(r.torch.equal(checkpoint["client_bn"][0]["network.1.running_mean"], checkpoint["client_bn"][1]["network.1.running_mean"]))
                self.assertTrue(r.train_job(config, str(path))["skipped"])
                with self.assertRaises(ValueError):
                    r.valid_result(path, {**config, "seed": 321001})


if __name__ == "__main__":
    unittest.main()
