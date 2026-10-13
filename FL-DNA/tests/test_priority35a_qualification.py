import unittest
from collections import OrderedDict
import torch
from experiments.priority35a_qualification import cells, cell_id, client_step
from experiments.priority34a_local_bn import p, domains


class QualificationTest(unittest.TestCase):
    def test_complete_cell_grid(self):
        self.assertEqual(len(cells()), 72)
        self.assertEqual(len(set(map(cell_id, cells()))), 72)
        self.assertEqual(set(c["batch"] for c in cells()), {4, 8, 16})

    def test_private_client_upload_replay_and_bn_firewall(self):
        torch.set_num_threads(1)
        torch.manual_seed(125)
        model = p.FraudMLP(3)
        bn, allowed = domains(model)
        public = OrderedDict((k, model.state_dict()[k].clone()) for k in allowed)
        private_bn = OrderedDict((k, model.state_dict()[k].clone()) for k in sorted(bn))
        features, labels = torch.randn(4, 3), torch.tensor([[0.], [0.], [1.], [0.]])
        first = client_step(public, private_bn, features, labels, 571)
        second = client_step(public, private_bn, features, labels, 571)
        self.assertEqual(tuple(first), allowed)
        self.assertFalse(set(first) & bn)
        self.assertTrue(all(torch.equal(first[k], second[k]) for k in first))
        private_bn["network.1.running_var"][0] = -1.
        with self.assertRaises(ValueError):
            client_step(public, private_bn, features, labels, 571)
