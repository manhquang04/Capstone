"""Synthetic only: actual masks, exact replay, ensemble and spawn determinism."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import unittest
import torch
from experiments import priority34a_local_bn as old
from experiments.priority35a_d1prime_estimator import capture_masks, mask_model, reconstruct, aggregate
from experiments.priority35a_diagnostic_estimator import operator
from experiments.priority35a_estimator import flatten


def synthetic_payload():
    torch.set_num_threads(1)
    torch.manual_seed(849)
    model = old.p.FraudMLP(3).cpu()
    bn, allowed = old.domains(model)
    shared = OrderedDict((k, model.state_dict()[k].clone()) for k in allowed)
    bnstate = OrderedDict((k, model.state_dict()[k].clone()) for k in sorted(bn))
    private = dict(features=torch.randn(4, 3), labels=torch.tensor([[0.], [1.], [0.], [0.]]),
                   client_bn=bnstate, dropout_seed=887)
    masks, grads = capture_masks(dict(shared=shared), private)
    return dict(shared=shared, gradient=grads, bn=bnstate, masks=masks, labels=private['labels'],
                meta=dict(feature_names=['a', 'b', 'c'], one_hot_levels={}), ordinal='synthetic'), private


def small_reconstruct(payload):
    result = reconstruct(payload, 'tableak', 'a', synthetic=True, test_iterations=2)
    result.pop('elapsed_seconds')
    return result


class D1PrimeTests(unittest.TestCase):
    def test_native_actual_mask_gradient_replay_bit_exact(self):
        payload, private = synthetic_payload()
        model = mask_model(payload['shared'], payload['bn'], 3, 0, payload['masks'])
        gradient = operator(model, private['features'], private['labels'], payload['shared'], 'D1').detach()
        self.assertTrue(torch.equal(gradient, flatten(payload['gradient'])))
        self.assertTrue(all(m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)))
        self.assertFalse(any(isinstance(m, torch.nn.Dropout) for m in model.modules()))
        self.assertTrue(torch.equal(payload['labels'], private['labels']))

    def test_mask_bn_nonfinite_fail_closed(self):
        payload, _ = synthetic_payload()
        payload['bn']['network.1.running_var'][0] = -1.
        with self.assertRaises(ValueError):
            small_reconstruct(payload)
        payload, _ = synthetic_payload()
        payload['masks']['network.3'][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            small_reconstruct(payload)

    def test_budget_whitelist_and_default_iteration_contract(self):
        payload, _ = synthetic_payload()
        with self.assertRaises(ValueError):
            reconstruct(payload, 'wan', 'b', test_iterations=1)
        payload['features'] = torch.zeros(4, 3)
        with self.assertRaisesRegex(ValueError, 'whitelist'):
            small_reconstruct(payload)

    def test_reference_lower_median_once_softmax_and_no_truth_alignment(self):
        blocks = [dict(indices=[0], categories=None), dict(indices=[1, 2], categories=['x', 'y'])]
        members = [dict(raw=torch.tensor([[float(i), float(i), -float(i)]]).repeat(4, 1), final_objective=float(30-i))
                   for i in range(30)]
        chosen, soft, rows = aggregate(members, blocks, 'b')
        expected = torch.stack([torch.cat((m['raw'][:, :1], m['raw'][:, 1:].softmax(1)), 1)
                                for m in members]).median(0).values
        self.assertIsNone(chosen)
        self.assertTrue(torch.equal(soft, expected))
        self.assertEqual(float(rows[0, 0]), 14.)
        self.assertEqual(rows[:, 1:].argmax(1).tolist(), [0]*4)

    def test_serial_spawn_identical_results(self):
        payload, _ = synthetic_payload()
        serial = small_reconstruct(payload)
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn')) as pool:
            parallel = list(pool.map(small_reconstruct, [payload, payload]))
        def equal(left, right):
            if torch.is_tensor(left):
                return torch.equal(left, right)
            if isinstance(left, dict):
                return set(left) == set(right) and all(equal(v, right[k]) for k, v in left.items())
            if isinstance(left, list):
                return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
            return left == right
        self.assertTrue(all(equal(serial, p) for p in parallel))


if __name__ == '__main__':
    unittest.main()
