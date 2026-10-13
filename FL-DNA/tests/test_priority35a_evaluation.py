import unittest
import numpy as np
from scipy.stats import binomtest
from experiments.priority35a_evaluation import sign_test, gate, reference_statistics, quality, marginal
from experiments.priority35a_checkpoints import exact_states
import torch
from collections import OrderedDict


class EvaluationTest(unittest.TestCase):
    def test_exact_sign_and_intersection_gate(self):
        for values in ([1]*8, [1]*7+[-1], [1]*6+[-1]*2, [0]*8, [1]*3+[0]*5):
            result = sign_test(values)
            n = result["wins"]+result["losses"]
            independent = binomtest(result["wins"], n, alternative="greater").pvalue if n else 1.
            self.assertEqual(result["p_one_sided"], independent)
        rows = [dict(index=i, estimator={"feature_accuracy": 1.}, mean_mode={"feature_accuracy": 0.},
                     empirical_marginal={"feature_accuracy": 0.}, decoy={"feature_accuracy": 1.}) for i in range(8)]
        self.assertFalse(gate(rows, 8)["passed"])
        rows[0]["index"] = 1
        with self.assertRaises(ValueError):
            gate(rows, 8)

    def test_reference_alignment(self):
        training = np.array([[0., 1., 0.], [2., 0., 1.], [4., 1., 0.]])
        meta = dict(feature_names=["x", "c=a", "c=b"], one_hot_levels={"c": ["a", "b"]})
        stats = reference_statistics(training, meta)
        self.assertEqual(stats["prior"], [2., 1., 0.])
        truth = training[:2]
        result = quality(truth, truth[::-1], stats, 3.)
        self.assertEqual(result["feature_accuracy"], 1.)
        self.assertEqual(result["standardized_mse"], 0.)
        self.assertEqual(result["assignment"], [1, 0])
        self.assertTrue(np.array_equal(marginal(training, stats, 4, "fixture"), marginal(training, stats, 4, "fixture")))

    def test_checkpoint_exact_not_approximate(self):
        state = dict(global_non_bn=OrderedDict(x=torch.ones(2)), client_bn=[OrderedDict(y=torch.ones(1)) for _ in range(3)])
        self.assertTrue(exact_states(state, state))
        other = dict(global_non_bn=OrderedDict(x=torch.ones(2)), client_bn=[OrderedDict(y=torch.ones(1)) for _ in range(3)])
        other["global_non_bn"]["x"][0] = torch.nextafter(torch.tensor(1.), torch.tensor(2.))
        with self.assertRaises(ValueError):
            exact_states(state, other)
