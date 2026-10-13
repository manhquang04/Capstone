import unittest
import numpy as np
from experiments.priority35a_evaluation import reference_statistics, quality, gate
from experiments.priority35a_verify import independent_quality, independent_gate, compare_metrics


class IndependentTest(unittest.TestCase):
    def test_metric_and_gate_independent_paths(self):
        rng = np.random.default_rng(913)
        for categorical in (False, True):
            training = rng.normal(size=(100, 5))
            meta = dict(feature_names=[f"x{i}" for i in range(5)], one_hot_levels={})
            if categorical:
                training[:, 3:] = np.eye(2)[rng.integers(0, 2, size=100)]
                meta = dict(feature_names=["x0", "x1", "x2", "c=a", "c=b"], one_hot_levels={"c": ["a", "b"]})
            stats = reference_statistics(training, meta)
            truth = training[:8]
            guess = training[20:28]
            compare_metrics(quality(truth, guess, stats, 2.), independent_quality(truth, guess, stats, 2.))
            rows = [dict(index=i, estimator={"feature_accuracy": float(i)/8}, mean_mode={"feature_accuracy": .25},
                         empirical_marginal={"feature_accuracy": .125}, decoy={"feature_accuracy": 0.}) for i in range(8)]
            self.assertEqual(gate(rows, 8), independent_gate(rows, 8))
    def test_tampering_fails(self):
        metric = dict(feature_accuracy=.5, standardized_mse=1., bounded_score=1., assignment=[0], absent_type_denominator=4)
        bad = dict(metric, feature_accuracy=.6)
        with self.assertRaises(ValueError):
            compare_metrics(metric, bad)
