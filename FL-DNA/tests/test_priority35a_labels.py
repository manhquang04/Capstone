"""Synthetic label-rule tests, not real-record privacy measurements."""
import unittest
import torch
from experiments.fraud_fl_common import BinaryFocalLoss
from experiments.priority35a_adam import fresh_adam_parameters
from experiments.priority35a_labels import canonical_labels, estimate_label_count


def synthetic_bias_upload(labels, logit=0.):
    # A public constant-logit synthetic model, with no unknown feature or BN.
    bias = torch.tensor([logit], dtype=torch.float64, requires_grad=True)
    logits = bias.expand(len(labels), 1)
    loss = BinaryFocalLoss(alpha=0.95, gamma=2.)(logits, labels.double())
    gradient, = torch.autograd.grad(loss, (bias,))
    updated = fresh_adam_parameters({"bias": bias}, {"bias": gradient})["bias"]
    return (updated - bias).detach()


class LabelRuleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_all_counts_with_focal_and_exact_adam(self):
        for batch in (4, 8, 16):
            for count in range(batch + 1):
                truth = synthetic_bias_upload(canonical_labels(batch, count))
                labels, receipt = estimate_label_count(batch, lambda candidate:
                    (synthetic_bias_upload(candidate) - truth).square().sum())
                self.assertEqual(receipt["estimated_fraud_count"], count)
                self.assertEqual(int(labels.sum()), count)
                self.assertEqual(len(receipt["candidates"]), batch + 1)

    def test_aggregate_sign_is_ambiguous_for_mixed_batches(self):
        one = synthetic_bias_upload(canonical_labels(8, 1))
        four = synthetic_bias_upload(canonical_labels(8, 4))
        self.assertTrue(torch.equal(one.sign(), four.sign()))
        self.assertFalse(torch.equal(one, four))

    def test_permutation_invariance_and_tie_receipt(self):
        labels = canonical_labels(8, 3)
        self.assertTrue(torch.equal(synthetic_bias_upload(labels),
                                    synthetic_bias_upload(labels.flip(0))))
        result, receipt = estimate_label_count(4, lambda candidate: 1.)
        self.assertEqual(int(result.sum()), 0)
        self.assertEqual(receipt["exact_tied_counts"], list(range(5)))

    def test_no_bad_count_exclusion(self):
        with self.assertRaises(ValueError):
            estimate_label_count(4, lambda labels: float("nan") if labels.sum() == 2 else 0.)
        with self.assertRaises(ValueError):
            canonical_labels(4, 5)
        with self.assertRaises(ValueError):
            estimate_label_count(True, lambda labels: 0.)


if __name__ == "__main__":
    unittest.main()
