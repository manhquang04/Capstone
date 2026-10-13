"""Synthetic correctness checks for the approved infrastructure-only rule."""
from collections import OrderedDict
import unittest
import torch
from experiments.priority35a_cosine_verifier import cosine_audit, RECORDS


class CosineVerifierTests(unittest.TestCase):
    def fixture(self, zero=False):
        torch.set_num_threads(1)
        g = torch.zeros(4) if zero else torch.tensor([.1, -.2, .3, -.4])
        d = -g
        saved = {k: float(torch.nn.functional.cosine_similarity(v, g.sign(), dim=0, eps=1e-12))
                 for k, v in dict(gradient_vs_sign=g, adam_vs_sign=d, descent_aligned_adam_vs_sign=-d).items()}
        saved.update(zero_gradient_norm=zero, zero_upload_norm=zero)
        return saved, OrderedDict(g=g), OrderedDict(d=d)

    def test_exact_float32_and_float64_crosscheck(self):
        RECORDS.clear()
        cosine_audit(*self.fixture())
        self.assertEqual(len(RECORDS), 3)
        self.assertTrue(all(r['absolute_difference'] <= 1e-5 for r in RECORDS))

    def test_real_stored_mismatch_fails_even_inside_tolerance(self):
        saved, g, d = self.fixture()
        saved['adam_vs_sign'] += 1e-7
        with self.assertRaisesRegex(ValueError, 'exact float32'):
            cosine_audit(saved, g, d)

    def test_zero_nonfinite_and_wrong_zero_flag_fail_closed(self):
        cosine_audit(*self.fixture(zero=True))
        saved, g, d = self.fixture()
        saved['zero_gradient_norm'] = True
        with self.assertRaisesRegex(ValueError, 'zero flag'):
            cosine_audit(saved, g, d)
        saved, g, d = self.fixture()
        g['g'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            cosine_audit(saved, g, d)


if __name__ == '__main__':
    unittest.main()
