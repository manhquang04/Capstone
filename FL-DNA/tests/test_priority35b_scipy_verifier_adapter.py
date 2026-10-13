import unittest
from experiments import priority35b_scipy_verifier_adapter as a


class SciPyAdapterTests(unittest.TestCase):
    def test_float64_only_crosscheck(self):
        self.assertTrue(a.close(.5,.5000000000000002))
        self.assertFalse(a.close(.5,.500001))

    def test_threshold_margin_fail_closed(self):
        for p in (.05,.05+1e-15):
            with self.assertRaises(ValueError):a.check_margin(p)
        a.check_margin(.0625);a.check_margin(.01969072292558849)

    def test_exact_counts_and_family(self):
        from experiments import priority35b_design as d
        pairs={};reasons={k[:-1]:'NA' for k in d.family()}
        key=('ieee_cis','tableak_batch1','dna_v2_0p95','global','bounded_mse')
        reasons.pop(key);pairs[key]=([0.]*19+[1.]*20,[.5]*39)
        rows=a.stats.evaluate(pairs,reasons)
        self.assertEqual(a.independently_verify(rows,pairs,reasons)['status'],'PASS')
        rows[0]['raw_p']=.2
        with self.assertRaises(ValueError):a.independently_verify(rows,pairs,reasons)


if __name__=='__main__':unittest.main()
