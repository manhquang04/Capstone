import unittest
import numpy as np
from scipy.stats import binomtest
from experiments import priority34d_statistics as s


class StatisticsTests(unittest.TestCase):
    def test_family(self):
        rows=s.family()
        self.assertEqual(len(rows),76)
        self.assertEqual(sum(row['instrument']=='image' for row in rows),60)
        self.assertEqual(sum(row['instrument']=='BN' for row in rows),16)

    def test_rank39(self):
        ci=s.median_interval(np.arange(39))
        self.assertEqual((ci['lower_rank'],ci['upper_rank']),(13,27))
        self.assertGreaterEqual(ci['coverage'],.95)

    def test_pooled_ranks(self):
        ci=s.median_interval(np.arange(117))
        self.assertNotEqual((ci['lower_rank'],ci['upper_rank']),(13,27))
        self.assertGreaterEqual(ci['coverage'],.95)

    def test_exact_sign_independent(self):
        for n in (1,8,24,39,117):
            for wins in range(n+1):
                a=np.concatenate([np.ones(wins),-np.ones(n-wins),np.zeros(3)])
                row=s.exact_sign(a,np.zeros(n+3),'dna_greater')
                self.assertAlmostEqual(row['p_raw'],binomtest(wins,n,.5,alternative='greater').pvalue)
                self.assertEqual(row['ties'],3)

    def test_holm(self):
        self.assertTrue(np.allclose(s.holm([.01,.04,.03,1.]),[.04,.09,.09,1.]))

    def test_ties(self):
        self.assertEqual(s.exact_sign(np.zeros(39),np.zeros(39),'dna_less')['p_raw'],1.)

    def test_no_partial(self):
        with self.assertRaises(ValueError):
            s.comparison(s.family()[0],np.zeros(38),np.ones(38))

    def test_combined211(self):
        rows=[s.comparison(spec) for spec in s.family()]
        prior=[dict(hypothesis_id='prior'+str(i),p_raw=.5) for i in range(135)]
        own,combined=s.finish(rows,prior)
        self.assertEqual(len(own),76)
        self.assertEqual(len(combined),211)
        self.assertTrue(all(row['holm_p34d']==1. for row in own))


if __name__=='__main__':
    unittest.main()
