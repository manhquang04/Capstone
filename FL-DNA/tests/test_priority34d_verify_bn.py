import unittest
from unittest.mock import patch
import numpy as np
from scipy.stats import binomtest
from experiments import priority34d_verify_bn as verifier


class BNVerifierTests(unittest.TestCase):
    def test_comparison_precision(self):
        verifier.close([1.],[1.+1e-8],'array')
        with self.assertRaises(ValueError):
            verifier.close([1.],[1.1],'array')

    def test_nonfinite_shape(self):
        for a,b in (([np.inf],[1.]),([1.],[1.,2.])):
            with self.assertRaises(ValueError):
                verifier.close(a,b)

    def test_exact_control_sign(self):
        left=np.r_[np.ones(21),-np.ones(3),np.zeros(2)]
        doc=verifier.sign(left,np.zeros(26))
        self.assertEqual((doc['wins'],doc['losses'],doc['ties']),(21,3,2))
        self.assertEqual(doc['p'],binomtest(21,24,.5,alternative='greater').pvalue)

    def test_all_ties(self):
        self.assertEqual(verifier.sign(np.ones(8),np.ones(8))['p'],1.)

    def test_no_early_target_access(self):
        with patch.object(verifier.d,'read',side_effect=FileNotFoundError('stage not complete')):
            with patch.object(verifier.d,'model') as model:
                with self.assertRaises(FileNotFoundError):
                    verifier.verify()
                model.assert_not_called()


if __name__=='__main__':
    unittest.main()
