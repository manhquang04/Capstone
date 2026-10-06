import unittest
import numpy as np
from experiments import priority34d_independent_metrics as m


class IndependentTests(unittest.TestCase):
    def test_images(self):
        truth=np.random.default_rng(1).uniform(.1,.8,size=(4,3,32,32)).astype(np.float32)
        means,records,pairing=m.images(truth,truth[[2,0,3,1]]+.01)
        self.assertEqual(pairing,[1,3,0,2])
        self.assertAlmostEqual(means['psnr_db'],40.,places=4)
        self.assertAlmostEqual(means['mse'],.0001,places=7)

    def test_nonfinite_before_clipping(self):
        with self.assertRaises(ValueError):
            m.images(np.zeros((1,3,32,32)),np.full((1,3,32,32),np.inf))

    def test_bn_mse(self):
        self.assertAlmostEqual(m.standardized_mse([1,2],[0,0],[1,2]),1.)

    def test_accounting_monotonic(self):
        self.assertGreater(m.gaussian_epsilon(.001),m.gaussian_epsilon(.01))
        self.assertGreater(m.gaussian_epsilon(.01,3),m.gaussian_epsilon(.01,1))


if __name__=='__main__':
    unittest.main()
