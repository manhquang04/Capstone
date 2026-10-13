"""Utility integration tests on synthetic data only, before execution freeze."""
import multiprocessing as mp
import time
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from experiments import priority35b_utility as u
from experiments.priority35b_utility_analysis import summary


class UtilityTests(unittest.TestCase):
    def test_serial_spawn_bit_exact_training(self):
        np=u.local.np
        folder=u.OUT/'synthetic_tests'/str(time.time_ns())
        prepared=folder/'data/prepared/synthetic';prepared.mkdir(parents=True)
        rng=np.random.RandomState(44)
        for split,n in (('train',120),('validation',24),('test',24)):
            np.save(prepared/f'{split}_x.npy',rng.normal(size=(n,6)).astype(np.float32))
            np.save(prepared/f'{split}_y.npy',(np.arange(n)%3==0).astype(np.float32))
        np.save(prepared/'train_categories.npy',np.asarray(['A','B','C']*40))
        tasks=[(str(folder/'data'),str(folder/f'parallel_{c}'),c) for c in ('global','per_tensor')]
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
            parallel=list(pool.map(u.synthetic_identity_fixture,tasks))
        serial=[u.synthetic_identity_fixture((str(folder/'data'),str(folder/f'serial_{c}'),c)) for c in ('global','per_tensor')]
        self.assertEqual(parallel,serial)
        self.assertTrue(all(x['threads']==x['inter_op']==1 for x in parallel))

    def test_account_extended_grid_same_historical_values(self):
        for epsilon in (1,3,10):
            self.assertEqual(u.account(epsilon)['noise_sd'],u.local.ORIGINAL_ACCOUNT(epsilon)['sigma_sensitivity']*.02)
        for epsilon in u.design.EPSILONS:
            self.assertEqual(u.account(epsilon)['mechanism_location'],'client before transmission')
            self.assertFalse(u.account(epsilon)['record_level_dp'])

    def test_t_interval_recomputed_independently(self):
        from scipy.stats import t
        import math
        values=[i/100 for i in range(-5,6)];doc=summary(values)
        half=float(t.isf(.025,10))*math.sqrt(sum(x*x for x in values)/110)
        self.assertAlmostEqual(doc['ci_high'],half,places=15)
        self.assertEqual(summary([0.]*11)['ci_low'],0)
        with self.assertRaises(ValueError):summary([0.]*10)
        with self.assertRaises(ValueError):summary([float('nan')]*11)


if __name__=='__main__':unittest.main()
