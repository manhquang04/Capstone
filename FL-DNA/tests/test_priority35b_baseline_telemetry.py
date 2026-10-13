import multiprocessing as mp
import time
import unittest
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from experiments import priority35b_baseline_telemetry as b

class BaselineTelemetryTests(unittest.TestCase):
    def test_copy_norm_rng_dtype_and_input_preservation(self):
        t=b.torch;b.configure()
        before=OrderedDict(x=t.tensor([1.,2.]),y=t.tensor([3.]))
        after=OrderedDict(x=t.tensor([4.,6.]),y=t.tensor([3.]))
        originals=[v.clone() for v in [*before.values(),*after.values()]]
        rng=t.get_rng_state().clone()
        self.assertEqual(b.norm_copy(after,before),5.)
        self.assertTrue(t.equal(rng,t.get_rng_state()))
        for old,new in zip(originals,[*before.values(),*after.values()]):
            self.assertTrue(t.equal(old,new));self.assertEqual(old.dtype,new.dtype);self.assertEqual(old.device,new.device)

    def test_nonfinite_fail_closed(self):
        t=b.torch
        with self.assertRaises(ValueError):b.norm_copy(OrderedDict(x=t.tensor([float('nan')])),OrderedDict(x=t.tensor([0.])))

    def test_serial_parallel_and_uninstrumented_bit_exact(self):
        np=b.u.local.np;folder=b.OUT/'synthetic_tests'/str(time.time_ns())
        prepared=folder/'data/prepared/synthetic';prepared.mkdir(parents=True)
        rng=np.random.RandomState(17)
        for split,n in [('train',120),('validation',24),('test',24)]:
            np.save(prepared/f'{split}_x.npy',rng.normal(size=(n,6)).astype(np.float32))
            np.save(prepared/f'{split}_y.npy',(np.arange(n)%3==0).astype(np.float32))
        np.save(prepared/'train_categories.npy',np.array(['A','B','C']*40))
        data=str(folder/'data')
        with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn')) as pool:
            parallel=pool.submit(b.synthetic_fixture,(data,str(folder/'parallel'),True)).result()
        serial=b.synthetic_fixture((data,str(folder/'serial'),True))
        plain=b.synthetic_fixture((data,str(folder/'plain'),False))
        self.assertEqual(parallel,serial)
        self.assertEqual({k:v for k,v in serial.items() if k!='norms'},{k:v for k,v in plain.items() if k!='norms'})
        self.assertEqual(len(serial['norms']),6)
        self.assertEqual(serial['threads'],1);self.assertEqual(serial['interop'],1)
        current=b.u.read(folder/'serial/jobs/synthetic/321000.json')
        reference=b.u.read(folder/'plain/jobs/synthetic/321000.json')
        self.assertEqual(b.exact_pair(current,reference,folder/'serial',folder/'plain')['probability_arrays'],6)
        altered=dict(current,test={'f1':-1})
        with self.assertRaises(ValueError):b.exact_pair(altered,reference,folder/'serial',folder/'plain')

if __name__=='__main__':unittest.main()
