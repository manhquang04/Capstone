"""Synthetic correctness and deterministic-pool tests, no benchmark measurement."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import unittest
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import torch
from tests.test_priority35a_d1prime import synthetic_payload
from experiments.priority35a_d1prime_estimator import fit
from experiments.priority35a_sanity_t_estimator import initial,fit_initial,reconstruct,at_rows,ratio
from experiments.priority35a_estimator import flatten,schema_blocks
from experiments.priority35a_verify import independent_quality


def small_t(args):
    payload,truth=args
    result=reconstruct(payload,truth,[1.,1.,1.],'tableak',.1,synthetic=True,test_iterations=2)
    result.pop('elapsed_seconds')
    return result


def equal(a,b):
    if torch.is_tensor(a):
        return torch.equal(a,b)
    if isinstance(a,dict):
        return set(a)==set(b) and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,list):
        return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


class SanityTTests(unittest.TestCase):
    def test_identical_fitting_to_frozen_d1prime(self):
        payload,_=synthetic_payload()
        blocks=schema_blocks(payload['meta'])
        seed=143
        raw=torch.randn(4,3,generator=torch.Generator().manual_seed(seed))
        old=fit(payload['shared'],flatten(payload['gradient']),payload['bn'],payload['masks'],blocks,payload['labels'],'tableak',seed,3)
        new=fit_initial(payload,blocks,'tableak',seed,3,raw)
        self.assertTrue(equal(old,new))

    def test_truth_forward_exact_and_metric_one(self):
        payload,private=synthetic_payload()
        gradient,value=at_rows(payload,private['features'],'tableak')
        self.assertTrue(torch.equal(gradient,flatten(payload['gradient'])))
        self.assertLessEqual(abs(value),1e-6)
        stats=dict(blocks=schema_blocks(payload['meta']),tolerances=[.319]*3,std=[1.]*3)
        self.assertEqual(independent_quality(private['features'],private['features'],stats)['feature_accuracy'],1.)

    def test_noise_scaling_zero_sd_and_categorical_mapping(self):
        blocks=[dict(indices=[0],categories=None),dict(indices=[1,2],categories=['a','b'])]
        truth=torch.tensor([[2.,1.,0.]]).repeat(4,1)
        result=initial(truth,[0.,1.,1.],blocks,.1,42)
        self.assertTrue(torch.equal(result['noise'][:,0],torch.zeros(4)))
        self.assertTrue(torch.equal(result['raw'][:,0],truth[:,0]))
        values=result['perturbed'][:,1:].clamp_min(1e-6)
        self.assertTrue(torch.equal(result['raw'][:,1:],(values/values.sum(1,keepdim=True)).log()))
        self.assertTrue(torch.isfinite(result['raw']).all())

    def test_ratio_zero_is_na_not_infinity(self):
        self.assertIsNone(ratio(1.,0.)[0])
        self.assertIsNone(ratio(1.,-1e-7)[0])
        self.assertEqual(ratio(2.,.1),(20.,None))

    def test_independent_native_forward_and_objective(self):
        from experiments.priority35a_sanity_t_verify import independent_gradient,gradient_objective
        payload,private=synthetic_payload()
        public=dict(shared=payload['shared'])
        gradient=independent_gradient(public,private,payload['masks'],private['features'])
        expected,value=at_rows(payload,private['features'],'tableak')
        self.assertTrue(torch.equal(gradient,expected))
        self.assertEqual(value,gradient_objective(public,private,payload['masks'],private['features'],'tableak',payload['gradient']))

    def test_fail_closed_and_budget(self):
        payload,private=synthetic_payload()
        with self.assertRaises(ValueError):
            reconstruct(payload,private['features'],[1.]*3,'wan',.1,test_iterations=1)
        payload['bn']['network.1.running_var'][0]=-1.
        with self.assertRaises(ValueError):
            small_t((payload,private['features']))
        with self.assertRaises(ValueError):
            initial(torch.full((4,3),float('nan')),[1.]*3,schema_blocks(payload['meta']),.1,1)

    def test_serial_spawn_identical(self):
        payload,private=synthetic_payload()
        args=(payload,private['features'])
        serial=small_t(args)
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
            parallel=list(pool.map(small_t,[args,args]))
        self.assertTrue(all(equal(serial,p) for p in parallel))


if __name__=='__main__':
    unittest.main()
