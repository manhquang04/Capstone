import unittest
import numpy as np
from experiments import priority34d_independent_bn as independent
from experiments import priority34d_bn as runner


class IndependentBNTests(unittest.TestCase):
    def test_dense_projection(self):
        for size in (13,58,128):
            q,meta=runner.b.v2.transform_update_array_v2(np.arange(size,dtype=np.float32),runner.b.V2)
            np.testing.assert_allclose(independent.projection(meta),runner.b.projection(meta),atol=1e-14,rtol=1e-14)

    def test_raw_closed_form(self):
        rng=np.random.default_rng(123)
        w=rng.normal(size=(16,5)); b=rng.normal(size=16); mean=rng.normal(size=16)
        truth=rng.normal(size=5); delta=.1*(w@truth+b-mean)
        recovered=independent.decode(w,b,mean,.1,dict(kind='raw',q=delta))
        np.testing.assert_allclose(recovered,truth,atol=1e-12)

    def test_sketch_through_linear_system(self):
        rng=np.random.default_rng(124)
        w=rng.normal(size=(16,5)); b=rng.normal(size=16); mean=rng.normal(size=16)
        truth=rng.normal(size=5); delta=.1*(w@truth+b-mean)
        _,meta=runner.b.v2.transform_update_array_v2(delta.astype(np.float32),runner.b.V2)
        matrix=independent.projection(meta)
        recovered=independent.decode(w,b,mean,.1,dict(kind='v2',q=matrix@delta,metadata=meta))
        np.testing.assert_allclose(recovered,truth,atol=1e-12)

    def test_distortion_decoder_only(self):
        q,meta=runner.b.v2.transform_update_array_v2(np.arange(58,dtype=np.float32),runner.b.V2)
        expected,_=runner.b.v2.reconstruct_update_array_v2(q,meta)
        np.testing.assert_allclose(independent.defender_decode_for_distortion(dict(kind='v2',q=q,metadata=meta)),expected,atol=1e-6)

    def test_privacy_firewall(self):
        with self.assertRaises(ValueError):
            independent.decode(np.eye(2),np.zeros(2),np.zeros(2),.1,
                               dict(kind='raw',q=np.zeros(2),truth=np.zeros(2)))

    def test_nonfinite(self):
        with self.assertRaises(ValueError):
            independent.decode(np.eye(2),np.zeros(2),np.zeros(2),.1,dict(kind='raw',q=[np.nan,0]))


if __name__=='__main__':
    unittest.main()
