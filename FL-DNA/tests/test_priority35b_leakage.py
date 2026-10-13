"""Synthetic B3 mechanism, scoring, fixed family and spawn equivalence checks."""
import hashlib
import math
import multiprocessing as mp
import unittest
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch
from experiments import priority35b_leakage as b

def fixture(mode):
    b.configure();net=b.core.ratio_model(13)
    x=b.torch.linspace(-.2,.2,13).reshape(1,13);labels=b.torch.tensor([1])
    raw,_=b.core.gradient(net,x,labels,ratio=True)
    arm=dict(C=.7420758605003357,clipping=mode,noise_sd=.007445980477175384,epsilon=1000000)
    payload,receipt=b.clip_payload(raw,arm,123456)
    data=SimpleNamespace(mean=b.torch.zeros(13),std=b.torch.ones(13))
    candidates,selected=b.reference.recover(net,data,list(raw),payload,labels,987,'dp_fixture',None,None,True)
    arrays=[g.numpy().tobytes() for g in payload]+[candidates[selected]['reconstruction'].numpy().tobytes()]
    return hashlib.sha256(b''.join(arrays)).hexdigest(),receipt,b.torch.get_num_threads(),b.torch.get_num_interop_threads()

def native_fixture(_):
    """Full original budget on synthetic rows, never a sampled benchmark target."""
    b.configure();data=b.core.PreparedDataset('paysim','tableak_batch2')
    truth=b.torch.zeros(2,data.num_features)
    truth[0]=.03;truth[1]=-.03
    labels=b.torch.tensor([0,1]);net=b.core.native.make_model(data.num_features)
    raw,_=b.core.gradient(net,truth,labels)
    rec,ensemble,losses=b.core.native.native_recover(net,data,list(raw.values()),labels,351777)
    digest=hashlib.sha256(rec.numpy().tobytes()+b''.join(x.numpy().tobytes() for x in ensemble)+b.np.array(losses).tobytes()).hexdigest()
    return digest,len(ensemble),b.torch.get_num_threads(),b.torch.get_num_interop_threads()

class Tests(unittest.TestCase):
    def test_exact_reference_clip_noise(self):
        raw=OrderedDict(a=b.torch.tensor([3.,4.]),z=b.torch.tensor([12.]))
        for mode in b.design.CLIPS:
            arm=dict(C=.01,clipping=mode,noise_sd=b.design.account(10)['noise_sd'],epsilon=10)
            payload,r=b.clip_payload(raw,arm,42)
            gen=b.torch.Generator().manual_seed(42)
            bound=.01/math.sqrt(2) if mode=='per_tensor' else .01
            for g,q in zip(raw.values(),payload):
                n=13. if mode=='global' else float(g.double().norm())
                clipped=(g.double()*min(1.,bound/n)*(1-4*b.np.finfo(b.np.float32).eps)).to(g)
                expected=(clipped.double()+b.torch.randn(g.shape,dtype=b.torch.float64,generator=gen)*arm['noise_sd']).to(g)
                self.assertTrue(b.torch.equal(q,expected))
            self.assertLessEqual(r['after_norm'],.01)

    def test_clip_only_no_randomness_and_fail_closed(self):
        raw=OrderedDict(a=b.torch.ones(3));arm=dict(C=.1,clipping='global',noise_sd=0.,epsilon=None)
        state=b.torch.random.get_rng_state().clone()
        with patch.object(b.torch,'randn',side_effect=AssertionError('no Gaussian draw')):b.clip_payload(raw,arm,None)
        self.assertTrue(b.torch.equal(state,b.torch.random.get_rng_state()))
        with self.assertRaises(FloatingPointError):b.clip_payload(dict(a=b.torch.tensor([float('nan')])),arm,None)
        net=b.core.ratio_model(13);net.network[1].running_var[0]=-1
        with self.assertRaises(FloatingPointError):b.core.strict_model(net)

    def test_serial_spawn_identical(self):
        serial=[fixture(m) for m in b.design.CLIPS]
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
            parallel=list(pool.map(fixture,b.design.CLIPS))
        self.assertEqual(serial,parallel)
        self.assertTrue(all(x[2:]==(1,1) for x in parallel))

    def test_full_native_serial_spawn_identical(self):
        serial=native_fixture(0)
        with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn')) as pool:
            parallel=pool.submit(native_fixture,0).result()
        self.assertEqual(serial,parallel)
        self.assertEqual(serial[1:],(30,1,1))

    def test_metric_native_alignment(self):
        data=SimpleNamespace(train_features=OrderedDict(x=None,c=['a','b']),train_feature_index_map=OrderedDict(x=('cont',[0]),c=('cat',[1,2])),
            standardized=True,standardized_continuous_bounds={'x':(-10,10)},
            decode_batch=lambda x:b.np.array([[float(r[0]),'a' if r[1]>r[2] else 'b'] for r in x],dtype=object),
            create_tolerance_map=lambda:[.319,'cat'])
        truth=b.torch.tensor([[0.,1.,0.],[1.,0.,1.]])
        result=b.metric(data,truth,truth.flip(0))
        self.assertEqual(result['feature_accuracy'],1.)
        self.assertEqual(result['standardized_mse'],0.)
        self.assertEqual(result['assignment'],[1,0])
        score=b.core.native.measure(data,truth,truth.flip(0))
        self.assertEqual(score['accuracy_percent']/100,result['feature_accuracy'])

    def test_arm_job_family_and_matching_coverage(self):
        self.assertEqual(len(b.configs()),9984)
        for d in b.design.DATASETS:
            arms=b.arms(d);self.assertEqual(len(arms),32)
            self.assertEqual(sum(a['C'] is not None and a['epsilon'] is None for a in arms),4)
            for r in b.read(b.UTILITY/'matching.json'):
                if r['dataset']==d and r['matched_arm'] is not None:self.assertIn(r['matched_arm'],[a['method'] for a in arms])
        self.assertEqual(len(b.design.family()),144)
        self.assertEqual(b.design.holm([1.]*144),[1.]*144)

    def test_native_budget_and_observable_interface(self):
        cfg=b.core.native.official_config()
        self.assertEqual(cfg['max_iterations'],1500)
        self.assertEqual(cfg['learning_rates'],.06)
        self.assertEqual(cfg['post_selection'],30)
        import inspect
        self.assertNotIn('truth',inspect.signature(b.reference.recover).parameters)
        self.assertNotIn('noise_seed',inspect.signature(b.reference.recover).parameters)

    def test_firewall_rejects_overlap(self):
        rows=[[i] for i in range(39)]
        self.assertTrue(b.firewall.assert_fresh('baf',rows,[100],range(101)))
        with self.assertRaises(ValueError):b.firewall.assert_fresh('baf',rows,[1],range(101))
        with self.assertRaises(ValueError):b.firewall.assert_fresh('baf',[[0]]*39,[],range(101))

if __name__=='__main__':unittest.main()
