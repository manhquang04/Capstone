"""Pre-run synthetic checks, no benchmark arm measurements."""
import multiprocessing as mp
import time
import unittest
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from experiments import priority35b_b2b as b
from experiments.priority35b_b2b_analysis import match_union

class B2bTests(unittest.TestCase):
    def test_original_noise_bit_exact_c001(self):
        b.configure();t=b.torch
        state=OrderedDict(w=t.tensor([3.,4.]),bias=t.tensor([.2]))
        for e in (10,1000,1000000):
            # Historical extended-grid implementation, same private fixture seed.
            original=b.u.local.account;b.u.local.account=b.u.account
            try: old,_=b.u.local.noise_upload(state,'global',e,999,1,0)
            finally: b.u.local.account=original
            actual,_=b.upload_noise(state,.01,e,999,1,0)
            self.assertTrue(all(t.equal(old[k],actual[k]) for k in old))
            self.assertEqual(b.accounting(.1,e)['noise_sd'],10*b.accounting(.01,e)['noise_sd'])

    def test_clip_only_no_rng(self):
        t=b.torch;t.manual_seed(991);before=t.get_rng_state().clone()
        state=OrderedDict(w=t.tensor([3.,4.]))
        got,r=b.upload_noise(state,.1,None,None,1,0)
        expected,_=b.u.local.old.clip(state,'global',radius=.1)
        self.assertTrue(t.equal(before,t.get_rng_state()));self.assertTrue(t.equal(got['w'],expected['w']))
        self.assertEqual(r['noise_sd'],0);self.assertFalse(r['gaussian_draws'])
        with self.assertRaises(FloatingPointError):b.upload_noise(OrderedDict(w=t.tensor([float('nan')])),.1,None,None,1,0)

    def test_serial_spawn_identical(self):
        np=b.u.local.np;folder=b.OUT/'synthetic_tests'/str(time.time_ns());prepared=folder/'data/prepared/synthetic';prepared.mkdir(parents=True)
        rng=np.random.RandomState(44)
        for split,n in (('train',120),('validation',24),('test',24)):
            np.save(prepared/f'{split}_x.npy',rng.normal(size=(n,6)).astype(np.float32));np.save(prepared/f'{split}_y.npy',(np.arange(n)%3==0).astype(np.float32))
        np.save(prepared/'train_categories.npy',np.asarray(['A','B','C']*40))
        tasks=[(str(folder/'data'),str(folder/f'parallel_{e}'),e) for e in (None,10)]
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool: parallel=list(pool.map(b.synthetic_fixture,tasks))
        serial=[b.synthetic_fixture((str(folder/'data'),str(folder/f'serial_{e}'),e)) for e in (None,10)]
        self.assertEqual(serial,parallel);self.assertTrue(all(x['threads']==x['interop']==1 for x in serial))

    def test_union_matches_old_rule(self):
        d=b.u.design;target={s:.8 for s in d.SEEDS};grid={e:{s:(.7 if e<100 else .81) for s in d.SEEDS} for e in d.EPSILONS}
        pts=[dict(method=str(e),epsilon=e,C=.01,noise_sd=d.account(e)['noise_sd'],f1=grid[e]) for e in d.EPSILONS]
        self.assertEqual(match_union(target,pts)['epsilon'],d.match(target,grid)['epsilon'])
        # Equal-SD clip-only controls are one tier; never imply a DP guarantee.
        pts=[dict(method='noisy',epsilon=10,C=.1,noise_sd=.2,f1={s:.7 for s in d.SEEDS}),dict(method='clip',epsilon=None,C=.1,noise_sd=0.,f1={s:.8 for s in d.SEEDS})]
        self.assertEqual(match_union(target,pts)['status'],'MATCHED_CLIP_ONLY_NO_DP')
        pts[0]['f1']={s:.8 for s in d.SEEDS}
        self.assertEqual(match_union(target,pts)['status'],'NOT_ASSESSABLE')
        pts[1]['f1'].pop(d.SEEDS[0])
        with self.assertRaises(ValueError):match_union(target,pts)

    def test_job_count_and_baseline_values(self):
        values=b.u.read(b.telemetry.OUT/'baseline_clip_values.json')['values'];jobs=b.job_list(values)
        self.assertEqual(len(jobs),429);self.assertEqual(sum(j['epsilon'] is None for j in jobs),132)
        self.assertTrue(all(j['C']==.01 for j in jobs if j['C_tag']=='c001'))

if __name__=='__main__':unittest.main()
