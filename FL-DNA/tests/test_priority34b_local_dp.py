"""Local-release sensitivity, independence and immutable runner integration."""
import copy
import math
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch
from scipy.optimize import minimize_scalar
from experiments import priority34b_local_dp as l
torch,np=l.torch,l.np


class LocalDPTests(unittest.TestCase):
    def state(self): return OrderedDict(a=torch.tensor([3.,4.]),b=torch.tensor([12.]))

    def test_independent_accounting_and_unit(self):
        for epsilon in (1,3,10):
            row=l.account(epsilon); sigma=row['sigma_sensitivity']
            minimum=minimize_scalar(lambda alpha:50*alpha/(2*sigma*sigma)+math.log(1e5)/(alpha-1),
                bounds=(1.00001,5000),method='bounded')
            self.assertAlmostEqual(minimum.fun,epsilon,places=8)
            self.assertEqual(row['sensitivity'],.02)
            self.assertAlmostEqual(row['noise_sd'],2*l.old.C*sigma)
            self.assertIn('NOT record-level',row['unit'])

    def test_noise_rng_separation_and_domains(self):
        torch.manual_seed(5); before=torch.get_rng_state().clone()
        first,_=l.noise_upload(self.state(),'global',10,123,1,0)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        repeat,_=l.noise_upload(self.state(),'global',10,123,1,0)
        other_client,_=l.noise_upload(self.state(),'global',10,123,1,1)
        other_round,_=l.noise_upload(self.state(),'global',10,123,2,0)
        self.assertTrue(torch.equal(first['a'],repeat['a']))
        self.assertFalse(torch.equal(first['a'],other_client['a']))
        self.assertFalse(torch.equal(first['a'],other_round['a']))

    def test_joint_clipping_and_before_transmission(self):
        for mechanism in ('global','per_tensor'):
            noisy,row=l.noise_upload(self.state(),mechanism,10,123,1,0)
            self.assertLessEqual(row['after_norm'],.01)
            self.assertTrue(row['noise_before_transmission'])
            self.assertEqual(row['client'],0)
            self.assertTrue(all(torch.isfinite(x).all() for x in noisy.values()))

    def test_server_average_no_extra_noise(self):
        previous=OrderedDict(a=torch.tensor([2.]))
        uploads=[OrderedDict(a=torch.tensor([x])) for x in (1.,3.,5.)]
        torch.manual_seed(2); before=torch.get_rng_state().clone()
        actual=l.mean_noisy_uploads(uploads,[.2,.3,.5],previous)
        self.assertAlmostEqual(actual['a'].item(),5.6,places=6)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_local_sensitivity_not_weighted(self):
        row=l.account(10)
        central=l.ORIGINAL_ACCOUNT(10)['sigma_sensitivity']*2*l.old.C/3
        self.assertAlmostEqual(row['noise_sd'],3*central)

    def test_nonfinite_and_negative_bn_fail_closed(self):
        with self.assertRaises(FloatingPointError):
            l.noise_upload(OrderedDict(a=torch.tensor([float('nan')])),'global',10,123,1,0)
        model=l.p.FraudMLP(13); model.network[1].running_var.fill_(-1)
        with self.assertRaises(ValueError): l.a.guard(model,'negative synthetic')

    def test_synthetic_same_frozen50round_tabular_runner(self):
        # No research record, gate or tuning outcome; mechanics-only fixture.
        with tempfile.TemporaryDirectory(prefix='p34b_local_unit_') as tmp:
            folder=Path(tmp); prepared=folder/'prepared/paysim'; prepared.mkdir(parents=True)
            rng=np.random.default_rng(17)
            for split,n in (('train',60),('validation',16),('test',16)):
                np.save(prepared/f'{split}_x.npy',rng.normal(size=(n,13)).astype('float32'))
                np.save(prepared/f'{split}_y.npy',np.array([0,1]*(n//2),dtype='float32'))
            np.save(prepared/'train_categories.npy',np.array(['TRANSFER']*60))
            job=dict(dataset='paysim',stage='tabular',mechanism='global',epsilon=10,seed=321000,
                method='local_dp_global_eps10',result='jobs/paysim/local_dp_global_eps10/321000.json')
            (folder/'private_noise_seeds.json').write_text('{"'+job['result']+'":12345}')
            with patch.object(l,'OUT',folder),patch.object(l.p,'OUT',folder):
                first=l.run_job(job); self.assertFalse(first['skipped'])
                doc=l.read(folder/job['result']); self.assertEqual(len(doc['local_upload_audits']),150)
                self.assertFalse(doc['server_noise_added']); self.assertEqual(doc['payload_assertions'],150)
                self.assertTrue(doc['no_bn_transmitted']); self.assertTrue(l.run_job(job)['skipped'])

    def test_noise_distribution_scale(self):
        zero=OrderedDict(a=torch.zeros(100000))
        noisy,row=l.noise_upload(zero,'global',10,123,1,0)
        self.assertLess(abs(float(noisy['a'].double().std())/row['noise_sd']-1),.015)
        self.assertLess(abs(float(noisy['a'].double().mean()))/row['noise_sd'],.015)

    def test_synthetic_same50round_image_wrapper(self):
        from experiments import priority34b_images as image
        # Exercise the original image loop with synthetic features and a tiny
        # stand-in classifier. No public dataset target or utility result is used.
        with tempfile.TemporaryDirectory(prefix='p34b_local_image_unit_') as tmp:
            folder=Path(tmp); (folder/'split.json').write_text('{"train_ids":[],"validation_ids":[]}')
            generator=torch.Generator().manual_seed(9)
            x=torch.randn(12000,2,generator=generator); y=torch.arange(12000)%2
            vx=torch.randn(16,2,generator=generator); vy=torch.arange(16)%2
            batches=iter([(x,y),(vx,vy)])
            job=dict(dataset='cifar10',stage='image_utility',mechanism='per_tensor',epsilon=10,seed=321000,
                method='local_dp_per_tensor_eps10',result='jobs/cifar10/local_dp_per_tensor_eps10/321000.json')
            (folder/'private_noise_seeds.json').write_text('{"'+job['result']+'":12345}')
            with patch.object(l,'OUT',folder),patch.object(image.p31,'OUT',folder),\
                 patch.object(image.p31.dp,'CIFAR10',return_value=object()),\
                 patch.object(image,'tensors',side_effect=lambda *_:next(batches)),\
                 patch.object(image,'image_model',side_effect=lambda:torch.nn.Linear(2,2)):
                l.run_job(job)
                doc=l.read(folder/job['result'])
                self.assertEqual(len(doc['round_audit']),50)
                self.assertEqual(len(doc['local_upload_audits']),150)
                self.assertTrue(all(not row['server_noise_added'] for row in doc['round_audit']))
                self.assertEqual(doc['accounting']['mechanism_location'],'client before transmission')
                self.assertTrue(l.run_job(job)['skipped'])


if __name__=='__main__': unittest.main()
