import math
import unittest
import tempfile
from pathlib import Path
from collections import OrderedDict
from unittest.mock import patch
from scipy.optimize import minimize_scalar
from experiments import priority34b_core as b
from experiments import priority34b_analysis as analysis
from experiments import priority34b_images as image
torch=b.torch


class MeaningfulDPTests(unittest.TestCase):
    def state(self):
        return OrderedDict(a=torch.tensor([3.,4.]),bb=torch.tensor([12.]))

    def test_accounting_independent_optimizer(self):
        for epsilon in [1,3,10]:
            row=b.account(epsilon)
            sigma=row['sigma_sensitivity']
            objective=lambda alpha:50*alpha/(2*sigma*sigma)+math.log(1e5)/(alpha-1)
            independent=minimize_scalar(objective,bounds=(1.0001,5000),method='bounded')
            self.assertAlmostEqual(independent.fun,epsilon,places=8)
            self.assertEqual(row['q'],1)

    def test_composition_not_single_round(self):
        self.assertAlmostEqual(b.account(10)['sigma_sensitivity']/b.account(10,rounds=1)['sigma_sensitivity'],math.sqrt(50))

    def test_global_clip(self):
        result,receipt=b.clip(self.state(),'global',radius=1)
        self.assertLessEqual(math.sqrt(sum(float(v.double().norm())**2 for v in result.values())),1)
        self.assertEqual(receipt['before_norm'],13)

    def test_per_tensor_joint_bound(self):
        result,receipt=b.clip(self.state(),'per_tensor',radius=1)
        for value in result.values():
            self.assertLessEqual(float(value.double().norm()),1/math.sqrt(2))
        self.assertLessEqual(receipt['after_norm'],1)

    def test_replace_one_sensitivity(self):
        weights=[.2,.3,.5]; C=b.C
        x=[torch.tensor([C,0.]) for _ in range(3)]
        before=sum(w*v for w,v in zip(weights,x))
        x[2]=-x[2]
        after=sum(w*v for w,v in zip(weights,x))
        self.assertAlmostEqual(float((before-after).norm()),2*max(weights)*C,places=8)

    def test_server_once_noise_deterministic_and_separate_rng(self):
        previous=OrderedDict(a=torch.zeros(3))
        clipped=[OrderedDict(a=torch.zeros(3)) for _ in range(3)]
        torch.manual_seed(123); rng=torch.get_rng_state().clone()
        first,receipt=b.aggregate(clipped,[.2,.3,.5],10,1234,1,previous)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        second,_=b.aggregate(clipped,[.2,.3,.5],10,1234,1,previous)
        self.assertTrue(torch.equal(first['a'],second['a']))
        self.assertAlmostEqual(receipt['noise_sd'],b.account(10)['sigma_sensitivity']*2*.5*b.C)

    def test_nonfinite_rejected(self):
        with self.assertRaises(FloatingPointError):
            b.clip(OrderedDict(a=torch.tensor([float('nan')])),'global')

    def test_bn_domain_and_negative_gate_unchanged(self):
        model=b.p.FraudMLP(13)
        bn,allowed=b.a.domains(model)
        self.assertEqual(len(allowed),8)
        self.assertFalse(set(allowed)&bn)
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
                module.running_var.fill_(-1)
                break
        with self.assertRaises(ValueError):
            b.a.guard(model,'unit negative fixture')

    def test_descriptive_ci_sign_and_holm(self):
        result=analysis.paired([i/100 for i in range(11)])
        self.assertAlmostEqual(result['mean_delta'],.05)
        row=analysis.sign([1.]*39,1)
        self.assertEqual(row['raw_p'],2**-39)
        self.assertEqual(row['order_interval'],[1.,1.])
        rows=[dict(raw_p=.01),dict(raw_p=.04),dict(raw_p=.5)]
        analysis.holm(rows)
        self.assertEqual([r['holm_p'] for r in rows],[.03,.08,.5])

    def test_image_no_bn_and_contiguous(self):
        model=image.image_model()
        self.assertFalse(list(model.buffers()))
        x=torch.randn(2,3,32,32).contiguous()
        with torch.no_grad():
            output=model(x)
        self.assertTrue(torch.isfinite(output).all())

    def test_payload_only_modes_frozen(self):
        self.assertEqual(image.recovery.native.IMAGE_CONFIG['max_iterations'],4800)
        self.assertEqual(image.recovery.native.IMAGE_CONFIG['total_variation'],.01)
        self.assertEqual(image.recovery.native.IMAGE_CONFIG['restarts'],1)

    def test_project_models_namespace_restored(self):
        from models.fraud_mlp import FraudMLP
        self.assertIs(FraudMLP,b.p.FraudMLP)

    def test_synthetic_tabular_wrapper_full50_round_and_resume(self):
        # Synthetic matrices only, never a new research-data outcome.
        with tempfile.TemporaryDirectory(prefix='p34b_unit_') as tmp:
            folder=Path(tmp)
            prepared=folder/'prepared/paysim'; prepared.mkdir(parents=True)
            rng=b.np.random.default_rng(7)
            for split,n in [('train',60),('validation',16),('test',16)]:
                b.np.save(prepared/(split+'_x.npy'),rng.normal(size=(n,13)).astype('float32'))
                b.np.save(prepared/(split+'_y.npy'),b.np.array([0,1]*(n//2),dtype='float32'))
            b.np.save(prepared/'train_categories.npy',b.np.array(['TRANSFER']*60))
            job=dict(stage='tabular',dataset='paysim',method='dp_global_eps10',mechanism='global',
                     epsilon=10,seed=321000,result='jobs/paysim/dp_global_eps10/321000.json')
            with patch.object(b,'OUT',folder),patch.object(b.p,'OUT',folder),patch.object(b,'noise_key',return_value=887766):
                first=b.tabular_job(job)
                self.assertFalse(first['skipped'])
                doc=b.read(folder/job['result'])
                self.assertEqual(len(doc['dp_round_audit']),50)
                self.assertEqual(doc['payload_assertions'],150)
                self.assertEqual(len(doc['clients']),3)
                self.assertTrue(all(row['sensitivity']>0 for row in doc['dp_round_audit']))
                second=b.tabular_job(job)
                self.assertTrue(second['skipped'])


if __name__=='__main__':
    unittest.main()
