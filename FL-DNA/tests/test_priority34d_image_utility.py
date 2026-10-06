"""Synthetic-only checks: no benchmark data or recovery targets."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments import priority34d_image_utility as u


class UtilityTests(unittest.TestCase):
    def test_schedule(self):
        jobs = u.jobs()
        self.assertEqual(len(jobs),1216)
        self.assertEqual(len({j['id'] for j in jobs}),1216)
        for setting in u.SETTINGS:
            for method in ['baseline']+u.DNA:
                self.assertEqual(sum(j['setting']==setting and j['method']==method for j in jobs),16)

    def test_extension(self):
        jobs = u.jobs(u.EXTENSION,('single',),['trained_batch4'],False)
        self.assertEqual(len(jobs),48)
        self.assertTrue(all(j['setting']=='trained_batch4' and j['mechanism']=='single' for j in jobs))

    def test_bracket(self):
        row = u.bracket(.45,-.02,{.001:-.01,.003:-.024,.01:-.026})
        self.assertEqual(row['status'],'MATCHED')
        self.assertEqual(row['sigma'],.003)
        self.assertEqual(row['next_sigma'],.01)

    def test_baseline_gate(self):
        row = u.bracket(.39,-.02,{.001:0,.003:-.04})
        self.assertEqual(row['status'],'NOT_ASSESSABLE')
        self.assertFalse(row['extension_needed'])

    def test_extension_trigger(self):
        self.assertTrue(u.bracket(.45,-.02,{.001:0,.003:-.01})['extension_needed'])
        self.assertFalse(u.bracket(.45,-.02,{.001:-.1,.003:-.2})['extension_needed'])

    def test_nonfinite(self):
        for v in [np.nan,np.inf,-np.inf]:
            with self.assertRaises(FloatingPointError):
                u.bracket(.45,-.02,{.001:v})

    def test_threads(self):
        self.assertEqual(torch.get_num_threads(),1)
        self.assertEqual(torch.get_num_interop_threads(),1)

    def test_synthetic_probe(self):
        model = torch.nn.Linear(2,2)
        before = {k:v.clone() for k,v in model.state_dict().items()}
        x = torch.tensor([[0.,1.],[1.,0.],[1.,1.],[-1.,0.]])
        y = torch.tensor([0,1,0,1])
        clip = u.probe(model,x,y)
        self.assertEqual(len(clip['global_norms']),2)
        self.assertEqual(len(clip['clips']),2)
        self.assertTrue(all(c>0 for c in clip['clips']))
        for k,v in model.state_dict().items():
            self.assertTrue(torch.equal(v,before[k]))

    def test_nonfinite_probe(self):
        model = torch.nn.Linear(2,2)
        with self.assertRaises(FloatingPointError):
            u.probe(model,torch.full((2,2),float('nan')),torch.tensor([0,1]))

    def test_write_firewall(self):
        with self.assertRaises(ValueError):
            u.write(u.ROOT/'artifacts/priority33c/forbidden.json',{})

    def test_negative_bn(self):
        model = torch.nn.BatchNorm1d(2)
        model.running_var[0] = -1
        with self.assertRaises(FloatingPointError):
            u.check(model)

    def test_no_truth_in_jobs(self):
        self.assertTrue(all(set(j)=={'id','setting','method','mechanism','sigma','seed'} for j in u.jobs()))


if __name__ == '__main__':
    unittest.main()
