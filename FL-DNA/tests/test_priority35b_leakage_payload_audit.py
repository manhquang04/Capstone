"""Synthetic independent mechanism/gradient/budget checks only."""
import unittest
from collections import OrderedDict
from experiments import priority35b_leakage as driver
from experiments import priority35b_leakage_payload_audit as audit


class Tests(unittest.TestCase):
    def test_independent_dp_bytes_and_receipts(self):
        driver.configure()
        raw=OrderedDict(a=driver.torch.tensor([3.,4.]),b=driver.torch.tensor([0.,12.]))
        for mode in ('global','per_tensor'):
            for sd in (0.,.003):
                arm=dict(C=.1,clipping=mode,noise_sd=sd,epsilon=None if not sd else 1000)
                x,r=driver.clip_payload(raw,arm,42)
                y,s=audit.independent_dp(raw,arm,42)
                self.assertEqual(r,s)
                for a,b in zip(x,y):audit.tensor_equal(a,b)

    def test_independent_gradient(self):
        driver.configure()
        for ratio in (True,False):
            net=driver.core.ratio_model(13) if ratio else driver.core.native.make_model(13)
            x=driver.torch.linspace(-.1,.1,13).reshape(1,13);labels=driver.torch.tensor([1])
            raw,receipt=driver.core.gradient(net,x,labels,ratio=ratio)
            separate,omitted,loss=audit.independent_gradient(net,x,labels,ratio)
            self.assertEqual(list(raw),list(separate));self.assertEqual(receipt['loss'],loss)
            self.assertEqual(receipt['omitted_bn_parameters'],omitted)
            for name in raw:audit.tensor_equal(raw[name],separate[name])

    def test_fail_closed_and_signed_zero(self):
        with self.assertRaises(ValueError):audit.finite_tree({'running_var':driver.torch.tensor([-1.])})
        with self.assertRaises(ValueError):audit.tensor_equal(driver.torch.tensor([0.]),driver.torch.tensor([-0.]))
        with self.assertRaises(ValueError):audit.finite_tree([float('nan')])

    def test_selection_and_budget(self):
        c=[dict(name='plain',objective=2.,reconstruction=driver.torch.ones(1),ensemble=[],objectives=[]),
           dict(name='structure_debias',objective=1.,reconstruction=driver.torch.ones(1),ensemble=[],objectives=[])]
        receipt=dict(candidates=c,selected=1);result=dict(selected_variant='structure_debias')
        audit.audit_candidate_receipt(receipt,result,True,'dna_v1_conservative')
        receipt['selected']=0
        with self.assertRaises(ValueError):audit.audit_candidate_receipt(receipt,result,True,'dna_v1_conservative')


if __name__=='__main__':unittest.main()
