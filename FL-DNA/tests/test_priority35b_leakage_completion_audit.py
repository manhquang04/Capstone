"""Synthetic ratio/pooling correctness checks; no measurement data accessed."""
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from experiments import priority35b_leakage as b
from experiments import priority35b_leakage_completion_audit as audit


class Tests(unittest.TestCase):
    def test_ratio_formula_receipt_and_objective(self):
        b.configure();net=b.core.ratio_model(13)
        x=b.torch.linspace(-.2,.2,13).reshape(1,13);labels=b.torch.tensor([1])
        raw,_=b.core.gradient(net,x,labels,ratio=True)
        data=SimpleNamespace(mean=b.torch.zeros(13),std=b.torch.ones(13))
        candidates,_=b.reference.recover(net,data,list(raw),list(raw.values()),labels,1,'unprotected',None,None,True)
        server=dict(names=list(raw),payload=list(raw.values()),metadata=None,labels=labels)
        audit.ratio_check(b,net,data,server,candidates[0])
        candidates[0]['objective']+=.01
        with self.assertRaises(ValueError):audit.ratio_check(b,net,data,server,candidates[0])

    def test_pooling_alignment_reference(self):
        from attacks.ensembling import pooled_ensemble
        data=SimpleNamespace(standardized=False,mean=b.torch.zeros(2),std=b.torch.ones(2),train_features=OrderedDict(x=None,y=None),
            train_feature_index_map=OrderedDict(x=('cont',[0]),y=('cont',[1])),
            decode_batch=lambda x:x.numpy().copy(),create_tolerance_map=lambda:[.319,.319])
        members=[b.torch.tensor([[0.,0.],[2.,2.]]).flip(0) if i%2 else b.torch.tensor([[0.,0.],[2.,2.]]) for i in range(30)]
        rec=pooled_ensemble(members,members[0],data,pooling='median+softmax')
        c=dict(ensemble=members,objectives=[0.]*30,reconstruction=rec)
        audit.pooling_check(data,c)
        c['reconstruction']=rec+.01
        with self.assertRaises(ValueError):audit.pooling_check(data,c)


if __name__=='__main__':unittest.main()
