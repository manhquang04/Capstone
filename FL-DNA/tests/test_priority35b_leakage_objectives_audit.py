"""Synthetic native objective checks; no new optimization or real target access."""
import unittest
from types import SimpleNamespace
from experiments import priority35b_leakage as b
from experiments import priority35b_leakage_objectives_audit as audit


class Tests(unittest.TestCase):
    def test_native_plain_objective_exact(self):
        b.configure();net=b.core.native.make_model(13)
        x=b.torch.linspace(-.1,.1,13).reshape(1,13);labels=b.torch.tensor([1])
        raw,_=b.core.gradient(net,x,labels)
        target=[g+.001 for g in raw.values()]
        native=float(b.core.native.gia._cosine_similarity_loss(list(raw.values()),target,'cpu'))
        self.assertEqual(audit.cosine_plain(list(raw.values()),target),native)
        data=SimpleNamespace(standardized=True,project_batch=lambda member,standardized:member)
        c=dict(name='plain',ensemble=[x.clone() for _ in range(30)],objectives=[native]*30,objective=native)
        self.assertEqual(audit.verify_native(net,data,labels,target,None,[c]),0.)
        c['objectives'][0]+=.01
        with self.assertRaises(ValueError):audit.verify_native(net,data,labels,target,None,[c])

    def test_debias_exact(self):
        from experiments.priority33b_comparators import debias_blocks
        values=[b.torch.linspace(-1.,1.,301)]
        for x,y in zip(audit.debias(values),debias_blocks(values)):
            b.torch.testing.assert_close(x,y,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
