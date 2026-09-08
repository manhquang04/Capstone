import torch
from types import SimpleNamespace

from experiments.phase3_followup import decode


def test_categorical_relaxation_preserves_numeric_and_has_derivative():
    z=torch.arange(39,dtype=torch.float64).reshape(3,13).requires_grad_(True)
    x=decode(z,'categorical')
    assert torch.equal(x[:,:8],z[:,:8])
    torch.testing.assert_close(x[:,8:].sum(1),torch.ones(3,dtype=torch.float64))
    assert (x[:,8:]>0).all()
    gradient,=torch.autograd.grad(x[:,8].sum(),z)
    assert torch.isfinite(gradient).all()
    assert gradient[:,8:].abs().sum()>0
    assert torch.equal(gradient[:,:8],torch.zeros_like(gradient[:,:8]))


def test_raw_and_scaled_share_representation():
    x=torch.randn(4,13)
    assert decode(x,'raw') is x
    assert decode(x,'scaled') is x


def test_balance_constraint_uses_raw_units_and_keeps_categories():
    meta=SimpleNamespace(numeric_center=[0,0,10,3,4,9,1,2],numeric_scale=[1,2,3,4,5,6,7,8])
    z=torch.zeros(2,13,dtype=torch.float64,requires_grad=True)
    x=decode(z,'balanced',meta)
    torch.testing.assert_close(x[:,6]*7+1,torch.full((2,),7.,dtype=torch.float64))
    torch.testing.assert_close(x[:,7]*8+2,torch.full((2,),5.,dtype=torch.float64))
    assert torch.equal(x[:,8:],z[:,8:])
    gradient,=torch.autograd.grad(x[:,6].sum(),z)
    torch.testing.assert_close(gradient[:,2],torch.full((2,),3/7,dtype=torch.float64))
    torch.testing.assert_close(gradient[:,3],torch.full((2,),-4/7,dtype=torch.float64))
    assert torch.equal(gradient[:,6],torch.zeros(2,dtype=torch.float64))
