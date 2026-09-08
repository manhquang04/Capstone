import torch

from experiments.phase3_bounded_validation import _align_for_evaluation, _binomial_tail, _decode, update_objective


def test_decode_preserves_numeric_and_relaxes_categories():
    latent=torch.randn(4,13,dtype=torch.float64,requires_grad=True)
    decoded=_decode(latent)
    torch.testing.assert_close(decoded[:,:8],latent[:,:8])
    torch.testing.assert_close(decoded[:,8:].sum(1),torch.ones(4,dtype=torch.float64))
    decoded.sum().backward(); assert latent.grad is not None


def test_update_objective_includes_buffer_keys():
    candidate={"weight":torch.tensor([1.]),"running_mean":torch.tensor([3.])}
    observed={"weight":torch.tensor([1.]),"running_mean":torch.tensor([1.])}
    assert update_objective(candidate,observed,["weight","running_mean"]).item()==2.0


def test_cosine_magnitude_objective_prefers_exact_signal():
    observed={"weight":torch.tensor([1.,-2.]),"running_mean":torch.tensor([.5])}
    zero={key:torch.zeros_like(value) for key,value in observed.items()}
    exact=update_objective(observed,observed,list(observed),reference=observed,mode="cosine_magnitude")
    wrong=update_objective(zero,observed,list(observed),reference=observed,mode="cosine_magnitude")
    assert exact.item()<wrong.item()


def test_exact_sign_tail_known_values():
    assert _binomial_tail(10,10)==1/1024
    assert _binomial_tail(5,10)==0.623046875


def test_zero_control_does_not_depend_on_real_update_scale():
    candidate={'w':torch.tensor([1.,2.],requires_grad=True)}
    zero={'w':torch.zeros(2)}
    a=update_objective(candidate,zero,['w'],reference={'w':torch.ones(2)},mode='cosine_magnitude')
    b=update_objective(candidate,zero,['w'],reference={'w':torch.ones(2)*100},mode='cosine_magnitude')
    assert a.item()==b.item()==5.0
    torch.testing.assert_close(torch.autograd.grad(a,candidate['w'])[0],torch.tensor([2.,4.]))


def test_hungarian_alignment_only_permutates_within_label():
    original=torch.tensor([[0.,0.],[1.,1.],[9.,9.],[2.,2.]])
    candidate=torch.tensor([[1.,1.],[0.,0.],[9.,9.],[2.,2.]])
    labels=torch.tensor([[0.],[0.],[1.],[0.]])
    torch.testing.assert_close(_align_for_evaluation(original,candidate,labels),original)
