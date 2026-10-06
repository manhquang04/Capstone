"""Synthetic checks; no CIFAR training or attack runs."""
import math
import numpy as np
import torch
from experiments import priority31_image_utility_dp as p


def test_split_and_selection_contract():
    train,val=p.split_ids(np.tile(np.arange(10),5000))
    assert len(train)==12000 and len(val)==3000
    assert not set(train)&set(val)
    assert np.array_equal(train,p.split_ids(np.tile(np.arange(10),5000))[0])


def test_matching_largest_and_bracket():
    rows=[]
    values={'baseline':0.,p.TRANSFORMS[0]:-.01,p.TRANSFORMS[1]:-.02,
            p.sigma_id(.001):-.005,p.sigma_id(.01):-.018,p.sigma_id(.1):-.05}
    for method,delta in values.items():
        for seed in p.SEEDS:
            rows.append(dict(method=method,validation_accuracy=.6+delta,paired_validation_delta=delta))
    matched=p.match_grid(rows,[.001,.01,.1])
    assert matched['baseline_gate']
    assert matched['matches'][p.TRANSFORMS[0]]['sigma']==.001
    assert matched['matches'][p.TRANSFORMS[1]]['sigma']==.01
    assert all(m['bracketed'] for m in matched['matches'].values())


def test_dp_clip_and_noise_accountant():
    observed=[torch.tensor([3.,4.])]
    result=p.dp.add_clipped_noise(observed,1.,0.,123)
    assert torch.allclose(result[0],torch.tensor([.6,.8]))
    a=p.dp.add_clipped_noise(observed,1.,.1,123)
    b=p.dp.add_clipped_noise(observed,1.,.1,123)
    assert torch.equal(a[0],b[0])
    eps=p.dp.epsilon_from_rdp(.1,1.,1e-5,1,p.dp.alpha_grid())
    assert eps['epsilon']>0


def test_sign_holm_and_order_interval():
    assert p.dp.exact_p(39,0)==2**-39
    rows=[dict(p_value=.01),dict(p_value=.04)]
    p.dp.holm(rows)
    assert rows[0]['holm_p']==.02 and rows[1]['holm_p']==.04
    coverage=1-2*sum(math.comb(39,k) for k in range(13))/2**39
    assert .95<coverage<1
    model,_=p.dp.inversefed.construct_model('LeNetZhu',seed=42)
    assert sum(x.numel() for x in model.parameters())==15826
    assert not list(model.buffers())


if __name__=='__main__':
    for name,value in list(globals().items()):
        if name.startswith('test_'):
            value()
            print(name+' PASS')
