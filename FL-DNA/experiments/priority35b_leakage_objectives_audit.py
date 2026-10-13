"""Post-exit native ensemble objective audit; no optimization is repeated."""
import math
import torch
from experiments import priority35b_leakage_payload_audit as payload


def cosine_plain(left,right):
    if len(left)!=len(right) or not left:raise ValueError('gradient inventory')
    dot=torch.tensor([0.],device='cpu');a=torch.tensor([0.]);b=torch.tensor([0.])
    for x,y in zip(left,right):
        if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('gradient domain')
        dot+=(x*y).sum();a+=x.pow(2).sum();b+=y.pow(2).sum()
    value=1-dot/(a.sqrt()*b.sqrt())
    payload.finite_tree(value)
    return float(value)


def debias(observed):
    return [torch.cat([(part-.08*part.mean())/.92 for part in g.flatten().split(256)]).reshape_as(g) for g in observed]


def verify_native(net,data,labels,observed,plans,candidates,tolerance=0.):
    """Recompute each saved member's projected-gradient objective, exact.

    Plain cosine separately implemented. Sketch projection/cosine reuses frozen
    helper; source hashes and transmitted plan bytes are audited separately.
    Pooling reconstruction is not an ensemble member and is not substituted for
    the native minimum member objective. No decoded truth enters this function.
    """
    from experiments.priority30_native_defenses.run_audit import v2_sketch_loss
    worst=0.
    for candidate in candidates:
        members=candidate['ensemble'];stored=candidate['objectives']
        if len(members)!=30 or len(stored)!=30:raise ValueError('30 native members required')
        target=debias(observed) if candidate['name']=='structure_debias' else observed
        if candidate['name'] not in ('plain','structure_debias'):raise ValueError('unknown variant')
        for member,expected in zip(members,stored):
            payload.finite_tree(member)
            projected=data.project_batch(member.detach().clone(),standardized=data.standardized)
            raw,_,_=payload.independent_gradient(net,projected,labels,ratio=False)
            value=float(v2_sketch_loss(list(raw.values()),target,plans)) if plans is not None else cosine_plain(list(raw.values()),target)
            error=abs(value-expected)
            if not math.isfinite(error) or error>tolerance:raise ValueError('native objective recomputation mismatch')
            worst=max(worst,error)
        if candidate['objective']!=min(stored):raise ValueError('native minimum selection objective')
    return worst
