"""Post-hoc privileged T: identical fitting, explicitly near-truth starts."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import time
import torch
from collections import OrderedDict
from experiments.priority35a_adam import require_finite_cpu
from experiments.priority35a_d1prime_estimator import mask_model, aggregate
from experiments.priority35a_diagnostic_estimator import operator
from experiments.priority35a_estimator import candidate_rows, schema_blocks, public_seed, flatten, objective

SCALES = (.01, .1, .5)
PRECISION = 1e-6


def initial(truth, std, blocks, scale, seed):
    if scale not in SCALES:
        raise ValueError('fixed T scale')
    std = torch.as_tensor(std, dtype=torch.float32)
    require_finite_cpu(dict(truth=truth, std=std), 'T initialization')
    if truth.dtype != torch.float32 or truth.shape != (4,len(std)) or (std < 0).any():
        raise ValueError('T truth/SD shape')
    noise = torch.randn(truth.shape, generator=torch.Generator().manual_seed(seed)) * std * scale
    perturbed = truth + noise
    columns = []
    for block in blocks:
        values = perturbed[:,block['indices']]
        if block['categories'] is not None:
            values = values.clamp_min(1e-6)
            values = (values / values.sum(1,keepdim=True)).log()
        columns.append(values)
    raw = torch.cat(columns,1)
    require_finite_cpu(dict(noise=noise, perturbed=perturbed,raw=raw), 'T finite initial logits')
    return dict(seed=seed,noise=noise,perturbed=perturbed,raw=raw,
                soft=candidate_rows(raw,blocks).detach())


def fit_initial(payload, blocks, instrument, seed, iterations, initial_raw):
    shared, bn = payload['shared'], payload['bn']
    observed = flatten(payload['gradient']).detach()
    model = mask_model(shared,bn,initial_raw.shape[1],seed,payload['masks'])
    raw = torch.nn.Parameter(initial_raw.detach().clone())
    optimizer = torch.optim.Adam([raw], lr=.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=iterations,eta_min=0.)
    trace, zeros = [], 0
    for iteration in range(iterations):
        optimizer.zero_grad()
        model.zero_grad()
        candidate = operator(model,candidate_rows(raw,blocks),payload['labels'],shared,'D1')
        zeros += int(float(candidate.detach().norm()) == 0)
        value = objective(candidate,observed,instrument)
        value.backward()
        if raw.grad is None or not torch.isfinite(raw.grad).all():
            raise ValueError('nonfinite/missing candidate derivative')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('nonfinite model derivative')
        optimizer.step()
        scheduler.step()
        require_finite_cpu(model.state_dict(),'T optimized model')
        require_finite_cpu({'raw':raw},'T optimized candidate')
        if iteration == 0 or (iteration+1)%500 == 0 or iteration+1 == iterations:
            trace.append(dict(iteration=iteration+1,before_step_objective=float(value.detach())))
    final = objective(operator(model,candidate_rows(raw,blocks),payload['labels'],shared,'D1'),observed,instrument)
    params = dict(model.named_parameters())
    if any(not torch.equal(params[k].detach(),v) for k,v in shared.items()):
        raise ValueError('fixed shared parameters changed')
    affine = OrderedDict((k,p.detach().clone()) for k,p in params.items() if k in bn)
    if any(params[k].requires_grad or not torch.equal(v,bn[k]) for k,v in affine.items()):
        raise ValueError('true BN affine changed')
    return dict(raw=raw.detach().clone(),final_objective=float(final.detach()),trace=trace,
                oracle_affine=affine,zero_candidate_norms=zeros)


def at_rows(payload, rows, instrument):
    require_finite_cpu(dict(rows=rows), 'T direct rows')
    model = mask_model(payload['shared'],payload['bn'],rows.shape[1],0,payload['masks'])
    gradient = operator(model,rows,payload['labels'],payload['shared'],'D1').detach()
    return gradient, float(objective(gradient,flatten(payload['gradient']),instrument).detach())


def ratio(output, truth):
    return (None,'numerical-zero truth objective (abs<=1e-6)') if abs(truth)<=PRECISION else (output/truth,None)


def reconstruct(payload, truth, std, instrument, scale, *, synthetic=False, test_iterations=None):
    torch.set_num_threads(1)
    if set(payload)!={'shared','gradient','bn','masks','labels','meta','ordinal'}:
        raise ValueError('privileged T payload whitelist')
    if test_iterations is not None and not synthetic:
        raise ValueError('fixed scientific budget')
    if payload['labels'].shape!=(4,1) or not ((payload['labels']==0)|(payload['labels']==1)).all():
        raise ValueError('ordered labels')
    require_finite_cpu({'labels':payload['labels']},'T labels')
    require_finite_cpu(payload['shared'],'T shared')
    require_finite_cpu(payload['gradient'],'T gradient')
    began = time.time()
    blocks = schema_blocks(payload['meta'])
    iterations = 4000 if test_iterations is None else test_iterations
    starts, members = [], []
    for i in range(2):
        seed = public_seed('sanity-T',payload['ordinal'],instrument,scale,i)
        init = initial(truth,std,blocks,scale,seed)
        starts.append(init)
        members.append(fit_initial(payload,blocks,instrument,seed,iterations,init['raw']))
    chosen,soft,rows = aggregate(members,blocks,'a')
    _,final = at_rows(payload,soft,instrument)
    return dict(rows=rows,pooled_soft=soft,members=members,initializations=starts,chosen_member=chosen,
                final_objective=final,scale=scale,starts=2,iterations=iterations,labels=payload['labels'].clone(),
                torch_threads=1,synthetic=synthetic,elapsed_seconds=time.time()-began)
