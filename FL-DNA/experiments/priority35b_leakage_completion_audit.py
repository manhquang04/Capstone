"""Ratio and native pooling checks for post-exit scientific audit.

Never optimizes/retrains, never uses target rows to choose an estimator variant.
LSMR is replayed only as a deterministic decoding audit of transmitted sketches.
"""
import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from experiments import priority35b_leakage_payload_audit as tensors
from experiments import priority35b_leakage_metrics_audit as metrics
from experiments import priority35b_leakage_objectives_audit as objectives


def ratio_check(driver,net,data,server,candidate):
    observed=objectives.debias(server['payload']) if candidate['name']=='structure_debias' else server['payload']
    decoded=[];receipts=[]
    if server['metadata'] is not None:
        from experiments.priority34c_solvers import known_key_least_squares
        if len(server['metadata'])!=len(observed):raise ValueError('LS sketch inventory')
        for tensor,meta in zip(observed,server['metadata']):
            array,receipt=known_key_least_squares(tensor.numpy(),meta)
            decoded.append(torch.from_numpy(array));receipts.append(receipt)
    else:decoded=observed
    visible=dict(zip(server['names'],decoded))
    w=visible['network.0.weight'].numpy().astype(np.float64)
    bias=visible['network.0.bias'].numpy().astype(np.float64)
    row=int(np.argmax(np.abs(bias)))
    if not bias[row] or not np.isfinite([bias[row]]).all():raise ValueError('undefined ratio denominator')
    prepared=torch.from_numpy(w[row]/bias[row]).float().unsqueeze(0)
    tensors.finite_tree(prepared)
    tensors.tensor_equal(candidate['reconstruction'],(prepared-data.mean)/data.std)
    receipt=dict(ratio=dict(selected_row=row,observed_denominator=float(bias[row])),least_squares=receipts)
    if candidate['solver']!=receipt:raise ValueError('ratio solver receipt mismatch')
    if receipts:
        value=sum(r['residual'] for r in receipts)
    else:
        raw,_,_=tensors.independent_gradient(net,prepared,server['labels'],ratio=True)
        a=torch.cat([g.double().reshape(-1) for g in raw.values()]);b=torch.cat([g.double().reshape(-1) for g in observed])
        denominator=torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)
        if denominator<=0:raise ValueError('undefined ratio cosine')
        value=float(1-torch.dot(a,b)/denominator)
    if not math.isfinite(value) or value!=candidate['objective']:raise ValueError('ratio objective mismatch')


def pooling_check(data,candidate):
    from utils import categorical_softmax
    members=candidate['ensemble']
    if len(members)!=30 or len(candidate['objectives'])!=30:raise ValueError('30-member pool')
    best=min(range(30),key=lambda i:candidate['objectives'][i])
    decoded_best=data.decode_batch(members[best].detach().clone());aligned=[]
    for member in members:
        costs=metrics.decoded_costs(decoded_best,data.decode_batch(member.detach().clone()),data.create_tolerance_map())
        _,columns=linear_sum_assignment(costs)
        reordered=member[columns].detach().clone()
        aligned.append(categorical_softmax(reordered,data))
    pooled=torch.stack(aligned).median(dim=0).values
    tensors.tensor_equal(candidate['reconstruction'],pooled)


def audit_saved_objectives(driver,config,data):
    result=driver.read(driver.folder(config)/'result.json');directory=driver.folder(config)/result['attempt']
    server=torch.load(directory/'server_receipt.pt',map_location='cpu',weights_only=False)
    saved=torch.load(directory/'reconstruction.pt',map_location='cpu',weights_only=False)
    ratio=config['cell']=='ratio_batch1'
    net=driver.core.ratio_model(data.num_features) if ratio else driver.core.native.make_model(data.num_features)
    if ratio:
        for candidate in saved['candidates']:ratio_check(driver,net,data,server,candidate)
    else:
        objectives.verify_native(net,data,server['labels'],server['payload'],server['plans'],saved['candidates'])
        for candidate in saved['candidates']:pooling_check(data,candidate)
    return dict(ratio=ratio,variants=len(saved['candidates']),optimization_repeated=False)
