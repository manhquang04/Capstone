"""Read-only B3 gradient and mechanism audit; never reruns optimization.

Private noise receipts are used only for verification, never estimator inputs.
The frozen transform implementation is reused and its source hash must pass.
DP clipping/noise and the native loss gradient are separately implemented here.
"""
import math
from collections import OrderedDict
import numpy as np
import torch


def finite_tree(value):
    if torch.is_tensor(value):
        if value.device.type != 'cpu' or not torch.isfinite(value).all():
            raise ValueError('non-finite or non-CPU tensor evidence')
    elif isinstance(value, (float, np.floating)):
        if not math.isfinite(value): raise ValueError('non-finite scalar evidence')
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item)
            if str(key).endswith('running_var') and torch.is_tensor(item) and (item < 0).any():
                raise ValueError('negative BN variance evidence')
    elif isinstance(value, (list, tuple)):
        for item in value: finite_tree(item)


def tensor_equal(actual, expected):
    """Tensor byte identity, not torch.equal's numerical zero equivalence."""
    if not torch.is_tensor(actual) or not torch.is_tensor(expected):
        raise ValueError('tensor evidence required')
    finite_tree(actual); finite_tree(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError('tensor shape/dtype mismatch')
    if actual.detach().contiguous().numpy().tobytes() != expected.detach().contiguous().numpy().tobytes():
        raise ValueError('tensor byte mismatch')


def independent_gradient(net, inputs, labels, ratio):
    finite_tree(net.state_dict()); finite_tree(inputs)
    logits = net(inputs)
    finite_tree(logits)
    if ratio:
        targets = labels.float().reshape_as(logits)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probabilities = torch.sigmoid(logits)
        pt = probabilities * targets + (1-probabilities) * (1-targets)
        alpha = .95 * targets + .05 * (1-targets)
        loss = (alpha * (1-pt).pow(2) * bce).mean()
    else:
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
    finite_tree(loss)
    omitted = {prefix+'.'+name if prefix else name
               for prefix, layer in net.named_modules()
               if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
               for name, _ in layer.named_parameters(recurse=False)}
    pairs = [(n,p) for n,p in net.named_parameters() if n not in omitted]
    values = torch.autograd.grad(loss, [p for _,p in pairs])
    result = OrderedDict((n,g.detach().clone()) for (n,_),g in zip(pairs,values))
    finite_tree(result); finite_tree(net.state_dict())
    return result, sorted(omitted), float(loss)


def independent_dp(raw, arm, private_seed):
    finite_tree(raw)
    C, mode, sd = arm['C'], arm['clipping'], arm['noise_sd']
    if not raw or mode not in ('global','per_tensor') or not math.isfinite(C) or C<=0 or not math.isfinite(sd) or sd<0:
        raise ValueError('invalid mechanism')
    before = [float(v.to(torch.float64).norm()) for v in raw.values()]
    total = math.sqrt(math.fsum(n*n for n in before))
    radius = C if mode=='global' else C/math.sqrt(len(raw))
    clipped = []
    for tensor,norm in zip(raw.values(),before):
        denominator = total if mode=='global' else norm
        factor = min(1.,radius/max(denominator,1e-300))
        clipped.append((tensor.double()*factor*(1-4*np.finfo(np.float32).eps)).to(tensor))
    after = [float(v.double().norm()) for v in clipped]
    after_total = math.sqrt(math.fsum(n*n for n in after))
    if after_total>C or mode=='per_tensor' and any(v>radius for v in after):
        raise ValueError('clip bound violated')
    output = clipped
    if sd:
        if type(private_seed) is not int or not 0<=private_seed<2**63:
            raise ValueError('missing private noise receipt')
        generator = torch.Generator(device='cpu').manual_seed(private_seed)
        output = [(v.double()+torch.randn(v.shape,dtype=torch.float64,generator=generator)*sd).to(v) for v in clipped]
    finite_tree(output)
    return output, dict(C=C,clipping=mode,noise_sd=sd,sensitivity=2*C,tensor_radius=radius,
                        before_norm=total,after_norm=after_total,tensor_before=before,tensor_after=after,
                        epsilon=arm['epsilon'],record_level_dp=False,clip_only=arm['epsilon'] is None,no_bn_payload=True)


def audit_candidate_receipt(receipt, result, ratio, arm):
    finite_tree(receipt)
    expected = ['plain','structure_debias'] if arm=='dna_v1_conservative' else ['plain']
    candidates=receipt['candidates']
    if [c['name'] for c in candidates]!=expected: raise ValueError('variant inventory')
    selected=min(range(len(candidates)),key=lambda i:candidates[i]['objective'])
    if selected!=receipt['selected'] or candidates[selected]['name']!=result['selected_variant']:
        raise ValueError('observable variant selection')
    for candidate in candidates:
        if ratio:
            if candidate['ensemble'] or candidate['objectives']:
                raise ValueError('unexpected ratio optimization budget')
        else:
            if len(candidate['ensemble'])!=30 or len(candidate['objectives'])!=30:
                raise ValueError('native30 budget evidence')
            if candidate['objective']!=min(candidate['objectives']):
                raise ValueError('native minimum observable objective')
    return candidates[selected]['reconstruction']


def audit_job(driver, config, target, data, private_seeds):
    """Intended post-exit only; callers must first pass evidence/no-live audit."""
    result=driver.read(driver.folder(config)/'result.json')
    directory=driver.folder(config)/result['attempt']
    server=torch.load(directory/'server_receipt.pt',map_location='cpu',weights_only=False)
    private=torch.load(directory/'private_truth.pt',map_location='cpu',weights_only=False)
    reconstruction=torch.load(directory/'reconstruction.pt',map_location='cpu',weights_only=False)
    finite_tree(server);finite_tree(private);finite_tree(reconstruction)
    truth=data.Xtrain[target['indices']].clone(); labels=data.ytrain[target['indices']].clone()
    tensor_equal(private['truth'],truth);tensor_equal(private['labels'],labels);tensor_equal(server['labels'],labels)
    if private['source_ids']!=target['source_ids'] or private['indices']!=target['indices']:
        raise ValueError('private target identity')
    ratio=config['cell']=='ratio_batch1'
    net=driver.core.ratio_model(data.num_features) if ratio else driver.core.native.make_model(data.num_features)
    if set(net.state_dict())!=set(server['public_model']):raise ValueError('public model keys')
    for name,value in net.state_dict().items():tensor_equal(server['public_model'][name],value)
    raw,omitted,loss=independent_gradient(net,data.de_standardize(truth) if ratio else truth,labels,ratio)
    if list(raw)!=server['names'] or list(raw)!=list(private['raw']):raise ValueError('non-BN gradient domain')
    for name,value in raw.items():tensor_equal(private['raw'][name],value)
    if server['bn']!=dict(no_bn_payload=True,omitted_bn_parameters=omitted,payload_names=list(raw),loss=loss):
        raise ValueError('BN exclusion receipt')
    arm=next(a for a in driver.read(driver.OUT/'arms.json')[config['dataset']] if a['method']==config['arm'])
    if arm['C'] is not None:
        payload,mechanism=independent_dp(raw,arm,private_seeds.get(driver.key(config)))
        if mechanism!=server['mechanism'] or mechanism!=result['mechanism']:
            raise ValueError('independent DP mechanism receipt')
        if server['plans'] is not None or server['metadata'] is not None:raise ValueError('unexpected DP transform')
    else:
        payload,plans,metadata=driver.core.defend(raw,config['arm'])
        if server['mechanism'] is not None or result['mechanism'] is not None:raise ValueError('unexpected DP receipt')
        if metadata!=server['metadata']:raise ValueError('transform metadata')
        if plans is None:
            if server['plans'] is not None:raise ValueError('unexpected sketch plans')
        else:
            if len(plans)!=len(server['plans']):raise ValueError('sketch plan count')
            for p,q in zip(plans,server['plans']):
                for name,value in p.items():
                    if torch.is_tensor(value):tensor_equal(q[name],value)
                    elif q[name]!=value:raise ValueError('sketch plan mismatch')
    if len(payload)!=len(server['payload']):raise ValueError('payload count')
    for observed,expected in zip(server['payload'],payload):tensor_equal(observed,expected)
    rec=audit_candidate_receipt(reconstruction,result,ratio,config['arm'])
    return result,truth,rec,reconstruction['controls']
