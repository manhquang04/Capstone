"""Independent image receipt/array audit after completed recovery, no attack replay."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments import priority34d_image_recovery as r
from experiments.priority34d_continue import live
from experiments.priority34d_independent_metrics import images


def compare(actual,expected,truth=False):
    a,b=np.asarray(actual),np.asarray(expected)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid image audit shape/finite values')
    rtol,atol=(2e-7,2e-7) if truth else (1e-6,1e-8)
    if not np.allclose(a,b,rtol=rtol,atol=atol):
        raise ValueError('independent image audit mismatch')


def selected_mode(objectives):
    order=[mode for mode in ('plain','v1_debias','v2_sketch') if mode in objectives]
    if set(order)!=set(objectives) or not order or not all(np.isfinite(v) for v in objectives.values()):
        raise ValueError('invalid observable objective record')
    return min(order,key=lambda mode:(objectives[mode],order.index(mode)))


def same_plans(left,right):
    if r.torch.is_tensor(left) or r.torch.is_tensor(right):
        return r.torch.is_tensor(left) and r.torch.is_tensor(right) and r.torch.equal(left,right)
    if isinstance(left,dict) or isinstance(right,dict):
        return isinstance(left,dict) and isinstance(right,dict) and set(left)==set(right) and all(same_plans(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)) or isinstance(right,(list,tuple)):
        return isinstance(left,(list,tuple)) and isinstance(right,(list,tuple)) and len(left)==len(right) and all(same_plans(a,b) for a,b in zip(left,right))
    return left==right


def expected_dp(values,clip,sigma,seed,mechanism):
    generator=r.torch.Generator().manual_seed(seed)
    if mechanism=='single':
        norm=float(r.u.old.p31.dp.flatten(values).norm())
        factor=1. if norm<=clip['C'] or norm==0 else clip['C']/norm
        return [value.detach().cpu()*factor+r.torch.randn(value.shape,generator=generator,dtype=value.dtype)*(sigma*clip['C']) for value in values]
    if mechanism=='per_tensor':
        if len(values)!=len(clip['clips']):
            raise ValueError('clip dimensions differ')
        expected=[]
        for value,limit in zip(values,clip['clips']):
            factor=min(1.,limit/max(float(value.norm()),1e-12))
            expected.append(value.detach().cpu()*factor+r.torch.randn(value.shape,generator=generator,dtype=value.dtype)*sigma*limit)
        return expected
    raise ValueError('unknown audit DP mechanism')


def verify():
    complete=r.read(r.BASE/'IMAGE_COMPLETE.json')
    if live():
        raise RuntimeError('scientific workers alive; defer image audit')
    freeze=r.verify(); assert complete['freeze_sha256']==r.sha(r.FREEZE)
    assert complete['jobs']==len(freeze['jobs'])
    targets=r.read(r.BASE/'targets.json')
    private=r.read(r.BASE/'private_noise_seeds.json')  # never passed to attacker or emitted
    data=r.u.old.p31.dp.CIFAR10(str(r.ROOT/'datasets/cifar10'),train=False,download=False)
    endpoints=[]
    for job in freeze['jobs']:
        assert r.valid(job), 'missing validated image output'
        parent=r.BASE/'jobs'/job['id']; receipt=r.read(parent/'validated.json')
        result_names=[name for name in receipt['outputs'] if name.endswith('/result.json')]
        assert len(result_names)==1
        folder=(parent/result_names[0]).parent; result=r.read(folder/'result.json')
        assert result['job']==job and result['status']=='PASS'
        assert result['iterations']==4800 and result['restarts']==1
        assert result['device']=='cpu' and result['intra_threads']==result['inter_threads']==1
        assert job['indices']==targets['groups'][job['setting']][job['target']]
        arrays=np.load(folder/'private_arrays.npz',allow_pickle=False)
        assert arrays['indices'].tolist()==job['indices']
        assert arrays['labels'].tolist()==[int(data.targets[index]) for index in job['indices']]
        raw_pixels=np.asarray(data.data[job['indices']],dtype=np.float32).transpose(0,3,1,2)/255
        compare(arrays['truth'],raw_pixels,truth=True)
        metrics,records,pairing=images(arrays['truth'],arrays['reconstruction'])
        assert pairing==result['pairing'], 'independent Hungarian pairing mismatch'
        assert set(metrics)==set(result['metrics'])
        for key in metrics:
            compare(metrics[key],result['metrics'][key])
        assert len(records)==len(result['record_metrics'])
        for a,b in zip(records,result['record_metrics']):
            for key in a:
                compare(a[key],b[key])
        assert result['selected_mode']==selected_mode(result['observable_objectives'])
        observed=r.torch.load(folder/'received.pt',map_location='cpu',weights_only=False)
        assert set(observed)=={'kind','payload','projection_plans'} and observed['kind']=='transmitted_only'
        normal,labels=r.u.old.p31.tensors(data,job['indices']); normal=normal.contiguous()
        public=r.u.starting_model(job['setting'])
        public.load_state_dict(r.torch.load(r.u.BASE/job['setting']/'initial_state.pt',map_location='cpu',weights_only=False))
        r.u.check(public); public.eval()
        # Audit-only derivative recomputation. No training or recovery optimizer.
        logits=public(normal); r.u.finite(logits,'audit logits')
        loss=r.torch.nn.functional.cross_entropy(logits,labels); r.u.finite(loss,'audit loss')
        gradients=r.torch.autograd.grad(loss,list(public.parameters()))
        state=dict(zip([name for name,_ in public.named_parameters()],gradients))
        if job['arm']=='unprotected':
            expected=list(state.values()); plans=None
        elif job['arm']==r.u.DNA[0]:
            expected=list(r.u.old.transform(state,r.u.DNA[0]).values()); plans=None
        elif job['arm']==r.u.DNA[1]:
            expected,plans=r.native.v2_sketch_payload(state)
        else:
            clip=r.read(r.u.BASE/job['setting']/'clip.json')
            values=list(state.values())
            expected=expected_dp(values,clip,job['sigma'],private[job['id']],job['mechanism'])
            plans=None
        assert len(expected)==len(observed['payload'])
        for a,b in zip(expected,observed['payload']):
            r.u.finite(a,'audit expected payload'); r.u.finite(b,'audit observed payload')
            assert r.torch.equal(a,b), 'received payload not bit-exact'
        assert same_plans(observed['projection_plans'],plans)
        endpoints.append(dict(setting=job['setting'],arm=job['arm'],target=job['target'],
                              indices=job['indices'],metrics=metrics,record_metrics=records))
    doc=dict(status='PASS_IMAGE_STAGE_ONLY',priority34d_complete=False,at=time.time(),
             freeze_sha256=r.sha(r.FREEZE),jobs=len(endpoints),endpoints=endpoints,
             independent_metrics=True,received_payloads_bit_exact=True)
    r.write(r.OUT/'audits/independent_image_results.json',doc)
    return doc


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if not args.verify:
        parser.error('choose --verify only after completed image science')
    try:
        doc=verify(); print(json.dumps(dict(status=doc['status'],jobs=doc['jobs'])))
    except Exception as error:
        r.write(r.OUT/('audits/image_verifier_failure_'+str(time.time_ns())+'.json'),
                dict(error=repr(error),at=time.time(),priority34d_complete=False))
        raise
