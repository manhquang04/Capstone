"""P34D BAF BN-only instrument, separated from image import namespaces."""
import argparse
import copy
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import priority34d_checkpoints as c
from experiments import priority34c_history as history
b, torch, np = c.b, c.torch, c.b.np
OUT, sha, read = c.OUT, c.audit.sha, lambda p:json.loads(Path(p).read_text())
write, append = c.write, c.append
BASE = OUT/'bn/baf'
FREEZE = BASE/'execution_freeze.json'
SEEDS = c.SEEDS
SIZES = {'n8':8,'n24':24,'development':24,'confirmatory':39}
METHODS = b.METHODS
ANNEX = ROOT/'protocols/amendments/2026-10-04_priority34d_bn_execution.md'


def finite(value, label):
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise FloatingPointError('nonfinite '+label)
    return value


def model(seed):
    assert c.valid(seed), 'checkpoint validation missing'
    prepared = b.p.OUT/'prepared/baf'
    with torch.random.fork_rng(devices=[]):
        result = b.p.FraudMLP(np.load(prepared/'train_x.npy',mmap_mode='r').shape[1])
    result.load_state_dict(torch.load(c.folder(seed)/'final_state.pt',map_location='cpu',weights_only=False))
    b.strict_state(result.state_dict(),'P34D fixed BN checkpoint')
    return result


def verify():
    doc = read(FREEZE)
    for name,digest in {**doc['sources'],**doc['inputs']}.items():
        if sha(ROOT/name) != digest:
            raise ValueError('BN freeze drift: '+name)
    return doc


def reserve(source, excluded):
    available = np.flatnonzero(~np.isin(source,list(excluded)))
    count = 3*sum(SIZES.values())*4
    if len(available)<count:
        raise ValueError('insufficient fresh BAF sources')
    selected = available[np.random.default_rng(342321).permutation(len(available))[:count]]
    groups, position = {},0
    for seed in SEEDS:
        groups[str(seed)] = {}
        for stage,size in SIZES.items():
            indices = selected[position:position+size*4].reshape(size,4)
            position += size*4
            groups[str(seed)][stage] = dict(indices=indices.tolist(),
                source_ids=[[int(source[i]) for i in row] for row in indices])
    ids = [i for checkpoint in groups.values() for stage in checkpoint.values()
           for row in stage['source_ids'] for i in row]
    assert len(ids)==len(set(ids))==count and not set(ids)&set(excluded)
    return dict(seed=342321,groups=groups,total=count,excluded_count=len(excluded))


def prepare():
    if FREEZE.exists():
        return verify()
    assert not (BASE/'PREFLIGHT_FAILURE.json').exists()
    c.verify(); assert all(c.valid(seed) for seed in SEEDS)
    paths = {Path(__file__).resolve(),ANNEX,ROOT/'tests/test_priority34d_bn.py',
             ROOT/'protocols/amendments/2026-10-04_priority34d_checkpoint_robustness.md'}
    for module in list(sys.modules.values()):
        raw = getattr(module,'__file__',None)
        if raw:
            path = Path(raw).resolve()
            if path.is_relative_to(ROOT) and '.venv' not in str(path) and path.suffix=='.py':
                paths.add(path)
    try:
        provenance = history.audit('baf')
        p34c_targets = ROOT/'artifacts/priority34c/baf/targets.json'
        excluded = set(provenance['exclusions'])|set(history.extract(read(p34c_targets)))
        source = np.load(b.p.OUT/'prepared/baf/test_source_ids.npy')
        targets = reserve(source,excluded)
        if (BASE/'targets.json').exists() or (BASE/'private_noise_seeds.json').exists():
            raise RuntimeError('partial preparation preserved; no overwrite')
        write(BASE/'history.json',provenance)
        write(BASE/'targets.json',targets)
        seeds = {str(seed):{stage:{method:[secrets.randbits(63) for _ in range(size)]
                    for method in METHODS} for stage,size in SIZES.items()}
                 for seed in SEEDS}
        private = BASE/'private_noise_seeds.json'
        descriptor = os.open(str(private),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'w') as stream:
            json.dump(seeds,stream)
        inputs = list((b.p.OUT/'prepared/baf').glob('*'))
        inputs += [p34c_targets,BASE/'targets.json',BASE/'history.json',private,
                   c.FREEZE,OUT/'CHECKPOINTS_COMPLETE.json']
        for seed in SEEDS:
            inputs += [c.folder(seed)/name for name in ('final_state.pt','result.json','checkpoint_validated.json')]
        inputs += [ROOT/row['path'] for row in provenance['provenance']]
        write(FREEZE,dict(at=time.time(),stage='BN instrument only',seeds=list(SEEDS),sizes=SIZES,
              sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)},
              inputs={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(inputs)) if p.is_file()}))
        print(json.dumps(dict(freeze_sha256=sha(FREEZE),sources=len(paths),reserved=targets['total'])))
    except Exception as error:
        write(BASE/'PREFLIGHT_FAILURE.json',dict(error=repr(error),at=time.time(),science_started=False))
        raise


def folder(seed,stage,index):
    return BASE/str(seed)/stage/('target_%03d'%index)


def valid(seed,stage,index):
    path = folder(seed,stage,index)/'validated.json'
    if not path.exists():
        return False
    doc = read(path)
    if doc['identity'] != [seed,stage,index] or doc['freeze_sha256'] != sha(FREEZE):
        raise ValueError('BN receipt identity drift')
    for name,digest in doc['outputs'].items():
        if sha(path.parent/name) != digest:
            raise ValueError('BN output hash mismatch')
    return True


def capture(public_model,seed,stage,index):
    target = read(BASE/'targets.json')['groups'][str(seed)][stage]
    group = target['indices'][index]
    prepared = b.p.OUT/'prepared/baf'
    x = np.load(prepared/'test_x.npy',mmap_mode='r')
    y = np.load(prepared/'test_y.npy',mmap_mode='r')
    local = copy.deepcopy(public_model).train()
    torch.manual_seed(b.derive_seed(342322,seed,stage,index))
    inputs = torch.from_numpy(np.array(x[group],copy=True))
    labels = torch.from_numpy(np.array(y[group],copy=True)[:,None])
    finite(inputs.numpy(),'batch inputs')
    optimizer = torch.optim.Adam(local.parameters(),lr=.001)
    optimizer.zero_grad()
    b.strict_state(local.state_dict(),'pre-step')
    logits = local(inputs)
    finite(logits.detach().numpy(),'logits'); b.strict_state(local.state_dict(),'actual forward')
    loss = b.p.BinaryFocalLoss(alpha=.95,gamma=2.)(logits,labels)
    finite(loss.detach().numpy(),'loss'); loss.backward()
    for name,param in local.named_parameters():
        finite(param.grad.numpy(),'gradient '+name)
    optimizer.step(); b.strict_state(local.state_dict(),'post-step')
    raw = local.state_dict()[b.BN_KEY]-public_model.state_dict()[b.BN_KEY]
    finite(raw.numpy(),'BN mean delta')
    return dict(source_ids=target['source_ids'][index],raw=raw,
                true_mean=np.asarray(inputs.numpy(),dtype=np.float64).mean(axis=0))


def calibrate(public_model,captures,seed):
    clip = 1.01*max(float(row['raw'].norm()) for row in captures)
    finite(clip,'development clip')
    if clip<=0:
        raise FloatingPointError('zero development clip')
    private = read(BASE/'private_noise_seeds.json')[str(seed)]['development']
    result = {}
    for method in METHODS:
        differences,noise_norms = [],[]
        for index,row in enumerate(captures):
            received = b.payload(public_model,row['raw'],method,index)
            if method==METHODS[0]:
                decoded = received['q'].numpy()
            else:
                decoded,_ = b.v2.reconstruct_update_array_v2(received['q'],received['metadata'])
            differences.append(float(np.linalg.norm(decoded-row['raw'].numpy())))
            gen = torch.Generator().manual_seed(private[method][index])
            noise_norms.append(float(torch.randn(row['raw'].shape,generator=gen,dtype=torch.float64).norm()))
        sigma = float(np.median(differences)/(clip*np.median(noise_norms)))
        finite(sigma,'distortion sigma')
        if sigma<=0:
            raise FloatingPointError('nonpositive distortion sigma')
        result[method] = dict(clip=clip,sigma=sigma,distortions=differences,unit_noise_norms=noise_norms,
                             target_median_distortion=float(np.median(differences)),development_n=24)
    return result


def run_target(public_model,seed,stage,index,std,calibration=None):
    if valid(seed,stage,index):
        return torch.load(folder(seed,stage,index)/'private_capture.pt',map_location='cpu',weights_only=False)
    target_folder = folder(seed,stage,index)
    if target_folder.exists():
        raise RuntimeError('preserved partial BN target; require direction')
    target_folder.mkdir(parents=True)
    try:
        row = capture(public_model,seed,stage,index)
        recovered = finite(b.recover_mean(public_model,row['raw']),'unprotected recovery')
        arrays = dict(unprotected=np.asarray(recovered))
        payloads = dict(unprotected=dict(kind='raw',q=row['raw']))
        if stage=='confirmatory':
            private = read(BASE/'private_noise_seeds.json')[str(seed)][stage]
            for method in METHODS:
                received = b.payload(public_model,row['raw'],method,index)
                arrays[method] = finite(b.recover_payload(public_model,received),'protected recovery')
                payloads[method] = received
                cal = calibration[method]
                raw = row['raw'].double()
                clipped = raw*min(1.,cal['clip']/max(float(raw.norm()),1e-12))
                gen = torch.Generator().manual_seed(private[method][index])
                q = (clipped+torch.randn(raw.shape,generator=gen,dtype=torch.float64)*cal['clip']*cal['sigma']).to(row['raw'].dtype)
                finite(q.numpy(),'DP transmitted BN delta')
                # Only q/public checkpoint reaches this decoder; no seed/truth/raw.
                arrays['dp_'+method] = finite(b.recover_mean(public_model,q),'DP recovery')
                payloads['dp_'+method] = dict(kind='raw',q=q)
        torch.save(row,target_folder/'private_capture.pt')
        os.chmod(target_folder/'private_capture.pt',0o600)
        torch.save(payloads,target_folder/'received_payloads.pt')
        np.savez(target_folder/'recoveries.npz',**arrays)
        metrics = {name:float(b.mse_std(value,row['true_mean'],std)) for name,value in arrays.items()}
        finite(list(metrics.values()),'recovery MSE')
        write(target_folder/'result.json',dict(seed=seed,stage=stage,index=index,source_ids=row['source_ids'],
              metrics=metrics,freeze_sha256=sha(FREEZE),device='cpu',intra_threads=1,inter_threads=1))
        write(target_folder/'validated.json',dict(identity=[seed,stage,index],freeze_sha256=sha(FREEZE),
              outputs={p.name:sha(p) for p in target_folder.iterdir() if p.is_file()}))
        return row
    except Exception as error:
        write(target_folder/'failure.json',dict(error=repr(error),at=time.time(),identity=[seed,stage,index]))
        raise


def gate(public_model,captures,prior,std):
    recovered = [finite(b.recover_mean(public_model,row['raw']),'qualification recovery') for row in captures]
    rows = [dict(recovered_mse=float(b.mse_std(recovered[i],row['true_mean'],std)),
                 prior_mse=float(b.mse_std(prior,row['true_mean'],std)),
                 decoy_mse=float(b.mse_std(recovered[(i+1)%len(captures)],row['true_mean'],std)))
            for i,row in enumerate(captures)]
    finite([value for row in rows for value in row.values()],'qualification metrics')
    tests = {name:b.sign([row[name+'_mse'] for row in rows],[row['recovered_mse'] for row in rows])
             for name in ('prior','decoy')}
    return dict(rows=rows,tests=tests,passed=all(row['p']<.05 for row in tests.values()))


def supervise():
    verify(); c.verify()
    rows = subprocess.check_output(['ps','-axo','pid,command'],text=True).splitlines()
    if any('priority34d_image_utility.py' in row and ('--supervise' in row or '--job' in row) for row in rows):
        raise RuntimeError('image utility occupies CPU budget; do not overlap')
    guard = socket.socket(); guard.bind(('127.0.0.1',43468)); guard.listen(1)
    write(BASE/'supervisor.lock.json',dict(pid=os.getpid(),at=time.time()))
    checklist = read(OUT/'checklist.json'); checklist['part3_qualification_calibration']['status']='in_progress'
    write(OUT/'checklist.json',checklist)
    prior,std = b.population('baf')
    summary,done = {},0
    for seed in SEEDS:
        public_model = model(seed)
        outcome = dict(status='NOT_ASSESSABLE')
        for stage in ('n8','n24','development','confirmatory'):
            if stage=='confirmatory':
                cal = read(BASE/str(seed)/'distortion_calibration.json')
            else:
                cal = None
            captures = []
            for index in range(SIZES[stage]):
                captures.append(run_target(public_model,seed,stage,index,std,cal))
                done += 1
                progress = dict(part='part3_bn',checkpoint_seed=seed,stage=stage,done=done,
                                maximum_total=3*sum(SIZES.values()),last_update=time.time(),failed=0)
                write(OUT/'progress.json',progress); append(OUT/'progress.log',progress)
                append(OUT/'runs.jsonl',dict(event='bn_target_finished',seed=seed,stage=stage,index=index,at=time.time()))
            if stage in ('n8','n24'):
                result = gate(public_model,captures,prior,std)
                path = BASE/str(seed)/(stage+'_gate.json')
                if path.exists():
                    assert read(path)==result
                else:
                    write(path,result)
                if not result['passed']:
                    outcome.update(reason='qualification failed '+stage,gate=result['tests'])
                    break
            elif stage=='development':
                result = calibrate(public_model,captures,seed)
                path = BASE/str(seed)/'distortion_calibration.json'
                if path.exists():
                    assert read(path)==result
                else:
                    write(path,result)
            else:
                outcome = dict(status='CONFIRMATORY_COMPLETE',n=39)
        summary[str(seed)] = outcome
    write(BASE/'BN_COMPLETE.json',dict(stage_only=True,checkpoints=summary,done=done,
                                      freeze_sha256=sha(FREEZE),at=time.time()))
    checklist['part3_qualification_calibration']['status']='completed'
    checklist['part3_bn_outcomes']=summary
    write(OUT/'checklist.json',checklist)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for flag in ('prepare','launch','supervise','resume'):
        parser.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    if args.prepare:
        prepare()
    elif args.supervise:
        try:
            supervise()
        except Exception as error:
            write(BASE/'REQUIRES_DIRECTION.json',dict(error=repr(error),at=time.time(),pool_drained=True))
            raise
    elif args.launch:
        verify()
        assert not (BASE/'BN_COMPLETE.json').exists() and not (BASE/'REQUIRES_DIRECTION.json').exists()
        if list(BASE.glob('launch_*.json')) and not args.resume:
            raise RuntimeError('previous launch; verified infrastructure resume only')
        rows=subprocess.check_output(['ps','-axo','pid,command'],text=True).splitlines()
        assert not any('priority34d_bn.py' in row and '--supervise' in row for row in rows)
        with (BASE/'detached_stdout.log').open('a') as so,(BASE/'detached_stderr.log').open('a') as se:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--supervise'],
                                  cwd=ROOT,stdout=so,stderr=se,start_new_session=True)
        doc=dict(pid=proc.pid,at=time.time(),freeze_sha256=sha(FREEZE),resume=args.resume)
        write(BASE/('launch_'+str(proc.pid)+'.json'),doc); print(json.dumps(doc))
    else:
        parser.error('choose prepare/launch/supervise')
