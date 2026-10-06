"""P34B central Gaussian DP, immutable-input IO and accounting."""
from __future__ import annotations
import copy
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import priority34a_local_bn as a
np, torch, p = a.np, a.torch, a.p
OUT = ROOT / 'artifacts/priority34b'
PROTOCOL = ROOT / 'protocols/amendments/2026-10-03_priority34b_meaningful_dp.md'
SEEDS = list(range(321000, 321011))
EPSILONS = (1, 3, 10)
MECHANISMS = ('global', 'per_tensor')
C = .01


def finite(value, location):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError('nonfinite '+location)


def check(model):
    return a.guard(model,'P34B model guard')


def save(path, obj):
    path = Path(path)
    if OUT.resolve() not in path.resolve().parents:
        raise ValueError('P34B write outside new namespace')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def noise_key(job):
    # Never derive privacy noise from the public training/target seed.
    path=OUT/'private_noise_seeds.json'
    return int(read(path)[job['result']])


def event(kind, **fields):
    row = dict(at=p.now(), event=kind, **fields)
    with (OUT/'runs.jsonl').open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')
    print(json.dumps(row), flush=True)


def progress(part, done, total, failed=0, job=None, start=None):
    job = job or {}
    elapsed = time.time()-start if start else 0
    row = dict(part=part, dataset=job.get('dataset','all'), defense=job.get('method','all'),
               comparator='local_bn_baseline', done=done, total=total, failed=failed,
               last_update=p.now(), eta_minutes=elapsed/(done+failed)*(total-done-failed)/60
               if start and done+failed else None)
    save(OUT/'progress.json', row)
    with (OUT/'progress.log').open('a') as stream:
        stream.write(json.dumps(row)+'\n')


def checklist(item, status):
    doc = read(OUT/'checklist.json')
    doc[str(item)].update(status=status, last_update=p.now())
    save(OUT/'checklist.json', doc)


def account(epsilon, rounds=50, delta=1e-5):
    if epsilon <= 0 or rounds < 1 or not 0 < delta < 1:
        raise ValueError('invalid accountant inputs')
    logdelta = math.log(1/delta)
    # Rationalized expression avoids cancellation for small epsilon.
    rho = (epsilon/(math.sqrt(logdelta+epsilon)+math.sqrt(logdelta)))**2
    sigma = math.sqrt(rounds/(2*rho))
    alpha = 1+math.sqrt(logdelta/rho)
    bound = rounds*alpha/(2*sigma*sigma)+logdelta/(alpha-1)
    if abs(bound-epsilon) > 1e-10:
        raise ValueError('accounting inversion failure')
    return dict(target_epsilon=epsilon, certified_epsilon=bound, delta=delta,
                sigma_sensitivity=sigma, alpha=alpha, rho=rho, rounds=rounds, q=1,
                adjacency='fixed-slot replace-one client/update-generating dataset',
                release='50 noisy non-BN global broadcasts; trusted central aggregation',
                conversion='Gaussian RDP composed at q=1, standard delta conversion')


def clip(state, mechanism, radius=C):
    if mechanism not in MECHANISMS or not state or radius <= 0:
        raise ValueError('invalid clipping contract')
    for value in state.values():
        finite(value, 'clipping input')
        if value.device.type != 'cpu':
            raise ValueError('CPU only')
    before = [float(value.double().norm()) for value in state.values()]
    total = math.sqrt(math.fsum(value*value for value in before))
    bound = radius if mechanism == 'global' else radius/math.sqrt(len(state))
    factors = [min(1., radius/max(total,1e-300))]*len(state) if mechanism == 'global' else [min(1.,bound/max(n,1e-300)) for n in before]
    # Deterministic small inward rounding: never exceeds intended L2 radius.
    result = OrderedDict((name, (value.double()*factor*(1-4*np.finfo(np.float32).eps)).to(value))
                         for (name,value),factor in zip(state.items(),factors))
    after = [float(value.double().norm()) for value in result.values()]
    norm = math.sqrt(math.fsum(n*n for n in after))
    if norm > radius or (mechanism=='per_tensor' and any(n > bound for n in after)):
        raise ValueError('clipping bound failed')
    return result, dict(mechanism=mechanism, C=radius, tensor_radius=bound,
                       before_norm=total, after_norm=norm, tensor_before=before,
                       tensor_after=after, tensor_count=len(state))


def aggregate(clipped, weights, epsilon, seed, round_number, previous):
    if len(clipped)!=3 or len(weights)!=3 or min(weights)<=0 or abs(math.fsum(weights)-1)>1e-12:
        raise ValueError('fixed three-client weights invalid')
    accounting = account(epsilon)
    sensitivity = 2*max(weights)*C
    sd = accounting['sigma_sensitivity']*sensitivity
    generator = torch.Generator().manual_seed(p.derive_seed(seed,'p34b-server-gaussian',round_number))
    result = OrderedDict()
    for name in previous:
        mean = sum((row[name].double()*weight for row,weight in zip(clipped,weights)), torch.zeros_like(previous[name],dtype=torch.float64))
        noise = torch.randn(mean.shape, generator=generator, dtype=torch.float64)*sd
        result[name] = (previous[name].double()+mean+noise).to(previous[name])
        finite(result[name], 'noisy server aggregate')
    return result, dict(round=round_number, weights=weights, sensitivity=sensitivity,
                        noise_sd=sd, **accounting)


def valid(path, job):
    path = Path(path)
    if not path.exists():
        return False
    doc = read(path)
    if doc.get('status')!='COMPLETED' or doc.get('job_sha256')!=p.canonical_sha(job):
        raise ValueError('incompatible/corrupt existing result '+str(path))
    for relative,digest in doc['artifact_sha256'].items():
        if p.sha(OUT/relative)!=digest:
            raise ValueError('changed result artifact '+relative)
    if doc['torch_threads']!=1 or not doc['no_bn_transmitted']:
        raise ValueError('thread/payload contract')
    if job['stage']=='tabular' and len(doc['dp_round_audit'])!=50:
        raise ValueError('missing DP round audit')
    return True


def tabular_job(job):
    path = OUT/job['result']
    if valid(path, job):
        return dict(path=str(path), skipped=True)
    seed, dataset = job['seed'], job['dataset']
    private_seed=noise_key(job)
    frozen = read(ROOT/'artifacts/priority34a/execution_freeze.json')
    config = copy.deepcopy(next(j['config'] for j in frozen['jobs']
                   if j['config']['dataset']==dataset and j['config']['method']=='baseline' and j['config']['seed']==seed))
    config.update(method=job['method'], protocol_variant='p34b_local_bn_central_DP',
                  dp_job=job, p34b_protocol_sha256=p.sha(PROTOCOL))
    original_out, original_upload, original_average = a.OUT, a.upload, a.p.fed_avg
    a.OUT = OUT
    audits, pending = [], []
    context = {}

    def upload(local, previous, method, seed_value, round_number, client, bn, allowed):
        state = OrderedDict((name,local.state_dict()[name].detach().clone()) for name in allowed)
        a.assert_payload(state,bn,allowed)
        delta = OrderedDict((name,state[name]-previous[name]) for name in allowed)
        clipped, receipt = clip(delta,job['mechanism'])
        pending.append(clipped)
        context.update(previous=previous,round=round_number,bn=bn,allowed=allowed)
        audits.append(dict(client=client,round=round_number,no_bn_transmitted=True,**receipt))
        # Original runner's wire firewall is checked; server uses exact stored
        # clipped deltas, not cancellation-prone absolute-state subtraction.
        return OrderedDict((name,previous[name]+clipped[name]) for name in allowed)

    server_audit = []
    def average(states, sizes):
        weights = [size/sum(sizes) for size in sizes]
        result, audit = aggregate(pending,weights,job['epsilon'],private_seed,context['round'],context['previous'])
        audit['client_clip_audits'] = audits[-3:]
        server_audit.append(audit)
        pending.clear()
        a.assert_payload(result,context['bn'],context['allowed'])
        return result

    a.upload, a.p.fed_avg = upload, average
    try:
        a.train_job(config,str(path))
        doc = read(path)
        doc.update(job=job,job_sha256=p.canonical_sha(job),dp_round_audit=server_audit,
                   accounting=account(job['epsilon']))
        save(path,doc)
        valid(path,job)
        return dict(path=str(path),skipped=False)
    finally:
        a.OUT, a.upload, a.p.fed_avg = original_out, original_upload, original_average


def verified_freeze():
    frozen = read(OUT/'execution_freeze.json')
    for relative,digest in {**frozen['inputs'],**frozen['sources']}.items():
        if p.sha(ROOT/relative)!=digest:
            raise ValueError('frozen source/input changed '+relative)
    return frozen
