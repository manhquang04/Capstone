"""P34A baseline replay with read-only telemetry; no DP or B3 entry point."""
from __future__ import annotations
import argparse
import json
import math
import multiprocessing as mp
import os
import socket
import statistics
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[name]='1'
from experiments import priority35b_utility as u
from experiments import priority34a_local_bn as a
torch=u.local.torch
OUT=ROOT/'artifacts/priority35b/b2b/baseline_telemetry'
PROTOCOL=ROOT/'protocols/amendments/2026-10-10_priority35b_b2b_baseline_telemetry.md'
PORT=43573

def configure():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)

def stage(name,**fields):
    u.stage(name,B2_complete=False,B2b_complete=False,B3_started=False,baseline_telemetry_only=True,**fields)

def norm_copy(upload,previous):
    if list(upload)!=list(previous): raise ValueError('telemetry tensor domain/order')
    copies=[]
    for name,value in upload.items():
        before=previous[name]
        if value.dtype!=before.dtype or value.device!=before.device or value.device.type!='cpu': raise ValueError('dtype/device mismatch')
        if value.dtype!=torch.float32 or not torch.isfinite(value).all() or not torch.isfinite(before).all(): raise ValueError('nonfinite/unsupported telemetry input')
        copies.append((value.detach().clone()-before.detach().clone()).reshape(-1))
    result=float(torch.linalg.vector_norm(torch.cat(copies)))
    if not math.isfinite(result): raise ValueError('nonfinite telemetry norm')
    return result

def train(config,path,with_telemetry=True):
    configure(); previous_out,previous_upload=a.OUT,a.upload
    a.OUT=OUT
    records=[]
    def upload(model,previous,method,seed,round_number,client,bn,allowed):
        state=previous_upload(model,previous,method,seed,round_number,client,bn,allowed)
        # Never return or write telemetry copies back into the training state.
        records.append(dict(round=round_number,client=client,norm=norm_copy(state,previous),
            tensor_keys=list(state),dtype='torch.float32',device='cpu'))
        return state
    if with_telemetry: a.upload=upload
    try:
        a.train_job(config,str(path))
        doc=u.read(path)
        if with_telemetry:
            if len(records)!=config['training']['rounds']*3: raise ValueError('telemetry missing')
            telemetry=path.with_suffix('.norms.json')
            u.save(telemetry,dict(dataset=config['dataset'],seed=config['seed'],records=records,
                statistic='unweighted median of all client/round/seed non-BN update norms',
                source_config_sha256=doc['config_sha256'],rng_consumed=False))
        return doc,records
    finally: a.OUT,a.upload=previous_out,previous_upload

def exact_pair(current,reference,base_current=OUT,base_reference=ROOT/'artifacts/priority34a'):
    if current['status']!='COMPLETED' or current['config_sha256']!=reference['config_sha256']: raise ValueError('config/completion mismatch')
    for doc,base in ((current,base_current),(reference,base_reference)):
        for name,h in doc['artifact_sha256'].items():
            if u.sha(base/name)!=h: raise ValueError('artifact hash drift '+name)
        if doc['torch_threads']!=1 or doc['payload_assertions']!=doc['config']['training']['rounds']*3 or not doc['no_bn_transmitted']: raise ValueError('thread/payload contract')
        journal=next(n for n in doc['artifact_sha256'] if n.endswith('rounds.jsonl'))
        rr=[json.loads(x) for x in (base/journal).read_text().splitlines()]
        if [r['round'] for r in rr]!=list(range(1,doc['config']['training']['rounds']+1)): raise ValueError('round coverage')
        for r in rr:
            for minima in r['bn_minima_all_steps']:
                if any(not math.isfinite(x) or x<0 for x in minima.values()): raise ValueError('BN fail closed')
    for key in ('partition_sha256','payload_keys','bn_keys','client_rows','client_fraud_counts','clients','test','validation'):
        if current[key]!=reference[key]: raise ValueError('bit-exact metadata/metrics '+key)
    states=[]
    for doc,base in ((current,base_current),(reference,base_reference)):
        ck=next(n for n in doc['artifact_sha256'] if n.endswith('final_checkpoint.pt'))
        states.append(torch.load(base/ck,map_location='cpu',weights_only=False))
    left,right=states
    if set(left)!=set(right): raise ValueError('checkpoint domain')
    pairs=[(left['global_non_bn'],right['global_non_bn']),*zip(left['client_bn'],right['client_bn'])]
    if len(left['client_bn'])!=3 or len(right['client_bn'])!=3: raise ValueError('BN clients')
    count=0
    for l,r in pairs:
        if tuple(l)!=tuple(r): raise ValueError('checkpoint key order')
        for k in l:
            x,y=l[k],r[k]
            if x.dtype!=y.dtype or x.shape!=y.shape or x.device!=y.device or x.numpy().tobytes()!=y.numpy().tobytes(): raise ValueError('checkpoint bits '+k)
            if not torch.isfinite(x).all() or ('running_var' in k and (x<0).any()): raise ValueError('nonfinite/negative checkpoint')
            count+=1
    for client in range(3):
        for split in ('validation','test'):
            suffix=f'client{client}_{split}_probabilities.npy'
            p=next(n for n in current['artifact_sha256'] if n.endswith(suffix))
            q=next(n for n in reference['artifact_sha256'] if n.endswith(suffix))
            x=u.local.np.load(base_current/p);y=u.local.np.load(base_reference/q)
            if x.dtype!=y.dtype or x.shape!=y.shape or x.tobytes()!=y.tobytes() or not u.local.np.isfinite(x).all(): raise ValueError('probability bits '+suffix)
    return dict(checkpoint_tensors=count,probability_arrays=6,metrics_exact=True,thresholds_exact=True)

def synthetic_fixture(task):
    global OUT
    prepared,output,telemetry=task
    previous_out,previous_prepared=OUT,a.p.OUT
    OUT=Path(output);a.p.OUT=Path(prepared)
    try:
        config=dict(dataset='synthetic',method='baseline',seed=321000,training=dict(alpha=.95,lr=.001,rounds=2))
        doc,records=train(config,OUT/'jobs/synthetic/321000.json',telemetry)
        ck=next(n for n in doc['artifact_sha256'] if n.endswith('final_checkpoint.pt'))
        state=torch.load(OUT/ck,map_location='cpu',weights_only=False)
        return dict(global_digest=u.local.tensor_digest(state['global_non_bn']),
            bn_digests=[u.local.tensor_digest(v) for v in state['client_bn']],
            probability_digests={Path(n).name:u.sha(OUT/n) for n in doc['artifact_sha256'] if n.endswith('.npy')},
            clients=doc['clients'],validation=doc['validation'],test=doc['test'],norms=records,
            threads=torch.get_num_threads(),interop=torch.get_num_interop_threads())
    finally: OUT,a.p.OUT=previous_out,previous_prepared

def verify_freeze():
    f=u.read(OUT/'execution_freeze.json')
    for name,h in {**f['sources'],**f['inputs']}.items():
        if u.sha(ROOT/name)!=h: raise ValueError('freeze drift '+name)
    return f

def valid(job):
    path=OUT/job['result']; receipt=path.with_suffix('.verified.json')
    if not receipt.exists(): return False
    d=u.read(receipt)
    if d['status']!='VERIFIED_BIT_EXACT' or d['job']!=job: raise ValueError('invalid verified job')
    for rel,h in d['hashes'].items():
        if u.sha(ROOT/rel)!=h: raise ValueError('completed job hash drift '+rel)
    return True

def worker(job):
    configure(); path=OUT/job['result']
    u.save(path.with_suffix('.process.json'),dict(pid=os.getpid(),job=job))
    if path.with_suffix('.failure.json').exists(): raise ValueError('preserved failure requires direction')
    if valid(job): return dict(skipped=True)
    try:
        f=u.read(OUT/'execution_freeze.json')
        if job not in f['jobs']: raise ValueError('job not frozen')
        reference=u.read(ROOT/job['reference'])
        doc,_=train(reference['config'],path)
        checks=exact_pair(doc,reference)
        paths=[path,path.with_suffix('.norms.json')]+[OUT/p for p in doc['artifact_sha256']]
        u.save(path.with_suffix('.verified.json'),dict(status='VERIFIED_BIT_EXACT',job=job,checks=checks,
            hashes={str(p.relative_to(ROOT)):u.sha(p) for p in paths}))
        return dict(skipped=False)
    except Exception:
        u.save(path.with_suffix('.failure.json'),dict(job=job,traceback=traceback.format_exc()));raise

def live():
    raw=subprocess.check_output(['ps','-axo','pid=,command='],text=True)
    found=[]
    for line in raw.splitlines():
        x=line.strip().split(None,1)
        if len(x)!=2 or int(x[0])==os.getpid(): continue
        if 'python' in Path(x[1].split()[0]).name.lower() and 'priority35b_baseline_telemetry.py' in x[1] and '--supervise' in x[1]: found.append(line)
    for p in [OUT/'supervisor_pid.json',*OUT.glob('jobs/**/*.process.json')]:
        if not p.exists(): continue
        pid=u.read(p)['pid']
        try: os.kill(pid,0)
        except ProcessLookupError: pass
        else:
            if pid!=os.getpid(): found.append(f'PID {pid} alive')
    return found

def prepare():
    if (OUT/'execution_freeze.json').exists(): raise ValueError('freeze already exists')
    configure();u.verify_freeze()
    stage('B2B_BASELINE_TELEMETRY_PREFLIGHT',done=0,total=33)
    checks=[]
    for command in ([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_priority35b_baseline_telemetry.py','-v'],['git','diff','--check']):
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        checks.append(dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        u.save(OUT/'preflight_checks.json',checks)
        if result.returncode: raise ValueError('preflight checks failed; logs retained')
    import py_compile
    for p in (Path(__file__),ROOT/'tests/test_priority35b_baseline_telemetry.py'):
        py_compile.compile(str(p),cfile=str(OUT/'compile'/(p.stem+'.pyc')),doraise=True)
    jobs=[dict(dataset=d,seed=s,reference=f'artifacts/priority34a/jobs/{d}/baseline/{s}.json',result=f'jobs/{d}/{s}.json') for d in u.design.DATASETS for s in u.design.SEEDS]
    sources=dict(u.verify_freeze()['sources']);sources[str(PROTOCOL.relative_to(ROOT))]=u.sha(PROTOCOL)
    for p in [Path(__file__),ROOT/'tests/test_priority35b_baseline_telemetry.py']: sources[str(p.relative_to(ROOT))]=u.sha(p)
    inputs=dict(u.verify_freeze()['inputs'])
    inputs[str((OUT/'preflight_checks.json').relative_to(ROOT))]=u.sha(OUT/'preflight_checks.json')
    for j in jobs:
        p=ROOT/j['reference'];inputs[j['reference']]=u.sha(p);doc=u.read(p)
        for rel,h in doc['artifact_sha256'].items():
            p=ROOT/'artifacts/priority34a'/rel
            if u.sha(p)!=h: raise ValueError('historical artifact drift')
            inputs[str(p.relative_to(ROOT))]=h
    u.save(OUT/'execution_freeze.json',dict(sources=sources,inputs=inputs,jobs=jobs,workers=4,
        protocol_sha256=u.sha(PROTOCOL),norm_statistic='median1650perdataset',device='cpu',threads=1))
    verify_freeze();stage('B2B_BASELINE_TELEMETRY_FROZEN',done=0,total=33)

def supervise():
    configure();f=verify_freeze()
    with socket.socket() as lock:
        lock.bind(('127.0.0.1',PORT));lock.listen(1)
        u.save(OUT/'supervisor_pid.json',dict(pid=os.getpid(),at=u.local.p.now()))
        if list(OUT.glob('jobs/**/*.failure.json')): raise ValueError('preserved failure')
        jobs=[j for j in f['jobs'] if not valid(j)];done=33-len(jobs);failures=[];started=time.time()
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
            futures={pool.submit(worker,j):j for j in jobs}
            for future in as_completed(futures):
                j=futures[future]
                try: future.result();done+=1
                except Exception: failures.append(dict(job=j,traceback=traceback.format_exc()))
                stage('B2B_BASELINE_TELEMETRY_RUNNING',done=done,total=33,failed=len(failures),last_job=j,
                    eta_minutes=(time.time()-started)/max(1,done)*max(0,33-done)/60)
        if failures:
            u.save(OUT/'REQUIRES_DIRECTION.json',failures);stage('B2B_BASELINE_REQUIRES_DIRECTION',done=done,failed=len(failures));return
        # Deliberately no final verification or median calculation inside supervisor.
        stage('B2B_BASELINE_REPLAYS_FINISHED',done=33,total=33,needs_postexit_verifier=True)

def finish():
    configure()
    if live() or u.live(): raise ValueError('scientific processes still alive')
    f=verify_freeze()
    if list(OUT.glob('jobs/**/*.failure.json')) or (OUT/'REQUIRES_DIRECTION.json').exists(): raise ValueError('failed replay; do not compute C')
    by_dataset={d:[] for d in u.design.DATASETS}; hashes={};checks=[]
    for j in f['jobs']:
        if not valid(j): raise ValueError('missing replay')
        path=OUT/j['result'];reference=ROOT/j['reference']
        result=exact_pair(u.read(path),u.read(reference));telemetry=path.with_suffix('.norms.json')
        records=u.read(telemetry)['records']
        if len(records)!=150 or {(r['round'],r['client']) for r in records}!={(r,c) for r in range(1,51) for c in range(3)}: raise ValueError('norm coverage')
        if not all(math.isfinite(r['norm']) and r['norm']>=0 and r['dtype']=='torch.float32' and r['device']=='cpu' for r in records): raise ValueError('norm nonfinite/device')
        by_dataset[j['dataset']].extend(r['norm'] for r in records)
        hashes.update(u.read(path.with_suffix('.verified.json'))['hashes']);hashes[j['reference']]=u.sha(reference)
        checks.append(dict(job=j,checks=result))
    if any(len(x)!=1650 for x in by_dataset.values()): raise ValueError('median population incomplete')
    values={d:dict(n=1650,C_med=statistics.median(v),C_med_half=statistics.median(v)/2,C_fixed=.1,
        ranks=[825,826],DP_norms_used=False) for d,v in by_dataset.items()}
    annex=OUT/'baseline_clip_values.json'
    if annex.exists(): raise ValueError('final values already exist; preserve')
    u.save(annex,dict(statistic='prespecified pooled median over3clients50rounds11seeds, perdataset',values=values,
        source_hashes=hashes,independent_bit_exact_checks=checks,protocol_sha256=u.sha(PROTOCOL),live_scientific_workers=[]))
    text=['# B2b baseline-derived C values — 2026-10-10, post hoc','',
        'Values derived only after all33baselines passed independent post-exit bit-exact verification. No DP norm used. These values do not authorize or freeze B2b arms by themselves.','',
        '| Dataset | Norms | C_med | C_med/2 | Fixed C |','| --- | --- | --- | --- | --- |']
    for d,v in values.items(): text.append(f"| {d} | 1650 | {v['C_med']:.17g} | {v['C_med_half']:.17g} | 0.1 |")
    text+=['','Complete source-file hashes: `artifacts/priority35b/b2b/baseline_telemetry/baseline_clip_values.json` SHA '+u.sha(annex)+'.',
        'Statistic fixed before replay in `'+str(PROTOCOL.relative_to(ROOT))+'`, SHA '+u.sha(PROTOCOL)+'.']
    addendum=ROOT/'protocols/amendments/2026-10-10_priority35b_b2b_baseline_clip_values.md'
    with addendum.open('x') as handle: handle.write('\n'.join(text)+'\n')
    u.save(OUT/'BASELINE_TELEMETRY_VERIFIED.json',dict(status='BASELINE_TELEMETRY_VERIFIED',jobs=33,
        probability_arrays=198,values_sha256=u.sha(annex),values_annex_sha256=u.sha(addendum),
        output_manifest=hashes,B2_complete=False,B2b_complete=False,B3_started=False))
    stage('B2B_BASELINE_TELEMETRY_VERIFIED',done=33,total=33,values=values,next='write full B2b execution amendment and freeze before arms')

def launch(resume=False):
    verify_freeze()
    if live() or u.live(): raise ValueError('live science; no duplicate')
    if (OUT/'launch_receipt.json').exists() and not resume: raise ValueError('already launched')
    if resume and (list(OUT.glob('jobs/**/*.failure.json')) or (OUT/'REQUIRES_DIRECTION.json').exists()): raise ValueError('preserved failure')
    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--supervise']
    with (OUT/'detached_stdout.log').open('a') as so,(OUT/'detached_stderr.log').open('a') as se:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True)
    u.save(OUT/f'launch_receipt_{time.time_ns()}.json',dict(pid=process.pid,command=command,freeze_sha256=u.sha(OUT/'execution_freeze.json')))
    # Stable administrative marker only; prior receipts retained.
    u.save(OUT/'launch_receipt.json',dict(pid=process.pid))
    print(json.dumps(dict(supervisor_pid=process.pid,jobs=33)))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for flag in ('prepare','launch','supervise','finish','resume'):parser.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.launch:launch(args.resume)
    elif args.supervise:supervise()
    elif args.finish:finish()
    else:parser.error('explicit stage required')
