"""Frozen post-hoc B2b utility arms; no leakage entry point."""
from __future__ import annotations
import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import random
import socket
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[name]='1'
from experiments import priority35b_utility as u
from experiments import priority35b_baseline_telemetry as telemetry
torch=u.local.torch
OUT=ROOT/'artifacts/priority35b/b2b/arms'
PROTOCOL=ROOT/'protocols/amendments/2026-10-10_priority35b_b2b_execution.md'
PORT=43574

def configure():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)

def stage(name,**fields):
    u.stage(name,B2_complete=False,B2b_complete=False,B3_started=False,**fields)

def accounting(C,epsilon):
    if not math.isfinite(C) or C<=0: raise ValueError('invalid radius')
    if epsilon is None:
        return dict(C=C,sensitivity=2*C,noise_sd=0.,epsilon=None,record_level_dp=False,
                    guarantee='NO_DP_CLIP_ONLY',mechanism_location='client before transmission')
    doc=u.account(epsilon)
    doc.update(C=C,sensitivity=2*C,noise_sd=2*C*doc['sigma_sensitivity'])
    return doc

def upload_noise(state,C,epsilon,key,r,c):
    clipped,receipt=u.local.old.clip(state,'global',radius=C)
    account=accounting(C,epsilon)
    if epsilon is None:
        noisy=clipped
    else:
        if type(key) is not int: raise ValueError('private seed required')
        gen=torch.Generator().manual_seed(u.local.p.derive_seed(key,'p34b-local-gaussian',r,c))
        noisy=OrderedDict()
        for name,value in clipped.items():
            noise=torch.randn(value.shape,generator=gen,dtype=torch.float64)*account['noise_sd']
            noisy[name]=(value.double()+noise).to(value)
            u.local.old.finite(noisy[name],'B2b client noisy upload')
    receipt.update(round=r,client=c,noise_before_transmission=True,no_bn_transmitted=True,
        noise_sd=account['noise_sd'],sensitivity=2*C,noise_domain=f'round{r}/client{c}',
        noisy_delta_sha256=u.local.tensor_digest(noisy),clipped=receipt['before_norm']>C,
        gaussian_draws=epsilon is not None,clip_only=epsilon is None)
    return noisy,receipt

def train(job,config,key):
    configure()
    previous=(u.OUT,u.PROTOCOL,u.account,u.local.noise_upload,u.local.OUT,u.local.PROTOCOL,u.local.account)
    base_account=u.account
    def wrapped_account(e):
        # Avoid recursive reference when u.account is temporarily adapted.
        if e is None: return accounting(job['C'],None)
        doc=base_account(e);doc.update(C=job['C'],sensitivity=2*job['C'],noise_sd=2*job['C']*doc['sigma_sensitivity'])
        return doc
    # upload_noise uses u.account through accounting; it already returns the same
    # sigma under wrapped_account, so radius scaling remains explicit and exact.
    u.OUT,u.PROTOCOL,u.account=OUT,PROTOCOL,wrapped_account
    u.local.noise_upload=lambda state,mechanism,e,k,r,c: upload_noise(state,job['C'],e,k,r,c)
    try:
        doc=u.train(job,config,key)
    finally:
        u.OUT,u.PROTOCOL,u.account,u.local.noise_upload,u.local.OUT,u.local.PROTOCOL,u.local.account=previous
    doc.update(extension='B2b_posthoc',accounting=accounting(job['C'],job['epsilon']))
    u.save(OUT/job['result'],doc)
    return doc

def verify_freeze():
    f=u.read(OUT/'execution_freeze.json')
    for p,h in {**f['sources'],**f['inputs']}.items():
        if u.sha(ROOT/p)!=h: raise ValueError('B2b freeze drift '+p)
    return f

def valid(job):
    p=OUT/job['result'];v=p.with_suffix('.verified.json')
    if not v.exists(): return False
    receipt=u.read(v)
    if receipt['job']!=job or receipt['status']!='PASS': raise ValueError('receipt mismatch')
    for q,h in receipt['hashes'].items():
        if u.sha(ROOT/q)!=h: raise ValueError('completed job drift '+q)
    doc=u.read(p)
    if doc['status']!='COMPLETED' or doc['job_sha256']!=u.local.p.canonical_sha(job): raise ValueError('job mismatch')
    if not doc['no_bn_transmitted'] or doc['torch_threads']!=1 or doc['inter_op_threads']!=1 or doc['payload_assertions']!=150: raise ValueError('thread/payload failure')
    audits=doc['local_upload_audits'];acc=accounting(job['C'],job['epsilon'])
    if len(audits)!=150 or {(r['round'],r['client']) for r in audits}!={(r,c) for r in range(1,51) for c in range(3)}: raise ValueError('upload coverage')
    for r in audits:
        if r['C']!=job['C'] or r['after_norm']>job['C'] or r['noise_sd']!=acc['noise_sd'] or r['sensitivity']!=2*job['C']: raise ValueError('clip/accounting mismatch')
        if r['clipped']!=(r['before_norm']>job['C']) or r['gaussian_draws']!=(job['epsilon'] is not None): raise ValueError('telemetry mismatch')
        if not all(math.isfinite(x) and x>=0 for x in [r['before_norm'],r['after_norm'],*r['tensor_before'],*r['tensor_after']]): raise ValueError('nonfinite norms')
    return True

def worker(job):
    configure();path=OUT/job['result']
    u.save(path.with_suffix('.process.json'),dict(pid=os.getpid(),job=job,at=u.local.p.now()))
    if path.with_suffix('.failure.json').exists(): raise ValueError('preserved failure')
    try:
        f=u.read(OUT/'execution_freeze.json')
        if job not in f['jobs']: raise ValueError('unfrozen job')
        if valid(job): return dict(skipped=True)
        key=None if job['epsilon'] is None else u.read(OUT/'private_noise_seeds.json')[job['result']]
        start=time.monotonic()
        doc=train(job,f['baseline_configs'][job['dataset']+'/'+str(job['seed'])],key)
        hashes={str(path.relative_to(ROOT)):u.sha(path)}
        hashes.update({'artifacts/priority35b/b2b/arms/'+p:h for p,h in doc['artifact_sha256'].items()})
        u.save(path.with_suffix('.verified.json'),dict(status='PASS',job=job,hashes=hashes,elapsed_seconds=time.monotonic()-start))
        if not valid(job): raise ValueError('invalid output')
        return dict(skipped=False)
    except Exception:
        u.save(path.with_suffix('.failure.json'),dict(job=job,at=u.local.p.now(),traceback=traceback.format_exc()));raise

def live():
    found=[]
    for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        q=line.strip().split(None,1)
        if len(q)==2 and int(q[0])!=os.getpid() and 'python' in Path(q[1].split()[0]).name.lower() and 'priority35b_b2b.py' in q[1] and '--supervise' in q[1]: found.append(line)
    for p in [OUT/'supervisor_pid.json',*OUT.glob('jobs/**/*.process.json')]:
        if not p.exists(): continue
        pid=u.read(p)['pid']
        try: os.kill(pid,0)
        except ProcessLookupError: pass
        else:
            if pid!=os.getpid(): found.append(f'PID {pid} alive')
    return found

def job_list(values):
    jobs=[]
    for dataset in u.design.DATASETS:
        radii=dict(c001=.01,c01=.1,cmed_half=values[dataset]['C_med_half'],cmed=values[dataset]['C_med'])
        for tag,C in radii.items():
            for epsilon in ((None,) if tag=='c001' else (None,10,1000,1000000)):
                method=f'b2b_global_{tag}_'+('clip_only' if epsilon is None else f'eps{epsilon}')
                for seed in u.design.SEEDS:
                    jobs.append(dict(dataset=dataset,C=C,C_tag=tag,epsilon=epsilon,mechanism='global',method=method,seed=seed,
                        result=f'jobs/{dataset}/{method}/{seed}.json',stage='tabular'))
    if len(jobs)!=429: raise ValueError('job population')
    return jobs

def prepare():
    if (OUT/'execution_freeze.json').exists(): raise ValueError('freeze exists')
    configure()
    if live() or telemetry.live() or u.live(): raise ValueError('live prerequisite')
    stage('B2B_ARM_PREPARING',done=0,total=429)
    tf=telemetry.verify_freeze();uf=u.verify_freeze()
    values_path=telemetry.OUT/'baseline_clip_values.json';values=u.read(values_path)
    vr=u.read(telemetry.OUT/'BASELINE_TELEMETRY_VERIFIED.json')
    if vr['status']!='BASELINE_TELEMETRY_VERIFIED' or vr['jobs']!=33 or u.sha(values_path)!=vr['values_sha256']: raise ValueError('baseline prerequisite')
    inputs={**uf['inputs'],**tf['inputs'],**vr['output_manifest']}
    for p in (values_path,telemetry.OUT/'BASELINE_TELEMETRY_VERIFIED.json',telemetry.OUT/'execution_freeze.json',
              ROOT/'protocols/amendments/2026-10-10_priority35b_b2b_baseline_clip_values.md',u.OUT/'utility_per_seed.csv',u.OUT/'utility_table.csv',u.OUT/'matching.json',
              u.OUT/'utility_manifest.json',u.OUT/'supplementary_2026-10-10/infrastructure_seal.json'):
        inputs[str(p.relative_to(ROOT))]=u.sha(p)
    # Approved seal, not silent modification of the original B2 manifest.
    seal=u.read(u.OUT/'supplementary_2026-10-10/infrastructure_seal.json')
    for p,h in seal['effective_postexit_hashes'].items(): inputs[p]=h
    checks=[]
    for command in ([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_priority35b_b2b*.py','-v'],['git','diff','--check']):
        r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        checks.append(dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));u.save(OUT/'preflight_checks.json',checks)
        if r.returncode: raise ValueError('preflight failure preserved')
    import py_compile
    files=[ROOT/'experiments/priority35b_b2b.py',ROOT/'experiments/priority35b_b2b_analysis.py',ROOT/'tests/test_priority35b_b2b.py']
    for p in files: py_compile.compile(str(p),cfile=str(OUT/'compile'/f'{p.stem}.pyc'),doraise=True)
    jobs=job_list(values['values'])
    keys={j['result']:int.from_bytes(os.urandom(8),'big')&((1<<63)-1) for j in jobs if j['epsilon'] is not None}
    u.save(OUT/'private_noise_seeds.json',keys);(OUT/'private_noise_seeds.json').chmod(0o600)
    for p in (OUT/'private_noise_seeds.json',OUT/'preflight_checks.json'): inputs[str(p.relative_to(ROOT))]=u.sha(p)
    sources={**uf['sources'],**tf['sources'],**{str(p.relative_to(ROOT)):u.sha(p) for p in [*files,PROTOCOL]}}
    u.save(OUT/'execution_freeze.json',dict(at=u.local.p.now(),sources=sources,inputs=inputs,jobs=jobs,
        baseline_configs=uf['baseline_configs'],protocol_sha256=u.sha(PROTOCOL),workers=4,threads=1,inter_op_threads=1,device='cpu'))
    verify_freeze();stage('B2B_ARMS_FROZEN',done=0,total=429,protocol_sha256=u.sha(PROTOCOL),freeze_sha256=u.sha(OUT/'execution_freeze.json'))

def ownership():
    s=socket.socket()
    try: s.bind(('127.0.0.1',PORT));s.listen(1)
    except Exception: s.close();raise
    return s

def supervise():
    configure()
    with ownership():
        u.save(OUT/'supervisor_pid.json',dict(pid=os.getpid(),at=u.local.p.now()))
        f=verify_freeze();missing=[];done=0
        if (OUT/'REQUIRES_DIRECTION.json').exists() or list(OUT.glob('jobs/**/*.failure.json')): raise ValueError('preserved failure')
        for j in f['jobs']:
            if valid(j): done+=1
            else: missing.append(j)
        # Dataset-balanced first cohort makes the initial throughput useful.
        random.Random(353501).shuffle(missing)
        first=[next((j for j in missing if j['dataset']==d and j['C_tag']=='c01' and j['epsilon']==10),None) for d in u.design.DATASETS]
        first=[j for j in first if j is not None]
        missing=first+[j for j in missing if j not in first]
        start=time.monotonic();initial=done;failures=[]
        stage('B2B_ARMS_RUNNING',done=done,total=429,failed=0,eta_minutes=None,supervisor=os.getpid())
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
            futures={pool.submit(worker,j):j for j in missing}
            for future in as_completed(futures):
                j=futures[future]
                try: future.result();done+=1
                except Exception: failures.append(dict(job=j,traceback=traceback.format_exc()))
                elapsed=time.monotonic()-start;successes=done-initial
                eta=elapsed/successes*(429-done-len(failures))/60 if successes else None
                if successes and not (OUT/'first_runtime_estimate.json').exists():
                    cadence=30 if eta>120 else max(5,min(30,5*math.ceil(eta/20)))
                    u.save(OUT/'first_runtime_estimate.json',dict(at=u.local.p.now(),done=done,total=429,
                        elapsed_seconds=elapsed,eta_minutes=eta,monitor_minutes=cadence,first_job=j,
                        warning='preliminary completed-job four-worker throughput; heterogeneous dataset runtimes'))
                    if 'P35B_B2B_FIRST_RUNTIME_FD' in os.environ:
                        fd=int(os.environ.pop('P35B_B2B_FIRST_RUNTIME_FD'))
                        os.write(fd,json.dumps(u.read(OUT/'first_runtime_estimate.json')).encode()+b'\n');os.close(fd)
                stage('B2B_ARMS_RUNNING',done=done,total=429,failed=len(failures),last_job=j,elapsed_seconds=elapsed,eta_minutes=eta)
        if failures:
            u.save(OUT/'REQUIRES_DIRECTION.json',dict(failures=failures,done=done,total=429))
            stage('B2B_ARMS_REQUIRES_DIRECTION',done=done,total=429,failed=len(failures));return
        stage('B2B_ARMS_MEASUREMENTS_FINISHED',done=429,total=429,needs_postexit_verifier=True)

def launch(resume=False,wait_first=False):
    verify_freeze()
    if live() or u.live() or telemetry.live(): raise ValueError('live processes; no duplicate')
    with ownership(): pass
    prior=list(OUT.glob('launch_receipt_*.json'))
    if bool(prior)!=resume: raise ValueError('explicit reviewed resume only')
    if list(OUT.glob('jobs/**/*.failure.json')) or (OUT/'REQUIRES_DIRECTION.json').exists(): raise ValueError('preserved failures require direction')
    if resume:
        for j in verify_freeze()['jobs']: valid(j)
        u.save(OUT/f'resume_audit_{time.time_ns()}.json',dict(at=u.local.p.now(),no_live=True,no_measurement_replacement=True))
    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--supervise']
    env=os.environ.copy();read_fd=write_fd=None
    if wait_first:
        read_fd,write_fd=os.pipe();env['P35B_B2B_FIRST_RUNTIME_FD']=str(write_fd)
    with (OUT/'detached_stdout.log').open('a') as so,(OUT/'detached_stderr.log').open('a') as se:
        p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True,
                           env=env,pass_fds=(() if write_fd is None else (write_fd,)))
    if write_fd is not None: os.close(write_fd)
    receipt=dict(pid=p.pid,at=u.local.p.now(),freeze_sha256=u.sha(OUT/'execution_freeze.json'),command=command)
    u.save(OUT/f'launch_receipt_{time.time_ns()}.json',receipt);print(json.dumps(receipt),flush=True)
    if wait_first:
        # Event-driven blocking pipe, not a repeated progress/process/file poll.
        result=os.read(read_fd,65536);os.close(read_fd)
        if not result: raise ValueError('supervisor exited without successful runtime sample')
        print(result.decode(),flush=True)

def synthetic_fixture(task):
    global OUT
    data,output,epsilon=task;previous=(OUT,u.local.p.OUT)
    OUT=Path(output);u.local.p.OUT=Path(data)
    try:
        job=dict(dataset='synthetic',C=.1,C_tag='c01',epsilon=epsilon,mechanism='global',method='synthetic_b2b',seed=321000,result='result.json',stage='tabular')
        config=dict(dataset='synthetic',method='baseline',seed=321000,training=dict(alpha=.95,lr=.001,rounds=2))
        doc=train(job,config,891012 if epsilon else None)
        ck=next(k for k in doc['artifact_sha256'] if k.endswith('final_checkpoint.pt'))
        state=torch.load(OUT/ck,map_location='cpu',weights_only=False)
        return dict(global_digest=u.local.tensor_digest(state['global_non_bn']),bn_digests=[u.local.tensor_digest(x) for x in state['client_bn']],
            predictions={Path(k).name:u.sha(OUT/k) for k in doc['artifact_sha256'] if k.endswith('.npy')},
            audits=doc['local_upload_audits'],clients=doc['clients'],test=doc['test'],validation=doc['validation'],threads=doc['torch_threads'],interop=doc['inter_op_threads'])
    finally: OUT,u.local.p.OUT=previous

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for flag in ('prepare','launch','supervise','resume','finish','wait-first'):p.add_argument('--'+flag,action='store_true')
    args=p.parse_args()
    try:
        if args.prepare: prepare()
        elif args.launch: launch(args.resume,args.wait_first)
        elif args.supervise: supervise()
        elif args.finish:
            from experiments.priority35b_b2b_analysis import finish
            finish()
        else: p.error('explicit stage required')
    except Exception:
        u.save(OUT/f'FAILURE_{time.time_ns()}.json',dict(at=u.local.p.now(),traceback=traceback.format_exc(),stage=vars(args)));raise
