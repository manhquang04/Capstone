"""Additive P34B client-side DP extension; immutable P34A/B runner reuse."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from experiments import priority34b_core as old
np,torch,p,a=old.np,old.torch,old.p,old.a
torch.set_num_threads(1)
OUT=ROOT/'artifacts/priority34b_local_dp'
PROTOCOL=ROOT/'protocols/amendments/2026-10-03_priority34b_local_dp_extension.md'
REPORT=ROOT/'reports/priority34b_local_dp_extension_report.md'
ORIGINAL_ACCOUNT,ORIGINAL_CLIP=old.account,old.clip
PORT=43461  # Loopback-only exclusive process ownership, portable across OSs.


def read(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path)
    if OUT.resolve() not in path.resolve().parents:
        raise ValueError('local extension output firewall')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def event(kind,**fields):
    value=dict(at=p.now(),event=kind,**fields)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'runs.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,allow_nan=False)+'\n')
    print(json.dumps(value),flush=True)


def progress(part,done,total,failed=0,job=None,start=None):
    job=job or {}; elapsed=time.time()-start if start else 0
    row=dict(part=part,dataset=job.get('dataset','all'),defense=job.get('method','all'),
        comparator='client_side_DP_vs_local_BN_baseline',done=done,total=total,failed=failed,
        last_update=p.now(),eta_minutes=elapsed/max(1,done+failed)*(total-done-failed)/60 if start else None)
    save(OUT/'progress.json',row)
    with (OUT/'progress.log').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row)+'\n')


def check_item(key,status):
    doc=read(OUT/'checklist.json'); doc[key].update(status=status,last_update=p.now())
    save(OUT/'checklist.json',doc)


def account(epsilon,rounds=50,delta=1e-5):
    result=ORIGINAL_ACCOUNT(epsilon,rounds,delta)
    result.update(adjacency='per-client replace-one clipped update contribution, conditional on public history',
        release='one client:50 independently noised individual non-BN uploads before transmission',
        unit='one client contribution transcript across training; NOT record-level DP',
        mechanism_location='client before transmission',sensitivity=2*old.C,
        noise_sd=2*old.C*result['sigma_sensitivity'])
    return result


def tensor_digest(state):
    h=hashlib.sha256()
    for name,value in state.items():
        h.update(name.encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def noise_upload(state,mechanism,epsilon,key,round_number,client):
    clipped,receipt=ORIGINAL_CLIP(state,mechanism)
    accounting=account(epsilon)
    generator=torch.Generator().manual_seed(p.derive_seed(key,'p34b-local-gaussian',round_number,client))
    noisy=OrderedDict()
    for name,value in clipped.items():
        noise=torch.randn(value.shape,generator=generator,dtype=torch.float64)*accounting['noise_sd']
        noisy[name]=(value.double()+noise).to(value)
        old.finite(noisy[name],'client-side noisy upload '+name)
    receipt.update(client=client,round=round_number,noise_before_transmission=True,
        no_bn_transmitted=True,noise_sd=accounting['noise_sd'],sensitivity=2*old.C,
        noise_domain=f'round{round_number}/client{client}',noisy_delta_sha256=tensor_digest(noisy))
    return noisy,receipt


def mean_noisy_uploads(uploads,weights,previous):
    if len(uploads)!=3 or len(weights)!=3 or abs(math.fsum(weights)-1)>1e-12 or min(weights)<=0:
        raise ValueError('three-client weighted average contract')
    result=OrderedDict()
    for name in previous:
        delta=sum((row[name].double()*w for row,w in zip(uploads,weights)),torch.zeros_like(previous[name],dtype=torch.float64))
        result[name]=(previous[name].double()+delta).to(previous[name])
        old.finite(result[name],'postprocess weighted noisy uploads')
    return result


def valid(path,job):
    path=Path(path)
    if not path.exists(): return False
    doc=read(path)
    if doc.get('status')!='COMPLETED' or doc.get('job_sha256')!=p.canonical_sha(job):
        raise ValueError('existing result incompatible '+str(path))
    if not doc['no_bn_transmitted'] or doc['torch_threads']!=1 or doc['payload_assertions']!=150:
        raise ValueError('noBN/thread/assertion contract')
    if doc['accounting']['mechanism_location']!='client before transmission':
        raise ValueError('central/local confusion')
    audits=doc['local_upload_audits']
    if len(audits)!=150 or {(x['round'],x['client']) for x in audits}!={(r,c) for r in range(1,51) for c in range(3)}:
        raise ValueError('missing150 individual noise audits')
    sd=account(job['epsilon'])['noise_sd']
    for row in audits:
        if not row['noise_before_transmission'] or not row['no_bn_transmitted'] or row['after_norm']>old.C:
            raise ValueError('local clipping/upload order failed')
        if row['noise_sd']!=sd or row['sensitivity']!=2*old.C:
            raise ValueError('incorrect local sensitivity or noise scale')
    for relative,digest in doc['artifact_sha256'].items():
        if p.sha(OUT/relative)!=digest: raise ValueError('result artifact changed '+relative)
    return True


def run_job(job):
    path=OUT/job['result']
    if valid(path,job): return dict(skipped=True)
    private_seed=int(read(OUT/'private_noise_seeds.json')[job['result']])
    if job['stage']=='tabular':
        frozen=read(ROOT/'artifacts/priority34a/execution_freeze.json')
        config=copy.deepcopy(next(j['config'] for j in frozen['jobs'] if j['config']['dataset']==job['dataset']
            and j['config']['method']=='baseline' and j['config']['seed']==job['seed']))
        config.update(method=job['method'],protocol_variant='client_side_local_BN_DP',dp_job=job,
            local_dp_protocol_sha256=p.sha(PROTOCOL))
        original_out,original_upload,original_avg=a.OUT,a.upload,p.fed_avg
        audits,pending=[],[]; context={}
        def upload(model,previous,method,seed,round_number,client,bn,allowed):
            state=OrderedDict((n,model.state_dict()[n].detach().clone()-previous[n]) for n in allowed)
            a.assert_payload(state,bn,allowed)
            noisy,receipt=noise_upload(state,job['mechanism'],job['epsilon'],private_seed,round_number,client)
            a.assert_payload(noisy,bn,allowed)
            wire=OrderedDict((n,previous[n]+noisy[n]) for n in allowed)
            a.assert_payload(wire,bn,allowed)
            receipt['wire_absolute_state_sha256']=tensor_digest(wire)
            audits.append(receipt); pending.append(noisy); context.update(previous=previous,bn=bn,allowed=allowed)
            return wire
        def average(states,sizes):
            if len(pending)!=len(states): raise ValueError('noise before aggregation contract')
            weights=[s/sum(sizes) for s in sizes]
            result=mean_noisy_uploads(pending,weights,context['previous'])
            a.assert_payload(result,context['bn'],context['allowed']); pending.clear()
            return result
        a.OUT,a.upload,p.fed_avg=OUT,upload,average
        try:
            a.train_job(config,str(path)); doc=read(path)
        finally:
            a.OUT,a.upload,p.fed_avg=original_out,original_upload,original_avg
    else:
        from experiments import priority34b_images as image
        original_out,old_out,old_clip,old_average,old_valid,old_account=(image.OUT,old.OUT,old.clip,old.aggregate,old.valid,old.account)
        audits=[]; counter=[0]
        def noisy_clip(state,mechanism):
            round_number,client=counter[0]//3+1,counter[0]%3; counter[0]+=1
            noisy,receipt=noise_upload(state,mechanism,job['epsilon'],private_seed,round_number,client)
            audits.append(receipt)
            embedded={k:v for k,v in receipt.items() if k not in ('client','no_bn_transmitted')}
            return noisy,embedded
        def average(uploads,weights,epsilon,key,round_number,previous):
            if len(audits)!=round_number*3: raise ValueError('image noise upload sequence')
            return mean_noisy_uploads(uploads,weights,previous),dict(round=round_number,weights=weights,
                server_noise_added=False,**account(epsilon))
        # The old validity checker runs before the wrapper adds its local audits.
        # Only this isolated job process overrides it; final local validator is strict.
        old.OUT,image.OUT,old.clip,old.aggregate,old.valid,old.account=OUT,OUT,noisy_clip,average,lambda *_:False,account
        try:
            image.utility_job(job); doc=read(path)
        finally:
            image.OUT,old.OUT,old.clip,old.aggregate,old.valid,old.account=(original_out,old_out,old_clip,old_average,old_valid,old_account)
    doc.update(job=job,job_sha256=p.canonical_sha(job),local_upload_audits=audits,
        accounting=account(job['epsilon']),server_noise_added=False)
    save(path,doc); valid(path,job)
    return dict(skipped=False)


def verify_freeze():
    frozen=read(OUT/'execution_freeze.json')
    for relative,digest in {**frozen['sources'],**frozen['inputs']}.items():
        if p.sha(ROOT/relative)!=digest: raise ValueError('freeze mismatch '+relative)
    return frozen


def freeze():
    if (OUT/'execution_freeze.json').exists(): raise ValueError('immutable freeze already exists')
    if read(OUT/'preflight_checks.json')['status']!='PASS': raise ValueError('preflight prerequisite')
    for name in ('final_verified_receipt','administrative_verified_receipt'):
        prior=read(ROOT/f'artifacts/priority34b/{name}.json')
        if prior['status']!='PASS' or prior['jobs']!=470 or prior['live_workers']:
            raise ValueError('P34B prerequisite failed')
    original=old.verified_freeze()
    inputs={k:v for k,v in original['inputs'].items() if not k.endswith('private_noise_seeds.json')}
    for path in (ROOT/'artifacts/priority34b/jobs/cifar10/baseline').glob('*.json'):
        doc=read(path); inputs[str(path.relative_to(ROOT))]=p.sha(path)
        for relative,digest in doc['artifact_sha256'].items(): inputs['artifacts/priority34b/'+relative]=digest
    for name in ('reports/priority34b_report.md','artifacts/priority34b/administrative_verified_receipt.json',
                 'artifacts/priority34b/final_verified_receipt.json'):
        inputs[name]=p.sha(ROOT/name)
    sources=dict(original['sources'])
    for relative in ('experiments/priority34b_local_dp.py','experiments/analyze_priority34b_local_dp.py',
        'tests/test_priority34b_local_dp.py',str(PROTOCOL.relative_to(ROOT)),
        'protocols/amendments/2026-10-03_priority34c_local_dp_authorized.md',
        'protocols/amendments/2026-10-03_priority34b_local_dp_preflight_annex.md',
        'reports/priority34b_central_dp_scope_addendum.md'):
        sources[relative]=p.sha(ROOT/relative)
    jobs=[]
    for dataset in (*p.DATASETS,'cifar10'):
        for mechanism in old.MECHANISMS:
            for epsilon in old.EPSILONS:
                for seed in old.SEEDS:
                    method=f'local_dp_{mechanism}_eps{epsilon}'
                    jobs.append(dict(dataset=dataset,stage='image_utility' if dataset=='cifar10' else 'tabular',
                        mechanism=mechanism,epsilon=epsilon,seed=seed,method=method,
                        result=f'jobs/{dataset}/{method}/{seed}.json'))
    if len(jobs)!=264: raise ValueError('job count')
    keys={j['result']:int.from_bytes(os.urandom(8),'big')&((1<<63)-1) for j in jobs}
    if (OUT/'private_noise_seeds.json').exists(): raise ValueError('do not replace private keys')
    save(OUT/'private_noise_seeds.json',keys); (OUT/'private_noise_seeds.json').chmod(0o600)
    inputs[str((OUT/'private_noise_seeds.json').relative_to(ROOT))]=p.sha(OUT/'private_noise_seeds.json')
    save(OUT/'execution_freeze.json',dict(status='FROZEN',at=p.now(),jobs=jobs,workers=4,sources=sources,
        inputs=inputs,accounting=[account(e) for e in old.EPSILONS],platform=platform.platform(),
        python=sys.version,torch=torch.__version__,device='cpu',threads=1,inter_op=torch.get_num_interop_threads()))
    verify_freeze(); check_item('freeze','complete'); progress('frozen',0,264)


def ownership():
    owner=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    if os.name=='nt': owner.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
    try: owner.bind(('127.0.0.1',PORT)); owner.listen(1)
    except Exception: owner.close(); raise RuntimeError('local supervisor already active or reserved port occupied')
    return owner


def execute(job):
    path=OUT/job['result']
    if valid(path,job): return dict(skipped=True)
    if path.with_suffix('.failure.json').exists(): raise ValueError('scientific failure requires direction')
    path.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--job',json.dumps(job,sort_keys=True)]
    with path.with_suffix('.stdout.log').open('a') as so,path.with_suffix('.stderr.log').open('a') as se:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=so,stderr=se)
        save(path.with_suffix('.process.json'),dict(pid=process.pid,command=command,at=p.now()))
        code=process.wait()
    if code or not valid(path,job): raise RuntimeError(f'job failed exit{code}: {job["result"]}')
    return dict(skipped=False)


def supervise():
    with ownership():
        save(OUT/'supervisor.lock.json',dict(pid=os.getpid(),at=p.now(),port=PORT))
        frozen=verify_freeze(); jobs=frozen['jobs']; done=0; missing=[]
        for job in jobs:
            if valid(OUT/job['result'],job): done+=1
            elif (OUT/job['result']).with_suffix('.failure.json').exists():
                raise ValueError('scientific failure requires direction')
            else: missing.append(job)
        random.Random(345034).shuffle(missing); start=time.time(); failures=[]
        check_item('training','in_progress'); progress('local_dp_utility',done,len(jobs),start=start)
        event('stage_started',part='local_dp_utility',remaining=len(missing))
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(execute,j):j for j in missing}
            for future in as_completed(futures):
                job=futures[future]
                try: future.result(); done+=1; event('job_complete',job=job)
                except Exception:
                    failure=dict(job=job,at=p.now(),traceback=traceback.format_exc())
                    save((OUT/job['result']).with_suffix('.failure.json'),failure)
                    failures.append(failure); event('job_failed',result=job['result'])
                progress('local_dp_utility',done,len(jobs),len(failures),job,start)
        if failures:
            save(OUT/'REQUIRES_DIRECTION.json',dict(done=done,total=len(jobs),failed=failures)); return
        check_item('training','complete'); progress('analysis',264,264)
        from experiments.analyze_priority34b_local_dp import finish
        finish()
        save(OUT/'READY_FOR_FINAL_AUDIT.json',dict(at=p.now(),jobs=264,supervisor=os.getpid(),
            command=[sys.executable,'-B','-u','experiments/analyze_priority34b_local_dp.py','--verify']))
        progress('ready_for_post_exit_audit',264,264)


def launch(resume):
    verify_freeze()
    with ownership(): pass
    if (OUT/'COMPLETE.json').exists() or (OUT/'REQUIRES_DIRECTION.json').exists():
        raise ValueError('complete or unresolved failure, no relaunch')
    receipts=list(OUT.glob('launch_receipt_*.json'))
    if bool(receipts)!=resume: raise ValueError('explicit --resume required only after interruption review')
    if resume:
        # All prior per-job process records must be audited externally before
        # resume; live OS process identity validation is implemented in auditor.
        from experiments.analyze_priority34b_local_dp import live_workers
        if live_workers(): raise ValueError('live extension process, refuse duplicate')
        save(OUT/f'interruption_review_{time.time_ns()}.json',dict(at=p.now(),reason='explicit infrastructure resume',
            policy='same frozen keys/config; immutable partial attempt journals retained'))
    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--supervise']
    kwargs=dict(cwd=ROOT,stdin=subprocess.DEVNULL,close_fds=True)
    if os.name=='nt': kwargs['creationflags']=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS
    else: kwargs['start_new_session']=True
    with (OUT/'detached_stdout.log').open('a') as so,(OUT/'detached_stderr.log').open('a') as se:
        process=subprocess.Popen(command,stdout=so,stderr=se,**kwargs)
    receipt=dict(pid=process.pid,at=p.now(),command=command,resume=resume,freeze_sha256=p.sha(OUT/'execution_freeze.json'))
    save(OUT/f'launch_receipt_{time.time_ns()}.json',receipt); print(json.dumps(receipt),flush=True)


def preflight():
    if (OUT/'execution_freeze.json').exists(): raise ValueError('preflight only before freeze')
    import py_compile
    import shutil
    checks=[]
    for command in ([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_priority34b_local_dp.py','-v'],['git','diff','--check']):
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        checks.append(dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:
            save(OUT/f'PREFLIGHT_FAILURE_{time.time_ns()}.json',dict(checks=checks)); raise ValueError('preflight failure')
    folder=OUT/'compile_preflight'; folder.mkdir(exist_ok=True)
    for relative in ('experiments/priority34b_local_dp.py','experiments/analyze_priority34b_local_dp.py','tests/test_priority34b_local_dp.py'):
        py_compile.compile(str(ROOT/relative),cfile=str(folder/(Path(relative).stem+'.pyc')),doraise=True)
    free=shutil.disk_usage(ROOT).free
    if free<30*1024**3 or torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1:
        raise ValueError('disk/thread gate')
    save(OUT/'preflight_checks.json',dict(status='PASS',at=p.now(),checks=checks,py_compile='PASS',
        free_bytes=free,threads=1,inter_op=1,torch=torch.__version__,python=sys.version))
    check_item('preflight','complete'); progress('preflight_PASS',0,264)
    print('PREFLIGHT_PASS',flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--supervise',action='store_true'); parser.add_argument('--launch',action='store_true')
    parser.add_argument('--resume',action='store_true'); parser.add_argument('--job')
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if args.job:
        job=json.loads(args.job)
        try: run_job(job)
        except Exception:
            save((OUT/job['result']).with_suffix('.failure.json'),dict(job=job,at=p.now(),traceback=traceback.format_exc())); raise
    elif args.preflight: preflight()
    elif args.freeze: freeze()
    elif args.launch: launch(args.resume)
    elif args.supervise:
        try: supervise()
        except Exception:
            save(OUT/f'SUPERVISOR_INTERRUPTION_{time.time_ns()}.json',dict(at=p.now(),traceback=traceback.format_exc())); raise
    else: parser.error('select a mode')


if __name__=='__main__': main()
