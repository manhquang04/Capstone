"""Frozen, detached, attempt-preserving descriptive sanity diagnostic T."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import json
import multiprocessing as mp
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments.priority35a_runtime import ROOT,OUT,Q,load,write,tensor_save,sha,check_hashes,now
from experiments.priority35a_d1prime import P as PREVIOUS,receipt as previous_receipt
from experiments.priority35a_diagnostic import selected_cells
from experiments.priority35a_qualification import cell_id,valid_receipt
from experiments.priority35a_evaluation import quality
from experiments.priority35a_estimator import flatten
from experiments.priority35a_sanity_t_estimator import SCALES,at_rows,ratio,reconstruct

P=OUT/'sanity_t'
AMENDMENT=ROOT/'protocols/amendments/2026-10-10_priority35a_sanity_t.md'
REPORT=ROOT/'reports/priority35a_report.md'
PORT=35938


def checks(folder):
    sources=sorted([*(ROOT/'experiments').glob('priority35a_*.py'),*(ROOT/'tests').glob('test_priority35a*.py')])
    commands=[[sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_priority35a*.py','-v'],
              [sys.executable,'-m','py_compile',*[str(p.relative_to(ROOT)) for p in sources]],['git','diff','--check']]
    for i,command in enumerate(commands):
        result=subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONPYCACHEPREFIX=str(P/'bytecode')),capture_output=True,text=True)
        write(folder/f'command_{i}.json',dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),exclusive=True)
        if result.returncode:
            raise ValueError('preflight/verification command failed; logs preserved')
    for source in sources:
        if not source.read_text().endswith('\n') or any(line.rstrip(' \t')!=line for line in source.read_text().splitlines()):
            raise ValueError('source whitespace')
    return sources


def progress(stage,**fields):
    doc=load(OUT/'progress.json')
    doc.update(sanity_t_stage=stage,sanity_t_updated=now(),sanity_t_complete=fields.get('sanity_t_complete',False))
    doc.update(fields)
    write(OUT/'progress.json',doc)
    event=dict(at=now(),diagnostic='T',stage=stage,**fields)
    with (OUT/'progress.log').open('a') as f:
        f.write(json.dumps(event,allow_nan=False)+'\n')
    check=load(OUT/'checklist.json')
    check.update(sanity_t_stage=stage,sanity_t_complete=fields.get('sanity_t_complete',False),
                 sanity_t_pending=fields.get('sanity_t_pending',['independent verification','append-only report']))
    write(OUT/'checklist.json',check)
    print(event,flush=True)


def no_live():
    found=[]
    lines=subprocess.run(['ps','-axo','pid=,args='],capture_output=True,text=True,check=True).stdout.splitlines()
    for line in lines:
        parts=line.strip().split(None,1)
        if len(parts)!=2 or int(parts[0])==os.getpid():
            continue
        cmd=parts[1]
        if 'python' in Path(cmd.split()[0]).name.lower() and any(s in cmd for s in
                ('priority35a_sanity_t.py','priority35a_d1prime.py','priority35a_qualification.py','priority35a_diagnostic.py')):
            found.append(line)
    for folder in (P,PREVIOUS):
        for path in folder.glob('worker_pool_*.json'):
            for pid in load(path)['worker_pids']:
                try:
                    os.kill(pid,0)
                except ProcessLookupError:
                    continue
                found.append('recorded worker '+str(pid))
    if found:
        raise ValueError('live scientific process; no duplicate: '+repr(found))
    with socket.socket() as lock:
        lock.bind(('127.0.0.1',PORT))
    return dict(at=now(),no_live_scientific_workers=True)


def frozen():
    doc=load(P/'freeze.json')
    check_hashes(doc['hashes'])
    if sha(REPORT)!=doc['base_report_sha256']:
        raise ValueError('historical report changed before append')
    return doc


def prepare():
    no_live()
    if (P/'freeze.json').exists():
        raise ValueError('T already frozen')
    complete=load(PREVIOUS/'COMPLETE.json')
    if complete['status']!='VERIFIED_COMPLETE' or sha(PREVIOUS/'manifest.json')!=complete['manifest_sha256']:
        raise ValueError('D1-prime not verified')
    check_hashes(load(PREVIOUS/'manifest.json')['hashes'])
    if sha(REPORT)!=complete['report_sha256']:
        raise ValueError('D1-prime report mismatch')
    P.mkdir(parents=True,exist_ok=True)
    with (P/'base_report.md').open('xb') as f:
        f.write(REPORT.read_bytes())
    folder=P/'preflight'/str(time.time_ns())
    sources=checks(folder)
    hashes=load(PREVIOUS/'manifest.json')['hashes'].copy()
    # Prior frozen input manifests bind all original sources/targets/checkpoints.
    hashes.update({k:v for k,v in load(PREVIOUS/'freeze.json')['hashes'].items() if k!=str(REPORT.relative_to(ROOT))})
    paths=sources+[AMENDMENT,P/'base_report.md',PREVIOUS/'COMPLETE.json',PREVIOUS/'manifest.json',PREVIOUS/'freeze.json']+list(folder.glob('*.json'))
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    check_hashes(hashes)
    write(P/'freeze.json',dict(at=now(),hashes=hashes,amendment_sha256=sha(AMENDMENT),
          base_report_sha256=sha(REPORT),base_report_bytes=REPORT.stat().st_size,cells=selected_cells(),
          scales=list(SCALES),jobs=144,starts=2,iterations=4000,workers=4,threads=1,descriptive_only=True),exclusive=True)
    progress('FROZEN',sanity_t_jobs_total=144,sanity_t_jobs_done=0,sanity_t_failures=0,
             sanity_t_freeze_sha256=sha(P/'freeze.json'),sanity_t_pending=['T0/T1','144 T2 jobs','independent verification','report append'])


def inputs(cell,index):
    identity=f"{cell['dataset']}_r10_b4_n8_{index:02d}"
    original=valid_receipt(Q/'uploads'/identity/'receipt.json')
    private=torch.load(ROOT/original['private_path'],map_location='cpu',weights_only=False)
    observation=previous_receipt(PREVIOUS/'observations'/identity/'receipt.json')
    payload=torch.load(ROOT/observation['payload_path'],map_location='cpu',weights_only=False)
    stats=load(Q/'evaluator'/(cell['dataset']+'.json'))
    return identity,private,payload,stats,observation


def truth_checks():
    rows=[]
    for cell in selected_cells():
        for index in range(8):
            identity,private,payload,stats,_=inputs(cell,index)
            previous=previous_receipt(PREVIOUS/'jobs/a'/cell_id(cell)/f'{index:02d}.json')
            output=torch.load(ROOT/previous['reconstruction_path'],map_location='cpu',weights_only=False)
            gradient,value=at_rows(payload,private['features'],cell['instrument'])
            observed=flatten(payload['gradient'])
            denominator=float(observed.double().norm())
            diff=float((gradient.double()-observed.double()).norm())
            relative=diff/denominator if denominator else None
            _,soft=at_rows(payload,output['pooled_soft'],cell['instrument'])
            _,hard=at_rows(payload,output['rows'],cell['instrument'])
            ratio_value,reason=ratio(soft,value)
            rows.append(dict(cell=cell,index=index,identity=identity,source_ids=private['target']['source_ids'],
                 truth=quality(private['features'].numpy(),private['features'].numpy(),stats),
                 original=quality(private['features'].numpy(),output['rows'].numpy(),stats),
                 truth_objective=value,original_soft_objective=soft,original_hard_objective=hard,
                 output_truth_ratio=ratio_value,ratio_na_reason=reason,truth_strictly_lower=value<soft,
                 forward_relative_difference=relative,forward_bit_exact=torch.equal(gradient,observed),
                 forward_max_abs=float((gradient-observed).abs().max()),zero_observed_norm=denominator==0))
    write(P/'truth_checks.json',dict(at=now(),rows=rows,count=48),exclusive=True)


def receipt(path):
    doc=load(path)
    if doc['status']!='COMPLETED' or doc['freeze_sha256']!=sha(P/'freeze.json'):
        raise ValueError('T result contract')
    check_hashes(doc['hashes'])
    return doc


def worker(job):
    torch.set_num_threads(1)
    attempt=(ROOT/job['result_path']).parent/('attempt_'+str(time.time_ns()))
    attempt.mkdir(parents=True,exist_ok=False)
    write(attempt/'job.json',job,exclusive=True)
    try:
        identity,private,payload,stats,observation=inputs(job['cell'],job['index'])
        if identity!=job['identity'] or sha(ROOT/observation['payload_path'])!=job['payload_sha256']:
            raise ValueError('T input hash binding')
        result=reconstruct(payload,private['features'],stats['std'],job['cell']['instrument'],job['scale'])
        path=attempt/'reconstruction.pt'
        tensor_save(path,result,private=True)
        doc=dict(status='COMPLETED',job=job,freeze_sha256=job['freeze_sha256'],reconstruction_path=str(path.relative_to(ROOT)),
                 elapsed_seconds=result['elapsed_seconds'],hashes={str(p.relative_to(ROOT)):sha(p) for p in (path,attempt/'job.json')})
        write(ROOT/job['result_path'],doc,exclusive=True)
        return doc
    except BaseException:
        write(attempt/'failure.json',dict(at=now(),traceback=traceback.format_exc()),exclusive=True)
        raise


def supervisor():
    torch.set_num_threads(1)
    with socket.socket() as lock:
        lock.bind(('127.0.0.1',PORT))
        try:
            frozen()
            progress('TRUTH_CHECKS',sanity_t_supervisor_pid=os.getpid())
            if not (P/'truth_checks.json').exists():
                truth_checks()
            jobs=[]
            for cell in selected_cells():
                for i in range(8):
                    identity,_,_,_,observation=inputs(cell,i)
                    for scale in SCALES:
                        path=P/'jobs'/cell_id(cell)/str(scale)/f'{i:02d}.json'
                        if path.exists():
                            receipt(path)
                            continue
                        jobs.append(dict(cell=cell,index=i,scale=scale,identity=identity,
                              result_path=str(path.relative_to(ROOT)),payload_sha256=sha(ROOT/observation['payload_path']),
                              freeze_sha256=sha(P/'freeze.json')))
            done,errors=144-len(jobs),[]
            with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
                futures={pool.submit(worker,j):j for j in jobs}
                write(P/('worker_pool_'+str(time.time_ns())+'.json'),dict(at=now(),worker_pids=sorted(pool._processes),supervisor_pid=os.getpid()),exclusive=True)
                for future in as_completed(futures):
                    try:
                        future.result()
                        done+=1
                    except BaseException:
                        errors.append(dict(job=futures[future],traceback=traceback.format_exc()))
                    progress('T2_RUNNING',sanity_t_jobs_done=done,sanity_t_failures=len(errors))
            if errors:
                write(P/('REQUIRES_DIRECTION_'+str(time.time_ns())+'.json'),dict(errors=errors),exclusive=True)
                raise ValueError('submitted pool drained; no replay')
            frozen()
            write(P/'MEASUREMENTS_FINISHED.json',dict(at=now(),jobs=144,truth_checks=48,failures=0),exclusive=True)
            progress('MEASUREMENTS_FINISHED',sanity_t_jobs_done=144,sanity_t_pending=['independent verification','report append'])
        except BaseException:
            write(P/('SUPERVISOR_FAILURE_'+str(time.time_ns())+'.json'),dict(at=now(),traceback=traceback.format_exc()),exclusive=True)
            progress('REQUIRES_DIRECTION')
            raise


def launch(resume):
    frozen()
    audit=no_live()
    if (P/'launch_receipt.json').exists() and not resume:
        raise ValueError('no implicit restart')
    if list(P.rglob('failure.json')) or list(P.glob('*FAILURE*.json')) or list(P.glob('REQUIRES_DIRECTION*.json')):
        raise ValueError('failure requires direction')
    if (P/'MEASUREMENTS_FINISHED.json').exists():
        raise ValueError('already measured')
    folder=P/'launches'/str(time.time_ns())
    folder.mkdir(parents=True,exist_ok=False)
    cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--supervisor']
    with (folder/'stdout.log').open('x') as stdout,(folder/'stderr.log').open('x') as stderr:
        child=subprocess.Popen(cmd,cwd=ROOT,env=os.environ.copy(),stdout=stdout,stderr=stderr,start_new_session=True)
    doc=dict(at=now(),pid=child.pid,command=cmd,resume=resume,no_live_audit=audit,freeze_sha256=sha(P/'freeze.json'),
             stdout=str((folder/'stdout.log').relative_to(ROOT)),stderr=str((folder/'stderr.log').relative_to(ROOT)))
    write(folder/'receipt.json',doc,exclusive=True)
    if not (P/'launch_receipt.json').exists():
        write(P/'launch_receipt.json',doc,exclusive=True)
    progress('LAUNCHED',sanity_t_supervisor_pid=child.pid)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for flag in ('prepare','launch','supervisor','finish','resume'):
        parser.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    if sum((args.prepare,args.launch,args.supervisor,args.finish))!=1:
        parser.error('one action required')
    if args.prepare:
        prepare()
    elif args.launch:
        launch(args.resume)
    elif args.supervisor:
        supervisor()
    else:
        from experiments.priority35a_sanity_t_verify import finish
        finish()
