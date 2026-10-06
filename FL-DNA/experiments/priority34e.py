"""Priority34E immutable configuration, detached/resumable execution and progress."""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
OUT = ROOT/'artifacts/priority34e'
PROTOCOL = ROOT/'protocols/amendments/2026-10-06_priority34e_clients_seed_search.md'
SEEDS = tuple(range(343000,343011))
KS = (10,20)

def read(path):
    return json.loads(Path(path).read_text())

def write(path, doc, once=False):
    path=Path(path)
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError('writes restricted to P34E')
    path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if once:
        with path.open('x') as f: f.write(content)
    else:
        temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(content);temp.replace(path)

def progress(stage,done=0,failed=0,**fields):
    doc=dict(stage=stage,utility_done=done,utility_total=198,failed=failed,at=time.time(),**fields)
    write(OUT/'progress.json',doc)
    with (OUT/'progress.log').open('a') as f: f.write(json.dumps(doc,allow_nan=False)+'\n')
    print(json.dumps(doc),flush=True)

def checklist(item,status):
    path=OUT/'checklist.json'
    doc=read(path) if path.exists() else {name:'pending' for name in ('amendment','preflight_freeze','seed_search','utility','independent_audit','report_seal')}
    doc[item]=status;write(path,doc)

def alive(pid):
    import psutil
    try:
        process=psutil.Process(int(pid))
        return process.is_running() and process.status()!=psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess,ValueError):return False
    except psutil.AccessDenied:return True

def freeze():
    from experiments import priority34a_local_bn as a
    p=a.p
    if (OUT/'execution_freeze.json').exists(): raise RuntimeError('freeze exists')
    seal=read(ROOT/'artifacts/priority34d/COMPLETE.json')
    assert seal['status']=='verified COMPLETE'
    assert p.sha(ROOT/'reports/priority34d_report.md')==seal['report_sha256']
    assert p.sha(ROOT/'artifacts/priority34d/audits/full_immutable_manifest.json')==seal['full_manifest_sha256']
    old=read(a.OUT/'execution_freeze.json')
    for name,digest in {**old['sources'],**old['inputs']}.items():
        if p.sha(ROOT/name)!=digest: raise RuntimeError('P34A immutable mismatch: '+name)
    sources={str(path.relative_to(ROOT)):p.sha(path) for path in list((ROOT/'experiments').glob('priority34e*.py'))+list((ROOT/'tests').glob('test_priority34e*.py'))+[PROTOCOL]}
    sources.update(old['sources'])
    inputs=dict(old['inputs'])
    for path in (a.OUT/'execution_freeze.json',ROOT/'artifacts/priority34d/COMPLETE.json',ROOT/'reports/priority34d_report.md'):
        inputs[str(path.relative_to(ROOT))]=p.sha(path)
    chosen=read(p.OUT/'execution_freeze.json')['chosen']
    jobs=[];partition_audit=[]
    for dataset in p.DATASETS:
        folder=p.OUT/'prepared'/dataset
        y=p.np.load(folder/'train_y.npy');cats=p.np.load(folder/'train_categories.npy')
        for k in KS:
            for seed in SEEDS:
                parts=p._mild_non_iid_client_indices(p.pd.DataFrame({'type':cats}),y,k,seed)
                assert len(parts)==k and sum(map(len,parts))==len(y)
                assert p.np.unique(p.np.concatenate(parts)).size==len(y)
                if any(len(v)==0 or len(v)%1024==1 for v in parts):
                    raise RuntimeError(f'empty/singleton partition {dataset}/{k}/{seed}; requires direction')
                counts=[int(y[v].sum()) for v in parts]
                assert max(counts)-min(counts)<=1
                partition_audit.append(dict(dataset=dataset,K=k,seed=seed,rows=list(map(len,parts)),fraud=counts))
                for method in p.METHODS:
                    config=dict(dataset=dataset,K=k,seed=seed,method=method,training=chosen[dataset]['training'],
                        evaluation='unweighted_mean_K_local_BN_client_metrics',source_sha256=sources)
                    assert config['training']==dict(rounds=50,lr=.001,alpha=.95)
                    jobs.append(dict(config=config,result=f'jobs/{dataset}/K{k}/{method}/{seed}.json'))
    write(OUT/'partition_preflight.json',dict(status='PASS',cells=partition_audit),once=True)
    write(OUT/'execution_freeze.json',dict(sources=sources,inputs=inputs,jobs=jobs,workers=4,
        seed_search=dict(bits=20,total=2**20,chunk_size=16384,true_base_seed=343101),
        environment=dict(python=sys.version,torch=p.torch.__version__,numpy=p.np.__version__,platform=platform.platform(),processor=platform.processor(),cpu_count=os.cpu_count(),torch_threads=1,interop_threads=1,device='cpu')),once=True)
    checklist('amendment','complete');checklist('preflight_freeze','complete');progress('FROZEN')

def verify_freeze():
    from experiments.priority32_multidataset import sha
    doc=read(OUT/'execution_freeze.json')
    for name,digest in {**doc['inputs'],**doc['sources']}.items():
        if sha(ROOT/name)!=digest:raise RuntimeError('freeze mismatch: '+name)
    return doc

def supervise(resume=False):
    lock=OUT/'supervisor.lock.json'
    if lock.exists():
        if alive(read(lock)['pid']):raise RuntimeError('live supervisor; no duplicate')
        if not resume:raise RuntimeError('stale supervisor lock requires explicit resume')
        # Preserve interrupted lock rather than deleting it.
        lock.rename(OUT/('supervisor.lock.interrupted_'+str(time.time_ns())+'.json'))
    write(lock,dict(pid=os.getpid(),started=time.time(),command=sys.argv),once=True)
    try:
        manifest=verify_freeze()
        if list(OUT.rglob('*.failure.json')) or (OUT/'REQUIRES_DIRECTION.json').exists():
            raise RuntimeError('retained failure requires direction')
        from experiments.priority34e_seed_search import run_search
        checklist('seed_search','in_progress');progress('seed_search')
        run_search();checklist('seed_search','complete')
        from experiments.priority34e_utility import valid_result,train_job
        pending=[];done=0
        for job in manifest['jobs']:
            if valid_result(OUT/job['result'],job['config']):done+=1
            else:pending.append(job)
        random.Random(343999).shuffle(pending)
        start=time.time();failed=0;initial_done=done
        checklist('utility','in_progress');progress('utility',done)
        with ProcessPoolExecutor(max_workers=manifest['workers'],mp_context=mp.get_context('spawn')) as pool:
            futures={pool.submit(train_job,j['config'],str(OUT/j['result'])):j for j in pending}
            for future in as_completed(futures):
                job=futures[future]
                try:future.result();done+=1
                except Exception:
                    failed+=1
                    write((OUT/job['result']).with_suffix('.failure.json'),dict(job=job,traceback=traceback.format_exc(),at=time.time()),once=True)
                elapsed=time.time()-start
                progress('utility',done,failed,last_job=job['result'],eta_seconds=(198-done-failed)*elapsed/(done-initial_done+failed) if done-initial_done+failed else None)
        if failed:
            write(OUT/'REQUIRES_DIRECTION.json',dict(failed=failed,done=done),once=True)
            checklist('utility','requires_direction');progress('REQUIRES_DIRECTION',done,failed);return
        write(OUT/'UTILITY_COMPLETE.json',dict(jobs=done,stage_only=True),once=True)
        checklist('utility','complete');progress('READY_FOR_INDEPENDENT_AUDIT',done)
        from experiments.priority34e_analysis import finish
        finish()
    except Exception:
        write(OUT/('supervisor_'+str(time.time_ns())+'.failure.json'),dict(traceback=traceback.format_exc(),at=time.time()),once=True)
        progress('REQUIRES_DIRECTION',failed=1);raise
    finally:
        if lock.exists() and read(lock)['pid']==os.getpid():
            lock.rename(OUT/('supervisor.lock.exited_'+str(time.time_ns())+'.json'))

def launch(resume=False):
    verify_freeze()
    if (OUT/'launch.json').exists() and not resume:raise RuntimeError('existing launch; explicit resume required')
    if (OUT/'supervisor.lock.json').exists() and alive(read(OUT/'supervisor.lock.json')['pid']):raise RuntimeError('supervisor alive')
    # A launch receipt also guards the brief interval before the child acquires its lock.
    if (OUT/'launch.json').exists() and alive(read(OUT/'launch.json')['pid']):raise RuntimeError('launched process alive')
    stamp=str(time.time_ns());command=[sys.executable,'-B',str(Path(__file__).resolve()),'--supervise']+(['--resume'] if resume else [])
    with (OUT/('stdout_'+stamp+'.log')).open('x') as out,(OUT/('stderr_'+stamp+'.log')).open('x') as err:
        process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,start_new_session=True)
    receipt=dict(pid=process.pid,command=command,resume=resume,at=time.time())
    write(OUT/('launch_'+stamp+'.json'),receipt,once=True);write(OUT/'launch.json',receipt)
    print(json.dumps(receipt))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--launch',action='store_true');parser.add_argument('--supervise',action='store_true');parser.add_argument('--resume',action='store_true');parser.add_argument('--status',action='store_true')
    args=parser.parse_args()
    if args.freeze:freeze()
    elif args.launch:launch(args.resume)
    elif args.supervise:supervise(args.resume)
    elif args.status:print(json.dumps(read(OUT/'progress.json')))
    else:parser.error('choose action')
if __name__=='__main__':main()
