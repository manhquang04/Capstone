"""P34D native image instrument; attacker sees only received payload."""
import argparse
import json
import os
import random
import secrets
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import priority34d_image_utility as u
from experiments import priority33c_image_recovery as instrument
from scipy.optimize import linear_sum_assignment
torch,np=u.torch,u.np
native=instrument.native
OUT=u.OUT
BASE=OUT/'image_recovery'
FREEZE=BASE/'execution_freeze.json'
ANNEX=ROOT/'protocols/amendments/2026-10-04_priority34d_image_recovery_execution.md'
read,write,append,sha=u.read,u.write,u.append,u.sha


def leaves(value):
    if isinstance(value,list):
        return [i for child in value for i in leaves(child)]
    if isinstance(value,int) and not isinstance(value,bool) and 0<=value<10000:
        return [value]
    return []


def collect(value):
    result=set()
    if isinstance(value,dict):
        for key,child in value.items():
            if key in ('indices','target_indices','image_source_ids','cifar10_index',
                       'decoy_cifar10_index','excluded','C1','C2_batch4','C2_trained',
                       'development','targets'):
                result.update(leaves(child))
            result.update(collect(child))
    elif isinstance(value,list):
        for child in value:
            result.update(collect(child))
    return result


def reserve(excluded):
    available=np.array(sorted(set(range(10000))-set(excluded)))
    if len(available)<273:
        raise ValueError('insufficient fresh CIFAR targets')
    order=np.random.default_rng(342600).permutation(available)
    position=0; groups={}
    for setting in u.SETTINGS:
        count=4 if setting=='trained_batch4' else 1
        groups[setting]=order[position:position+39*count].reshape(39,count).tolist()
        position+=39*count
    selected=[i for rows in groups.values() for row in rows for i in row]
    assert len(selected)==len(set(selected))==273 and not set(selected)&set(excluded)
    return dict(source='CIFAR10/test',seed=342600,groups=groups,excluded=sorted(excluded),total_records=273)


def history():
    excluded=set(); evidence=[]
    for base in ('artifacts','results'):
        for path in sorted((ROOT/base).rglob('*.json')):
            if OUT in path.parents or path.name=='split.json':
                continue
            if not any(tag in str(path).lower() for tag in ('image','cifar','priority29','priority30')):
                continue
            ids=collect(read(path))
            if ids:
                excluded.update(ids)
                evidence.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),count=len(ids)))
    for seed,count in ((29000801,1000),(30400,3100)):
        population=list(range(10000)); random.Random(seed).shuffle(population)
        excluded.update(population[:count])
    # Explicit mandatory reservation manifests, not only per-target results.
    for name in ('artifacts/priority33c/image_targets.json','artifacts/priority34b/image_targets.json'):
        path=ROOT/name
        ids=collect(read(path)); assert ids, 'historical manifest schema not parsed'
        excluded.update(ids)
        if not any(row['path']==name for row in evidence):
            evidence.append(dict(path=name,sha256=sha(path),count=len(ids)))
    return excluded,evidence


def build_jobs(targets,matches):
    jobs=[]; missing=[]
    for setting in u.SETTINGS:
        arms=[dict(arm='unprotected',mechanism=None,sigma=None),
              *[dict(arm=method,mechanism=None,sigma=None) for method in u.DNA]]
        for method in u.DNA:
            for mechanism in ('single','per_tensor'):
                key=setting+'::'+method+'::'+mechanism
                match=matches[key]
                if match['status']=='MATCHED':
                    arms.append(dict(arm='dp_'+mechanism+'_for_'+method,mechanism=mechanism,sigma=match['sigma']))
                else:
                    missing.append(dict(setting=setting,method=method,mechanism=mechanism,reason=match))
        for arm in arms:
            for index,ids in enumerate(targets['groups'][setting]):
                jobs.append(dict(id=setting+'__'+arm['arm']+'__'+str(index),setting=setting,
                                  target=index,indices=ids,attack_seed=342700+index,**arm))
    return jobs,missing


def verify():
    doc=read(FREEZE)
    for name,digest in {**doc['sources'],**doc['inputs']}.items():
        if sha(ROOT/name)!=digest:
            raise ValueError('image recovery freeze drift: '+name)
    return doc


def prepare():
    if FREEZE.exists():
        return verify()
    u.verify()
    complete=read(u.BASE/'UTILITY_COMPLETE.json')
    assert complete['freeze_sha256']==sha(u.FREEZE)
    assert complete['calibration_sha256']==sha(u.BASE/'calibration.json')
    assert not u.live(), 'utility still active'
    if (BASE/'PREFLIGHT_FAILURE.json').exists():
        raise RuntimeError('preserved preflight failure')
    try:
        calibration=read(u.BASE/'calibration.json')
        pairs=[tuple(p) for p in calibration['conditional_pairs']]
        assert calibration['matches']==u.selection(pairs), 'independent selection reload failed'
        excluded,evidence=history()
        targets=reserve(excluded); targets['evidence']=evidence
        if (BASE/'targets.json').exists():
            raise RuntimeError('partial target reservation; no overwrite')
        write(BASE/'targets.json',targets)
        jobs,missing=build_jobs(targets,calibration['matches'])
        private=BASE/'private_noise_seeds.json'
        descriptor=os.open(str(private),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'w') as stream:
            json.dump({job['id']:secrets.randbits(63) for job in jobs if job['mechanism'] is not None},stream)
        sources={Path(__file__).resolve(),ANNEX,ROOT/'tests/test_priority34d_image_recovery.py'}
        for module in list(sys.modules.values()):
            raw=getattr(module,'__file__',None)
            if raw:
                path=Path(raw).resolve()
                if path.is_relative_to(ROOT) and '.venv' not in str(path) and path.suffix=='.py':
                    sources.add(path)
        sources.update((ROOT/'external_defenses/invertinggradients').rglob('*.py'))
        source_map={str(p.relative_to(ROOT)):sha(p) for p in sorted(sources)}
        utility_freeze=read(u.FREEZE)
        source_map.update(utility_freeze['sources'])
        inputs=dict(utility_freeze['inputs'])
        for path in (BASE/'targets.json',private,u.FREEZE,u.BASE/'UTILITY_COMPLETE.json',u.BASE/'calibration.json'):
            inputs[str(path.relative_to(ROOT))]=sha(path)
        for row in evidence:
            inputs[row['path']]=row['sha256']
        write(FREEZE,dict(at=time.time(),sources=source_map,inputs=inputs,jobs=jobs,missing=missing,
                          instrument=dict(native.IMAGE_CONFIG),maximum_jobs=1092))
        print(json.dumps(dict(freeze_sha256=sha(FREEZE),jobs=len(jobs),missing=len(missing),targets=273)))
    except Exception as error:
        write(BASE/'PREFLIGHT_FAILURE.json',dict(error=repr(error),at=time.time(),science_started=False))
        raise


def valid(job):
    path=BASE/'jobs'/job['id']/'validated.json'
    if not path.exists():
        return False
    doc=read(path)
    if doc['job']!=job or doc['freeze_sha256']!=sha(FREEZE):
        raise ValueError('image receipt identity drift')
    for name,digest in doc['outputs'].items():
        if sha(path.parent/name)!=digest:
            raise ValueError('image receipt output mismatch')
    return True


def reconstruct(model,normalizer,received,labels,attack_seed,modes,mix=0,plans=None):
    # Deliberately no truth/raw gradient/private noise argument.
    candidates=[]
    for mode in modes:
        torch.manual_seed(attack_seed); np.random.seed(attack_seed)
        evaluator=instrument.ObservableReconstructor(model,normalizer,native.IMAGE_CONFIG.copy(),
                    received,mode,mix=mix,plans=plans,count=len(labels))
        result,stats=evaluator.reconstruct(received,labels,img_shape=(3,32,32))
        candidates.append((float(stats['opt']),mode,result.detach()))
    selected=min(enumerate(candidates),key=lambda pair:(pair[1][0],pair[0]))[1]
    return selected, {mode:objective for objective,mode,_ in candidates}


def score(truth,reconstructed):
    u.finite(truth,'metric truth'); u.finite(reconstructed,'pre-clipping reconstruction')
    matrix=((truth[:,None]-reconstructed[None,:])**2).mean(dim=(2,3,4)).numpy()
    _,columns=linear_sum_assignment(matrix)
    aligned=reconstructed[columns]
    records=[native.image_metrics(truth[i:i+1],aligned[i:i+1]) for i in range(len(truth))]
    means={key:float(np.mean([record[key] for record in records])) for key in records[0]}
    if not all(np.isfinite(value) for value in means.values()):
        raise FloatingPointError('nonfinite image metrics')
    return means,records,columns.tolist()


def attack(job,attempt):
    verify()
    if valid(job):
        return
    parent=BASE/'jobs'/job['id']; folder=parent/attempt
    if (folder/'result.json').exists():
        raise RuntimeError('attempt already completed')
    folder.mkdir(parents=True,exist_ok=True)
    started=time.time()
    try:
        data=u.old.p31.dp.CIFAR10(str(ROOT/'datasets/cifar10'),train=False,download=False)
        normal,labels=u.old.p31.tensors(data,job['indices']); normal=normal.contiguous()
        u.finite(normal,'input'); model=u.starting_model(job['setting'])
        model.load_state_dict(torch.load(u.BASE/job['setting']/'initial_state.pt',map_location='cpu',weights_only=False))
        u.check(model); model.eval()
        state=native.gradient_dict(model,normal,labels,torch.nn.CrossEntropyLoss())
        for name,value in state.items():
            u.finite(value,'raw gradient '+name)
        plans=None; mix=0
        if job['arm']=='unprotected':
            received=list(state.values()); modes=['plain']
        elif job['arm']==u.DNA[0]:
            received=list(u.old.transform(state,u.DNA[0]).values()); modes=['plain','v1_debias']; mix=.08
        elif job['arm']==u.DNA[1]:
            received,plans=native.v2_sketch_payload(state); modes=['v2_sketch']
        else:
            private_seed=read(BASE/'private_noise_seeds.json')[job['id']]
            clip=read(u.BASE/job['setting']/'clip.json')
            if job['mechanism']=='single':
                received=u.old.p31.dp.add_clipped_noise(list(state.values()),clip['C'],job['sigma'],private_seed)
            elif job['mechanism']=='per_tensor':
                received=u.old.per_tensor_noise(list(state.values()),clip['clips'],job['sigma'],private_seed)
            else:
                raise ValueError('unregistered arm')
            modes=['plain']
        for value in received:
            u.finite(value,'received payload')
        del state  # protected decoder receives no raw gradient reference
        dm=torch.tensor(native.inversefed.consts.cifar10_mean)[:,None,None]
        ds=torch.tensor(native.inversefed.consts.cifar10_std)[:,None,None]
        selected,objectives=reconstruct(model,(dm,ds),received,labels,job['attack_seed'],modes,mix,plans)
        reconstructed=selected[2]*ds+dm
        truth=normal*ds+dm
        metrics,records,pairing=score(truth,reconstructed)
        torch.save(dict(kind='transmitted_only',payload=received,projection_plans=plans),folder/'received.pt')
        np.savez(folder/'private_arrays.npz',truth=truth.numpy(),reconstruction=reconstructed.numpy(),
                 indices=job['indices'],labels=labels.numpy())
        os.chmod(folder/'private_arrays.npz',0o600)
        write(folder/'result.json',dict(status='PASS',job=job,metrics=metrics,record_metrics=records,
              pairing=pairing,selected_mode=selected[1],observable_objectives=objectives,
              freeze_sha256=sha(FREEZE),iterations=4800,restarts=1,device='cpu',
              intra_threads=torch.get_num_threads(),inter_threads=torch.get_num_interop_threads(),
              elapsed_seconds=time.time()-started))
        write(parent/'validated.json',dict(job=job,freeze_sha256=sha(FREEZE),
              outputs={str(p.relative_to(parent)):sha(p) for p in folder.iterdir()
                       if p.is_file() and p.name not in ('stdout.log','stderr.log')}))
    except Exception as error:
        write(folder/'failure.json',dict(error=repr(error),job=job,at=time.time()))
        raise


def execute(job):
    if valid(job):
        return dict(job=job,status='PASS',skipped=True)
    parent=BASE/'jobs'/job['id']
    if list(parent.glob('attempt_*/failure.json')):
        raise RuntimeError('preserved failed attack; no automatic replay')
    attempt='attempt_'+str(time.time_ns()); folder=parent/attempt; folder.mkdir(parents=True)
    append(OUT/'runs.jsonl',dict(event='image_recovery_started',job=job,attempt=attempt,at=time.time()))
    with (folder/'stdout.log').open('a') as so,(folder/'stderr.log').open('a') as se:
        proc=subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--job',job['id'],
                              '--attempt',attempt],cwd=ROOT,stdout=so,stderr=se)
    row=dict(job=job,status='PASS' if proc.returncode==0 and valid(job) else 'FAILED',
             returncode=proc.returncode,attempt=attempt,at=time.time())
    append(OUT/'runs.jsonl',dict(event='image_recovery_finished',**row))
    return row


def live():
    rows=subprocess.check_output(['ps','-axo','pid,command'],text=True).splitlines()
    return [row for row in rows if 'priority34d_image_recovery.py' in row
            and ('--supervise' in row or '--job' in row) and int(row.split()[0])!=os.getpid()]


def supervise():
    guard=socket.socket(); guard.bind(('127.0.0.1',43469)); guard.listen(1)
    assert not live() and not u.live()
    rows=subprocess.check_output(['ps','-axo','pid,command'],text=True).splitlines()
    assert not any('priority34d_bn.py' in row and '--supervise' in row for row in rows)
    doc=verify(); write(BASE/'supervisor.lock.json',dict(pid=os.getpid(),at=time.time()))
    checklist=read(OUT/'checklist.json'); checklist['confirmatory']['status']='in_progress'
    checklist['recovery_drivers_targets_freezes']['status']='completed'; write(OUT/'checklist.json',checklist)
    completed=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(execute,job):job for job in doc['jobs']}
        for future in as_completed(futures):
            try:
                row=future.result()
            except Exception as error:
                row=dict(job=futures[future],status='FAILED',error=repr(error))
            completed.append(row)
            progress=dict(part='image_confirmatory',done=len(completed),total=len(doc['jobs']),
                          failed=sum(row['status']!='PASS' for row in completed),last_update=time.time())
            write(OUT/'progress.json',progress); append(OUT/'progress.log',progress)
    failed=[row for row in completed if row['status']!='PASS']
    if failed:
        write(BASE/'REQUIRES_DIRECTION.json',dict(failed=failed,pool_drained=True))
        raise RuntimeError('image failures; direction required')
    write(BASE/'IMAGE_COMPLETE.json',dict(stage_only=True,jobs=len(completed),at=time.time(),freeze_sha256=sha(FREEZE)))
    checklist['confirmatory']['status']='completed'; write(OUT/'checklist.json',checklist)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for flag in ('prepare','launch','supervise','resume'):
        parser.add_argument('--'+flag,action='store_true')
    parser.add_argument('--job'); parser.add_argument('--attempt')
    args=parser.parse_args()
    if args.prepare:
        prepare()
    elif args.job:
        selected=[job for job in read(FREEZE)['jobs'] if job['id']==args.job]
        assert len(selected)==1 and args.attempt
        attack(selected[0],args.attempt)
    elif args.supervise:
        try:
            supervise()
        except Exception as error:
            write(BASE/'SUPERVISOR_FAILURE.json',dict(error=repr(error),at=time.time())); raise
    elif args.launch:
        verify(); assert not live() and not u.live()
        assert not (BASE/'IMAGE_COMPLETE.json').exists() and not (BASE/'REQUIRES_DIRECTION.json').exists()
        assert not (BASE/'SUPERVISOR_FAILURE.json').exists()
        if list(BASE.glob('launch_*.json')) and not args.resume:
            raise RuntimeError('previous launch; verified infrastructure resume only')
        with (BASE/'detached_stdout.log').open('a') as so,(BASE/'detached_stderr.log').open('a') as se:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--supervise'],
                                  cwd=ROOT,stdout=so,stderr=se,start_new_session=True)
        doc=dict(pid=proc.pid,at=time.time(),freeze_sha256=sha(FREEZE),resume=args.resume)
        write(BASE/('launch_'+str(proc.pid)+'.json'),doc); print(json.dumps(doc))
    else:
        parser.error('choose prepare/launch/supervise/job')
