"""Source-frozen B3 single-gradient evaluation; never writes historical artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import secrets
import socket
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from experiments import priority34c_core as core
from experiments import priority34c_confirmatory as reference
from experiments import priority35b_design as design
from experiments import priority35b_firewall as firewall
from privacy.seed_manager import derive_seed
torch,np=core.torch,core.np
OUT=ROOT/'artifacts/priority35b/b3'
BASE=ROOT/'artifacts/priority35b'
UTILITY=BASE/'b2b/arms'
PROTOCOL=ROOT/'protocols/amendments/2026-10-11_priority35b_b3_execution.md'
FREEZE=OUT/'execution_freeze.json'
PORT=43576

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def save(p,doc):
    p=Path(p)
    if not p.resolve().is_relative_to(BASE.resolve()):raise ValueError('P35B write boundary')
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def stage(name,**fields):
    doc=dict(at=now(),stage=name,scope='B3 single-gradient leakage',B2_complete=True,B2b_complete=True,
             B3_started=True,full_priority35b_complete=False,**fields)
    save(BASE/'progress.json',doc)
    with (BASE/'progress.log').open('a') as h:h.write(json.dumps(doc,allow_nan=False)+'\n')
    check=read(BASE/'checklist.json');check['scope']='B3 authorized; full P35B pending';check['B3']=doc
    check['items'].update(B3_protocol_and_arm_freeze='DONE',B3_measurements=name,
                          B3_independent_verification='PENDING',B3_figure_report_manifest='PENDING')
    save(BASE/'checklist.json',check)
    print(json.dumps(doc),flush=True)

def configure():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)

def arms(dataset):
    jobs=read(UTILITY/'execution_freeze.json')['jobs']
    result=[dict(method=m,C=None,epsilon=None,clipping=None,noise_sd=0.) for m in ('unprotected',*design.TRANSFORMS)]
    result += [dict(method=f'local_dp_{c}_eps{e}',C=.01,epsilon=e,clipping=c,noise_sd=design.account(e)['noise_sd']) for c in design.CLIPS for e in design.EPSILONS]
    for j in jobs:
        if j['dataset']!=dataset or j['seed']!=design.SEEDS[0]:continue
        sd=0. if j['epsilon'] is None else 2*j['C']*design.account(j['epsilon'])['sigma_sensitivity']
        result.append(dict(method=j['method'],C=j['C'],epsilon=j['epsilon'],clipping='global',noise_sd=sd))
    if len(result)!=32 or len({a['method'] for a in result})!=32:raise ValueError('exact32 arm coverage')
    return result

def clip_payload(raw,arm,private_seed):
    C=arm['C'];mode=arm['clipping'];sd=arm['noise_sd']
    if mode not in design.CLIPS or not raw or not math.isfinite(C) or C<=0 or not math.isfinite(sd) or sd<0:raise ValueError('mechanism contract')
    for g in raw.values():core.native.finite(g,'clipping input')
    before=[float(g.double().norm()) for g in raw.values()];norm=math.sqrt(math.fsum(n*n for n in before))
    bound=C if mode=='global' else C/math.sqrt(len(raw))
    factors=[min(1.,C/max(norm,1e-300))]*len(raw) if mode=='global' else [min(1.,bound/max(n,1e-300)) for n in before]
    clipped=[(g.double()*f*(1-4*np.finfo(np.float32).eps)).to(g) for g,f in zip(raw.values(),factors)]
    after=[float(g.double().norm()) for g in clipped];after_norm=math.sqrt(math.fsum(n*n for n in after))
    if after_norm>C or mode=='per_tensor' and any(n>bound for n in after):raise ValueError('clip bound')
    if sd:
        if type(private_seed) is not int:raise ValueError('missing private noise seed')
        gen=torch.Generator(device='cpu').manual_seed(private_seed)
        payload=[(g.double()+torch.randn(g.shape,generator=gen,dtype=torch.float64)*sd).to(g) for g in clipped]
    else:payload=clipped
    for g in payload:core.native.finite(g,'transmitted payload')
    return payload,dict(C=C,clipping=mode,noise_sd=sd,sensitivity=2*C,tensor_radius=bound,
                        before_norm=norm,after_norm=after_norm,tensor_before=before,tensor_after=after,
                        epsilon=arm['epsilon'],record_level_dp=False,clip_only=arm['epsilon'] is None,
                        no_bn_payload=True)

def metric(data,truth,recovered):
    core.native.finite(recovered,'score input')
    rec=core.native.post_process_continuous(recovered.detach().clone(),data)
    a,b=data.decode_batch(truth),data.decode_batch(rec)
    aligned,errors,_,_,rows,columns=core.native.match_reconstruction_ground_truth(a,b,data.create_tolerance_map(),return_indices=True)
    mse=float(((rec[columns].double()-truth[rows].double())**2).mean())
    acc=1-float(np.mean(errors));core.native.finite([acc,mse],'metrics')
    if not 0<=acc<=1 or mse<0:raise ValueError('invalid metric')
    return dict(feature_accuracy=acc,standardized_mse=mse,assignment=columns.tolist())

def live():
    if os.name=='nt':
        rows=read_process_windows()
        return [r for r in rows if r['ProcessId']!=os.getpid() and 'priority35b_leakage' in (r.get('CommandLine') or '')]
    rows=subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines()
    return [s for s in rows if s.strip() and int(s.split(None,1)[0])!=os.getpid() and
            'python' in Path(s.split(None,1)[1].split()[0]).name.lower() and
            'priority35b_leakage' in s and any(f in s for f in ('--supervise','--job'))]

def read_process_windows():
    return json.loads(subprocess.check_output(['powershell','-NoProfile','-Command','Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | ConvertTo-Json'],text=True))

def verify():
    f=read(FREEZE)
    for section in ('sources','inputs'):
        for name,digest in f[section].items():
            if core.sha(ROOT/name)!=digest:raise ValueError('B3 frozen drift '+name)
    return f

def configs():
    return [dict(dataset=d,cell=c,index=i,arm=a['method']) for d,c in design.QUALIFIED for i in range(39) for a in arms(d)]
def key(c):return '/'.join(str(c[k]) for k in ('dataset','cell','index','arm'))
def folder(c):return OUT/'jobs'/c['dataset']/c['cell']/('%03d'%c['index'])/c['arm']

def prepare():
    configure()
    if live():raise ValueError('active B3 workload')
    if FREEZE.exists():return verify()
    complete=read(UTILITY/'B2_B2B_COMPLETE.json')
    if complete['status']!='B2_B2B_UTILITY_VERIFIED_COMPLETE':raise ValueError('utility incomplete')
    if core.sha(UTILITY/'final_manifest.json')!=complete['manifest_sha256']:raise ValueError('utility manifest digest')
    um=read(UTILITY/'final_manifest.json')
    for p,h in um.items():
        if core.sha(ROOT/p)!=h:raise ValueError('utility output drift '+p)
    prior=reference.verify()
    n8=read(core.OUT/'gates_n8.json');n24=read(core.OUT/'gates_n24.json')
    actual={(k.split('/')[0],k.split('/')[1]) for k,v in n24.items() if v['passed'] and n8[k]['passed']}
    if actual!=set(design.QUALIFIED):raise ValueError('qualified cell mismatch')
    stage('B3_PREPARATION',measurement_runs=0,workers=8)
    hist=firewall.audit(ROOT);save(OUT/'historical_firewall.json',hist)
    targets=[]
    for d in design.DATASETS:
        source=np.load(core.PREPARED/d/'train_source_ids.npy');mapping={int(v):i for i,v in enumerate(source)}
        if len(mapping)!=len(source):raise ValueError('duplicate source mapping')
        available=sorted(set(mapping)-set(hist['exclusions'][d]));used=set()
        for dd,c in design.QUALIFIED:
            if dd!=d:continue
            b=2 if c.endswith('batch2') else 1
            rng=np.random.default_rng(derive_seed(351042,'p35b-b3-target',d,c))
            candidates=np.array([v for v in available if v not in used],dtype=np.int64)
            if len(candidates)<39*b:raise ValueError('insufficient fresh targets')
            chosen=rng.choice(candidates,39*b,replace=False).reshape(39,b).tolist()
            firewall.assert_fresh(d,chosen,hist['exclusions'][d],source.tolist())
            for i,ids in enumerate(chosen):targets.append(dict(dataset=d,cell=c,index=i,source_ids=ids,indices=[mapping[v] for v in ids]))
            used.update(v for row in chosen for v in row)
    if len(targets)!=312 or sum(len(r['source_ids']) for r in targets)!=390:raise ValueError('target coverage')
    save(OUT/'targets.json',dict(seed=351042,targets=targets))
    alljobs=configs();save(OUT/'arms.json',{d:arms(d) for d in design.DATASETS})
    keys={key(c):secrets.randbits(63) for c in alljobs if next(a for a in arms(c['dataset']) if a['method']==c['arm'])['noise_sd']>0}
    save(OUT/'private_noise.json',keys);(OUT/'private_noise.json').chmod(0o600)
    sources={**prior['sources']};inputs={**prior['inputs'],**um}
    for row in hist['inventory']:inputs[row['path']]=row['sha256']
    for path in [PROTOCOL,ROOT/'experiments/priority35b_design.py',ROOT/'experiments/priority35b_firewall.py',Path(__file__),*ROOT.glob('tests/test_priority35b_leakage*.py')]:sources[str(path.relative_to(ROOT))]=core.sha(path)
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.suffix=='.py' and p.is_relative_to(ROOT) and not p.relative_to(ROOT).parts[0].startswith('.venv'):sources[str(p.relative_to(ROOT))]=core.sha(p)
    for p in [UTILITY/'B2_B2B_COMPLETE.json',UTILITY/'final_manifest.json',UTILITY/'matching.json',UTILITY/'matching.csv',core.OUT/'gates_n8.json',core.OUT/'gates_n24.json',BASE/'b2b/baseline_telemetry/baseline_clip_values.json',ROOT/'reports/priority35b_report.md',OUT/'targets.json',OUT/'arms.json',OUT/'private_noise.json',OUT/'historical_firewall.json',OUT/'preflight_checks.json']:
        inputs[str(p.relative_to(ROOT))]=core.sha(p)
    save(FREEZE,dict(at=now(),sources=sources,inputs=inputs,jobs=alljobs,workers=8,threads=1,inter_threads=1,
                    device='cpu',fixed_holm=144,protocol_sha256=core.sha(PROTOCOL)))
    stage('B3_FROZEN',jobs=len(alljobs),targets=312,records=390,workers=8,measurement_runs=0,freeze_sha256=core.sha(FREEZE))
    return verify()

def valid(c):
    p=folder(c)/'result.json'
    if not p.exists():return False
    doc=read(p)
    if doc['config']!=c or doc['freeze_sha256']!=core.sha(FREEZE) or doc['status']!='COMPLETED':raise ValueError('result identity')
    if doc['threads']!=1 or doc['inter_threads']!=1:raise ValueError('thread contract')
    for name,h in doc['files'].items():
        if core.sha(p.parent/doc['attempt']/name)!=h:raise ValueError('result hash drift')
    return True

def job(c):
    configure();f=read(FREEZE)
    if c not in f['jobs']:raise ValueError('unfrozen job')
    path=folder(c);path.mkdir(parents=True,exist_ok=True)
    save(path/'process.json',dict(pid=os.getpid(),at=now()))
    if list(path.glob('attempt_*/failure.json')):raise ValueError('preserved failure; no retry')
    if valid(c):return dict(config=c,skipped=True)
    start=time.monotonic();dest=path/('attempt_'+str(time.time_ns()));dest.mkdir()
    try:
        data=core.PreparedDataset(c['dataset'],c['cell']);t=next(t for t in read(OUT/'targets.json')['targets'] if all(t[k]==c[k] for k in ('dataset','cell','index')))
        truth=data.Xtrain[t['indices']].clone();labels=data.ytrain[t['indices']].clone();ratio=c['cell']=='ratio_batch1'
        net=core.ratio_model(data.num_features) if ratio else core.native.make_model(data.num_features)
        raw,bn=core.gradient(net,data.de_standardize(truth) if ratio else truth,labels,ratio=ratio)
        a=next(a for a in read(OUT/'arms.json')[c['dataset']] if a['method']==c['arm'])
        if a['C'] is None:payload,plans,metadata=core.defend(raw,c['arm']);mechanism=None
        else:
            secret=read(OUT/'private_noise.json').get(key(c));payload,mechanism=clip_payload(raw,a,secret);plans=metadata=None
        seed=derive_seed(351400,'p35b-b3-reconstruction',c['dataset'],c['cell'],c['index'])
        candidates,selected=reference.recover(net,data,list(raw),payload,labels,seed,c['arm'],plans,metadata,ratio)
        score=metric(data,truth,candidates[selected]['reconstruction'])
        controls=None
        if c['arm']=='unprotected':
            cs=derive_seed(351500,'p35b-b3-control',c['dataset'],c['cell'],c['index'])
            scores,mm,emp=core.native.controls(data,truth,cs)
            prior_score=metric(data,truth,mm)
            controls=dict(scores=scores,mean_mode=mm,empirical_single=emp,control_seed=cs,mean_mode_metric=prior_score)
        else:
            u=dict(c,arm='unprotected')
            if not valid(u):raise ValueError('unprotected prerequisite missing')
            prior_score=read(folder(u)/'result.json')['controls']['mean_mode_metric']
        score['bounded_mse']=design.bounded_mse(score['standardized_mse'],prior_score['standardized_mse'])
        torch.save(dict(payload=payload,names=list(raw),labels=labels,public_model=net.state_dict(),plans=plans,metadata=metadata,bn=bn,mechanism=mechanism),dest/'server_receipt.pt')
        torch.save(dict(raw=raw,truth=truth,labels=labels,source_ids=t['source_ids'],indices=t['indices']),dest/'private_truth.pt')
        torch.save(dict(candidates=candidates,selected=selected,controls=controls),dest/'reconstruction.pt')
        save(path/'result.json',dict(status='COMPLETED',config=c,attempt=dest.name,freeze_sha256=core.sha(FREEZE),
          score=score,source_ids=t['source_ids'],mechanism=mechanism,selected_variant=candidates[selected]['name'],
          reconstruction_seed=seed,controls=None if controls is None else {k:v for k,v in controls.items() if k not in ('mean_mode','empirical_single')},
          threads=torch.get_num_threads(),inter_threads=torch.get_num_interop_threads(),elapsed_seconds=time.monotonic()-start,
          files={p.name:core.sha(p) for p in dest.glob('*.pt')},at=now()))
        if not valid(c):raise ValueError('missing output')
        return dict(config=c,elapsed_seconds=time.monotonic()-start,skipped=False)
    except Exception:
        save(dest/'failure.json',dict(config=c,at=now(),error=traceback.format_exc()));raise

def supervise():
    owner=socket.socket();owner.bind(('127.0.0.1',PORT));owner.listen(1)
    f=verify();save(OUT/'supervisor.json',dict(pid=os.getpid(),at=now(),port=PORT))
    runtimes=[];failures=[];started=time.monotonic();done=0
    for part in ('unprotected','protected'):
        jobs=sorted([c for c in f['jobs'] if (c['arm']=='unprotected')==(part=='unprotected')],key=lambda c:(c['index'],c['dataset'],c['cell'],c['arm']))
        if part=='protected' and not (OUT/'estimator_selection.json').exists():raise ValueError('selection must precede protected submission')
        with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn')) as pool:
            futures={pool.submit(job,c):c for c in jobs}
            for future in as_completed(futures):
                c=futures[future]
                try:r=future.result();runtimes.append(r)
                except Exception:failures.append(dict(config=c,error=traceback.format_exc()))
                done+=1
                save(OUT/'runtime_observations.json',runtimes)
                stage('B3_'+part.upper()+'_RUNNING',done=done,total=len(f['jobs']),failed=len(failures),elapsed_seconds=time.monotonic()-started)
        if failures:
            save(OUT/'REQUIRES_DIRECTION.json',dict(at=now(),failures=failures,pool_drained=True));stage('B3_REQUIRES_DIRECTION',done=done,total=len(f['jobs']),failed=len(failures));return
        if part=='unprotected':
            selection=[]
            for d in design.DATASETS:
                scores={c:[read(folder(dict(dataset=d,cell=c,index=i,arm='unprotected'))/'result.json')['score']['feature_accuracy'] for i in range(39)] for dd,c in design.QUALIFIED if dd==d}
                selection.append(design.select_estimator(d,scores))
            save(OUT/'estimator_selection.json',dict(at=now(),protected_submitted=False,panels=selection))
    stage('B3_MEASUREMENTS_FINISHED',done=done,total=len(f['jobs']),failed=0,needs_postexit_verifier=True)

def launch(resume=False):
    if live():raise ValueError('B3 already live')
    verify()
    if (OUT/'REQUIRES_DIRECTION.json').exists():raise ValueError('direction required')
    if (OUT/'launch.json').exists() and not resume:raise ValueError('existing launch; no duplicate')
    with (OUT/'detached_stdout.log').open('a') as stdout,(OUT/'detached_stderr.log').open('a') as stderr:
        p=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--supervise'],cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True)
    save(OUT/('launch_'+str(time.time_ns())+'.json'),dict(pid=p.pid,at=now(),resume=resume))
    if not (OUT/'launch.json').exists():save(OUT/'launch.json',dict(pid=p.pid,at=now(),workers=8))
    stage('B3_LAUNCHED',supervisor=p.pid,total=9984,workers=8)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--launch',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--supervise',action='store_true');args=p.parse_args()
    if args.prepare:prepare()
    elif args.launch:launch(args.resume)
    elif args.supervise:
        try:supervise()
        except Exception:save(OUT/'SUPERVISOR_FAILURE.json',dict(at=now(),error=traceback.format_exc()));raise
    else:p.error('choose --prepare or --launch')
