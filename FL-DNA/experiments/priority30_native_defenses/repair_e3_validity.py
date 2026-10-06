"""Priority 30 E3 repair. Old outputs are read-only; all writes are new."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.priority30_native_defenses import run_e3_utility_adult as legacy
from experiments.priority30_native_defenses import run_e3_dp_comparators as dp
from experiments.priority30_native_defenses import run_audit as audit
import numpy as np
import torch
from sklearn.model_selection import train_test_split

OLD = ROOT/'artifacts/priority30_native_defenses/audit'
OUT = OLD/'e3_repair_20261001'
STEPS = 400
SEEDS = list(range(42016,42032))
GRID = [1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1]
EXTENSION = [1.0,3.0,10.0]

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def log(message):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'journal.log').open('a') as f:
        f.write(time.strftime('%Y-%m-%dT%H:%M:%S%z')+' '+message+'\n')
    print(message,flush=True)

def data():
    old = Path.cwd()
    os.chdir(ROOT/'external_defenses/tableak')
    try:
        d = legacy.ADULT()
    finally:
        os.chdir(old)
    ids = np.arange(len(d.Xtrain))
    train, val = train_test_split(ids,test_size=0.2,random_state=43001,stratify=d.ytrain.numpy())
    mu = d.Xtrain[train].mean(0)
    sd = d.Xtrain[train].std(0).clamp_min(1e-6)
    return d, (d.Xtrain[train]-mu)/sd, d.ytrain[train], (d.Xtrain[val]-mu)/sd, d.ytrain[val], (d.Xtest-mu)/sd, d.ytest, train, val

def eval_accuracy(model,x,y,seed):
    model.eval()
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(seed)
        if isinstance(model,legacy.PrecodeAdultFC):
            probabilities = sum(model(x).softmax(1) for _ in range(8))/8
            pred = probabilities.argmax(1)
        else:
            pred = model(x).argmax(1)
    return float((pred==y).float().mean())

def train(seed, branch, sigma, clip, lr, path, test=True, rounds=STEPS):
    path = Path(path)
    if path.exists(): return json.loads(path.read_text())
    torch.set_num_threads(1)
    start=time.monotonic()
    torch.manual_seed(seed); np.random.seed(seed)
    d,x,y,v,vy,t,ty,_,_=data()
    model=legacy._new_model(d,'precode' if branch=='precode' else None)
    opt=torch.optim.SGD(model.parameters(),lr=lr)
    for step in range(rounds):
        model.train(); opt.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model(x),y)
        if branch=='precode': loss=loss+model.bottleneck.loss()
        params=[p for p in model.parameters() if p.requires_grad]
        grads=list(torch.autograd.grad(loss,params))
        if branch=='v1':
            named=dict(zip([n for n,p in model.named_parameters() if p.requires_grad],grads))
            transformed=legacy.dna_v1_gradient(named)
            grads=list(transformed.values())
        elif branch=='dp':
            grads=legacy._clip_and_noise_grads(grads,clip,sigma,seed*1000+step)
        legacy._assign_grads(model,grads); opt.step()
    r={'seed':seed,'branch':branch,'sigma':sigma,'clip_norm':clip,'lr':lr,'rounds':rounds,
       'validation_accuracy':eval_accuracy(model,v,vy,seed+500000),
       'elapsed_seconds':time.monotonic()-start}
    if test:r['test_accuracy']=eval_accuracy(model,t,ty,seed+600000)
    save(path,r); return r

def pool_jobs(jobs, workers, label):
    pending=[j for j in jobs if not Path(j[5]).exists()]
    log(f'{label} total={len(jobs)} pending={len(pending)} workers={workers}')
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(train,*j) for j in pending]
        for i,f in enumerate(as_completed(futures),1):
            f.result(); log(f'{label} completed={i}/{len(pending)} elapsed={time.monotonic()-start:.1f}')
    return [json.loads(Path(j[5]).read_text()) for j in jobs]

def baseline_gate(rows, majority_validation, majority_test):
    means={k:float(np.mean([r[k] for r in rows])) for k in ('validation_accuracy','test_accuracy')}
    passed=means['validation_accuracy']>=max(.8,majority_validation+.05) and means['test_accuracy']>=max(.8,majority_test+.05)
    return {'means':means,'majority_validation':majority_validation,'majority_test':majority_test,'passed':passed}

def select_bracket(grid,deltas,threshold):
    eligible=[s for s in grid if deltas[s]>=threshold]
    if not eligible:return None,False
    selected=max(eligible); i=grid.index(selected)
    return selected,i+1<len(grid) and deltas[grid[i+1]]<threshold

def calibrate_utility(workers):
    selected=OUT/'S4/selected.json'
    if selected.exists():return json.loads(selected.read_text())
    lr_path=OUT/'S4/lr_selection.json'
    if not lr_path.exists():
        jobs=[(seed,'baseline',None,0.,lr,str(OUT/f'S4/lr/seed_{seed}_r_{rounds}_lr_{lr}.json'),False,rounds) for seed in (42000,42001) for rounds in (200,400,800) for lr in (.1,.3,1.)]
        rows=pool_jobs(jobs,workers,'validation_lr')
        means={(rounds,lr):float(np.mean([r['validation_accuracy'] for r in rows if r['lr']==lr and r['rounds']==rounds])) for rounds in (200,400,800) for lr in (.1,.3,1.)}
        rounds,lr=max(means,key=lambda k:(means[k],-k[0],-k[1]));save(lr_path,{'lr':lr,'rounds':rounds,'validation_means':{str(k):v for k,v in means.items()},'test_used':False})
    choice=json.loads(lr_path.read_text());lr=choice['lr'];rounds=choice['rounds']
    jobs=[(seed,'baseline',None,0.,lr,str(OUT/f'S4/grid/seed_{seed}_baseline.json'),True,rounds) for seed in SEEDS]
    base=pool_jobs(jobs,workers,'baseline_gate')
    d,x,y,v,vy,t,ty,train_ids,val_ids=data()
    majv=float(torch.bincount(vy).max()/len(vy));majt=float(torch.bincount(ty).max()/len(ty))
    gate=baseline_gate(base,majv,majt)
    gate_path=OUT/'S4/baseline_gate.json'
    if not gate_path.exists():save(gate_path,gate)
    split_path=OUT/'S4/split.json'
    if not split_path.exists():save(split_path,{'train_ids':train_ids.tolist(),'validation_ids':val_ids.tolist(),'overlap':int(len(set(train_ids)&set(val_ids))),'train_only_scaling':True})
    if not gate['passed']:
        result={'status':'NOT_ASSESSABLE','reason':'baseline operating-point gate failed','baseline_gate':gate}
        save(selected,result);return result
    norms=[]
    for seed in SEEDS:
        torch.manual_seed(seed);model=legacy._new_model(d)
        grads=dp.gradient_dict(model,x,y,torch.nn.CrossEntropyLoss())
        norms.append(float(dp.flatten(list(grads.values())).norm()))
    clip=float(np.percentile(norms,95));clip_path=OUT/'S4/clip.json'
    if not clip_path.exists():save(clip_path,{'C':clip,'norms':norms,'seeds':SEEDS})
    b={r['seed']:r for r in base}
    grid=list(GRID)
    def collect(grid):
        jobs=[(s,branch,None,clip,lr,str(OUT/f'S4/grid/seed_{s}_{branch}.json'),True,rounds) for s in SEEDS for branch in ('v1','precode')]
        jobs += [(s,'dp',sigma,clip,lr,str(OUT/f'S4/grid/seed_{s}_dp_{sigma:.0e}.json'),True,rounds) for s in SEEDS for sigma in grid]
        return pool_jobs(jobs,workers,'utility_grid')
    rows=collect(grid)
    def selection(rows,grid):
        deltas={s:float(np.mean([r['validation_accuracy']-b[r['seed']]['validation_accuracy'] for r in rows if r['branch']=='dp' and r['sigma']==s])) for s in grid}
        specs={}
        for defense,branch in [('precode','precode'),('dna_v1_conservative','v1')]:
            threshold=float(np.mean([r['validation_accuracy']-b[r['seed']]['validation_accuracy'] for r in rows if r['branch']==branch]))-.005
            sigma,bracketed=select_bracket(grid,deltas,threshold)
            eps=dp.epsilon_from_rdp(sigma,1.,1e-5,1,dp.alpha_grid()) if sigma else None
            specs[defense]={'sigma':sigma,'bracketed':bracketed,'threshold':threshold,'mean_validation_delta':deltas.get(sigma),'epsilon_delta_1e_minus_5_one_release':eps['epsilon'] if eps else None}
        return specs
    specs=selection(rows,grid)
    if not all(r['bracketed'] for r in specs.values()):
        grid+=EXTENSION;rows=collect(grid);specs=selection(rows,grid)
    all_rows=base+rows
    for r in all_rows:
        for k in ('validation_accuracy','test_accuracy'):
            r['delta_'+k]=r[k]-b[r['seed']][k]
    csv_path=OUT/'S4/utility_grid.csv'
    with csv_path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in all_rows))));w.writeheader();w.writerows(all_rows)
    result={'status':'CALIBRATED','clip_norm':clip,'lr':lr,'rounds':rounds,'grid':grid,'seeds':SEEDS,'baseline_gate':gate,'selected':specs}
    save(selected,result);return result

def soteria_payload(target_id,indices):
    torch.manual_seed(42300+target_id);np.random.seed(42300+target_id)
    dataset=legacy._adult_data();x=dataset.Xtrain[indices];y=dataset.ytrain[indices]
    net=legacy._new_model(dataset);criterion=torch.nn.CrossEntropyLoss()
    base=dp.gradient_dict(net,x,y,criterion)
    defended={n:g.clone() for n,g in base.items()}
    rep_x=x.detach().clone().requires_grad_(True)
    rep=net.layers[:-1](rep_x);scores=[]
    for j in range(rep.shape[1]):
        if rep_x.grad is not None:rep_x.grad.zero_()
        net.zero_grad(set_to_none=True);rep[:,j].sum().backward(retain_graph=True)
        scores.append(rep_x.grad.detach().reshape(rep_x.shape[0],-1).norm(dim=1).sum()/(rep[:,j].detach().abs().sum()+.1))
    score=torch.stack(scores);mask=(score>=torch.quantile(score.float(),.40)).to(base['layers.3.weight'])
    defended['layers.3.weight']*=mask.unsqueeze(0)
    assert base.keys()==defended.keys()
    return list(base.values()),list(defended.values())

def recalibrate_soteria():
    path=OUT/'S3/calibration.json'
    if path.exists():return json.loads(path.read_text())
    norms=[];dist=[];dims=[]
    targets=audit.adult_targets('S2',39)
    for tid,indices in enumerate(targets):
        base,defended=soteria_payload(tid,indices);b=dp.flatten(base);d=dp.flatten(defended)
        norms.append(float(b.norm()));dist.append(float((d-b).norm()));dims.append(b.numel())
    assert len(set(dims))==1 and all(x>0 for x in dist)
    C=float(np.percentile(norms,95));med=float(np.median(dist));sigma=med/(C*math.sqrt(dims[0]))
    eps=dp.epsilon_from_rdp(sigma,1.,1e-5,1,dp.alpha_grid())
    spec={'clip_norm':C,'sigma':sigma,'median_defense_l2_distortion':med,'dimension':dims[0],'epsilon_delta_1e_minus_5_one_release':eps['epsilon'],'norms':norms,'distortions':dist,'initialization_seed':'42300+target_id','target_ids':list(range(39)),'source_ids':targets}
    calib={'adult':{'soteria':spec}}
    save(path,calib);return calib

def attack_job(kind,defense,tid,indices,spec):
    torch.set_num_threads(1);torch.manual_seed(42300+tid);np.random.seed(42300+tid)
    folder=OUT/('S3' if kind=='soteria' else 'S4')/'adult'/defense/f'target_{tid:03d}'/'result.json'
    if folder.exists():return {'skipped':True}
    current=json.loads((OLD/'S0u/adult'/f'target_{tid:03d}'/'result.json').read_text())
    assert current['source_ids']==list(indices)
    if kind=='soteria':
        return dp.run_adult_dp_attack('S3_repair','distortion_matched_clipped',defense,tid,list(indices),str(folder),spec)
    return legacy.run_adult_utility_attack('S4_repair',defense,tid,list(indices),str(folder),spec)

def run_attacks(jobs,workers):
    log(f'attacks total={len(jobs)} workers={workers}')
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(attack_job,*j) for j in jobs]
        for i,f in enumerate(as_completed(futures),1):
            f.result();log(f'attacks complete={i}/{len(jobs)} elapsed={time.monotonic()-start:.1f}s')

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['utility','soteria','all'],default='all');p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    torch.set_num_threads(1);log(f'BEGIN {a.stage}')
    targets=audit.adult_targets('S2',39)
    if a.stage in ('soteria','all'):
        spec=recalibrate_soteria();run_attacks([('soteria','soteria',tid,ids,spec) for tid,ids in enumerate(targets)],a.workers)
    if a.stage in ('utility','all'):
        selected=calibrate_utility(a.workers)
        if selected['status']=='CALIBRATED':
            jobs=[('utility',defense,tid,ids,selected) for defense,s in selected['selected'].items() if s['bracketed'] for tid,ids in enumerate(targets)]
            run_attacks(jobs,a.workers)
    log('COMPLETE '+a.stage)

if __name__=='__main__':main()
