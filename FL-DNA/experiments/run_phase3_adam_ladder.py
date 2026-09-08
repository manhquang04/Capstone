"""Close Phase 3 with a predeclared Adam update-reconstruction difficulty ladder."""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from attacks.local_update import simulate
from data.load_creditcard import BASE_FEATURE_COLUMNS, NUMERIC_COLUMNS, _build_features
from experiments import fraud_fl_common as common
from experiments.phase3_bounded_validation import _align_for_evaluation, _decode, update_objective
from experiments.run_phase3_full_client import checksum, dump, score
from models.fraud_mlp import FraudMLP
from privacy.seed_manager import derive_seed, generate_run_seed

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'artifacts/phase3/full_20260908T143837588533Z'
OUT=ROOT/'artifacts/phase3_closure/adam_ladder_v2'


def sign_tail(wins,n):
    return sum(math.comb(n,k) for k in range(wins,n+1))/2**n if n else 1.0


def metadata():
    return SimpleNamespace(**json.loads((REFERENCE/'preprocessing.json').read_text()))


def excluded_rows(labels):
    ids=np.arange(len(labels)); _,first=train_test_split(ids,test_size=500000,random_state=20260907,stratify=labels)
    excluded=set(first.tolist())
    for path in (ROOT/'artifacts').rglob('*targets.pt'):
        try:
            payload=torch.load(path,weights_only=False)
            if isinstance(payload,list):
                for group in payload: excluded.update(group.get('source_ids',[]))
            elif isinstance(payload,dict) and 'source_ids' in payload:
                excluded.update(payload['source_ids'].tolist())
        except (RuntimeError,EOFError,KeyError,TypeError):
            continue
    return excluded


def make_groups(frame, available, count, records, fraud, seed):
    labels=frame.isFraud.to_numpy(); rng=np.random.default_rng(seed)
    pos=np.asarray([i for i in available if labels[i]==1]); neg=np.asarray([i for i in available if labels[i]==0])
    chosen_pos=rng.choice(pos,count*fraud,replace=False).reshape(count,fraud)
    chosen_neg=rng.choice(neg,count*(records-fraud),replace=False).reshape(count,records-fraud)
    meta=metadata(); groups=[]
    for group_id in range(count):
        ids=np.concatenate((chosen_pos[group_id],chosen_neg[group_id])); rng.shuffle(ids)
        features=_build_features(frame.iloc[ids].copy())
        scaled=(features[NUMERIC_COLUMNS].to_numpy()-np.asarray(meta.numeric_center))/np.asarray(meta.numeric_scale)
        onehot=np.column_stack([(features.type.to_numpy()==value).astype(float) for value in meta.type_categories])
        groups.append(dict(source_ids=ids.tolist(),x=np.hstack((scaled,onehot)).astype('float32'),
                           y=labels[ids].astype('float32').reshape(-1,1)))
    return groups


def capture(group, seed, batch_size):
    x=torch.from_numpy(group['x']);y=torch.from_numpy(group['y']);batches=[]
    for start in range(0,len(x),batch_size): batches.append(slice(start,min(start+batch_size,len(x))))
    model=FraudMLP(x.shape[1]).train();model.load_state_dict(torch.load(REFERENCE/'pre_local.pt',weights_only=False))
    initial=copy.deepcopy(model.state_dict()); criterion=common.BinaryFocalLoss(); rng=torch.Generator().manual_seed(seed).get_state()
    observed=simulate(model,criterion,x,y,batches,rng)
    native=copy.deepcopy(model);optimizer=torch.optim.Adam(native.parameters(),lr=.001)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(rng)
        for ids in batches:
            optimizer.zero_grad();criterion(native(x[ids]),y[ids]).backward();optimizer.step()
    for key,value in native.state_dict().items():
        torch.testing.assert_close(observed[key],value-initial[key],atol=2e-7,rtol=2e-4)
    return model,criterion,x,y,batches,rng,observed


def run_pair(folder,group,group_id,restart,run_seed,lr,iterations,batch_size):
    folder.mkdir(parents=True,exist_ok=False); started=time.perf_counter()
    model,criterion,x,y,batches,rng,observed=capture(
        group, derive_seed(run_seed,'local',group_id,batch_size), batch_size)
    keys=[key for key,value in observed.items() if value.is_floating_point()]
    initial=torch.randn(x.shape,generator=torch.Generator().manual_seed(derive_seed(run_seed,'initial',group_id,restart,batch_size)))
    meta=metadata(); prior=_align_for_evaluation(x,_decode(initial),y); prior_metrics=score(x,prior,y,meta,folder/'prior.csv')
    rows=[]
    for method in ('baseline','zero_update'):
        signal=observed if method=='baseline' else {key:torch.zeros_like(value) for key,value in observed.items()}
        latent=initial.clone().requires_grad_(True);optimizer=torch.optim.Adam([latent],lr=lr)
        best=float('inf');best_step=0;candidate=latent.detach().clone();history=[]
        for step in range(iterations+1):
            delta=simulate(model,criterion,_decode(latent),y,batches,rng)
            loss=update_objective(delta,signal,keys,reference=observed,mode='cosine_magnitude')
            value=float(loss.detach());history.append(value)
            if value<best: best=value;best_step=step;candidate=latent.detach().clone()
            if step<iterations:
                gradient,=torch.autograd.grad(loss,latent);optimizer.zero_grad();latent.grad=gradient;optimizer.step()
        reconstruction=_decode(candidate);aligned=_align_for_evaluation(x,reconstruction,y)
        metrics=score(x,aligned,y,meta,folder/f'{method}.csv')
        artifact=dict(method=method,source_ids=group['source_ids'],original=x,labels=y,initial=initial,
                      reconstruction=reconstruction,aligned=aligned,best_objective=best,best_step=best_step,
                      history=history,observed=observed,keys=keys)
        torch.save(artifact,folder/f'{method}.pt');loaded=torch.load(folder/f'{method}.pt',weights_only=False)
        checked=update_objective(simulate(model,criterion,loaded['reconstruction'],y,batches,rng),signal,keys,
                                 reference=observed,mode='cosine_magnitude')
        torch.testing.assert_close(checked.detach(),torch.tensor(best),rtol=1e-5,atol=1e-10)
        rows.append(dict(method=method,group=group_id,restart=restart,objective=best,best_step=best_step,
                         metrics=metrics,prior=prior_metrics,seconds=time.perf_counter()-started))
    dump(folder/'results.json',rows);return rows


def evaluate(rows):
    selected=[]
    for group in sorted({row['group'] for row in rows}):
        baseline=min((row for row in rows if row['group']==group and row['method']=='baseline'),key=lambda row:row['objective'])
        zero=min((row for row in rows if row['group']==group and row['method']=='zero_update'),key=lambda row:row['objective'])
        priors=[row['prior']['mean_mse'] for row in rows if row['group']==group and row['method']=='baseline']
        selected.append(dict(group=group,baseline=baseline['metrics']['mean_mse'],prior=float(np.mean(priors)),
                             zero=zero['metrics']['mean_mse'],baseline_restart=baseline['restart'],zero_restart=zero['restart']))
    result={}
    for control in ('prior','zero'):
        values=np.asarray([row['baseline']-row[control] for row in selected]);nonzero=values[values!=0];wins=int((nonzero<0).sum())
        result[control]=dict(wins=wins,n=len(nonzero),mean=float(values.mean()),median=float(np.median(values)),p=sign_tail(wins,len(nonzero)))
    result['gate']=all(v['mean']<0 and v['median']<0 and v['p']<.05 for v in result.values())
    return result,selected


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True);torch.set_num_threads(1);common.DEVICE=torch.device('cpu');start=time.perf_counter()
    run_seed=generate_run_seed();frame=pd.read_csv(ROOT/'datasets/creditcard.csv',usecols=BASE_FEATURE_COLUMNS+['isFraud'])
    excluded=excluded_rows(frame.isFraud.to_numpy());available=np.asarray(sorted(set(range(len(frame)))-excluded))
    protocol=dict(run_seed=run_seed,tiers={'one_batch':{'records':4,'fraud':1,'batch_size':4},
             'four_batches':{'records':16,'fraud':4,'batch_size':4}},development_groups=4,
             confirmation_groups=10,iterations=300,restarts=3,learning_rates=[.01,.05,.1],
             optimizer='local Adam lr=.001; attack Adam',loss='focal alpha=.95 gamma=2',
             observation='full floating model state delta including differentiable BatchNorm buffers',
             gate='negative mean/median versus prior and zero plus one-sided tie-excluding sign p<.05',
             scope='development ladder; confirm one_batch only if development improves prior',
             dataset_sha256=checksum(ROOT/'datasets/creditcard.csv'),checkpoint_sha256=checksum(REFERENCE/'pre_local.pt'))
    dump(OUT/'protocol_lock.json',protocol);all_ids=set();development={}
    for tier,config in protocol['tiers'].items():
        groups=make_groups(frame,available,4,config['records'],config['fraud'],derive_seed(run_seed,'dev',tier));
        for group in groups: all_ids.update(group['source_ids'])
        available=np.asarray(sorted(set(available)-all_ids));torch.save(groups,OUT/f'{tier}_development_targets.pt')
        candidates=[]
        for lr in protocol['learning_rates']:
            rows=[]
            for group_id,group in enumerate(groups):
                rows+=run_pair(OUT/'development'/tier/f'lr_{lr}'/f'group_{group_id}',group,group_id,0,run_seed,lr,300,config['batch_size'])
            baseline=[row for row in rows if row['method']=='baseline']
            candidates.append(dict(lr=lr,mean_delta=float(np.mean([row['metrics']['mean_mse']-row['prior']['mean_mse'] for row in baseline])),rows=rows))
        development[tier]=candidates
    chosen=min(development['one_batch'],key=lambda item:item['mean_delta']);dump(OUT/'development.json',development)
    frozen=dict(lr=chosen['lr'],iterations=300,restarts=3,protocol_sha256=checksum(OUT/'protocol_lock.json'),
                proceed=chosen['mean_delta']<0);dump(OUT/'frozen.json',frozen)
    confirmation=[]
    if frozen['proceed']:
        config=protocol['tiers']['one_batch'];groups=make_groups(frame,available,10,4,1,derive_seed(run_seed,'confirmation'))
        assert not all_ids.intersection(i for group in groups for i in group['source_ids']);torch.save(groups,OUT/'confirmation_targets.pt')
        for group_id,group in enumerate(groups):
            for restart in range(3):
                confirmation+=run_pair(OUT/'confirmation'/f'group_{group_id}'/f'restart_{restart}',group,group_id,restart,
                                       run_seed,frozen['lr'],300,4)
    summary,selected=evaluate(confirmation) if confirmation else ({'gate':False,'reason':'development did not improve prior'},[])
    full_client=json.loads((ROOT/'artifacts/phase3_followup/adam_buffers_v6/adam_buffers/results.json').read_text())['rows']
    report=dict(protocol=protocol,frozen=frozen,development={tier:[{'lr':c['lr'],'mean_delta':c['mean_delta']} for c in values]
                for tier,values in development.items()},confirmation=summary,selected=selected,
                full_client_adam={r['method']:r['metrics']['mean_mse'] for r in full_client},
                closure='PASS_BOUNDED_ONLY' if summary.get('gate') else 'NEGATIVE_COMPLETE',
                phase4_ready=bool(summary.get('gate')),seconds=time.perf_counter()-start,
                interpretation='Full-client Adam remains negative regardless of bounded result.')
    dump(OUT/'phase3_closure.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
