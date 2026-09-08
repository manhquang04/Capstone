"""Prospective Phase 3 development and source-disjoint confirmation stages."""
import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from attacks.local_update import simulate, objective
from data.load_creditcard import BASE_FEATURE_COLUMNS, NUMERIC_COLUMNS, _build_features, _mild_non_iid_client_indices
from experiments import fraud_fl_common as common
from experiments.run_phase3_full_client import dump, checksum, score, Recorder
from models.fraud_mlp import FraudMLP
from privacy.seed_manager import derive_seed, generate_run_seed

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'artifacts/phase3'
RUNS=['full_20260908T143837588533Z','full_20260908T143843502722Z','full_20260908T143850567575Z']


def decode(z, variant, metadata=None):
    if variant in ('categorical', 'adam_buffers'):
        return torch.cat((z[:,:8],z[:,8:].softmax(-1)),1)
    if variant=='balanced':
        center=torch.as_tensor(metadata.numeric_center,dtype=z.dtype,device=z.device)
        scale=torch.as_tensor(metadata.numeric_scale,dtype=z.dtype,device=z.device)
        raw=z[:,:6]*scale[:6]+center[:6]
        origin=(raw[:,2]-raw[:,3]-center[6])/scale[6]
        destination=(raw[:,5]-raw[:,4]-center[7])/scale[7]
        return torch.cat((z[:,:6],origin[:,None],destination[:,None],z[:,8:]),1)
    return z


def attack(folder,cap,targets,metadata,variant,budget,seed):
    folder.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    x,y=targets['x'],targets['y']
    model=FraudMLP(x.shape[1]).train()
    model.load_state_dict(cap['pre'])
    criterion=common.BinaryFocalLoss()
    batches=[]; offset=0
    for size in cap['batch_sizes']:
        batches.append(slice(offset,offset+size)); offset+=size
    assert offset==len(x)
    replay=simulate(model,criterion,x,y,batches,cap['rng'])
    for k in replay:
        torch.testing.assert_close(replay[k],cap['raw'][k],atol=2e-7,rtol=2e-4)
    del replay
    initial=torch.randn(x.shape,generator=torch.Generator().manual_seed(seed))
    prior=score(x,decode(initial,variant,metadata),y,metadata,folder/'prior.csv')
    rows=[]
    names=list(dict(model.named_parameters()))
    if variant == 'adam_buffers':
        names = [k for k, v in cap['raw'].items() if v.is_floating_point()]
    for method in ('baseline','zero_update'):
        observed=cap['raw'] if method=='baseline' else {k:torch.zeros_like(v) for k,v in cap['raw'].items()}
        z=initial.clone().requires_grad_(True)
        optimizer=torch.optim.Adam([z],lr=.05)
        history=[]; best=float('inf'); best_step=0; candidate=None
        t=time.perf_counter()
        for step in range(budget+1):
            delta=simulate(model,criterion,decode(z,variant,metadata),y,batches,cap['rng'])
            loss=objective(delta,observed,names)
            # Public local learning-rate scale, identical for baseline and control.
            if variant=='scaled':
                loss=loss/(.001**2)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite objective')
            value=float(loss.detach()); history.append(value)
            if value<best:
                best=value; candidate=z.detach().clone(); best_step=step
            if step in (100,300,500) or step==budget:
                path=folder/f'{method}_{step}.pt'
                torch.save(dict(latent=candidate,initial=initial,candidate=decode(candidate,variant,metadata),best=best,
                                history=history.copy(),best_step=best_step,variant=variant,seed=seed),path)
                saved=torch.load(path,weights_only=False)
                checked=objective(simulate(model,criterion,saved['candidate'],y,batches,cap['rng']),observed,names)
                if variant=='scaled':
                    checked=checked/(.001**2)
                torch.testing.assert_close(checked.detach(),torch.tensor(best),atol=1e-12,rtol=1e-5)
                metrics=score(x,saved['candidate'],y,metadata,folder/f'{method}_{step}.csv')
                rows.append(dict(method=method,budget=step,best_step=best_step,objective=best,
                                 prior=prior,metrics=metrics,elapsed=time.perf_counter()-t))
                dump(folder/'results.json',dict(variant=variant,seed=seed,rows=rows))
            if step<budget:
                grad,=torch.autograd.grad(loss,z)
                if not torch.isfinite(grad).all():
                    raise RuntimeError('Nonfinite derivative')
                optimizer.zero_grad();z.grad=grad;optimizer.step()
            del loss,delta
            if step%50==0:
                print(folder.name,method,step,flush=True)
    dump(folder/'complete.json',dict(seconds=time.perf_counter()-started,rows=len(rows)))


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'protocol.json',dict(seed=generate_run_seed(),development_client=2,
        variants={'raw':500,'scaled':100,'categorical':100,'balanced':100},development_restarts=1,
        selection='minimum baseline-minus-own-prior mean MSE at 100; development only',
        evaluation_iterations=100,evaluation_restarts=2,evaluation_total_rows=500000,
        exclusion='entire original 500000-row subset; frozen scaler/checkpoint',
        gate='report paired benefit vs both controls and validity; no automatic significance threshold'))
    rows=[]
    for client,run in enumerate(RUNS):
        folder=OLD/run/f'client_{client}'
        target=torch.load(folder/'evaluator_targets.pt',weights_only=False)
        x=target['x'].numpy().astype('float64'); labels=target['y'].numpy().ravel()
        for restart in range(3):
            for method in ('baseline','zero_update','prior'):
                saved=torch.load(folder/f'{"baseline" if method=="prior" else method}_{restart}.pt',weights_only=False)
                z=saved['initial' if method=='prior' else 'candidate'].numpy().astype('float64')
                for label in (0,1):
                    error=x[labels==label]-z[labels==label]
                    rows.append(dict(client=client,restart=restart,method=method,label=label,
                        mse_by_feature=(error**2).mean(0).tolist(),mae_by_feature=np.abs(error).mean(0).tolist()))
    dump(out/'feature_analysis.json',dict(feature_names=json.loads((OLD/RUNS[0]/'preprocessing.json').read_text())['feature_names'],rows=rows))


def freeze(out):
    candidates=[]
    for variant in ('raw','scaled','categorical','balanced'):
        assert (out/variant/'complete.json').exists()
        rows=json.loads((out/variant/'results.json').read_text())['rows']
        baseline=next(r for r in rows if r['budget']==100 and r['method']=='baseline')
        candidates.append(dict(variant=variant,delta=baseline['metrics']['mean_mse']-baseline['prior']['mean_mse']))
    selected=min(candidates,key=lambda r:r['delta'])
    if (out/'frozen.json').exists():
        raise FileExistsError('Configuration already frozen')
    dump(out/'frozen.json',dict(selected=selected,development_candidates=candidates,frozen_at_unix=time.time(),
                             protocol_sha256=checksum(out/'protocol.json'),
                             iterations=100,restarts=2,rule='locked before loading new source targets'))


def holdout(out):
    assert (out/'frozen.json').exists()
    protocol=json.loads((out/'protocol.json').read_text())
    seed=protocol['seed']
    metadata=json.loads((OLD/RUNS[0]/'preprocessing.json').read_text())
    df=pd.read_csv(ROOT/'datasets/creditcard.csv',usecols=BASE_FEATURE_COLUMNS+['isFraud'])
    ids=np.arange(len(df)); labels=df.isFraud.to_numpy()
    _,excluded=train_test_split(ids,test_size=500000,random_state=20260907,stratify=labels)
    remaining=np.setdiff1d(ids,excluded)
    _,sampled=train_test_split(remaining,test_size=500000,random_state=seed,stratify=labels[remaining])
    assert not np.intersect1d(sampled,excluded).size
    train_ids,_=train_test_split(sampled,test_size=.35,random_state=seed,stratify=labels[sampled])
    frame=df.iloc[train_ids].copy()
    if frame.isna().any().any():
        raise ValueError('Missing values need train-only imputation metadata; do not fit on holdout')
    features=_build_features(frame)
    scaled=(features[NUMERIC_COLUMNS].to_numpy()-np.asarray(metadata['numeric_center']))/np.asarray(metadata['numeric_scale'])
    onehot=np.column_stack([(features.type.to_numpy()==v).astype(float) for v in metadata['type_categories']])
    values=torch.from_numpy(np.hstack((scaled,onehot)).astype('float32'))
    ys=torch.from_numpy(labels[train_ids].astype('float32')).reshape(-1,1)
    partitions=_mild_non_iid_client_indices(features,labels[train_ids],3,seed)
    common.DEVICE=torch.device('cpu')
    pre=torch.load(OLD/RUNS[0]/'pre_local.pt',weights_only=False)
    for client,indices in enumerate(partitions):
        folder=out/f'heldout_{client}';folder.mkdir(exist_ok=False)
        dataset=TensorDataset(values[indices],ys[indices],torch.from_numpy(train_ids[indices]))
        loader=DataLoader(dataset,batch_size=1024,shuffle=True,num_workers=0,
                          generator=torch.Generator().manual_seed(derive_seed(seed,'order',client)))
        recorder=Recorder(loader)
        model=FraudMLP(values.shape[1]);model.load_state_dict(pre)
        torch.manual_seed(derive_seed(seed,'training',client))
        common.train_local_model(model,recorder,torch.ones(1),1)
        raw={k:v.detach().clone()-pre[k] for k,v in model.state_dict().items()}
        torch.save(dict(pre=pre,raw=raw,rng=recorder.rng,batch_sizes=[len(p[0]) for p in recorder.parts]),folder/'capture.pt')
        torch.save(dict(x=torch.cat([p[0] for p in recorder.parts]),y=torch.cat([p[1] for p in recorder.parts]),
                        source_ids=torch.cat([p[2] for p in recorder.parts])),folder/'targets.pt')
    dump(out/'heldout_provenance.json',dict(seed=seed,source_disjoint=True,
         excluded_rows=len(excluded),training_rows=len(train_ids),client_sizes=[len(i) for i in partitions],
         dataset_sha256=checksum(ROOT/'datasets/creditcard.csv'),checkpoint_sha256=checksum(OLD/RUNS[0]/'pre_local.pt'),
         scaler_metadata=metadata))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','dev','freeze','holdout','evaluate'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',choices=['raw','scaled','categorical','balanced','adam_buffers'])
    p.add_argument('--client',type=int)
    args=p.parse_args();torch.set_num_threads(1)
    out=args.output
    if args.stage=='prepare':
        return prepare(out)
    if args.stage=='freeze':
        return freeze(out)
    if args.stage=='holdout':
        return holdout(out)
    metadata=SimpleNamespace(**json.loads((OLD/RUNS[0]/'preprocessing.json').read_text()))
    if args.stage=='dev':
        source=OLD/RUNS[2]/'client_2'
        cap=torch.load(source/'attacker_capture.pt',weights_only=False)
        target=torch.load(source/'evaluator_targets.pt',weights_only=False)
        # Match the original full-client restart 0 exactly.
        seed=derive_seed(1157354082,'init',2,0)
        attack(out/args.variant,cap,target,metadata,args.variant,500 if args.variant=='raw' else 100,seed)
    else:
        frozen=json.loads((out/'frozen.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
        folder=out/f'heldout_{args.client}'
        cap=torch.load(folder/'capture.pt',weights_only=False);target=torch.load(folder/'targets.pt',weights_only=False)
        for restart in range(frozen['restarts']):
            attack(folder/f'restart_{restart}',cap,target,metadata,frozen['selected']['variant'],frozen['iterations'],
                   derive_seed(protocol['seed'],'evaluation',args.client,restart))


if __name__=='__main__':
    main()
