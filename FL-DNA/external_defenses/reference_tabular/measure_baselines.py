"""Data-free mean/marginal controls against the exact saved native/PaySim batches."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['DATALOADER_NUM_WORKERS']='0'
import json
import numpy as np
import torch
from run_reference import HERE, ROOT, OFFICIAL, ADULT, PaySimAdapter, load_module, measure

torch.set_num_threads(2)
os.chdir(OFFICIAL)
adult=ADULT()
adult.standardize()
loader=load_module('baseline_paysim_loader', ROOT/'data/load_creditcard.py')
clients, _, _, _, _, metadata=loader.load_creditcard_data(batch_size=8,max_rows=100000,seed=42)
pooled=(torch.cat([c.dataset.tensors[0] for c in clients]),torch.cat([c.dataset.tensors[1] for c in clients]))
paysim=PaySimAdapter(pooled,metadata,loader.NUMERIC_COLUMNS)
results={}
for scenario in ['adult_native','paysim_official_fc','paysim_project_eval']:
    ds=adult if scenario=='adult_native' else paysim
    saved=np.load(HERE/'runs/default_batch8_seed42_v3'/scenario/'inputs.npz')
    truth=torch.from_numpy(saved['truth'])
    # One record repeated: numeric training means and categorical training modes.
    center=ds.Xtrain.mean(0)
    ptr=0
    for feature,categories in ds.train_features.items():
        if categories is None:
            ptr+=1
        else:
            block=center[ptr:ptr+len(categories)].clone()
            # Argmax category is unaffected by de-standardization? Decode instead
            # use true unscaled frequencies, then return standardized values.
            raw=ds.de_standardize(ds.Xtrain)
            mode=int(raw[:,ptr:ptr+len(categories)].mean(0).argmax())
            unscaled=torch.zeros(len(categories)); unscaled[mode]=1
            center[ptr:ptr+len(categories)]=(unscaled-ds.mean[ptr:ptr+len(categories)])/ds.std[ptr:ptr+len(categories)]
            ptr+=len(categories)
    baseline=center.expand_as(truth).clone()
    mean_metric=measure(ds,truth,baseline)
    rng=np.random.default_rng(42)
    # Independent per-feature empirical marginals; 30 random gradient-free guesses.
    marginal_scores=[]
    for repeat in range(30):
        guess=torch.empty_like(truth)
        ptr=0
        for feature,categories in ds.train_features.items():
            width=1 if categories is None else len(categories)
            indices=rng.integers(0,len(ds.Xtrain),len(truth))
            guess[:,ptr:ptr+width]=ds.Xtrain[indices,ptr:ptr+width]
            ptr+=width
        marginal_scores.append(measure(ds,truth,guess)['accuracy_percent'])
    results[scenario]=dict(training_mean_and_category_mode=mean_metric,
        empirical_marginal_accuracy_mean_percent=float(np.mean(marginal_scores)),
        empirical_marginal_accuracy_sd_percent=float(np.std(marginal_scores,ddof=1)),
        marginal_trials=30,seed=42)
(HERE/'baselines.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
