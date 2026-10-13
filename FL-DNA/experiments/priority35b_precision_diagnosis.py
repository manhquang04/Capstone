"""Read-only numerical/decision diagnosis; no measurements or tolerance patch."""
import json
import math
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from experiments import priority35b_leakage as driver
from experiments import priority35b_leakage_metrics_audit as metric
from experiments import priority35b_leakage_data_bridge as bridge
from experiments import priority35b_leakage_tables as tables
from experiments import priority35b_leakage_final_seal as seal


def run():
    seal.prepare.no_live(seal.prepare.ROOT)
    driver.configure()
    jobs=driver.read(driver.FREEZE)['jobs'];results=[];alternative={};cache={}
    errors=[];violations=[];groups={};native_bad=[];prior_cache={}
    for n,c in enumerate(jobs):
        k=(c['dataset'],c['cell'])
        if k not in cache:cache[k]=driver.core.PreparedDataset(*k)
        data=cache[k];r=driver.read(driver.folder(c)/'result.json');results.append(r)
        p=driver.folder(c)/r['attempt']
        truth=torch.load(p/'private_truth.pt',map_location='cpu',weights_only=False)['truth']
        bundle=torch.load(p/'reconstruction.pt',map_location='cpu',weights_only=False)
        rec=bundle['candidates'][bundle['selected']]['reconstruction']
        independent=metric.independently_score(data,truth,rec)
        native=driver.metric(data,truth,rec)
        if any(native[key]!=r['score'][key] for key in native):native_bad.append(c)
        if independent['feature_accuracy']!=r['score']['feature_accuracy'] or independent['assignment']!=r['score']['assignment']:
            violations.append(dict(config=c,kind='accuracy_or_alignment'))
        a=r['score']['standardized_mse'];b=independent['standardized_mse'];e=abs(a-b)
        # Read-only explanatory bound, NOT installed as verifier tolerance.
        eps=np.finfo(np.float64).eps;N=truth.numel();gamma=N*eps/(1-N*eps)
        bound=1e-12+2*gamma*max(abs(a),abs(b))
        if not math.isfinite(e) or e>bound:violations.append(dict(config=c,error=e,bound=bound))
        errors.append(e)
        g=groups.setdefault(c['dataset']+' / '+c['arm'],[]);g.append(e)
        pk=(*k,c['index'])
        if c['arm']=='unprotected':
            stored=r['controls']['mean_mode_metric']['standardized_mse']
            prior_rec=bundle['controls']['mean_mode']
            prior_cache[pk]=metric.independently_score(data,truth,prior_rec)['standardized_mse']
        alternative[(*k,c['index'],c['arm'])]=(independent,pk)
        if (n+1)%1000==0:print('checked',n+1,flush=True)
    arms=driver.read(driver.OUT/'arms.json')
    records=bridge.result_rows(results,arms);other=[]
    for r in records:
        s,pk=alternative[tuple(r[f] for f in ('dataset','cell','index','arm'))]
        other.append(dict(r,feature_accuracy=s['feature_accuracy'],standardized_mse=s['standardized_mse'],
                          prior_mse=prior_cache[pk],bounded_mse=min(s['standardized_mse'],prior_cache[pk])))
    matching=driver.read(driver.ROOT/'artifacts/priority35b/b2b/arms/matching.json')
    # Compare the two metric reductions directly; do not invoke the separate
    # prospective SciPy-bit-equality publication gate during this diagnosis.
    from experiments import priority35b_leakage_statistics as stats
    def summaries(rows):
        pairs,reasons=tables.pairs_from_records(tables.index_records(rows,arms),matching)
        return dict(holm=stats.evaluate(pairs,reasons))
    left=summaries(records);right=summaries(other)
    changes=[]
    for a,b in zip(left['holm'],right['holm']):
        fields=('dataset','cell','transform','clipping','metric','direction')
        ident={k:a[k] for k in fields}
        if a['status']!=b['status'] or a['raw_p']!=b['raw_p'] or a['holm_p']!=b['holm_p']:
            changes.append(dict(identity=ident,stored=a,recomputed=b))
        if a['summary'] is not None:
            for key in ('lower_wins','higher_wins','ties','effective_n'):
                if a['summary'][key]!=b['summary'][key]:changes.append(dict(identity=ident,field=key,stored=a['summary'][key],recomputed=b['summary'][key]))
    fa_equal=all(a['feature_accuracy']==b['feature_accuracy'] for a,b in zip(records,other))
    nonzero=[x for x in errors if x]
    out=dict(status='PRECISION_DIAGNOSIS_ONLY',jobs=len(jobs),current_absolute_tolerance=1e-12,
             max_absolute_difference=max(errors),median_absolute_difference=statistics.median(errors),
             nonzero_count=len(nonzero),median_nonzero_difference=statistics.median(nonzero) if nonzero else 0,
             failed_absolute_tolerance_count=sum(e>1e-12 for e in errors),
             affected_groups={k:dict(count=sum(x>1e-12 for x in v),maximum=max(v),median=statistics.median(v)) for k,v in groups.items() if max(v)>1e-12},
             native_exact_mismatches=native_bad,precision_bound_violations=violations,decision_changes=changes,
             all_feature_accuracy_exact=fa_equal,figure_points_unchanged=fa_equal,
             matching_unchanged=True,matching_reason='Matching uses immutable utility F1 only, not MSE.',
             qualification_unchanged=True,qualification_reason='Historical P34C gates are inputs, not recomputed from B3 MSE.',
             scientific_measurement_repeated=False,tolerance_changed=False)
    path=driver.OUT/('precision_diagnosis_'+str(time.time_ns())+'.json')
    with path.open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(dict(path=str(path),**{k:v for k,v in out.items() if k!='affected_groups'}),indent=2),flush=True)


if __name__=='__main__':run()
