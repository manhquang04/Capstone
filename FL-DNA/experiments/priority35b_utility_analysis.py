"""Independent utility-only metric/checkpoint reload, paired tables and matching."""
from __future__ import annotations
import csv
import math
import statistics
from pathlib import Path
from experiments import priority35b_utility as u
from experiments import priority35b_design as d
from scipy.stats import t
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


def summary(values):
    values = d.finite(values)
    if len(values)!=11: raise ValueError('all eleven paired replicates required')
    mean=statistics.fmean(values)
    half=float(t.ppf(.975,10))*statistics.stdev(values)/math.sqrt(11)
    return dict(n=11,df=10,mean_delta=mean,ci_low=mean-half,ci_high=mean+half)


def csv_save(path, rows):
    with path.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def check_doc(doc, base):
    np,torch,a=u.local.np,u.local.torch,u.local.a
    for name,expected in doc['artifact_sha256'].items():
        if u.sha(base/name)!=expected: raise ValueError('changed artifact '+name)
    if doc['status']!='COMPLETED' or not doc['no_bn_transmitted'] or doc['torch_threads']!=1:
        raise ValueError('completion/BN/thread contract')
    if doc['payload_assertions']!=150: raise ValueError('missing uploads')
    folder=u.local.p.OUT/'prepared'/doc['dataset']
    ck=next(name for name in doc['artifact_sha256'] if name.endswith('final_checkpoint.pt'))
    state=torch.load(base/ck,map_location='cpu',weights_only=False)
    width=np.load(folder/'train_x.npy',mmap_mode='r').shape[1]
    journal=next(name for name in doc['artifact_sha256'] if name.endswith('rounds.jsonl'))
    rounds=[u.json.loads(line) for line in (base/journal).read_text().splitlines()]
    if [r['round'] for r in rounds]!=list(range(1,51)): raise ValueError('round coverage')
    for row in rounds:
        if not row['no_bn_transmitted'] or set(row['transmitted_keys'])&set(row['bn_keys']): raise ValueError('BN upload')
        for minima in row['bn_minima_all_steps']:
            if any(not math.isfinite(v) or v<0 for v in minima.values()): raise ValueError('negative/nonfinite BN')
    for client in range(3):
        model=u.local.p.FraudMLP(width).cpu(); bn,allowed=a.domains(model)
        a.assert_payload(state['global_non_bn'],bn,allowed)
        if set(state['client_bn'][client])!=bn: raise ValueError('BN checkpoint domain')
        model_state=model.state_dict();model_state.update(state['global_non_bn']);model_state.update(state['client_bn'][client])
        model.load_state_dict(model_state);a.guard(model,'independent checkpoint')
        for split in ('validation','test'):
            filename=next(n for n in doc['artifact_sha256'] if n.endswith(f'client{client}_{split}_probabilities.npy'))
            stored=np.load(base/filename)
            actual=a.checked_probabilities(model,np.load(folder/f'{split}_x.npy',mmap_mode='r'))
            if not np.array_equal(actual,stored): raise ValueError('checkpoint predictions not bit-exact')
            labels=np.load(folder/f'{split}_y.npy');threshold=doc['clients'][client]['threshold']
            if split=='validation' and u.local.p.tune_threshold(labels,stored)!=threshold:
                raise ValueError('validation-only threshold mismatch')
            independent=dict(f1=float(f1_score(labels,stored>=threshold,zero_division=0)),
                             auc_roc=float(roc_auc_score(labels,stored)),pr_auc=float(average_precision_score(labels,stored)))
            for endpoint,value in independent.items():
                if not math.isfinite(value) or abs(value-doc['clients'][client][split][endpoint])>1e-12:
                    raise ValueError('independent prediction metric mismatch')
    for split in ('validation','test'):
        for endpoint in ('f1','auc_roc','pr_auc'):
            mean=statistics.fmean(c[split][endpoint] for c in doc['clients'])
            if abs(mean-doc[split][endpoint])>1e-12: raise ValueError('mean-client metric mismatch')


def finish():
    u.configure(); frozen=u.verify_freeze()
    if list(u.OUT.glob('jobs/**/*.failure.json')) or (u.OUT/'REQUIRES_DIRECTION.json').exists():
        raise ValueError('preserved failure; no partial-data analysis')
    registry=u.read(u.OUT/'reuse_registry.json')['reuse_jobs']
    entries=[]
    for row in registry:
        path=u.ROOT/row['path']
        if u.sha(path)!=row['sha256']: raise ValueError('historical receipt drift')
        base=u.ROOT/('artifacts/priority34a' if row['method'] in ('baseline',*d.TRANSFORMS) else 'artifacts/priority34b_local_dp')
        entries.append((row['dataset'],row['method'],row['seed'],path,base,True))
    for job in frozen['jobs']:
        if not u.local.valid(u.OUT/job['result'],job): raise ValueError('missing result')
        entries.append((job['dataset'],job['method'],job['seed'],u.OUT/job['result'],u.OUT,False))
    if len(entries)!=627: raise ValueError('require 627 utility receipts')
    docs={}; raw=[]
    for i,(dataset,method,seed,path,base,reused) in enumerate(entries):
        doc=u.read(path);check_doc(doc,base)
        baseline=u.read(u.ROOT/f'artifacts/priority34a/jobs/{dataset}/baseline/{seed}.json')
        if doc['partition_sha256']!=baseline['partition_sha256']: raise ValueError('paired partition mismatch')
        docs[(dataset,method,seed)]=doc
        if reused:
            receipt=dict(source=str(path.relative_to(u.ROOT)),source_sha256=u.sha(path),immutable_reuse=True,result=doc)
            target=u.OUT/f'reused_jobs/{dataset}/{method}/{seed}.json'
            if target.exists(): raise ValueError('reuse copy already exists; preserve attempts')
            u.save(target,receipt)
        for endpoint in ('f1','auc_roc','pr_auc'):
            value=doc['test'][endpoint];reference=baseline['test'][endpoint]
            raw.append(dict(dataset=dataset,method=method,seed=seed,endpoint=endpoint,value=value,
                baseline=reference,paired_delta=value-reference,reused=reused,
                source=str(path.relative_to(u.ROOT)),source_sha256=u.sha(path)))
        if i%22==0: print(f'Independent utility audit {i+1}/627',flush=True)
    methods=('baseline',*d.TRANSFORMS,*[f'local_dp_{c}_eps{e}' for c in d.CLIPS for e in d.EPSILONS])
    rows=[];matched=[]
    for dataset in d.DATASETS:
        for method in methods:
            for endpoint in ('f1','auc_roc','pr_auc'):
                a=[docs[(dataset,method,s)]['test'][endpoint] for s in d.SEEDS]
                b=[docs[(dataset,'baseline',s)]['test'][endpoint] for s in d.SEEDS]
                rows.append(dict(dataset=dataset,method=method,endpoint=endpoint,
                                 mean=statistics.fmean(a),baseline_mean=statistics.fmean(b),
                                 **summary([x-y for x,y in zip(a,b)])))
        for transform in d.TRANSFORMS:
            for clip in d.CLIPS:
                target={s:docs[(dataset,transform,s)]['test']['f1'] for s in d.SEEDS}
                grid={e:{s:docs[(dataset,f'local_dp_{clip}_eps{e}',s)]['test']['f1'] for s in d.SEEDS} for e in d.EPSILONS}
                matched.append(dict(dataset=dataset,transform=transform,clipping=clip,**d.match(target,grid)))
    csv_save(u.OUT/'utility_per_seed.csv',raw);csv_save(u.OUT/'utility_table.csv',rows)
    u.save(u.OUT/'matching.json',matched)
    csv_save(u.OUT/'matching.csv',[{k:v for k,v in m.items() if k!='points'} for m in matched])
    # Separate NumPy/isf recomputation from serialized CSV, and independent bracket selection.
    np=u.local.np
    reread=list(csv.DictReader((u.OUT/'utility_per_seed.csv').open()))
    table=list(csv.DictReader((u.OUT/'utility_table.csv').open()))
    if len(reread)!=1881 or len(table)!=171 or len(matched)!=12: raise ValueError('complete table counts')
    for row in table:
        subset=[r for r in reread if all(r[k]==row[k] for k in ('dataset','method','endpoint'))]
        if sorted(int(r['seed']) for r in subset)!=list(d.SEEDS): raise ValueError('CSV seed coverage')
        delta=np.asarray([float(r['value'])-float(r['baseline']) for r in subset])
        center=float(delta.mean());half=float(t.isf(.025,10))*float(delta.std(ddof=1))/math.sqrt(11)
        for key,value in dict(mean_delta=center,ci_low=center-half,ci_high=center+half).items():
            if abs(value-float(row[key]))>1e-12: raise ValueError('CSV independent t interval mismatch')
    for m in matched:
        means={e:math.fsum(docs[(m['dataset'],f'local_dp_{m["clipping"]}_eps{e}',s)]['test']['f1'] for s in d.SEEDS)/11 for e in d.EPSILONS}
        threshold=statistics.fmean(docs[(m['dataset'],m['transform'],s)]['test']['f1'] for s in d.SEEDS)-.005
        passing=[e for e in d.EPSILONS if means[e]>=threshold]
        choice=passing[0] if passing and passing[0]!=1 else None
        # Exact threshold implementation uses fmean; ordinary summation is only a numerical cross-check.
        if choice!=m['epsilon']: raise ValueError('independent matching mismatch')
    lines=['# Priority35B utility only','', 'Status: independently verified tables; full P35B NOT COMPLETE.',
        '', '627 receipts: 330 new DP jobs, 297 hash-identical historical reuses. '
        '11 paired seeds; mean of three local-BN clients. ROC-AUC is Delta-AUC; PR-AUC also shown. '
        'Intervals are descriptive two-sided Student-t, df10. No leakage measured; no privacy/NI claim.',
        '', '| Dataset | Method | Endpoint | Mean | Paired delta | 95% CI |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in rows: lines.append(f'| {r["dataset"]} | {r["method"]} | {r["endpoint"]} | {r["mean"]:.8g} | {r["mean_delta"]:+.8g} | [{r["ci_low"]:.8g}, {r["ci_high"]:.8g}] |')
    lines+=['','## Utility-only matching','', '| Dataset | Transform | Clipping | Status | Epsilon | Next noisier |', '| --- | --- | --- | --- | --- | --- |']
    for m in matched: lines.append(f'| {m["dataset"]} | {m["transform"]} | {m["clipping"]} | {m["status"]} | {m["epsilon"]} | {m["next_noisier_epsilon"]} |')
    lines+=['', 'Largest-noise passing mean F1 point and failing next-noisier neighbor; fixed margin .005. '
        'NOT_ASSESSABLE is no bracket, not privacy evidence. Per-client update-transcript DP, not record-level DP.',
        '', 'Commands: `.venv-phase1/bin/python -B experiments/priority35b_utility.py --prepare`; '
        '`--launch`. Frozen amendment SHA: '+u.sha(u.PROTOCOL)+'.',
        'Read-only historical/checkpoint/prediction/threshold/metric audits and CSV t intervals PASS. '
        'Stop after utility; no report outside P35B artifacts was edited.']
    (u.OUT/'utility_summary.md').write_text('\n'.join(lines)+'\n')
    u.save(u.OUT/'independent_audit.json',dict(status='PASS',jobs=627,new_jobs=330,reused=297,
        bit_exact_prediction_arrays=3762,paired_endpoints=171,matched_cells=12,leakage_runs=0))
    hashes={str(path.relative_to(u.ROOT)):u.sha(path) for path in u.OUT.rglob('*')
            if path.is_file() and not any(x in path.parts for x in ('compile','synthetic_tests'))
            and path.name not in ('detached_stdout.log','detached_stderr.log','supervisor.lock.json')}
    u.save(u.OUT/'utility_manifest.json',hashes)
    u.save(u.OUT/'UTILITY_COMPLETE.json',dict(status='UTILITY_VERIFIED_COMPLETE',full_priority35b_complete=False,
        new_jobs=330,reused=297,manifest_sha256=u.sha(u.OUT/'utility_manifest.json'),
        summary_sha256=u.sha(u.OUT/'utility_summary.md'),scope='utility only',leakage_runs=0,
        supervisor_must_exit_before_final_handoff=True))
