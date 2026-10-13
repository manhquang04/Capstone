"""Post-exit independent utility verification and descriptive B2b tables."""
from __future__ import annotations
import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path
from scipy.stats import t
from experiments import priority35b_b2b as b
from experiments.priority35b_utility_analysis import check_doc,summary,csv_save
u=b.u

def match_union(target,points):
    threshold=statistics.fmean(u.design.f1_values(target))-.005
    if not points or len({p['method'] for p in points})!=len(points): raise ValueError('missing/duplicate arms')
    rows=[]
    for p in points:
        if not math.isfinite(p['noise_sd']) or p['noise_sd']<0: raise ValueError('invalid SD')
        x=dict(p);x['mean_f1']=statistics.fmean(u.design.f1_values(x.pop('f1')));x['passes']=x['mean_f1']>=threshold;rows.append(x)
    rows.sort(key=lambda x:(-x['noise_sd'],x['C'],x['method']))
    tiers=sorted({x['noise_sd'] for x in rows},reverse=True)
    selected=next((sd for sd in tiers if any(x['passes'] and x['noise_sd']==sd for x in rows)),None)
    doc=dict(threshold=threshold,points=rows)
    if selected is None or selected==tiers[0]:
        return dict(doc,status='NOT_ASSESSABLE',reason='no passing point' if selected is None else 'no noisier neighbor',matched_arm=None,next_noisier_sd=None)
    choice=next(x for x in rows if x['passes'] and x['noise_sd']==selected)
    neighbor=tiers[tiers.index(selected)-1]
    if any(x['passes'] and x['noise_sd']==neighbor for x in rows): raise ValueError('no failing bracket')
    return dict(doc,status='MATCHED' if choice['epsilon'] is not None else 'MATCHED_CLIP_ONLY_NO_DP',
                matched_arm=choice['method'],epsilon=choice['epsilon'],C=choice['C'],noise_sd=selected,next_noisier_sd=neighbor,
                next_noisier_arms=[x['method'] for x in rows if x['noise_sd']==neighbor])

def audit_audits(doc,job):
    """Independent scalar clip reconstruction, not the runtime validator."""
    np=u.local.np;C=job['C'];eps=job['epsilon'];acc=b.accounting(C,eps)
    for r in doc['local_upload_audits']:
        before=math.sqrt(math.fsum(x*x for x in r['tensor_before']))
        after=math.sqrt(math.fsum(x*x for x in r['tensor_after']))
        if before!=r['before_norm'] or after!=r['after_norm'] or after>C: raise ValueError('independent norm audit')
        # Float32 casts can perturb each scaled norm; original implementation's
        # bound is exact and retained; this formula check uses only 8 eps rounding.
        factor=min(1.,C/max(before,1e-300))*(1-4*np.finfo(np.float32).eps)
        for original,actual in zip(r['tensor_before'],r['tensor_after']):
            if abs(actual-original*factor)>8*np.finfo(np.float32).eps*max(original*factor,1e-300): raise ValueError('clip scalar reconstruction')
        if r['noise_sd']!=acc['noise_sd'] or r['C']!=C or r['sensitivity']!=2*C or r['mechanism']!='global': raise ValueError('noise/accounting audit')
        if r['clipped']!=(before>C) or r['clip_only']!=(eps is None) or not r['no_bn_transmitted'] or not r['noise_before_transmission']: raise ValueError('upload metadata audit')

def independent_csv():
    raw=list(csv.DictReader((b.OUT/'utility_per_seed.csv').open()))
    table=list(csv.DictReader((b.OUT/'utility_table.csv').open()))
    np=u.local.np
    if len(raw)!=3168 or len(table)!=288: raise ValueError('union table coverage')
    for r in table:
        group=[x for x in raw if all(x[k]==r[k] for k in ('dataset','method','endpoint'))]
        if sorted(int(x['seed']) for x in group)!=list(u.design.SEEDS): raise ValueError('eleven seed coverage')
        delta=np.array([float(x['value'])-float(x['baseline']) for x in group]);mean=float(delta.mean())
        half=float(t.isf(.025,10))*float(delta.std(ddof=1))/math.sqrt(11)
        for k,v in dict(mean_delta=mean,ci_low=mean-half,ci_high=mean+half).items():
            if abs(v-float(r[k]))>1e-12: raise ValueError('independent t interval '+k)
    for r in csv.DictReader((b.OUT/'decomposition.csv').open()):
        groups={m:[x for x in raw if x['dataset']==r['dataset'] and x['method']==m and x['endpoint']==r['endpoint']] for m in (r['from_arm'],r['to_arm'])}
        by={m:{int(x['seed']):float(x['value']) for x in g} for m,g in groups.items()}
        delta=np.array([by[r['to_arm']][s]-by[r['from_arm']][s] for s in u.design.SEEDS]);mean=float(delta.mean());half=float(t.isf(.025,10))*float(delta.std(ddof=1))/math.sqrt(11)
        for k,v in dict(mean_delta=mean,ci_low=mean-half,ci_high=mean+half).items():
            if abs(v-float(r[k]))>1e-12: raise ValueError('decomposition CSV audit')
    for r in u.read(b.OUT/'matching.json'):
        pts=r['points'];target=r['threshold'];passing=[p for p in pts if p['mean_f1']>=target]
        # Independent bracket computed from serialized points and raw CSV means.
        for p in pts:
            a=[x for x in raw if x['dataset']==r['dataset'] and x['method']==p['method'] and x['endpoint']=='f1']
            mean=float(np.mean([float(x['value']) for x in a]))
            if len(a)!=11 or abs(mean-p['mean_f1'])>1e-12: raise ValueError('matching mean audit')
        sd=max((p['noise_sd'] for p in passing),default=None)
        noisier=sorted({p['noise_sd'] for p in pts if sd is not None and p['noise_sd']>sd})
        choice=sorted((p for p in passing if p['noise_sd']==sd),key=lambda p:(p['C'],p['method'])) if sd is not None else []
        expected=choice[0]['method'] if choice and noisier else None
        if r['matched_arm']!=expected or r['next_noisier_sd']!=(min(noisier) if expected else None): raise ValueError('independent bracket mismatch')

def finish():
    b.configure()
    if b.live() or b.telemetry.live() or u.live(): raise ValueError('workers alive; verifier forbidden')
    if (b.OUT/'B2_B2B_COMPLETE.json').exists(): raise ValueError('already complete; preserve')
    f=b.verify_freeze()
    if list(b.OUT.glob('jobs/**/*.failure.json')) or (b.OUT/'REQUIRES_DIRECTION.json').exists(): raise ValueError('preserved failure')
    entries={};methods={};raw=[];telemetry=[]
    registry=u.read(u.OUT/'reuse_registry.json')['reuse_jobs']
    for r in registry:
        base=u.ROOT/('artifacts/priority34a' if r['method'] in ('baseline',*u.design.TRANSFORMS) else 'artifacts/priority34b_local_dp')
        entries[(r['dataset'],r['method'],r['seed'])]=(u.ROOT/r['path'],base)
    for j in u.read(u.OUT/'execution_freeze.json')['jobs']:
        entries[(j['dataset'],j['method'],j['seed'])]=(u.OUT/j['result'],u.OUT)
    for j in f['jobs']:
        if not b.valid(j): raise ValueError('missing B2b output')
        entries[(j['dataset'],j['method'],j['seed'])]=(b.OUT/j['result'],b.OUT)
        methods[(j['dataset'],j['method'])]=dict(C=j['C'],epsilon=j['epsilon'],clipping='global',noise_sd=b.accounting(j['C'],j['epsilon'])['noise_sd'])
    if len(entries)!=1056: raise ValueError('627 historical plus429 new required')
    docs={}
    for i,(key,(p,base)) in enumerate(entries.items()):
        doc=u.read(p);check_doc(doc,base);dataset,method,seed=key
        baseline=u.read(u.ROOT/f'artifacts/priority34a/jobs/{dataset}/baseline/{seed}.json')
        if doc['partition_sha256']!=baseline['partition_sha256']: raise ValueError('paired partition')
        if base==b.OUT: audit_audits(doc,doc['job'])
        docs[key]=doc
        for endpoint in ('f1','auc_roc','pr_auc'):
            value=doc['test'][endpoint];ref=baseline['test'][endpoint]
            raw.append(dict(dataset=dataset,method=method,seed=seed,endpoint=endpoint,value=value,baseline=ref,paired_delta=value-ref,source=str(p.relative_to(u.ROOT)),source_sha256=u.sha(p)))
        if i%33==0: print(f'Independent B2/B2b verification {i+1}/1056',flush=True)
    oldmethods=['baseline',*u.design.TRANSFORMS,*[f'local_dp_{c}_eps{e}' for c in u.design.CLIPS for e in u.design.EPSILONS]]
    rows=[];matched=[];decomp=[]
    for dataset in u.design.DATASETS:
        newmethods=[m for d,m in methods if d==dataset]
        for method in [*oldmethods,*newmethods]:
            for endpoint in ('f1','auc_roc','pr_auc'):
                a=[docs[(dataset,method,s)]['test'][endpoint] for s in u.design.SEEDS];base=[docs[(dataset,'baseline',s)]['test'][endpoint] for s in u.design.SEEDS]
                rows.append(dict(dataset=dataset,method=method,endpoint=endpoint,mean=statistics.fmean(a),baseline_mean=statistics.fmean(base),**summary([x-y for x,y in zip(a,base)])))
            if method in newmethods:
                audits=[r for s in u.design.SEEDS for r in docs[(dataset,method,s)]['local_upload_audits']]
                telemetry.append(dict(dataset=dataset,method=method,**methods[(dataset,method)],n_uploads=len(audits),clipped_fraction=statistics.fmean(r['clipped'] for r in audits),median_preclip_norm=statistics.median(r['before_norm'] for r in audits)))
        for transform in u.design.TRANSFORMS:
            for clip in u.design.CLIPS:
                points=[]
                for e in u.design.EPSILONS:
                    method=f'local_dp_{clip}_eps{e}'
                    points.append(dict(method=method,C=.01,epsilon=e,noise_sd=u.account(e)['noise_sd'],f1={s:docs[(dataset,method,s)]['test']['f1'] for s in u.design.SEEDS}))
                if clip=='global':
                    for method in newmethods:
                        x=methods[(dataset,method)];points.append(dict(method=method,C=x['C'],epsilon=x['epsilon'],noise_sd=x['noise_sd'],f1={s:docs[(dataset,method,s)]['test']['f1'] for s in u.design.SEEDS}))
                matched.append(dict(dataset=dataset,transform=transform,clipping=clip,**match_union({s:docs[(dataset,transform,s)]['test']['f1'] for s in u.design.SEEDS},points)))
        clip='b2b_global_c001_clip_only'
        pairs=[('baseline',clip)]+[(clip,f'local_dp_global_eps{e}') for e in u.design.EPSILONS]
        for source,target in pairs:
            for endpoint in ('f1','auc_roc','pr_auc'):
                diffs=[docs[(dataset,target,s)]['test'][endpoint]-docs[(dataset,source,s)]['test'][endpoint] for s in u.design.SEEDS]
                decomp.append(dict(dataset=dataset,from_arm=source,to_arm=target,endpoint=endpoint,**summary(diffs)))
    csv_save(b.OUT/'utility_per_seed.csv',raw);csv_save(b.OUT/'utility_table.csv',rows);csv_save(b.OUT/'clipping_telemetry.csv',telemetry);csv_save(b.OUT/'decomposition.csv',decomp)
    u.save(b.OUT/'matching.json',matched)
    csv_save(b.OUT/'matching.csv',[{k:v for k,v in r.items() if k not in ('points','next_noisier_arms')} for r in matched])
    independent_csv()
    checks=[]
    for command in ([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_priority35b_b2b*.py','-v'],['git','diff','--check']):
        x=subprocess.run(command,cwd=u.ROOT,capture_output=True,text=True);checks.append(dict(command=command,returncode=x.returncode,stdout=x.stdout,stderr=x.stderr));u.save(b.OUT/'postexit_checks.json',checks)
        if x.returncode: raise ValueError('postexit tests fail')
    import py_compile
    for p in (u.ROOT/'experiments/priority35b_b2b.py',u.ROOT/'experiments/priority35b_b2b_analysis.py',u.ROOT/'tests/test_priority35b_b2b.py'):
        py_compile.compile(str(p),cfile=str(b.OUT/'compile'/f'postexit_{p.stem}.pyc'),doraise=True)
    lines=['# Priority35B B2/B2b utility summary — post hoc','',
        '429 new B2b jobs +627 unchanged B2/P34A/P34B jobs. n=11 paired seeds, mean of three local-BN clients. '
        'Two-sided descriptive Student-t95% intervals df10. No privacy/equivalence claim; no new Holm tests. '
        'Clipping includes inherited inward rounding and double-delta weighted aggregation. Clip-only has NO DP guarantee.',
        '', '| Dataset | Method | Endpoint | Mean | Delta vs baseline | 95% CI |','| --- | --- | --- | --- | --- | --- |']
    for r in rows: lines.append(f'| {r["dataset"]} | {r["method"]} | {r["endpoint"]} | {r["mean"]:.8g} | {r["mean_delta"]:+.8g} | [{r["ci_low"]:.8g}, {r["ci_high"]:.8g}] |')
    lines+=['','## Clipping telemetry','', '| Dataset | Method | C | epsilon | Clipped fraction | Median preclip norm |','| --- | --- | --- | --- | --- | --- |']
    for r in telemetry: lines.append(f'| {r["dataset"]} | {r["method"]} | {r["C"]:.8g} | {r["epsilon"]} | {r["clipped_fraction"]:.8g} | {r["median_preclip_norm"]:.8g} |')
    lines+=['','## C=.01 decomposition','', '| Dataset | From -> to | Endpoint | Paired difference | 95% CI |','| --- | --- | --- | --- | --- |']
    for r in decomp: lines.append(f'| {r["dataset"]} | {r["from_arm"]} -> {r["to_arm"]} | {r["endpoint"]} | {r["mean_delta"]:+.8g} | [{r["ci_low"]:.8g}, {r["ci_high"]:.8g}] |')
    lines+=['','## Union matching','', '| Dataset | Transform | Clip | Status | Matched arm | Next noisier SD |','| --- | --- | --- | --- | --- | --- |']
    for r in matched: lines.append(f'| {r["dataset"]} | {r["transform"]} | {r["clipping"]} | {r["status"]} | {r["matched_arm"]} | {r["next_noisier_sd"]} |')
    lines+=['','Matching is utility-only selection, not privacy evidence. Per-tensor original grid unchanged. '
        'If a zero-noise clip-only point is selected it is NOT a DP comparator. '
        'Update-transcript DP conditional on public history, NOT record-level DP.',
        '', 'Commands: `.venv-phase1/bin/python -B experiments/priority35b_b2b.py --prepare`, `--launch`, '
        '`--finish` only after supervisor/all workers exit. Frozen execution amendment SHA '+u.sha(b.PROTOCOL)+'.',
        'All historical hashes (including approved supplementary infrastructure log seal), checkpoint reloads, '
        'predictions, thresholds, metrics, CSV intervals, brackets, tests and compile verified. '
        'B2/B2b utility complete; fullP35B NOT COMPLETE; stop before B3.']
    (b.OUT/'utility_summary.md').write_text('\n'.join(lines)+'\n')
    if b.live() or u.live() or b.telemetry.live(): raise ValueError('unexpected live worker')
    b.verify_freeze()
    u.save(b.OUT/'independent_audit.json',dict(status='PASS',new_jobs=429,historical_jobs=627,
        bit_exact_prediction_arrays=6336,table_rows=288,per_seed_rows=3168,matched_cells=12,no_live=True))
    hashes={str(p.relative_to(u.ROOT)):u.sha(p) for p in b.OUT.rglob('*') if p.is_file() and not any(x in p.parts for x in ('compile','synthetic_tests'))}
    u.save(b.OUT/'final_manifest.json',hashes)
    u.save(b.OUT/'B2_B2B_COMPLETE.json',dict(status='B2_B2B_UTILITY_VERIFIED_COMPLETE',B2_complete=True,B2b_complete=True,B3_started=False,
        full_priority35b_complete=False,manifest_sha256=u.sha(b.OUT/'final_manifest.json'),summary_sha256=u.sha(b.OUT/'utility_summary.md')))
    u.stage('B2_B2B_UTILITY_VERIFIED_COMPLETE',B2_complete=True,B2b_complete=True,B3_started=False,done=429,total=429,stop='wait before B3')
