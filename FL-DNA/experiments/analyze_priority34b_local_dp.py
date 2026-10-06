"""Independent local-DP utility recomputation and post-exit checkpoint audit."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import priority34b_local_dp as l
from scipy.optimize import minimize_scalar
from scipy.stats import t
np,torch,OUT=l.np,l.torch,l.OUT


def live_workers():
    if os.name=='nt':
        command=['powershell','-NoProfile','-Command',
            'Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress']
        text=subprocess.check_output(command,text=True)
        raw=json.loads(text or '[]'); raw=raw if isinstance(raw,list) else [raw]
        return [row for row in raw if row['ProcessId']!=os.getpid() and
            'priority34b_local_dp.py' in (row.get('CommandLine') or '') and
            any(x in row['CommandLine'] for x in ('--job','--supervise'))]
    text=subprocess.check_output(['ps','-axo','pid=,ppid=,command='],text=True)
    rows=[]
    for line in text.splitlines():
        parts=line.strip().split(None,2)
        if len(parts)!=3 or int(parts[0])==os.getpid(): continue
        executable=Path(parts[2].split()[0]).name.lower()
        if 'python' in executable and 'priority34b_local_dp.py' in parts[2] and any(x in parts[2] for x in ('--job','--supervise')):
            rows.append(line)
    return rows


def paired(values):
    values=np.asarray(values,dtype=float)
    if len(values)!=11 or not np.isfinite(values).all(): raise ValueError('paired n11 finite')
    mean=float(values.mean()); width=float(t.ppf(.975,10))*float(values.std(ddof=1))/math.sqrt(11)
    independent_mean=math.fsum(values.tolist())/11
    independent_width=float(t.isf(.025,10))*math.sqrt(math.fsum((x-independent_mean)**2 for x in values)/110)
    if max(abs(mean-independent_mean),abs(width-independent_width))>1e-12:
        raise ValueError('independent paired recomputation mismatch')
    return dict(n=11,mean_delta=mean,median_delta=float(np.median(values)),ci_low=mean-width,ci_high=mean+width)


def accounting():
    rows=[]
    for epsilon in (1,3,10):
        doc=l.account(epsilon); sigma=doc['sigma_sensitivity']
        objective=lambda alpha:50*alpha/(2*sigma**2)+math.log(1e5)/(alpha-1)
        minimum=minimize_scalar(objective,bounds=(1.000001,5000),method='bounded')
        if abs(minimum.fun-epsilon)>1e-8: raise ValueError('independent RDP mismatch')
        rows.append(dict(**doc,independent_epsilon=float(minimum.fun),independent_alpha=float(minimum.x)))
    return rows


def artifacts(doc,base):
    for relative,digest in doc['artifact_sha256'].items():
        if l.p.sha(base/relative)!=digest: raise ValueError('input/output artifact altered '+relative)


def csv_save(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def remeasure_tabular(doc,base):
    maximum=0.
    for client in range(3):
        for split in ('validation','test'):
            filename=next(n for n in doc['artifact_sha256'] if n.endswith(f'client{client}_{split}_probabilities.npy'))
            probabilities=np.load(base/filename); labels=np.load(l.p.OUT/'prepared'/doc['dataset']/f'{split}_y.npy')
            if not np.isfinite(probabilities).all(): raise ValueError('prediction nonfinite')
            metrics=l.p.metrics(labels,probabilities,doc['clients'][client]['threshold'])
            for endpoint in ('f1','auc_roc','pr_auc'):
                error=abs(metrics[endpoint]-doc['clients'][client][split][endpoint]); maximum=max(maximum,error)
                if error>1e-12: raise ValueError('independent metrics mismatch')
    return maximum


def finish():
    frozen=l.verify_freeze(); rows=[]; per_seed=[]; maximum=0.
    if list(OUT.rglob('*.failure.json')) or (OUT/'REQUIRES_DIRECTION.json').exists():
        raise ValueError('failure gate unresolved')
    for dataset in (*l.p.DATASETS,'cifar10'):
        baseline_folder=ROOT/('artifacts/priority34b' if dataset=='cifar10' else 'artifacts/priority34a')
        baselines={s:l.read(baseline_folder/f'jobs/{dataset}/baseline/{s}.json') for s in l.old.SEEDS}
        for base in baselines.values():
            artifacts(base,baseline_folder)
            if dataset!='cifar10': maximum=max(maximum,remeasure_tabular(base,baseline_folder))
        endpoints=('validation_accuracy',) if dataset=='cifar10' else ('f1','auc_roc','pr_auc')
        for mechanism in l.old.MECHANISMS:
            for epsilon in l.old.EPSILONS:
                jobs=[j for j in frozen['jobs'] if j['dataset']==dataset and j['mechanism']==mechanism and j['epsilon']==epsilon]
                jobs.sort(key=lambda j:j['seed']); docs=[]
                for job in jobs:
                    if not l.valid(OUT/job['result'],job): raise ValueError('missing required result '+job['result'])
                    doc=l.read(OUT/job['result']); docs.append(doc)
                    if dataset!='cifar10': maximum=max(maximum,remeasure_tabular(doc,OUT))
                for endpoint in endpoints:
                    arm=[]; reference=[]; central=[]
                    for job,doc in zip(jobs,docs):
                        seed=job['seed']; base=baselines[seed]
                        cd=l.read(ROOT/f'artifacts/priority34b/jobs/{dataset}/dp_{mechanism}_eps{epsilon}/{seed}.json')
                        value=doc[endpoint] if dataset=='cifar10' else doc['test'][endpoint]
                        raw=base[endpoint] if dataset=='cifar10' else base['test'][endpoint]
                        cv=cd[endpoint] if dataset=='cifar10' else cd['test'][endpoint]
                        arm.append(value); reference.append(raw); central.append(cv)
                        per_seed.append(dict(dataset=dataset,mechanism=mechanism,epsilon=epsilon,seed=seed,
                            endpoint=endpoint,local_value=value,baseline=raw,central_value=cv,local_minus_baseline=value-raw))
                    rows.append(dict(dataset=dataset,mechanism=mechanism,epsilon=epsilon,endpoint=endpoint,
                        local_mean=float(np.mean(arm)),baseline_mean=float(np.mean(reference)),central_mean=float(np.mean(central)),
                        **paired([x-y for x,y in zip(arm,reference)])))
    dna=[]
    for dataset in l.p.DATASETS:
        for method in ('dna_v1_conservative','dna_v2_0p95'):
            for endpoint in ('f1','auc_roc','pr_auc'):
                values=[]; baseline=[]
                for seed in l.old.SEEDS:
                    values.append(l.read(ROOT/f'artifacts/priority34a/jobs/{dataset}/{method}/{seed}.json')['test'][endpoint])
                    baseline.append(l.read(ROOT/f'artifacts/priority34a/jobs/{dataset}/baseline/{seed}.json')['test'][endpoint])
                dna.append(dict(dataset=dataset,method=method,endpoint=endpoint,mean=float(np.mean(values)),
                    **paired([x-y for x,y in zip(values,baseline)])))
    if len(rows)!=60 or len(dna)!=18: raise ValueError('fixed60utility18DNA count')
    acct=accounting()
    l.save(OUT/'analysis/utility_summary.json',rows); l.save(OUT/'analysis/DNA_context.json',dna)
    csv_save(OUT/'analysis/utility_per_seed.csv',per_seed)
    l.save(OUT/'analysis/independent_recomputation.json',dict(status='PASS',jobs=264,paired_endpoints=78,
        max_metric_error=maximum,accounting=acct,baseline_reused=44))
    lines=['# Priority34B extension — LOCAL client-side update DP','',
        'Status: analysis complete; post-exit checkpoint audit pending.','',
        '264 new CPU/thread1 jobs;44 immutable baselines reused. Honest-but-curious server sees individual '+
        'uploads already clipped/noised on each client. BN never transmitted. No central noise added.',
        '', '## Accounting and release boundary','',
        'Unit: one client contribution transcript across50rounds, replace-one update-level adjacency; '+
        'NOT record-level DP. q1, delta1e-5, joint C=.01, sensitivity2C=.02. Per-tensor radiusC/sqrt(L). '+
        'Each arm/replicate is a separate hypothetical deployment; releasing all trials further composes. '+
        'Checkpoint/localBN/validation/evaluation/audit material and private noise keys are NOT private releases. '+
        'Ideal Gaussian simulation, not certified production finite-precision DP.',
        '', '| ε whole-training | σ / sensitivity | Noise SD EACH upload | Rounds |',
        '| --- | --- | --- | --- |']
    for row in acct: lines.append(f"| {row['target_epsilon']} | {row['sigma_sensitivity']:.12g} | {row['noise_sd']:.12g} | 50 |")
    lines+=['','## Descriptive paired n11 utility','',
        'Central column is historical context only: **central DP; requires secure aggregation to protect against the server**. '+
        'Never use central recovery as a P34C comparator. No new NI/equivalence claim.',
        '', '| Dataset | Mechanism | ε | Endpoint | Baseline | LOCAL DP | CENTRAL DP context | Local Δ [95% CI] |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['mechanism']} | {row['epsilon']} | {row['endpoint']} | {row['baseline_mean']:.8g} | {row['local_mean']:.8g} | {row['central_mean']:.8g} | {row['mean_delta']:+.8g} [{row['ci_low']:.8g},{row['ci_high']:.8g}] |")
    lines+=['','## Same eleven-seed local-BN DNA context','',
        '| Dataset | DNA | Endpoint | Mean | Paired Δ [95% CI] |','| --- | --- | --- | --- | --- |']
    for row in dna:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['endpoint']} | {row['mean']:.8g} | {row['mean_delta']:+.8g} [{row['ci_low']:.8g},{row['ci_high']:.8g}] |")
    lines+=['','## Execution, interpretation and preservation','',
        'Pre-run amendment: protocols/amendments/2026-10-03_priority34b_local_dp_extension.md. '+
        'Seeds321000–321010; unchanged data/partitions/model/50round training from P34A/B. '+
        'CIFAR uses P34B three-client50round FedAvg, NOT P31 centralized100epochs. '+
        'No C/sigma utility tuning, clamp, seed exclusion or scope reduction. Utility cost is not an optimal DP bound. '+
        'Training nonfinite/negative-BN fails closed. Earlier central sources/results/reports untouched; '+
        'scope label addendum reports/priority34b_central_dp_scope_addendum.md preserves prior hashes.',
        '', 'Synthetic preflight caught duplicate image receipt metadata before research-data runs. '+
        'The preflight failure and pre-correction annex remain preserved. Wrapper removed only duplicate '+
        'embedding fields, not numerical values, seeds, noise, training or objectives. All9 synthetic tests '+
        'must pass before execution freeze.',
        '', 'Independent prediction metric reload, Gaussian RDP optimizer and78 paired CIs PASS. '+
        'Full checkpoint bit-exact/post-exit audit remains mandatory. Commands/results/source hashes are '+
        'in runs.jsonl, execution_freeze.json and final manifest. P34C epsilon10 uses this LOCAL release, '+
        'not a noisy aggregate. No record-recovery claims from utility scores.']
    if REPORT.exists(): raise ValueError('do not overwrite generated report')
    REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    l.check_item('analysis','complete'); l.check_item('final_audit','pending')


def verify():
    if (OUT/'COMPLETE.json').exists(): raise ValueError('already sealed')
    if not (OUT/'READY_FOR_FINAL_AUDIT.json').exists() or live_workers(): raise ValueError('not post-exit ready')
    frozen=l.verify_freeze(); audit=l.read(OUT/'analysis/independent_recomputation.json')
    if audit['status']!='PASS' or audit['paired_endpoints']!=78: raise ValueError('analysis audit')
    from experiments import priority34b_images as image
    data=image.p31.dp.CIFAR10(str(ROOT/'datasets/cifar10'),train=True,download=False)
    split=l.read(image.p31.OUT/'split.json'); vx,vy=image.tensors(data,split['validation_ids'])
    arrays=0; assertions=0
    for index,job in enumerate(frozen['jobs']):
        if not l.valid(OUT/job['result'],job): raise ValueError('missing job')
        doc=l.read(OUT/job['result']); assertions+=doc['payload_assertions']
        filename=next(n for n in doc['artifact_sha256'] if n.endswith('final_checkpoint.pt'))
        saved=torch.load(OUT/filename,map_location='cpu')
        if job['stage']=='tabular':
            prepared=l.p.OUT/'prepared'/job['dataset']; width=np.load(prepared/'train_x.npy',mmap_mode='r').shape[1]
            journal=next(n for n in doc['artifact_sha256'] if n.endswith('rounds.jsonl'))
            rounds=[json.loads(x) for x in (OUT/journal).read_text().splitlines()]
            if [x['round'] for x in rounds]!=list(range(1,51)): raise ValueError('round coverage')
            for row in rounds:
                if not row['no_bn_transmitted'] or set(row['transmitted_keys'])&set(row['bn_keys']): raise ValueError('BN on wire')
                for client in row['bn_minima_all_steps']:
                    if any(not math.isfinite(x) or x<0 for x in client.values()): raise ValueError('BN invalid')
            for client in range(3):
                model=l.p.FraudMLP(width).cpu(); bn,allowed=l.a.domains(model)
                l.a.assert_payload(saved['global_non_bn'],bn,allowed)
                if set(saved['client_bn'][client])!=bn: raise ValueError('BN domain')
                state=model.state_dict(); state.update(saved['global_non_bn']); state.update(saved['client_bn'][client]); model.load_state_dict(state)
                for name in ('validation','test'):
                    path=next(n for n in doc['artifact_sha256'] if n.endswith(f'client{client}_{name}_probabilities.npy'))
                    actual=l.a.checked_probabilities(model,np.load(prepared/f'{name}_x.npy',mmap_mode='r'))
                    if not np.array_equal(actual,np.load(OUT/path)): raise ValueError('checkpoint array mismatch')
                    arrays+=1
        else:
            model=image.image_model(); model.load_state_dict(saved); model.eval(); logits=[]
            with torch.no_grad():
                for start in range(0,len(vx),512):
                    value=model(vx[start:start+512]); l.old.finite(value,'checkpoint image logits'); logits.append(value.numpy())
            path=next(n for n in doc['artifact_sha256'] if n.endswith('validation_logits.npy'))
            if not np.array_equal(np.concatenate(logits),np.load(OUT/path)): raise ValueError('image checkpoint mismatch')
            if float((np.concatenate(logits).argmax(1)==vy.numpy()).mean())!=doc['validation_accuracy']:
                raise ValueError('image metric reload mismatch')
            arrays+=1
        if index%22==0: print(json.dumps(dict(checkpoint_jobs=index+1,bit_exact_arrays=arrays)),flush=True)
    if arrays!=1254 or assertions!=39600: raise ValueError('completion counts')
    checks=[]
    for command in ([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_priority34b_local_dp.py','-v'],['git','diff','--check']):
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        checks.append(dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode: raise ValueError('final check failed '+str(command))
    folder=OUT/'compile_final'; folder.mkdir(exist_ok=True)
    for relative in frozen['sources']:
        if relative.endswith('.py'):
            py_compile.compile(str(ROOT/relative),cfile=str(folder/(relative.replace('/','_')+'c')),doraise=True)
    if live_workers(): raise ValueError('live workers after audit')
    archive=OUT/'report_pre_final_audit.md'
    if archive.exists(): raise ValueError('archive exists; review interrupted audit')
    shutil.copy2(REPORT,archive)
    text=REPORT.read_text(encoding='utf-8').replace('Status: analysis complete; post-exit checkpoint audit pending.',
        'Status: COMPLETE — independent post-exit audit PASS.',1)
    text+='\nFinal audit:264/264 jobs,1,254 checkpoint arrays bit-exact,39,600 noBN assertions;78paired CIs and per-client50round RDP PASS. Source/input/full result hashes, tests, py_compile, git diff --check PASS; no live workers.\n'
    REPORT.write_text(text,encoding='utf-8')
    l.save(OUT/'final_checks.json',dict(status='PASS',checks=checks,py_compile='PASS'))
    l.check_item('final_audit','complete'); l.progress('complete',264,264)
    receipt=dict(status='PASS',at=l.p.now(),jobs=264,baseline_reused=44,bit_exact_prediction_arrays=arrays,
        no_bn_assertions=assertions,independent=audit,live_workers=[],report=str(REPORT.relative_to(ROOT)),
        report_sha256=l.p.sha(REPORT))
    l.save(OUT/'COMPLETE.json',receipt)
    manifest=dict(frozen['sources']); manifest.update(frozen['inputs'])
    for path in OUT.rglob('*'):
        if path.is_file() and path.name!='sha256_manifest_final.json': manifest[str(path.relative_to(ROOT))]=l.p.sha(path)
    manifest[str(REPORT.relative_to(ROOT))]=l.p.sha(REPORT)
    l.save(OUT/'sha256_manifest_final.json',manifest)
    print(json.dumps(dict(status='PASS',jobs=264,arrays=arrays,noBN=assertions)),flush=True)


REPORT=l.REPORT
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--verify',action='store_true'); args=parser.parse_args()
    if args.verify: verify()
    else: parser.error('use --verify after supervisor exit')
