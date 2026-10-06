"""Independent P34B utility/accounting/recovery statistics and report."""
from __future__ import annotations
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest, t
from experiments import priority34b_core as b
np, torch, OUT, ROOT = b.np,b.torch,b.OUT,b.ROOT


def csv_write(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def paired(values):
    if not np.isfinite(values).all() or len(values)!=11:
        raise ValueError('paired n11 finite gate')
    mean = float(np.mean(values)); sd = float(np.std(values,ddof=1))
    width = float(t.ppf(.975,10))*sd/math.sqrt(11)
    independent_mean = math.fsum(values)/11
    independent_sd = math.sqrt(math.fsum((x-independent_mean)**2 for x in values)/10)
    independent_width = float(t.isf(.025,10))*independent_sd/math.sqrt(11)
    if max(abs(mean-independent_mean),abs(width-independent_width))>1e-12:
        raise ValueError('independent paired CI mismatch')
    return dict(n=11,mean_delta=mean,sd=sd,ci_low=mean-width,ci_high=mean+width,
                median_delta=float(np.median(values)),iqr=np.quantile(values,[.25,.75]).tolist())


def sign(values, direction):
    values = np.array(values,dtype=float)
    if len(values)!=39 or not np.isfinite(values).all():
        raise ValueError('recovery n39 finite gate')
    oriented = values*direction
    wins = int((oriented>0).sum()); losses = int((oriented<0).sum())
    n = wins+losses
    exact = sum(math.comb(n,k) for k in range(wins,n+1))/2**n if n else 1.
    independent = float(binomtest(wins,n,.5,alternative='greater').pvalue) if n else 1.
    if abs(exact-independent)>1e-14:
        raise ValueError('independent sign mismatch')
    order = sorted(values.tolist())
    return dict(wins=wins,losses=losses,ties=39-n,effective_n=n,raw_p=exact,
                median_effect=float(np.median(values)),order_interval=[order[12],order[26]])


def holm(rows):
    order = sorted(range(len(rows)),key=lambda i:(rows[i]['raw_p'],i))
    last = 0.
    for rank,index in enumerate(order):
        last = max(last,min(1.,rows[index]['raw_p']*(len(rows)-rank)))
        rows[index]['holm_p']=last
    # Independent every-prefix definition, not the in-place accumulator.
    for rank,index in enumerate(order):
        value = min(1.,max(rows[j]['raw_p']*(len(rows)-k) for k,j in enumerate(order[:rank+1])))
        if abs(rows[index]['holm_p']-value)>1e-14:
            raise ValueError('Holm recomputation mismatch')


def finish():
    frozen = b.verified_freeze()
    results = {}
    for job in frozen['jobs']:
        path = OUT/job['result']
        if not b.valid(path,job):
            raise ValueError('missing required job '+job['result'])
        results[(job['stage'],job['dataset'],job['method'],job.get('seed',job.get('target_id')))]=b.read(path)
    b.checklist(4,'complete'); b.checklist(5,'in_progress')
    accounting = []
    for epsilon in b.EPSILONS:
        row = b.account(epsilon); sigma=row['sigma_sensitivity']
        fn = lambda alpha:50*alpha/(2*sigma*sigma)+math.log(1e5)/(alpha-1)
        minimum = minimize_scalar(fn,bounds=(1.000001,5000),method='bounded',options={'xatol':1e-10})
        if abs(minimum.fun-epsilon)>1e-8:
            raise ValueError('independent accountant failure')
        accounting.append(dict(**row,independent_epsilon=float(minimum.fun),independent_alpha=float(minimum.x)))
    b.save(OUT/'analysis/accounting.json',accounting)
    utility_rows, per_seed, client_rows = [],[],[]
    baseline_docs = {}
    for dataset in b.p.DATASETS:
        for seed in b.SEEDS:
            doc = b.read(ROOT/f'artifacts/priority34a/jobs/{dataset}/baseline/{seed}.json')
            baseline_docs[(dataset,seed)]=doc
    # Reload every probability and recompute final client thresholds/metrics.
    max_metric_error = 0.
    for dataset in b.p.DATASETS:
        for method in ['baseline']+[f'dp_{m}_eps{e}' for m in b.MECHANISMS for e in b.EPSILONS]:
            for seed in b.SEEDS:
                doc = baseline_docs[(dataset,seed)] if method=='baseline' else results[('tabular',dataset,method,seed)]
                baseout = ROOT/'artifacts/priority34a' if method=='baseline' else OUT
                recomputed = {split:[] for split in ['validation','test']}
                for client in doc['clients']:
                    for split in ['validation','test']:
                        label = np.load(b.p.OUT/f'prepared/{dataset}/{split}_y.npy')
                        candidates = [name for name in doc['artifact_sha256'] if name.endswith(f"client{client['client']}_{split}_probabilities.npy")]
                        if len(candidates)!=1:
                            raise ValueError('missing client probabilities')
                        probability = np.load(baseout/candidates[0])
                        if split=='validation':
                            threshold = float(b.p.tune_threshold(label,probability))
                            if threshold!=client['threshold']:
                                raise ValueError('threshold replay mismatch')
                        score = b.p.metrics(label,probability,threshold)
                        for endpoint in ['f1','auc_roc','pr_auc']:
                            error=abs(score[endpoint]-client[split][endpoint]); max_metric_error=max(max_metric_error,error)
                            if error>1e-12:
                                raise ValueError('client metric replay mismatch')
                        recomputed[split].append(score)
                    client_rows.append(dict(dataset=dataset,method=method,seed=seed,client=client['client'],threshold=threshold,**score))
                for split in recomputed:
                    for endpoint in ['f1','auc_roc','pr_auc']:
                        mean = math.fsum(s[endpoint] for s in recomputed[split])/3
                        if abs(mean-doc[split][endpoint])>1e-12:
                            raise ValueError('client-mean error')
                if method!='baseline':
                    if doc['partition_sha256']!=baseline_docs[(dataset,seed)]['partition_sha256']:
                        raise ValueError('paired partition mismatch')
                    if len(doc['dp_round_audit'])!=50:
                        raise ValueError('50 DP rounds missing')
                    for audit in doc['dp_round_audit']:
                        if len(audit['client_clip_audits'])!=3 or abs(audit['sensitivity']-2*max(audit['weights'])*b.C)>1e-14:
                            raise ValueError('DP sensitivity/count failure')
                        for clip in audit['client_clip_audits']:
                            if clip['after_norm']>b.C or not clip['no_bn_transmitted']:
                                raise ValueError('clip/noBN audit failed')
                per_seed.append(dict(dataset=dataset,method=method,seed=seed,**doc['test']))
        for mechanism in b.MECHANISMS:
            for epsilon in b.EPSILONS:
                method=f'dp_{mechanism}_eps{epsilon}'
                for endpoint in ['f1','auc_roc','pr_auc']:
                    d=[results[('tabular',dataset,method,s)]['test'][endpoint] for s in b.SEEDS]
                    baseline=[baseline_docs[(dataset,s)]['test'][endpoint] for s in b.SEEDS]
                    utility_rows.append(dict(dataset=dataset,method=method,mechanism=mechanism,epsilon=epsilon,
                        endpoint=endpoint,baseline_mean=float(np.mean(baseline)),arm_mean=float(np.mean(d)),
                        **paired([x-y for x,y in zip(d,baseline)])))
    # Image validation logits are independently re-scored, not trusted JSON values.
    from experiments.priority34b_image_imports import p31, recovery
    data = p31.dp.CIFAR10(str(ROOT/'datasets/cifar10'),train=True,download=False)
    split = b.read(ROOT/'artifacts/priority31_image_utility_dp/split.json')
    labels=np.array(data.targets)[split['validation_ids']]
    for method in ['baseline']+[f'dp_{m}_eps{e}' for m in b.MECHANISMS for e in b.EPSILONS]:
        for seed in b.SEEDS:
            doc=results[('image_utility','cifar10',method,seed)]
            names=[name for name in doc['artifact_sha256'] if name.endswith('validation_logits.npy')]
            logits=np.load(OUT/names[0]); score=float(np.mean(logits.argmax(1)==labels))
            if not np.isfinite(logits).all() or score!=doc['validation_accuracy']:
                raise ValueError('image accuracy replay failed')
            per_seed.append(dict(dataset='cifar10',method=method,seed=seed,f1=None,auc_roc=None,pr_auc=None,validation_accuracy=score))
        if method!='baseline':
            values=[results[('image_utility','cifar10',method,s)]['validation_accuracy'] for s in b.SEEDS]
            base=[results[('image_utility','cifar10','baseline',s)]['validation_accuracy'] for s in b.SEEDS]
            spec=results[('image_utility','cifar10',method,b.SEEDS[0])]['job']
            utility_rows.append(dict(dataset='cifar10',method=method,mechanism=spec['mechanism'],epsilon=spec['epsilon'],
                endpoint='validation_accuracy',baseline_mean=float(np.mean(base)),arm_mean=float(np.mean(values)),
                **paired([x-y for x,y in zip(values,base)])))
    # Explicitly retain paired 11-seed local-BN DNA comparators, not an n21 CI.
    dna_rows=[]
    for dataset in b.p.DATASETS:
        for method in ['dna_v1_conservative','dna_v2_0p95']:
            for endpoint in ['f1','auc_roc','pr_auc']:
                values=[b.read(ROOT/f'artifacts/priority34a/jobs/{dataset}/{method}/{s}.json')['test'][endpoint] for s in b.SEEDS]
                base=[baseline_docs[(dataset,s)]['test'][endpoint] for s in b.SEEDS]
                dna_rows.append(dict(dataset=dataset,method=method,endpoint=endpoint,arm_mean=float(np.mean(values)),
                    baseline_mean=float(np.mean(base)),**paired([x-y for x,y in zip(values,base)])))
    recovery_rows, arm_scores, gates = [],[],{}
    native = recovery.native
    scores = {}
    arms=['unprotected','dna_v1_conservative','dna_v2_0p95','dp_global_eps10','dp_per_tensor_eps10']
    for arm in arms:
        scores[arm]={}
        for tid in range(39):
            doc=results[('recovery','cifar10',arm,tid)]
            filename=next(n for n in doc['artifact_sha256'] if n.endswith('arrays.npz'))
            array=np.load(OUT/filename)
            score=native.image_metrics(torch.from_numpy(array['truth']),torch.from_numpy(array['reconstruction']))
            if any(abs(score[k]-doc['metrics'][k])>1e-10 for k in score):
                raise ValueError('image recovery metrics do not replay')
            scores[arm][tid]=doc
        for metric in ['psnr_db','ssim','mse']:
            values=sorted(scores[arm][tid]['metrics'][metric] for tid in range(39))
            arm_scores.append(dict(arm=arm,metric=metric,n=39,median=float(np.median(values)),order_interval=[values[12],values[26]]))
        for reference in ['gray','cifar_mean']:
            values=[scores[arm][tid]['metrics']['psnr_db']-scores[arm][tid]['references'][reference]['psnr_db'] for tid in range(39)]
            gates[arm+'/'+reference]=sign(values,1)
    qualified=all(gates['unprotected/'+r]['raw_p']<.05 for r in ['gray','cifar_mean'])
    for dna in ['dna_v1_conservative','dna_v2_0p95']:
        for dp in ['dp_global_eps10','dp_per_tensor_eps10']:
            for metric in ['psnr_db','ssim','mse']:
                differences=[scores[dna][tid]['metrics'][metric]-scores[dp][tid]['metrics'][metric] for tid in range(39)]
                for direction in [1,-1]:
                    recovery_rows.append(dict(dna=dna,comparator=dp,metric=metric,direction=direction,
                        status='ASSESSABLE' if qualified else 'NOT_ASSESSABLE',**sign(differences,direction)))
    holm(recovery_rows)
    if len(recovery_rows)!=24:
        raise ValueError('fixed24test family changed')
    b.save(OUT/'analysis/utility_summary.json',utility_rows)
    b.save(OUT/'analysis/p34a_eleven_seed_comparison.json',dna_rows)
    b.save(OUT/'analysis/recovery_summary.json',dict(tests=recovery_rows,arm_medians=arm_scores,gates=gates,qualified=qualified,family=24))
    csv_write(OUT/'analysis/utility_per_seed.csv',per_seed)
    csv_write(OUT/'analysis/client_metrics.csv',client_rows)
    b.save(OUT/'analysis/independent_recomputation.json',dict(status='PASS',jobs=len(results),accounting=accounting,
        max_metric_error=max_metric_error,paired_endpoints=len(utility_rows)+len(dna_rows),sign_tests=24,holm_family=24))
    lines=['# Priority34B — utility at meaningful whole-training DP levels','',
           'Status: scientific analysis complete; final supervisor-exit audit pending.','',
           'Descriptive study, no new noninferiority/equivalence claim. '+str(len(results))+' new jobs; 33 immutable P34A baselines reused.',
           '', '## Formal mechanism and release boundary','',
           PROTOCOL_TEXT(),'', '| Target ε | Certified ε | σ (noise/sensitivity) | RDP order | Rounds | δ |',
           '| --- | --- | --- | --- | --- | --- |']
    for row in accounting:
        lines.append(f"| {row['target_epsilon']} | {row['certified_epsilon']:.10g} | {row['sigma_sensitivity']:.10g} | {row['alpha']:.8g} | 50 | 1e-5 |")
    lines+=['','## Utility versus epsilon — paired n11','',
            '| Dataset | Mechanism | ε | Endpoint | Baseline | DP mean | Mean Δ [95% CI] |',
            '| --- | --- | --- | --- | --- | --- | --- |']
    for row in utility_rows:
        lines.append(f"| {row['dataset']} | {row['mechanism']} | {row['epsilon']} | {row['endpoint']} | {row['baseline_mean']:.8g} | {row['arm_mean']:.8g} | {row['mean_delta']:+.8g} [{row['ci_low']:.8g}, {row['ci_high']:.8g}] |")
    lines+=['','## P34A local-BN DNA on the same eleven seeds','',
            'These n11 comparisons are not P34A\'s full21-seed NI claims, which remain unchanged in priority34a_report.md.',
            '', '| Dataset | DNA | Endpoint | Mean | Mean Δ [95% CI] |','| --- | --- | --- | --- | --- |']
    for row in dna_rows:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['endpoint']} | {row['arm_mean']:.8g} | {row['mean_delta']:+.8g} [{row['ci_low']:.8g}, {row['ci_high']:.8g}] |")
    lines+=['','## Frozen recovery — conditional single-round instrument','',
            'Not final-model or trained-client inversion. Same noise/sensitivity ratio as ε10 whole50-round accounting; known label and known-zero other contributions. C=.01 is in gradient coordinates here, versus update coordinates for utility: no utility-matched claim.',
            '', 'Unprotected qualification: '+str(qualified)+'. All reference gates (including DP instrument limitations): `'+json.dumps(gates,sort_keys=True)+'`.',
            '', '| Arm | Metric | Median [ranks13/27] |','| --- | --- | --- |']
    for row in arm_scores:
        lines.append(f"| {row['arm']} | {row['metric']} | {row['median']:.8g} {row['order_interval']} |")
    lines+=['','Positive effect = DNA metric minus DP metric; lower PSNR/SSIM or higher MSE indicates poorer recovery. Both signed directions reported, fixed Holm24 family.',
            '', '| DNA | DP | Metric | Direction | W/L/T | Median Δ [ranks13/27] | Raw p | Holm p | Status |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in recovery_rows:
        lines.append(f"| {row['dna']} | {row['comparator']} | {row['metric']} | {row['direction']} | {row['wins']}/{row['losses']}/{row['ties']} | {row['median_effect']:.8g} {row['order_interval']} | {row['raw_p']:.10g} | {row['holm_p']:.10g} | {row['status']} |")
    lines+=['','## Commands, preservation and limitations','',
            'Pre-run amendment: protocols/amendments/2026-10-03_priority34b_meaningful_dp.md. CPU/thread1; no negative-BN/nonfinite output permitted; no clamp, excluded/replaced seed or scientific tuning. Per-client scores and raw arrays remain in the new namespace. Metrics use the frozen image [0,1] metric clipping, not a repair of training or BN.',
            '', 'CIFAR uses P31 LeNet42/splits/lr.1 but50-round three-client FedAvg, NOT P31\'s100-epoch centralized SGD. Clipping C preselected, not optimized; three users offer no sampling amplification. Utility collapse or weak recovery cannot establish DNA privacy superiority at matched utility. Different epsilon arms are separate hypothetical deployments; releasing all arms/seeds composes further.',
            '', 'Privacy noise uses independent OS-entropy keys per DP job, privately retained mode0600; public training/noise namespace IDs do not determine it. Private research keys/targets/localBN/evaluation scores are outside the accounted release boundary; publishing them is not DP. Torch secret-seeded PRNG/float32 simulates ideal Gaussian DP, not a certified production sampler. Secure aggregation and cryptographically secret Gaussian noise are deployment assumptions, not implemented security claims.',
            '', 'Preflight initially exposed a generic models namespace collision between project FraudMLP and the read-only TabLeak import chain. A new P34B-only import isolation helper resolved it before freeze/training; no earlier source, data, budget, target or outcome changed. See preflight_import_failure.json and the preflight annex.',
            '', '[Gaussian RDP: Mironov2017](https://arxiv.org/abs/1702.07476), [user-level DP-FedAvg: McMahan et al., ICLR2018](https://arxiv.org/abs/1710.06963).',
            '', '```text','python -B -m unittest discover -s tests -p test_priority34b.py -v',
            'python -B experiments/supervise_priority34b.py --freeze',
            'python -B -u experiments/supervise_priority34b.py --supervise',
            'python -B experiments/verify_priority34b.py','git diff --check','```']
    report=ROOT/'reports/priority34b_report.md'
    if report.exists():
        raise ValueError('never overwrite existing report')
    report.write_text('\n'.join(lines)+'\n')
    checks=[]
    for command in [[sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_priority34b.py','-v'],
                    [sys.executable,'-B','-c','import py_compile,sys,pathlib; [py_compile.compile(p,cfile=str(pathlib.Path(sys.argv[1])/(pathlib.Path(p).stem+".pyc")),doraise=True) for p in sys.argv[2:]]',str(OUT/'compile')]+[str(ROOT/p) for p in frozen['sources'] if p.endswith('.py')],['git','diff','--check']]:
        (OUT/'compile').mkdir(exist_ok=True)
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        checks.append(dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:
            raise ValueError('final check failed')
    b.save(OUT/'final_checks.json',dict(status='PASS',checks=checks))
    manifest={**frozen['inputs'],**frozen['sources']}
    for path in OUT.rglob('*'):
        if path.is_file() and path.name not in ['sha256_manifest.json','progress.json','progress.log','runs.jsonl','supervisor.lock','supervisor.lock.json'] and not path.name.endswith('.log'):
            manifest[str(path.relative_to(ROOT))]=b.p.sha(path)
    manifest[str(report.relative_to(ROOT))]=b.p.sha(report)
    b.save(OUT/'sha256_manifest.json',manifest)
    b.save(OUT/'READY_FOR_FINAL_AUDIT.json',dict(jobs=len(results),status='READY',at=b.p.now()))
    b.progress('awaiting_final_exit_audit',len(results),len(results)); b.event('analysis_ready')


def PROTOCOL_TEXT():
    return ('q=1, fixed three slots, replace-one whole-client update-generating data/state; '+
            '50 adaptive releases, delta1e-5. Central trusted aggregator adds isotropic noise once after clipping/weighted averaging. '+
            'Global C=.01; per-tensor C/sqrt(L), concatenated radius C. Sensitivity2 max(w) C; sigma multiplies sensitivity, NOT C. '+
            'BN affine/buffers stay private/local. Global model transcript only is the accounted release. '+
            'LocalBN/evaluation/audit files and disclosed deterministic noise seeds are not protected releases. '+
            'Ideal Gaussian/RDP certificate; finite precision/PRNG simulation is not a certified private deployment.')
