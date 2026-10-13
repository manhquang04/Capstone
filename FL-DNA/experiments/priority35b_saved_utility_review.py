"""Approved additive infrastructure seal and post-hoc saved-data analysis only."""
import csv
import hashlib
import json
import math
import os
import statistics as st
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from scipy.stats import t, ttest_1samp, binomtest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/priority35b/utility'
DEST = OUT / 'supplementary_2026-10-10'
KNOWN = 'artifacts/priority35b/utility/postexit_stderr.log'
LOG_SHA = 'd5ec16c24c0ba2a2a51821d7e45437b7a5e6c0c93e91abd6f1c16512fefbd320'
MANIFEST_SHA = '12eec60ff241c41a364c95006ed065633c115381ed05f668d6299fa692fdf8fa'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def save(p, d):
    with p.open('x') as f: json.dump(d, f, indent=2, allow_nan=False); f.write('\n')

def rows(p):
    with p.open(newline='') as f: return list(csv.DictReader(f))

def ci(x):
    assert len(x) == 11 and all(math.isfinite(v) for v in x)
    m=st.fmean(x); h=float(t.ppf(.975,10))*st.stdev(x)/math.sqrt(11)
    return m,m-h,m+h

def main():
    import sys
    sys.path.insert(0,str(ROOT))
    from experiments import priority35b_utility as u
    u.configure()
    assert not u.live(), 'live scientific process'
    frozen=u.verify_freeze()
    assert sha(OUT/'utility_manifest.json') == MANIFEST_SHA
    manifest=json.loads((OUT/'utility_manifest.json').read_text())
    discrepancies=[]; effective={}
    for rel, expected in manifest.items():
        actual=sha(ROOT/rel)
        effective[rel]=actual
        if actual!=expected: discrepancies.append(dict(path=rel,original_sha256=expected,current_sha256=actual))
    assert discrepancies == [dict(path=KNOWN,original_sha256=hashlib.sha256(b'').hexdigest(),current_sha256=LOG_SHA)], discrepancies
    assert sha(OUT/'POSTEXIT_FAILURE_1791636740325019000.json')=='03a9f7581fefdfb454281e7df15922d89e3622b35a2e7dff971094bfc1d54f2f'
    assert sha(OUT/'UTILITY_COMPLETE.json')=='8ee6da0cea285e21fb306c1d01a37860d01cf66ceb53ed6207ec1ff084db3586'
    DEST.mkdir(exist_ok=False)
    proc=subprocess.check_output(['ps','-axo','pid=,ppid=,stat=,comm='],text=True)
    save(DEST/'infrastructure_seal.json',dict(date=datetime.now(timezone.utc).isoformat(),post_hoc=True,
        scope='infrastructure exception only; B2 NOT_COMPLETE; B3 NOT_STARTED',original_manifest_sha256=MANIFEST_SHA,
        entries_checked=len(manifest),matching_original_entries=len(manifest)-1,authorized_exception=discrepancies,
        other_discrepancies=[],sources_checked=len(frozen['sources']),inputs_checked=len(frozen['inputs']),
        live_scientific_processes=u.live(),process_snapshot=proc,measurement_runs=0,
        reason='manifest hashed empty error log before original verifier appended traceback',
        effective_postexit_hashes=effective))
    per=rows(OUT/'utility_per_seed.csv'); table=rows(OUT/'utility_table.csv')
    assert len(per)==1881 and len(table)==171
    groups={}; unique={}
    for r in per:
        k=(r['dataset'],r['method'],r['endpoint']); seed=int(r['seed'])
        assert seed not in groups.setdefault(k,{})
        v=float(r['value']); b=float(r['baseline']); delta=float(r['paired_delta'])
        assert all(math.isfinite(z) for z in (v,b,delta)) and abs(v-b-delta)<1e-12
        groups[k][seed]=r
        assert sha(ROOT/r['source'])==r['source_sha256']
        unique[(r['dataset'],r['method'],seed)]=r['source']
    assert len(unique)==627
    for r in table:
        g=groups[(r['dataset'],r['method'],r['endpoint'])]
        assert set(g)==set(range(321000,321011))
        m,lo,hi=ci([float(g[s]['paired_delta']) for s in sorted(g)])
        for key,val in [('mean_delta',m),('ci_low',lo),('ci_high',hi)]: assert abs(float(r[key])-val)<1e-12
    contrasts=[]
    for dataset in ('baf','ieee_cis','paysim'):
        for clipping in ('global','per_tensor'):
            base=groups[(dataset,f'local_dp_{clipping}_eps10','f1')]
            for eps in (10000,1000000):
                high=groups[(dataset,f'local_dp_{clipping}_eps{eps}','f1')]
                dif=[float(high[s]['paired_delta'])-float(base[s]['paired_delta']) for s in sorted(base)]
                m,lo,hi=ci(dif); result=ttest_1samp(dif,0)
                assert math.isfinite(float(result.pvalue))
                wins=sum(x>0 for x in dif); losses=sum(x<0 for x in dif)
                contrasts.append(dict(dataset=dataset,clipping=clipping,epsilon=eps,n=11,mean_difference=m,
                    ci_low=lo,ci_high=hi,t=float(result.statistic),df=10,p_raw=float(result.pvalue),
                    wins=wins,losses=losses,ties=11-wins-losses,
                    sign_p_two_sided=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.))
    order=sorted(range(12),key=lambda i:contrasts[i]['p_raw']); previous=0.
    for rank,i in enumerate(order):
        previous=max(previous,min(1.,(12-rank)*contrasts[i]['p_raw']))
        contrasts[i]['p_holm12']=previous
    clipping_rows=[]
    for dataset in ('baf','ieee_cis','paysim'):
        for mode in ('global','per_tensor'):
            for eps in (1,3,10,30,100,1000,10000,1000000):
                norms=[]; clipped=[]; tensor_clipped=0; tensors=0
                for seed in range(321000,321011):
                    source=unique[(dataset,f'local_dp_{mode}_eps{eps}',seed)]
                    doc=json.loads((ROOT/source).read_text()); audits=doc['local_upload_audits']
                    assert len(audits)==150 and {(a['round'],a['client']) for a in audits}=={(r,c) for r in range(1,51) for c in range(3)}
                    for a in audits:
                        norm=a['before_norm']; pre=a['tensor_before']; radius=a['tensor_radius']
                        assert math.isfinite(norm) and all(math.isfinite(z) for z in pre)
                        changed=norm>a['C'] if mode=='global' else any(z>radius for z in pre)
                        norms.append(norm); clipped.append(changed)
                        tensor_clipped+=sum(z>radius for z in pre); tensors+=len(pre)
                clipping_rows.append(dict(dataset=dataset,clipping=mode,epsilon=eps,client_uploads=len(norms),
                    clipped_updates=sum(clipped),fraction_clipped=st.fmean(clipped),median_preclip_norm=st.median(norms),
                    fraction_tensors_above_radius=tensor_clipped/tensors))
    matching=rows(OUT/'matching.csv'); assert len(matching)==12
    lines=['# B2 saved utility review — post hoc, 2026-10-10','',
        'B2 NOT_COMPLETE; B3 NOT_STARTED. No new measurements. Supplementary infrastructure seal: all3613 original entries rechecked; exactly the authorized error-log timing mismatch, no other discrepancies. Original manifest/log/results unchanged. Source/input freeze rechecked.',
        '', '## Utility table', '',
        'Method-minus-local-BN baseline, 11 paired seeds321000–321010; mean of three clients per seed. Entries are mean [95% paired Student-t CI], df10, in absolute score units. Delta-AUC is ROC-AUC. All57 dataset-method combinations are displayed, including baseline, v1, v2 and both DP clipping modes at every epsilon. No significance inferred from overlapping separate intervals.']
    for dataset in ('baf','ieee_cis','paysim'):
        lines+=['',f'### {dataset}', '', '| Method | Mean paired Delta-F1 [95% CI] | Mean paired Delta-AUC [95% CI] |','| --- | --- | --- |']
        methods=['baseline','dna_v1_conservative','dna_v2_0p95']+[f'local_dp_{c}_eps{e}' for c in ('global','per_tensor') for e in (1,3,10,30,100,1000,10000,1000000)]
        for method in methods:
            cells=[]
            for endpoint in ('f1','auc_roc'):
                r=next(r for r in table if (r['dataset'],r['method'],r['endpoint'])==(dataset,method,endpoint))
                cells.append(f"{float(r['mean_delta']):+.6f} [{float(r['ci_low']):+.6f}, {float(r['ci_high']):+.6f}]")
            lines.append('| '+method+' | '+' | '.join(cells)+' |')
    lines+=['','## Epsilon10000/1000000 versus epsilon10','',
        'Explicitly post-hoc exploratory comparisons of paired F1 differences (baseline cancels), n11, two-sided t test df10. Holm adjustment across these12 comparisons only, not the prospective leakage Holm family. Ordinary unadjusted95% intervals shown. Exact two-sided sign sensitivity is retained in the JSON. A non-rejection is not demonstrated equivalence. No equivalence margin was prespecified.','',
        '| Dataset | Clip | Epsilon | Mean F1 difference [95% CI] | Raw p | Holm12 p | Detectable at Holm .05 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in contrasts:
        lines.append(f"| {r['dataset']} | {r['clipping']} | {r['epsilon']} | {r['mean_difference']:+.6f} [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] | {r['p_raw']:.6g} | {r['p_holm12']:.6g} | {'yes' if r['p_holm12']<.05 else 'no'} |")
    lines+=['','## Saved clipping telemetry','',
        'Each dataset/epsilon/mode has1650 client uploads =11seeds x50rounds x3clients. This table pools uploads descriptively; uploads are not independent inferential replicates. Norm is the concatenated uploaded non-BN trainable delta before clipping, noise and aggregation. Joint clipping event: norm>C=.01; per-tensor event: any tensor norm>C/sqrt(L). Any clipping event is distinct from every tensor being clipped.','',
        '| Dataset | Clip | Epsilon | Clipped updates /1650 | Fraction | Median preclip norm |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in clipping_rows:
        lines.append(f"| {r['dataset']} | {r['clipping']} | {r['epsilon']} | {r['clipped_updates']} | {r['fraction_clipped']:.6f} | {r['median_preclip_norm']:.6f} |")
    lines+=['','## Matched utility and interpretation','',
        'All12 transform/dataset/clipping combinations remain NOT_ASSESSABLE: no grid point meets transform mean F1 minus.005. No matched epsilon; no interpolation, altered threshold or expanded grid.',
        '', 'High-epsilon comparisons and clipping telemetry can be consistent with a clipping bottleneck, but do not establish that clipping rather than noise caused the utility loss. At epsilon1e6 the noise SD remains positive; optimization paths and validation-selected thresholds may differ. There is no saved clipping-only/no-noise counterfactual in this design. Increasing epsilon does not remove C=.01 clipping. Utility and any future leakage axis are different configurations; no privacy conclusion is drawn here.',
        '', 'All prior artifacts/reports, failed outputs and error logs are preserved. No B3 run or B2 completion authorization is inferred.']
    with (DEST/'b2_saved_utility_review.md').open('x') as f: f.write('\n'.join(lines)+'\n')
    save(DEST/'saved_data_analysis.json',dict(contrasts=contrasts,clipping=clipping_rows,
        utility_rows_verified=171,paired_seed_rows_verified=1881,source_receipts_verified=627,
        matching=matching,B2_complete=False,B3_started=False,measurement_runs=0))
    # Recheck all original evidence after the analysis, not merely before it.
    for rel,h in effective.items(): assert sha(ROOT/rel)==h,rel
    assert not u.live()
    save(DEST/'supplementary_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(DEST.iterdir()) if p.is_file()})
    print(json.dumps(dict(status='SUPPLEMENTARY_SEAL_AND_SAVED_REVIEW_READY',B2_complete=False,B3_started=False,
        contrasts=contrasts),indent=2))

if __name__=='__main__': main()
