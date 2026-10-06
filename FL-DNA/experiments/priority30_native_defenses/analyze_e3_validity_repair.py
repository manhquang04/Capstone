"""Read-only re-analysis of P30, substituting corrected E3 outputs."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.priority30_native_defenses.analyze_s0u_pairs import collect_cell_results, exact_one_sided_p, holm_adjust
OLD=ROOT/'artifacts/priority30_native_defenses/audit'
NEW=OLD/'e3_repair_20261001'
OUT=NEW/'S6'

def load(p): return json.loads(Path(p).read_text())
def write_csv(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
def metric(domain,r):
    if domain=='image':return r['metrics']['mse'],r['metrics']['psnr_db'],r['metrics']['ssim']
    a=r['metric']['accuracy_percent'];return 100-a,a,None
def refs(domain,r):
    if domain=='image':return [r['baselines']['gray']['psnr_db'],r['baselines']['cifar_mean']['psnr_db']]
    return [r['baselines']['mean_mode']['accuracy_percent'],r['baselines']['empirical_marginal']['accuracy_percent']]
def paired(domain,defense,evaluation,comparator,a,b,source_a,source_b):
    ae,aq,ass=metric(domain,a);be,bq,bss=metric(domain,b)
    assert all(math.isfinite(value) for value in (ae,aq,be,bq)),(source_a,source_b)
    if domain=='image':assert math.isfinite(ass) and math.isfinite(bss),(source_a,source_b)
    tid=a['target_id']
    assert tid==b['target_id']
    key='cifar10_index' if domain=='image' else 'adult_indices'
    assert a[key]==b[key],(source_a,source_b)
    s0=load(OLD/'S0u'/domain/f'target_{tid:03d}'/'result.json')
    assert a[key]==s0[key]
    reference=refs(domain,s0)
    return dict(domain=domain,defense=defense,evaluation=evaluation,comparator=comparator,target_id=tid,
        defense_error=ae,comparator_error=be,difference_error=ae-be,
        defense_quality=aq,comparator_quality=bq,difference_quality=aq-bq,
        defense_ssim=ass,comparator_ssim=bss,
        reference_1=reference[0],reference_2=reference[1],source_defense=str(source_a.relative_to(ROOT)),source_comparator=str(source_b.relative_to(ROOT)))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    selection=load(NEW/'S4/selected.json')
    if selection['status']=='CALIBRATED':
        # Descriptive accounting for training composition, not selection.
        from experiments.dp_update_accounting import alpha_grid,epsilon_from_rdp
        for spec in selection['selected'].values():
            if spec['sigma'] is not None:
                spec['epsilon_50_releases']=epsilon_from_rdp(spec['sigma'],1.,1e-5,50,alpha_grid())['epsilon']
                spec['epsilon_training_rounds']=epsilon_from_rdp(spec['sigma'],1.,1e-5,selection['rounds'],alpha_grid())['epsilon']
        (OUT/'utility_accounting_descriptive.json').write_text(json.dumps(selection,indent=2)+'\n')
    paths=collect_cell_results(OLD)
    references={}
    rows=[]
    unavailable=[]
    for p in paths:
        a=load(p);domain=a['domain'];defense=a['defense'];ev=a['eval'];tid=a['target_id']
        if defense=='ats' and ev=='E2':continue
        references[domain,defense,ev,tid]=p
        s0=OLD/'S0u'/domain/f'target_{tid:03d}'/'result.json';b=load(s0)
        rows.append(paired(domain,defense,ev,'undefended',a,b,p,s0))
        for name in b['baselines']:
            adapted={**b}
            adapted['metrics' if domain=='image' else 'metric']=b['baselines'][name]
            rows.append(paired(domain,defense,ev,name,a,adapted,p,s0))
    # Match the corrected DP outputs to the SAME frozen E2 defense target.
    for p in sorted((OLD/'S3').glob('*/*/*/target_*/result.json')):
        b=load(p)
        if b.get('status')!='ran':continue
        domain=b['domain'];defense=b['defense'];c=b['comparator'];tid=b['target_id']
        if c=='distortion_matched_clipped' and defense=='precode':continue
        if domain=='adult' and defense=='soteria' and c=='distortion_matched_clipped':
            p=NEW/'S3/adult/soteria'/f'target_{tid:03d}'/'result.json';b=load(p)
        ap=references.get((domain,defense,'E2',tid))
        if ap is None:raise RuntimeError(f'missing E2 {domain} {defense} {tid}')
        a=load(ap);rows.append(paired(domain,defense,'E3',c,a,b,ap,p))
    for defense in ('precode','dna_v1_conservative'):
        available=selection['status']=='CALIBRATED' and selection['selected'][defense]['bracketed']
        if not available:
            unavailable.append({'domain':'adult','defense':defense,'comparator':'utility_matched_clipped','status':'NOT_ASSESSABLE','reason':'baseline operating-point gate or bracketing failed'})
            continue
        for tid in range(39):
            bp=NEW/'S4/adult'/defense/f'target_{tid:03d}'/'result.json';b=load(bp)
            ap=references['adult',defense,'E2',tid];a=load(ap)
            rows.append(paired('adult',defense,'E3','utility_matched_clipped',a,b,ap,bp))
    grouped=defaultdict(list)
    for r in rows:grouped[r['domain'],r['defense'],r['evaluation'],r['comparator']].append(r)
    summary=[];hypotheses=[]
    for key,cell in sorted(grouped.items()):
        assert len(cell)==39 and len({r['target_id'] for r in cell})==39,key
        domain,defense,ev,c=key;w=sum(r['difference_error']>0 for r in cell);l=sum(r['difference_error']<0 for r in cell);t=39-w-l
        med=lambda k:float(np.median([r[k] for r in cell]))
        ref1=med('reference_1');ref2=med('reference_2');a=med('defense_quality');b=med('comparator_quality')
        both=a<=max(ref1,ref2) and b<=max(ref1,ref2)
        entry=dict(domain=domain,defense=defense,evaluation=ev,comparator=c,n=39,defense_wins=w,comparator_wins=l,ties=t,
            p_defense_greater_error=exact_one_sided_p(w,l),median_defense_quality=a,median_comparator_quality=b,
            median_paired_quality_difference=med('difference_quality'),median_error_difference=med('difference_error'),
            median_reference_1=ref1,median_reference_2=ref2,defense_minus_reference_1=a-ref1,defense_minus_reference_2=a-ref2,
            comparator_minus_reference_1=b-ref1,comparator_minus_reference_2=b-ref2,
            reference_names='gray / CIFAR mean' if domain=='image' else 'mean-mode / empirical marginal',
            both_at_reference_level=both,reference_flag='both at reference level' if both else '')
        if domain=='image':
            entry['median_defense_ssim']=med('defense_ssim');entry['median_comparator_ssim']=med('comparator_ssim')
            entry['median_paired_ssim_difference']=float(np.median([r['defense_ssim']-r['comparator_ssim'] for r in cell]))
        name='::'.join(key)
        hypotheses.append((name+'::defense',entry['p_defense_greater_error']))
        entry['hypothesis_id']=name
        if ev=='E3':
            entry['p_comparator_greater_error']=exact_one_sided_p(l,w)
            hypotheses.append((name+'::comparator',entry['p_comparator_greater_error']))
        summary.append(entry)
    adjusted=holm_adjust(hypotheses)
    family_size=sum(math.isfinite(p) for _,p in hypotheses)
    for s in summary:
        name=s['hypothesis_id'];s['holm_p_defense_greater_error']=adjusted[name+'::defense']
        if s['evaluation']=='E3':s['holm_p_comparator_greater_error']=adjusted[name+'::comparator']
        s['holm_family_size']=family_size
    write_csv(OUT/'per_target.csv',rows);write_csv(OUT/'tests_with_effects_and_references.csv',summary)
    lookup={(r['domain'],r['defense'],r['evaluation'],r['comparator']):r for r in summary}
    old_verdict={('image','precode'):'SURVIVES',('image','gradient_pruning'):'SURVIVES',('image','dna_v1_conservative'):'SURVIVES',('adult','precode'):'SURVIVES',('image','ats'):'NOT_ASSESSABLE'}
    verdicts=[]
    for domain,defenses in [('image',['precode','soteria','gradient_pruning','ats','count_sketch','dna_v1_conservative','dna_v2_0p95']),('adult',['precode','soteria','gradient_pruning','count_sketch','dna_v1_conservative','dna_v2_0p95'])]:
        for defense in defenses:
            e2=lookup.get((domain,defense,'E2','undefended'))
            if e2 is None:v='NOT_ASSESSABLE';reason='ATS has no working adaptive attacker'
            elif not math.isfinite(e2['holm_p_defense_greater_error']) or e2['holm_p_defense_greater_error']>.05:v='DOES NOT SURVIVE';reason='E2 not significant under whole-family Holm'
            elif domain=='image' and defense=='precode':v='NOT_ASSESSABLE';reason='E2 passes; DP comparison NOT_ASSESSABLE (different parameter spaces)'
            else:
                required=['utility_matched_clipped'] if defense=='precode' else ['distortion_matched_clipped']
                if domain=='adult' and defense=='dna_v1_conservative':required+=['utility_matched_clipped']
                comparisons=[lookup.get((domain,defense,'E3',c)) for c in required]
                if any(r is None for r in comparisons):v='NOT_ASSESSABLE';reason='required matched-DP comparison unavailable'
                elif any(r['holm_p_comparator_greater_error']<=.05 for r in comparisons):v='DOES NOT SURVIVE';reason='matched DP significantly better: '+','.join(r['comparator'] for r in comparisons if r['holm_p_comparator_greater_error']<=.05)
                else:v='SURVIVES';reason='E2 passes; no significant loss to required matched DP'
            old=old_verdict.get((domain,defense),'DOES NOT SURVIVE')
            verdicts.append({'domain':domain,'defense':defense,'old_superseded_verdict':old,'verdict':v,'changed':old!=v,'reason':reason})
    write_csv(OUT/'verdicts.csv',verdicts)
    unavailable += [{'domain':d,'defense':'precode','comparator':'distortion_matched_clipped','status':'NOT_APPLICABLE','reason':'no common full parameter space'} for d in ('image','adult')]
    (OUT/'unavailable.json').write_text(json.dumps(unavailable,indent=2)+'\n')
    (OUT/'supersession.json').write_text(json.dumps({'old_report':'reports/priority30_native_audit_report_20260930.md','old_verdict':'superseded by this repair; preserved unchanged','superseded':['S4 all old utility calibration and dependent attack outputs','S3/adult/distortion_matched_clipped/soteria','S3/*/distortion_matched_clipped/precode','S6 old family-separated Holm and verdict table'],'holm_family_size':family_size},indent=2)+'\n')
    report=['# Priority 30 — E3 validity repair (2026-10-01)',
        '', 'Status: corrected analysis. The final verdict table in '
        '`reports/priority30_native_audit_report_20260930.md` is SUPERSEDED. '
        'That report and all old artifacts remain unchanged.', '',
        '## Protocol and execution', '',
        'Protocol: `protocols/amendments/2026-10-01_priority30_e3_validity_repair.md`. '
        'Utility length/lr selection was clarified before any utility output. '
        'No new source targets were drawn; attacks reuse the 39 S1/S2 targets. '
        'Every new result asserts source IDs equal the paired S0u artifact. '
        'No Latex/ or external_defenses/ edits; one torch thread per process.', '',
        'Commands executed:', '', '```sh',
        'external_defenses/.venv/bin/python -B experiments/priority30_native_defenses/repair_e3_validity.py --stage all --workers 8',
        'kill -INT 63696',
        '# Initial parent received SIGINT before utility execution after the continuation required validation-selected training length.',
        '# Its queued Soteria jobs were allowed to finish; completed files are preserved and never recomputed.',
        'external_defenses/.venv/bin/python -B experiments/priority30_native_defenses/repair_e3_validity.py --stage utility --workers 8',
        'external_defenses/.venv/bin/python -B experiments/priority30_native_defenses/analyze_e3_validity_repair.py',
        'external_defenses/.venv/bin/python -B experiments/priority30_native_defenses/finalize_e3_validity_repair.py', '```', '',
        'Execution journal: `artifacts/priority30_native_defenses/audit/e3_repair_20261001/journal.log`. '
        'An attempted pytest invocation failed because the read-only reference venv has no pytest; '
        'the synthetic gate/bracketing tests were invoked directly and passed. '
        'No package was installed. See final checks manifest for compile/diff checks.', '',
        '## PRECODE distortion is not defined in a common parameter space', '',
        'Old calibration (`run_e3_dp_comparators.py:310–312`) flattened both gradients, '
        'used `shared=min(b.numel(),d.numel())`, and took `norm(d[:shared]-b[:shared])`. '
        'LeNet-Zhu has 15,826 parameters; PRECODE has 606,930. The first six tensors '
        '(body convolutions, 8,136 coordinates) match names/shapes. After that, base '
        '`fc.0.weight` (7,680) and bias (10) were compared to the first 7,690 '
        'coordinates of PRECODE `bottleneck.encoder.weight` (393,216). Thus the reported '
        'median 30.9532146454 is a mismatched-prefix norm, not defense L2 distortion. '
        'The gradients through shared convolutions also differ due to a different '
        'stochastic forward path; they do not define the full-vector comparator. '
        'Adult old calibration used a pass-through (`run_e3_dp_comparators.py:279–284`), '
        'yielding zero. PRECODE E3(ii) is NOT_APPLICABLE in BOTH domains. '
        'Adult is assessed using repaired E3(iii) only; image has no assessable matched-DP arm.', '',
        '## Repaired Adult Soteria distortion calibration', '']
    sc=load(NEW/'S3/calibration.json')['adult']['soteria']
    report += ['| C | sigma | median L2 distortion | dimension | epsilon (one release, delta=1e-5) |',
        '|---:|---:|---:|---:|---:|',
        f"| {sc['clip_norm']:.12g} | {sc['sigma']:.12g} | {sc['median_defense_l2_distortion']:.12g} | {sc['dimension']} | {sc['epsilon_delta_1e_minus_5_one_release']:.12g} |", '',
        'Sensitivity mask uses the real Adult Soteria algorithm at 40th percentile '
        'on `layers.3.weight` (`run_audit.py:855–871`), not a pass-through. '
        'Seed 42300+target_id; base/defended gradient computed on the same model. '
        'Each of the 39 distortions is positive. Sigma matches expected Gaussian '
        'L2 scale using the inherited `median_distortion/(C*sqrt(d))` formula. '
        'C is the p95 norm; clipping in the top 5% can add distortion, so this '
        'is an expected-noise-scale match, not an exact match per target. '
        'Accounting: update-level add/remove, sensitivity ratio 1, Gaussian RDP, no subsampling.', '',
        '## Adult utility repair', '',
        'Re-reading the old CSV: v1 is exactly 0.7543160915 for all 16 seeds; '
        'baseline spans 0.7543160915–0.7544488907; DP sigma<=0.03 spans '
        '0.7543160915–0.7545152903. Thus the majority-class degeneracy is real, '
        'but the claim that every baseline/DP value is bit-identical is not exact. '
        'The old S4 selections/comparisons are superseded in full.', '',
        '```json',json.dumps(selection,indent=2), '```', '']
    if selection['status']=='CALIBRATED':
        utility=list(csv.DictReader((NEW/'S4/utility_grid.csv').open()))
        gridgroups=defaultdict(list)
        for r in utility:gridgroups[r['branch'],r['sigma']].append(r)
        report += ['Validation is stratified 20% of Adult train (seed 43001). '
            'Scaling is fit on the remaining train only. LR and rounds are selected '
            'on two validation-only seeds; DP selection uses 16 paired validation replicates. '
            'Test accuracy is descriptive and used only for the pre-frozen baseline quality gate. '
            'PRECODE includes official KL loss and 8-pass probability averaging for evaluation.', '',
            '| Branch | sigma | validation mean (SD) | paired validation delta | test mean (SD) | paired test delta |',
            '|---|---:|---:|---:|---:|---:|']
        for (branch,sigma),cell in sorted(gridgroups.items(),key=lambda kv:(kv[0][0],float(kv[0][1] or 0))):
            vals=lambda k:np.array([float(r[k]) for r in cell])
            v=vals('validation_accuracy');t=vals('test_accuracy')
            report.append(f"| {branch} | {sigma or '—'} | {v.mean():.6f} ({v.std(ddof=1):.6f}) | {vals('delta_validation_accuracy').mean():+.6f} | {t.mean():.6f} ({t.std(ddof=1):.6f}) | {vals('delta_test_accuracy').mean():+.6f} |")
    report += ['', '## Every E2/E3 test: effects, data-free references and whole-family Holm', '',
        f'Holm uses ONE family of {family_size} finite hypotheses: all valid E1/E2 '
        'vs undefended/data-free tests and both directions of all valid E3 tests. '
        'Invalid PRECODE distortion and old S4 tests are excluded. This is more '
        'conservative than the old separate-family output. Strict exact-zero ties; '
        'one-sided Binomial(non-tied, 0.5). E1 results remain in the family and full CSV.', '',
        'Quality: image PSNR dB (SSIM in parentheses), Adult feature accuracy %. '
        'Reference columns are gray/CIFAR mean for image and mean-mode/empirical '
        'marginal for Adult. Higher quality means more leakage. The reference flag '
        'compares both medians to the better baseline median, descriptively only. '
        'No practical-equivalence claim or verdict adjustment follows from this flag.', '',
        '| Domain | Defense | Evaluation/comparator | wins/losses/ties | defense quality (SSIM) | comparator quality (SSIM) | paired median quality delta | refs 1/2 | raw p defense / comparator | Holm p defense / comparator | reference flag |',
        '|---|---|---|---|---:|---:|---:|---:|---|---|---|']
    for r in summary:
        if r['evaluation'] not in ('E2','E3'):continue
        a=f"{r['median_defense_quality']:.4f}";b=f"{r['median_comparator_quality']:.4f}"
        if r['domain']=='image':
            a+=f" ({r['median_defense_ssim']:.4f})";b+=f" ({r['median_comparator_ssim']:.4f})"
        pr=f"{r['p_defense_greater_error']:.8g}";pa=f"{r['holm_p_defense_greater_error']:.8g}"
        if r['evaluation']=='E3':
            pr+=f" / {r['p_comparator_greater_error']:.8g}";pa+=f" / {r['holm_p_comparator_greater_error']:.8g}"
        report.append(f"| {r['domain']} | {r['defense']} | {r['evaluation']} / {r['comparator']} | {r['defense_wins']}/{r['comparator_wins']}/{r['ties']} | {a} | {b} | {r['median_paired_quality_difference']:+.4f} | {r['median_reference_1']:.4f}/{r['median_reference_2']:.4f} | {pr} | {pa} | {r['reference_flag']} |")
    report += ['', '## RQ4 re-issued verdicts', '',
        'This table supersedes the old verdict table. Failure of E2 takes precedence; '
        'missing required matched-DP evidence is NOT_ASSESSABLE. Paper-unclipped '
        'noise is descriptive, not the matched-DP criterion. Failure to reject '
        'a DP advantage is not proof of equivalence.', '',
        '| Domain | Defense | Old (superseded) | New | Changed | Reason |', '|---|---|---|---|---|---|']
    for r in verdicts:report.append(f"| {r['domain']} | {r['defense']} | {r['old_superseded_verdict']} | {r['verdict']} | {r['changed']} | {r['reason']} |")
    report += ['', 'Changed verdicts:', '']
    for r in verdicts:
        if r['changed']:report.append(f"- {r['domain']} / {r['defense']}: {r['old_superseded_verdict']} → {r['verdict']}; {r['reason']}.")
    report += ['', '## Artifacts and hashes', '',
        'All new per-target results, arrays, split IDs, per-seed grid results, '
        'analysis CSVs and their SHA-256 hashes are listed in '
        '`artifacts/priority30_native_defenses/audit/e3_repair_20261001/sha256_manifest.csv`. '
        'Full tests/effects/references: `S6/tests_with_effects_and_references.csv`; '
        'per-target comparisons: `S6/per_target.csv`; exclusions: `S6/unavailable.json`; '
        'supersession map: `S6/supersession.json`.', '',
        'Final verification is recorded in `checks.json`: py_compile, '
        'git diff --check, 3 direct unit/statistical tests, and unchanged old summary hashes. '
        'A final process check is performed after all workload exits.', '']
    report_path=ROOT/'reports/priority30_e3_validity_repair_report_20261001.md'
    report_path.write_text('\n'.join(report))
    print(json.dumps({'family_size':family_size,'verdicts':verdicts},indent=2))

if __name__=='__main__':main()
