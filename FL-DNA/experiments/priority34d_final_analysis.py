"""Additive final analysis orchestration; never trains or runs recovery."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'artifacts/priority34d'
FREEZE = OUT/'audits/final_analysis_freeze.json'
PHASES = ('checkpoint_execution_freeze.json', 'image_utility/preparation_freeze.json',
          'image_utility/execution_freeze.json', 'bn/baf/execution_freeze.json',
          'image_recovery/execution_freeze.json')
STAGES = ('CHECKPOINTS_COMPLETE.json', 'image_utility/UTILITY_COMPLETE.json',
          'bn/baf/BN_COMPLETE.json', 'image_recovery/IMAGE_COMPLETE.json')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_once(path, value):
    path = Path(path)
    if path.exists():
        raise RuntimeError('preserve existing artifact: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def no_live():
    from experiments.priority34d_continue import live
    if live():
        raise RuntimeError('scientific workload remains alive')
    for name in STAGES:
        if not (OUT/name).exists():
            raise RuntimeError('missing scientific stage: '+name)
    bad = [p for p in OUT.rglob('*.json') if p.name.startswith('REQUIRES_DIRECTION')
           or p.name == 'SUPERVISOR_FAILURE.json']
    if bad:
        raise RuntimeError('preserved science failure requires direction')


def verify_freezes():
    seen = {}
    for name in PHASES:
        doc = read(OUT/name)
        for key, digest in {**doc['sources'], **doc['inputs']}.items():
            if key in seen and seen[key] != digest:
                raise ValueError('inconsistent frozen digest: '+key)
            if sha(ROOT/key) != digest:
                raise ValueError('frozen source/input drift: '+key)
            seen[key] = digest
    return seen


def freeze():
    no_live(); verify_freezes()
    sources = list((ROOT/'experiments').glob('*priority34d*.py'))
    sources += list((ROOT/'tests').glob('*priority34d*.py'))
    sources += list((ROOT/'protocols/amendments').glob('*priority34d*.md'))
    sources += [OUT/'audits/json_key_report_disclosure.md']
    inputs = [OUT/name for name in PHASES+STAGES]
    inputs += [ROOT/'artifacts/priority33c/combined135_holm.csv',
               OUT/'image_utility/calibration.json', OUT/'audits/json_key_preflight_repair.json']
    write_once(FREEZE, dict(at=time.time(), sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs}, scope='administrative final audit only'))
    print(json.dumps(dict(stage='FINAL_ANALYSIS_FROZEN', sha256=sha(FREEZE))))


def verify_analysis():
    no_live(); verify_freezes()
    doc = read(FREEZE)
    for key, digest in {**doc['sources'], **doc['inputs']}.items():
        if sha(ROOT/key) != digest:
            raise ValueError('analysis freeze drift: '+key)


def descriptive(endpoints):
    from experiments.priority34d_assemble_statistics import image_registry
    from experiments import priority34d_statistics as s
    lookup = image_registry(endpoints); results = []; effects = []
    for setting, strata in [(model, (model,)) for model in s.MODELS]+[
            ('pooled_initializations',s.MODELS),('trained_batch4',('trained_batch4',))]:
        arms = set.intersection(*[{key[1] for key in lookup if key[0]==model} for model in strata])
        for arm in sorted(arms):
            values = {metric:[lookup[(model,arm,index)][metric] for model in strata for index in range(39)]
                      for metric in ('psnr_db','ssim','mse')}
            results.append(dict(setting=setting, arm=arm, metrics={m:s.median_interval(v) for m,v in values.items()}))
        for method in s.DNA:
            for arm in ('unprotected','dp_single_for_'+method,'dp_per_tensor_for_'+method):
                if arm not in arms:
                    continue
                values = {metric:[lookup[(model,method,index)][metric]-lookup[(model,arm,index)][metric]
                                  for model in strata for index in range(39)]
                          for metric in ('psnr_db','ssim','mse')}
                effects.append(dict(setting=setting, method=method, comparator=arm,
                    metrics={m:s.median_interval(v) for m,v in values.items()}, descriptive_only=True))
    return results, effects


def assemble():
    verify_analysis()
    from experiments.priority34d_assemble_statistics import assemble as stats
    from experiments import priority34d_report_text as report
    from experiments.priority34d_independent_metrics import gaussian_epsilon
    image = read(OUT/'audits/independent_image_results.json')
    bn = read(OUT/'audits/independent_bn_results.json')
    utility_audit = read(OUT/'audits/independent_utility_results.json')
    assert image['status']=='PASS_IMAGE_STAGE_ONLY' and image['jobs']==1092
    assert bn['status']=='PASS_BN_STAGE_ONLY' and utility_audit['status']=='PASS_UTILITY_STAGE_ONLY'
    assert utility_audit['jobs']==1216
    for doc, name in ((image, PHASES[4]), (bn,PHASES[3]),(utility_audit,PHASES[2])):
        assert doc['freeze_sha256']==sha(OUT/name)
    with (ROOT/'artifacts/priority33c/combined135_holm.csv').open(newline='') as stream:
        prior = list(csv.DictReader(stream))
    utility = read(OUT/'image_utility/calibration.json')
    bundle = stats(image['endpoints'], bn['endpoints'], bn['qualified_checkpoints'], utility['matches'], prior)
    bundle['summaries'], bundle['descriptive_metric_effects'] = descriptive(image['endpoints'])
    write_once(OUT/'audits/final_statistics.json', bundle)
    clips = {setting:read(OUT/'image_utility'/setting/'clip.json') for setting in
             ('init342042','init342043','init342044','trained_batch4')}
    bn_context = {}; gates = []
    for seed in (342000,342001,342002):
        entries = []
        for stage in ('n8','n24'):
            path = OUT/'bn/baf'/str(seed)/(stage+'_gate.json')
            if path.exists():
                gate = read(path)
                gates.append(dict(checkpoint=seed, stage=stage, tests=gate['tests'], passed=gate['passed']))
                entries.append(stage+': '+', '.join(name+' p='+report.number(test['p']) for name,test in gate['tests'].items()))
        path = OUT/'bn/baf'/str(seed)/'distortion_calibration.json'
        if path.exists():
            for method, cal in read(path).items():
                entries.append(method+': C='+report.number(cal['clip'])+', sigma='+report.number(cal['sigma'])+
                    ', one-release epsilon='+report.number(gaussian_epsilon(cal['sigma'])))
        bn_context[str(seed)] = '; '.join(entries)
    hashes = {name:sha(OUT/name) for name in PHASES+STAGES}
    for name in ('audits/final_analysis_freeze.json','audits/independent_image_results.json',
                 'audits/independent_bn_results.json','audits/independent_utility_results.json',
                 'audits/final_statistics.json','audits/json_key_preflight_repair.json'):
        hashes[name] = sha(OUT/name)
    chunks = [report.render(bundle,utility,clips,bn['endpoints'],bn['qualified_checkpoints'],bn_context,hashes,verified=True)]
    chunks.append('## Descriptive paired PSNR/SSIM/MSE effects\n\n'+report.table(
        ['Setting','DNA','Comparator','PSNR DNA−comparator','SSIM DNA−comparator','MSE DNA−comparator'],
        [[row['setting'],row['method'],row['comparator'],*[report.interval(row['metrics'][m]) for m in ('psnr_db','ssim','mse')]]
         for row in bundle['descriptive_metric_effects']]))
    chunks.append((OUT/'audits/json_key_report_disclosure.md').read_text())
    significant = [row for row in bundle['rows'] if row['holm_combined211']<.05]
    chunks.append('## Confirmatory decision summary\n\n'+str(len(significant))+
        ' of76 P34D directional tests have Holm211 p<0.05. Non-significance is not equivalence. '+
        'All76 tests remain reserved. Significant directions are listed below.\n\n'+report.table(
        ['Hypothesis','Holm211 p','Paired effect median'],
        [[row['hypothesis_id'],report.number(row['holm_combined211']),report.number(row['paired_effect_interval']['median'])]
         for row in significant]))
    destination = ROOT/'reports/priority34d_report.md'
    if destination.exists():
        raise RuntimeError('preserve existing report')
    destination.write_text('\n\n'.join(chunks))
    verify_analysis()
    print(json.dumps(dict(stage='INDEPENDENT_STATISTICS_REPORT_WRITTEN', complete=False,
        fixed_family=76, combined_family=211, significant_holm211=len(significant), report_sha256=sha(destination))))


if __name__=='__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--freeze',action='store_true'); parser.add_argument('--assemble',action='store_true')
    args = parser.parse_args()
    if args.freeze: freeze()
    elif args.assemble: assemble()
    else: parser.error('choose --freeze or --assemble')
