"""Full file manifest and final seal; requires independently audited report."""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.priority34d_final_analysis import OUT, sha, read, write_once, verify_analysis, no_live
MANIFEST=OUT/'audits/full_immutable_manifest.json'


def checkpoints():
    import torch
    for seed in (342000,342001,342002):
        base=OUT/'A2/baf/training'/('baseline_'+str(seed))
        receipt=read(base/'checkpoint_validated.json')
        assert receipt['seed']==seed and receipt['status']=='PASS' and receipt['rounds']==50
        assert receipt['device']=='cpu' and receipt['intra_threads']==receipt['inter_threads']==1
        for name,digest in receipt['outputs'].items(): assert sha(base/name)==digest
        rows=[json.loads(line) for line in (base/'bn_rounds.jsonl').read_text().splitlines()]
        assert [row['round'] for row in rows]==list(range(1,51))
        assert all(math.isfinite(v) and v>=0 for row in rows for v in row['bn_min'].values())
        state=torch.load(base/'final_state.pt',map_location='cpu',weights_only=False)
        assert all(torch.isfinite(value).all() for value in state.values())
        assert all((value>=0).all() for name,value in state.items() if name.endswith('running_var'))


def build():
    verify_analysis(); checkpoints()
    report=ROOT/'reports/priority34d_report.md'
    if not report.exists(): raise RuntimeError('report missing')
    stats=read(OUT/'audits/final_statistics.json')
    assert len(stats['rows'])==76 and len(stats['combined'])==211 and stats['independent_sign_holm_verified']
    for name,status in (('bn','PASS_BN_STAGE_ONLY'),('image','PASS_IMAGE_STAGE_ONLY'),('utility','PASS_UTILITY_STAGE_ONLY')):
        assert read(OUT/('audits/independent_'+name+'_results.json'))['status']==status
    checks=read(OUT/'audits/final_code_checks.json')
    assert checks['compile']=='PASS' and checks['diff']=='PASS' and checks['tests']=='PASS'
    tables=read(OUT/'audits/table_export_audit.json')
    assert tables['fixed76']==76 and tables['combined211']==211 and tables['roundtrip']=='PASS'
    snapshot=OUT/'audits/PROJECT_before_completion.md'
    if snapshot.exists(): raise RuntimeError('preserve existing PROJECT snapshot')
    shutil.copyfile(ROOT.parent/'PROJECT.md',snapshot)
    hashes={}
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path!=MANIFEST:
            hashes[str(path.relative_to(ROOT))]=sha(path)
    hashes[str(report.relative_to(ROOT))]=sha(report)
    # Include all external immutable dependencies, not just P34D output files.
    for path in list(OUT.rglob('*freeze.json')):
        doc=read(path)
        for name,digest in {**doc.get('sources',{}),**doc.get('inputs',{})}.items():
            assert sha(ROOT/name)==digest
            if name in hashes: assert hashes[name]==digest
            hashes[name]=digest
    for path in (Path(__file__),ROOT/'tests/test_priority34d_final_seal.py',OUT/'audits/table_builder.mjs'):
        hashes[str(path.relative_to(ROOT))]=sha(path)
    write_once(MANIFEST,dict(at=time.time(),files=hashes,count=len(hashes),
        scope='all P34D files plus report and immutable source/input dependencies; excludes this manifest and later completion seals'))
    verify_manifest()
    print(json.dumps(dict(stage='FULL_MANIFEST_PASS',files=len(hashes),sha256=sha(MANIFEST))))


def verify_manifest():
    doc=read(MANIFEST)
    assert doc['count']==len(doc['files'])
    for name,digest in doc['files'].items():
        if sha(ROOT/name)!=digest: raise ValueError('full manifest mismatch: '+name)
    no_live()


def complete():
    verify_analysis(); verify_manifest()
    # Read-only final code check after all files and administrative notes exist.
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    for path in list((ROOT/'experiments').glob('*priority34d*.py'))+list((ROOT/'tests').glob('*priority34d*.py')):
        compile(path.read_text(),str(path),'exec')
    project=ROOT.parent/'PROJECT.md'
    if 'P34D final independent verification COMPLETE' not in project.read_text():
        raise RuntimeError('final administrative PROJECT note missing')
    seal=dict(status='verified COMPLETE',at=time.time(),report_sha256=sha(ROOT/'reports/priority34d_report.md'),
        full_manifest_sha256=sha(MANIFEST),full_manifest_files=read(MANIFEST)['count'],
        PROJECT_snapshot_sha256=sha(OUT/'audits/PROJECT_before_completion.md'),
        PROJECT_after_note_sha256=sha(project),no_live_science=True,goal_changes=False,
        fixed_family=76,combined_family=211)
    no_live(); write_once(OUT/'COMPLETE.json',seal)
    print(json.dumps(seal))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--build',action='store_true');parser.add_argument('--complete',action='store_true');parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.build: build()
    elif args.complete: complete()
    elif args.verify: verify_manifest(); print('FULL_MANIFEST_PASS')
    else: parser.error('choose build/complete/verify')
