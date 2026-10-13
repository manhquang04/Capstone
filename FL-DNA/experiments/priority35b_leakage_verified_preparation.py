"""Additive one-shot audit wrapper with before/after scientific-file ledger.

Run ONLY after measurements finish and all processes exit. No measurements,
training, optimization, figure rendering or report publication are performed.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from experiments import priority35b_leakage_prepare_publication as prepare
from experiments import priority35b_leakage_manifest_audit as manifest

SEALS=prepare.SEALS+('manifest_audit_freeze.json','verified_preparation_freeze.json')


def check_stability(before, after):
    if len(before)!=49920 or len(after)!=49920:raise ValueError('all49920 measurement hashes required')
    if before!=after:raise ValueError('measurement evidence changed during independent verification')
    return dict(status='ALL_MEASUREMENT_HASHES_STABLE_ACROSS_VERIFICATION',entries=49920,
                measurement_repeated=False,full_priority35b_complete=False)


def verify_sources(root):
    prepare.scientific.verify_additive_sources(root,SEALS)
    out=Path(root)/'artifacts/priority35b/b3'
    for name in SEALS:
        for path,digest in prepare.integrity.read(out/name).get('inputs',{}).items():
            if prepare.integrity.sha(prepare.integrity.child(root,path))!=digest:
                raise ValueError('verified-preparation dependency drift: '+path)


def snapshot_measurements(root):
    root=Path(root);out=root/'artifacts/priority35b/b3'
    jobs=prepare.integrity.read(out/'execution_freeze.json')['jobs']
    results=[]
    for c in jobs:
        path=out/'jobs'/c['dataset']/c['cell']/('%03d'%c['index'])/c['arm']/'result.json'
        results.append(prepare.integrity.read(path))
    paths=manifest.measurement_inventory(jobs,results)
    return {p:prepare.integrity.sha(prepare.integrity.child(root,p)) for p in sorted(paths)}


def run(root=prepare.ROOT):
    root=Path(root).resolve();out=root/'artifacts/priority35b/b3'
    exited=prepare.no_live(root)  # No quality or scientific-file reads before this.
    prepare.require_finished(prepare.integrity.read(root/'artifacts/priority35b/progress.json'),
        list(out.glob('*.lock')),[p for p in (out/'REQUIRES_DIRECTION.json',out/'SUPERVISOR_FAILURE.json') if p.exists()])
    verify_sources(root)
    ledger=out/('verified_preparation_attempt_'+str(time.time_ns()));ledger.mkdir()
    prepare.new_json(ledger/'START.json',dict(at=prepare.now(),pid=__import__('os').getpid(),
        command=[sys.executable,*sys.argv],exited_pids=exited,measurement_repeated=False))
    try:
        before=snapshot_measurements(root)
        prepare.new_json(ledger/'measurement_hashes_before.json',before)
        prepared=prepare.audit(root)
        after=snapshot_measurements(root)
        stable=check_stability(before,after)
        prepare.new_json(ledger/'measurement_stability.json',stable)
        verify_sources(root);prepare.no_live(root)
        attempt=Path(prepared['attempt'])
        if not attempt.resolve().is_relative_to(out.resolve()):raise ValueError('publication attempt escaped')
        ready=dict(at=prepare.now(),status='VERIFIED_PREPARATION_READY_FOR_RENDER',
            publication_attempt=str(attempt.relative_to(root)),
            prepared_receipt_sha256=prepare.integrity.sha(attempt/'PREPARED.json'),
            ledger_path=str((ledger/'measurement_hashes_before.json').relative_to(root)),
            ledger_sha256=prepare.integrity.sha(ledger/'measurement_hashes_before.json'),
            stability_path=str((ledger/'measurement_stability.json').relative_to(root)),
            stability_sha256=prepare.integrity.sha(ledger/'measurement_stability.json'),
            measurements=9984,scientific_files=49920,measurement_repeated=False,
            full_priority35b_complete=False,remaining=['figure rendering/visual QA','report/full manifest/post-exit seal'])
        prepare.new_json(ledger/'READY_FOR_RENDER.json',ready)
        return dict(ready_receipt=str((ledger/'READY_FOR_RENDER.json').relative_to(root)),
                    status=ready['status'],full_priority35b_complete=False)
    except Exception:
        prepare.new_json(ledger/'FAILURE.json',dict(at=prepare.now(),traceback=traceback.format_exc(),
            measurement_repeated=False,requires_human_direction=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',required=True,action='store_true')
    parser.parse_args();print(json.dumps(run(),allow_nan=False))
