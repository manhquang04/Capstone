"""One-shot POST-EXIT audit/table preparation; never replays measurements.

This stage stops at FIGURE_AND_PUBLICATION_PENDING, not COMPLETE. A separate
publication seal requires rendered figure visual QA and final manifest checks.
"""
import argparse
import csv
import io
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import priority35b_leakage_integrity as integrity
from experiments import priority35b_leakage_postexit as postexit
from experiments import priority35b_leakage_scientific_verify as scientific

SEALS=scientific.SEALS+('scientific_coordinator_freeze.json','table_serialization_freeze.json',
    'report_guard_freeze.json','report_rendering_freeze.json','figure_rendering_freeze.json',
    'data_bridge_freeze.json','publication_preparation_freeze.json')


def now():return datetime.now(timezone.utc).isoformat()


def require_finished(progress, locks, failures):
    if progress.get('stage')!='B3_MEASUREMENTS_FINISHED' or progress.get('done')!=9984 or progress.get('failed')!=0:
        raise ValueError('complete measurements required before publication preparation')
    if locks or failures:raise ValueError('lock/failure evidence requires direction')


def no_live(root):
    out=Path(root)/'artifacts/priority35b/b3'
    exited=postexit.require_exit(postexit.process_snapshot(),postexit.recorded_pids(out))
    postexit.port_free(43576)
    if list(out.glob('*.lock')):raise ValueError('scientific lock remains')
    return exited


def verify_seal_inputs(root):
    out=Path(root)/'artifacts/priority35b/b3'
    for name in SEALS:
        for path,digest in integrity.read(out/name).get('inputs',{}).items():
            if integrity.sha(integrity.child(root,path))!=digest:
                raise ValueError('additive seal dependency drift: '+path)


def new_bytes(path, data):
    """Exclusive-create only; callers constrain paths to their new attempt."""
    if not isinstance(data,bytes):raise TypeError('bytes required')
    with Path(path).open('xb') as stream:stream.write(data)


def new_json(path, doc):
    new_bytes(path,(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())


def numeric_csv(text, fields, integers):
    reader=csv.DictReader(io.StringIO(text))
    if reader.fieldnames!=list(fields):raise ValueError('historical CSV schema drift')
    rows=[]
    for row in reader:
        if set(row)!=set(fields) or any(v is None for v in row.values()):raise ValueError('CSV shape')
        for name in fields:
            if name in ('dataset','method','endpoint','from_arm','to_arm'):continue
            row[name]=int(row[name]) if name in integers else float(row[name])
            if not math.isfinite(row[name]):raise ValueError('non-finite historical CSV')
        rows.append(row)
    return rows


def verified_utility(root):
    out=Path(root)/'artifacts/priority35b/b2b/arms'
    receipt=integrity.read(out/'B2_B2B_COMPLETE.json')
    if receipt['status']!='B2_B2B_UTILITY_VERIFIED_COMPLETE' or integrity.sha(out/'final_manifest.json')!=receipt['manifest_sha256']:
        raise ValueError('utility completion/manifest digest')
    manifest=integrity.read(out/'final_manifest.json')
    for path,digest in manifest.items():
        if integrity.sha(integrity.child(root,path))!=digest:raise ValueError('historical utility drift: '+path)
    utility=numeric_csv((out/'utility_table.csv').read_text(),
        ('dataset','method','endpoint','mean','baseline_mean','n','df','mean_delta','ci_low','ci_high'),{'n','df'})
    decomposition=numeric_csv((out/'decomposition.csv').read_text(),
        ('dataset','from_arm','to_arm','endpoint','n','df','mean_delta','ci_low','ci_high'),{'n','df'})
    if len(utility)!=288 or len(decomposition)!=81:raise ValueError('utility/decomposition row counts')
    return utility,decomposition,integrity.read(out/'matching.json'),dict(entries=len(manifest),manifest_sha256=receipt['manifest_sha256'],status='PASS')


def analysis_progress(root, attempt):
    """Called only after scientific verifier exits successfully and no-live recheck."""
    base=Path(root)/'artifacts/priority35b'
    doc=dict(at=now(),stage='B3_ANALYSIS_VERIFIED_FIGURE_PENDING',done=9984,total=9984,failed=0,
             B2_complete=True,B2b_complete=True,B3_started=True,full_priority35b_complete=False,
             publication_attempt=str(attempt.relative_to(root)))
    # Root progress/checklist are administrative live tracking, never scientific inputs.
    for path,data in ((base/'progress.json',doc),):
        temporary=path.with_suffix('.publication.tmp')
        new_json(temporary,data);temporary.replace(path)
    with (base/'progress.log').open('a') as stream:stream.write(json.dumps(doc)+'\n')
    checklist=integrity.read(base/'checklist.json');checklist['B3']=doc
    checklist['items'].update(B3_measurements='DONE',B3_independent_verification='SCIENTIFIC_AND_CSV_PASS',
                              B3_figure_report_manifest='PENDING')
    tmp=base/'checklist.publication.tmp';new_json(tmp,checklist);tmp.replace(base/'checklist.json')


def audit(root=ROOT):
    root=Path(root).resolve();out=root/'artifacts/priority35b/b3'
    exited=no_live(root)  # Before reading ANY protected quality.
    require_finished(integrity.read(root/'artifacts/priority35b/progress.json'),list(out.glob('*.lock')),
        [p for p in (out/'REQUIRES_DIRECTION.json',out/'SUPERVISOR_FAILURE.json') if p.exists()])
    scientific.verify_additive_sources(root,SEALS)
    verify_seal_inputs(root)
    attempt=out/('publication_attempt_'+str(time.time_ns()));attempt.mkdir()
    new_json(attempt/'START.json',dict(at=now(),command=' '.join(sys.argv),exited_pids=exited,
        stage='POSTEXIT_AUDIT_STARTED',measurement_repeated=False,full_priority35b_complete=False))
    try:
        verified=scientific.verify(root)
        utility,decomposition,matching,utility_receipt=verified_utility(root)
        from experiments import priority35b_leakage as driver
        from experiments import priority35b_leakage_tables as tables
        from experiments import priority35b_leakage_data_bridge as bridge
        from experiments import priority35b_leakage_report_guard as guard
        arms=integrity.read(out/'arms.json');selection=integrity.read(out/'estimator_selection.json')
        results=[integrity.read(driver.folder(c)/'result.json') for c in integrity.read(out/'execution_freeze.json')['jobs']]
        records=bridge.result_rows(results,arms);prepared=tables.prepare(records,arms,matching)
        for name in ('per_target','frontier','holm'):
            new_bytes(attempt/(name+'.csv'),tables.csv_string(prepared[name]).encode())
        csv_receipt=tables.verify_csvs(*[(attempt/(name+'.csv')).read_text() for name in ('per_target','frontier','holm')],arms,matching)
        figure_rows,controls=bridge.prepare_figure(records,arms,prepared['frontier'],utility,selection)
        new_bytes(attempt/'figure_data.csv',tables.csv_string(figure_rows).encode())
        figure_receipt=bridge.independently_verify_figure(tables.parse_csv((attempt/'figure_data.csv').read_text()),controls,records,utility)
        new_json(attempt/'data_free_lines.json',controls)
        new_json(attempt/'utility_and_matching.json',dict(utility=utility,decomposition=decomposition,matching=matching))
        new_json(attempt/'estimator_selection.json',selection)
        original=(root/'reports/priority35b_report.md').read_bytes();guard.assert_prefix(original)
        if len(original)!=guard.PREFIX_BYTES:raise ValueError('historical report already appended')
        new_bytes(attempt/'report_prefix_snapshot.md',original)
        new_json(attempt/'scientific_audit.json',verified)
        new_json(attempt/'csv_audit.json',csv_receipt)
        new_json(attempt/'figure_data_audit.json',figure_receipt)
        new_json(attempt/'utility_audit.json',utility_receipt)
        scientific.verify_additive_sources(root,SEALS);verify_seal_inputs(root);driver.verify();no_live(root)
        inputs={**integrity.read(out/'execution_freeze.json')['inputs'],**integrity.read(out/'execution_freeze.json')['sources']}
        for name in SEALS:
            inputs[str((out/name).relative_to(root))]=integrity.sha(out/name)
            inputs.update(integrity.read(out/name)['sources'])
        outputs={str(p.relative_to(root)):integrity.sha(p) for p in attempt.iterdir() if p.is_file()}
        receipt=dict(at=now(),status='B3_TABLES_VERIFIED_FIGURE_AND_PUBLICATION_PENDING',
            inputs=inputs,outputs=outputs,scientific_jobs=9984,controls=312,fixed_holm=144,
            csv_recomputation='PASS',figure_data='PASS',measurement_repeated=False,full_priority35b_complete=False)
        new_json(attempt/'PREPARED.json',receipt)
        analysis_progress(root,attempt)
        return dict(attempt=str(attempt),status=receipt['status'],full_priority35b_complete=False)
    except Exception:
        new_json(attempt/'FAILURE.json',dict(at=now(),traceback=traceback.format_exc(),
            measurement_repeated=False,requires_human_direction=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true',required=True)
    parser.parse_args();print(json.dumps(audit(),allow_nan=False))
