"""Authorized append-only report integration; never marks COMPLETE.

Requires post-exit scientific/table, measured visual-QA and validation receipts.
Final manifest closure and a fresh post-publication exit seal are separate.
"""
import argparse
import importlib.metadata
import json
import os
import sys
import time
import traceback
from pathlib import Path
from experiments import priority35b_leakage_final_validation as validation
from experiments import priority35b_leakage_publication_contract as contract
from experiments import priority35b_leakage_report_text as text

render=validation.render;prepare=render.prepare;integrity=render.integrity
SEAL='report_publication_integration_freeze.json'


def require_interpretation(doc,inputs):
    if doc.get('status')!='FULL_RESULTS_DESCRIPTIVE_READING' or doc.get('inputs')!=inputs:
        raise ValueError('interpretation must bind to the fully verified frontier and Holm CSVs')
    value=doc.get('text')
    if not isinstance(value,str) or not value.strip() or contract.guard.SECTION in value:
        raise ValueError('separate result-specific plain reading required')
    return value


def read_receipt(root,relative):
    path=render.evidence_path(root,relative)
    if not path.is_file():raise ValueError('missing publication receipt')
    return path,integrity.read(path)


def publish(ready_relative,render_relative,qa_relative,validation_relative,interpretation_relative,root=prepare.ROOT):
    root=Path(root).resolve();attempt,ready_path=render.verify_ready(root,ready_relative)
    validation.verify_sources(root)
    prepare.scientific.verify_additive_sources(root,(SEAL,))
    for path,digest in integrity.read(root/'artifacts/priority35b/b3'/SEAL)['inputs'].items():
        if integrity.sha(integrity.child(root,path))!=digest:raise ValueError('publication dependency drift')
    render_path,r=read_receipt(root,render_relative);qa_path,qa=read_receipt(root,qa_relative)
    val_path,v=read_receipt(root,validation_relative);reading_path,reading=read_receipt(root,interpretation_relative)
    for receipt in (r,v):
        render.require_actor_exit(prepare.postexit.process_snapshot(),receipt['pid'])
        if receipt.get('ready_receipt')!=ready_relative or receipt.get('ready_sha256')!=integrity.sha(ready_path):
            raise ValueError('receipt belongs to another preparation')
    contract.visual_qa(r,qa)
    render.wrapper.manifest.verify_mapping(r['outputs'],lambda p:integrity.sha(integrity.child(root,p)))
    render.wrapper.manifest.verify_mapping(v['log_hashes'],lambda p:integrity.sha(integrity.child(root,p)))
    count=0
    for i,command in enumerate(validation.test_commands(root/'.venv-phase1/bin/python')):
        log=integrity.read(val_path.parent/('suite_%02d.json'%i))
        if log['command']!=command:raise ValueError('validation suite command drift')
        count+=validation.successful_suite(log['returncode'],log['stdout']+'\n'+log['stderr'])
    diff=integrity.read(val_path.parent/'diff_check.json')
    if diff['command']!=['git','diff','--check'] or diff['returncode']!=0 or count!=v['tests_passed']:
        raise ValueError('validation receipt/log discrepancy')
    if len(list((val_path.parent/'compiled').glob('*.pyc')))!=len(v['compiled_sources']):
        raise ValueError('compiled-source inventory')
    render.recheck_data(root,attempt)
    scientific=integrity.read(attempt/'scientific_audit.json')
    checks=contract.publication_checks(scientific,integrity.read(attempt/'csv_audit.json'),
        integrity.read(attempt/'utility_audit.json'),integrity.read(attempt/'figure_data_audit.json'),r,qa,v,True,True)
    inputs={str((attempt/(name+'.csv')).relative_to(root)):integrity.sha(attempt/(name+'.csv')) for name in ('frontier','holm')}
    reading_text=require_interpretation(reading,inputs)
    bundle=integrity.read(attempt/'utility_and_matching.json')
    provenance=[integrity.read(ready_path.parent/'START.json')['command'],
        integrity.read(render_path.parent/'START.json')['command'],*v['commands'],[sys.executable,*sys.argv]]
    commands=[' '.join(c) if isinstance(c,list) else str(c) for c in provenance]
    amendments=[dict(path=str(p.relative_to(root)),sha256=integrity.sha(p))
        for p in sorted((root/'protocols/amendments').glob('*priority35b*.md'))]
    report_bundle=dict(commands=commands,amendment_hashes=amendments,
        software={name:importlib.metadata.version(name) for name in ('torch','numpy','scipy','matplotlib')},
        utility=bundle['utility'],decomposition=bundle['decomposition'],matching=bundle['matching'],
        frontier=render.tables.parse_csv((attempt/'frontier.csv').read_text()),
        holm=render.tables.parse_csv((attempt/'holm.csv').read_text()),
        selection=integrity.read(attempt/'estimator_selection.json')['panels'],
        figure_png=str(root/'artifacts/priority35b/figure/frontier.png'),
        figure_pdf=str(root/'artifacts/priority35b/figure/frontier.pdf'),
        figure_csv=str(root/'artifacts/priority35b/figure/frontier_data.csv'),
        verification=dict(scientific_jobs=9984,controls=312,csv_recomputation='PASS',checks=checks))
    section=text.section(report_bundle,checks)+'\n### Plain reading of the verified frontier\n\n'+reading_text+'\n'
    report=root/'reports/priority35b_report.md';original=report.read_bytes()
    proposed=contract.guard.prepare_append(original,section,checks)
    directory=attempt/('report_publication_attempt_'+str(time.time_ns()));directory.mkdir()
    receipt_paths=[ready_path,render_path,qa_path,val_path,reading_path]
    prepare.new_json(directory/'START.json',dict(at=prepare.now(),pid=os.getpid(),command=[sys.executable,*sys.argv],
        receipt_hashes={str(p.relative_to(root)):integrity.sha(p) for p in receipt_paths},measurement_repeated=False))
    try:
        prepare.new_bytes(directory/'proposed_report.md',proposed)
        render.verify_ready(root,ready_relative);validation.verify_sources(root);prepare.no_live(root)
        if report.read_bytes()!=original:raise ValueError('report changed before authorized append')
        with report.open('ab') as stream:stream.write(proposed[len(original):])
        prefix=contract.guard.verify_append(original,report.read_bytes())
        if report.read_bytes()!=proposed:raise ValueError('published report differs from checked proposal')
        prepare.no_live(root)
        receipt=dict(at=prepare.now(),status='REPORT_APPENDED_PENDING_POSTEXIT_FULL_SEAL',pid=os.getpid(),
            ready_receipt=ready_relative,report_sha256=integrity.sha(report),prefix=prefix,
            report_bundle_sha256=integrity.sha(directory/'proposed_report.md'),checks=checks,
            measurement_repeated=False,full_priority35b_complete=False)
        prepare.new_json(directory/'PUBLISHED.json',receipt)
        return dict(receipt=str((directory/'PUBLISHED.json').relative_to(root)),status=receipt['status'],full_priority35b_complete=False)
    except Exception:
        prepare.new_json(directory/'FAILURE.json',dict(at=prepare.now(),traceback=traceback.format_exc(),
            measurement_repeated=False,requires_human_direction=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('ready','render','qa','validation','interpretation'):parser.add_argument('--'+name,required=True)
    a=parser.parse_args();print(json.dumps(publish(a.ready,a.render,a.qa,a.validation,a.interpretation),allow_nan=False))
