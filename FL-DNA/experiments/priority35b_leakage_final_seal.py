"""Fresh post-publication full closure seal; no measurement replay.

The calling agent must independently check this process has exited and recheck
the receipt/manifest before notifying completion or pausing the heartbeat.
"""
import argparse
import json
import os
import py_compile
import subprocess
import sys
import time
import traceback
from pathlib import Path
from experiments import priority35b_leakage_publish_report as publication

render=publication.render;prepare=publication.prepare;integrity=publication.integrity
manifest=render.wrapper.manifest;contract=publication.contract
SEAL='final_seal_integration_freeze.json'
MANIFEST='artifacts/priority35b/b3/final_manifest.json'
COMPLETE='artifacts/priority35b/b3/COMPLETE.json'


def require_published(doc):
    if (doc.get('status')!='REPORT_APPENDED_PENDING_POSTEXIT_FULL_SEAL' or
        doc.get('measurement_repeated') is not False or doc.get('full_priority35b_complete') is not False):
        raise ValueError('append-only publication receipt required, not partial or COMPLETE')
    contract.guard.require_publication_checks(doc['checks'])


def verify_sources(root):
    publication.validation.verify_sources(root)
    names=(publication.SEAL,SEAL)
    prepare.scientific.verify_additive_sources(root,names)
    for name in names:
        for path,digest in integrity.read(Path(root)/'artifacts/priority35b/b3'/name)['inputs'].items():
            if integrity.sha(integrity.child(root,path))!=digest:raise ValueError('final seal dependency drift')


def verify_publication(root,relative):
    prepare.no_live(root)  # Before any receipt/evidence interpretation.
    verify_sources(root)
    path,doc=publication.read_receipt(root,relative);require_published(doc)
    render.require_actor_exit(prepare.postexit.process_snapshot(),doc['pid'])
    ready_path,ready=publication.read_receipt(root,doc['ready_receipt']);render.require_ready(ready)
    start=integrity.read(path.parent/'START.json')
    manifest.verify_mapping(start['receipt_hashes'],lambda p:integrity.sha(integrity.child(root,p)))
    # All earlier actors must have exited, not merely closed their sockets.
    bound_receipts=[]
    for receipt_path in start['receipt_hashes']:
        p=integrity.child(root,receipt_path);receipt=integrity.read(p)
        bound_receipts.append(receipt)
        if 'pid' in receipt:render.require_actor_exit(prepare.postexit.process_snapshot(),receipt['pid'])
        if receipt.get('status')=='RENDERED_AWAITING_VISUAL_QA':
            manifest.verify_mapping(receipt['outputs'],lambda p:integrity.sha(integrity.child(root,p)))
        if receipt.get('status')=='VALIDATION_PASS':
            manifest.verify_mapping(receipt['log_hashes'],lambda p:integrity.sha(integrity.child(root,p)))
    rendered=[r for r in bound_receipts if r.get('status')=='RENDERED_AWAITING_VISUAL_QA']
    visual=[r for r in bound_receipts if r.get('status')=='VISUAL_QA_PASS']
    if len(rendered)!=1 or len(visual)!=1:raise ValueError('exact measured render/QA receipts required')
    contract.visual_qa(rendered[0],visual[0])
    render.require_actor_exit(prepare.postexit.process_snapshot(),integrity.read(ready_path.parent/'START.json')['pid'])
    attempt=render.evidence_path(root,ready['publication_attempt'])
    if integrity.sha(attempt/'PREPARED.json')!=ready['prepared_receipt_sha256']:raise ValueError('preparation receipt drift')
    prepared=integrity.read(attempt/'PREPARED.json')
    snapshot=(attempt/'report_prefix_snapshot.md').read_bytes()
    manifest.verify_inputs(root,prepared['inputs'],snapshot)
    manifest.verify_mapping(prepared['outputs'],lambda p:integrity.sha(integrity.child(root,p)))
    hashes_path=render.evidence_path(root,ready['ledger_path'])
    if integrity.sha(hashes_path)!=ready['ledger_sha256']:raise ValueError('scientific hash ledger drift')
    hashes=integrity.read(hashes_path)
    if len(hashes)!=49920:raise ValueError('all49920 scientific files required')
    stability_path=render.evidence_path(root,ready['stability_path'])
    if integrity.sha(stability_path)!=ready['stability_sha256'] or integrity.read(stability_path)!=render.wrapper.check_stability(hashes,hashes):
        raise ValueError('stable-evidence receipt drift')
    manifest.verify_mapping(hashes,lambda p:integrity.sha(integrity.child(root,p)))
    report=integrity.child(root,manifest.REPORT)
    if integrity.sha(report)!=doc['report_sha256'] or integrity.sha(path.parent/'proposed_report.md')!=doc['report_bundle_sha256']:
        raise ValueError('published report binding')
    if report.read_bytes()!=(path.parent/'proposed_report.md').read_bytes():raise ValueError('report differs from approved proposal')
    render.recheck_data(root,attempt)
    prepare.verified_utility(root)
    return path,doc,prepared,snapshot,hashes


def seal(relative,root=prepare.ROOT):
    root=Path(root).resolve();out=root/'artifacts/priority35b/b3'
    if integrity.child(root,MANIFEST).exists() or integrity.child(root,COMPLETE).exists():
        raise ValueError('preserve existing final closure; no automatic replay')
    if list(out.glob('**/FAILURE.json')) or list(out.glob('**/failure.json')):
        raise ValueError('preserved operational failure requires direction')
    path,published,prepared,snapshot,measurement_hashes=verify_publication(root,relative)
    directory=out/('seal_attempt_'+str(time.time_ns()));directory.mkdir()
    prepare.new_json(directory/'START.json',dict(at=prepare.now(),pid=os.getpid(),command=[sys.executable,*sys.argv],
        publication_receipt=relative,publication_sha256=integrity.sha(path),measurement_repeated=False))
    failure=str((directory/'FAILURE.json').relative_to(root))
    try:
        # Additive final modules were implemented after the fixed twenty-suite
        # runner. Validate them separately without editing its frozen source.
        extra=('publish_report','final_seal')
        for name in extra:
            command=[str(root/'.venv-phase1/bin/python'),'-B','-m','unittest','tests.test_priority35b_leakage_'+name]
            result=subprocess.run(command,cwd=root,capture_output=True,text=True,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            prepare.new_json(directory/(name+'_tests.json'),dict(command=command,returncode=result.returncode,
                stdout=result.stdout,stderr=result.stderr))
            publication.validation.successful_suite(result.returncode,result.stdout+'\n'+result.stderr)
        compiled=directory/'compiled';compiled.mkdir()
        for index,name in enumerate(extra):
            for kind in ('experiments','tests'):
                filename=('priority35b_leakage_' if kind=='experiments' else 'test_priority35b_leakage_')+name+'.py'
                py_compile.compile(str(root/kind/filename),cfile=str(compiled/(str(index)+'_'+kind+'.pyc')),doraise=True)
        result=subprocess.run(['git','diff','--check'],cwd=root,capture_output=True,text=True)
        prepare.new_json(directory/'diff_check.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:raise ValueError('final diff check failed')
        verify_publication(root,relative);prepare.no_live(root)
        inputs=dict(prepared['inputs'])
        for name in render.wrapper.SEALS+(render.SEAL,'publication_contract_freeze.json',publication.validation.SEAL,publication.SEAL,SEAL):
            p=out/name;doc=integrity.read(p);inputs[str(p.relative_to(root))]=integrity.sha(p)
            inputs.update(doc['sources']);inputs.update(doc['inputs'])
        directories=['artifacts/priority35b/b3','artifacts/priority35b/figure']
        excluded=[MANIFEST,COMPLETE,failure]
        outputs=manifest.inventory_files(root,directories,excluded)
        outputs[manifest.REPORT]=integrity.sha(integrity.child(root,manifest.REPORT))
        doc=dict(schema='P35B_B3_FULL_CLOSURE_V1',at=prepare.now(),inputs=inputs,outputs=outputs,
            output_directories=directories,excluded_after_exit_only=excluded,
            exclusion_reason='Manifest self-reference, final completion receipt, and preserved failure log; no scientific file or report excluded.',
            scientific_jobs=9984,scientific_files=49920,fixed_holm=144,measurement_repeated=False)
        prepare.new_json(integrity.child(root,MANIFEST),doc)
        audit=manifest.verify_closure(root,doc,set(measurement_hashes),snapshot)
        verify_sources(root);prepare.no_live(root)
        receipt=dict(at=prepare.now(),status='PRIORITY35B_VERIFIED_COMPLETE',pid=os.getpid(),
            report_sha256=outputs[manifest.REPORT],manifest_sha256=integrity.sha(integrity.child(root,MANIFEST)),
            publication_receipt=relative,publication_sha256=integrity.sha(path),
            no_live_workers=True,postexit_seal=True,measurement_repeated=False,
            scientific_jobs=9984,scientific_files=49920,fixed_holm=144,closure=audit,
            next_action='Calling agent verifies this sealing process exited, rechecks closure/receipt, then updates administrative progress and pauses heartbeat; no Goal changes.')
        contract.final_receipt_checks(receipt,receipt['report_sha256'],receipt['manifest_sha256'])
        prepare.new_json(integrity.child(root,COMPLETE),receipt)
        return dict(status=receipt['status'],receipt=COMPLETE,requires_agent_postexit_recheck=True)
    except Exception:
        prepare.new_json(integrity.child(root,failure),dict(at=prepare.now(),traceback=traceback.format_exc(),
            measurement_repeated=False,requires_human_direction=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--published',required=True)
    a=parser.parse_args();print(json.dumps(seal(a.published),allow_nan=False))
