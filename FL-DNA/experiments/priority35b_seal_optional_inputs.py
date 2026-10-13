"""Approved seal-only normalization for thirteen exact historical documents."""
import json
import os
import py_compile
import sys
import time
import traceback
from pathlib import Path
from experiments import priority35b_physical_failure_seal as chain

FREEZE='artifacts/priority35b/b3/seal_optional_inputs_freeze.json'
PRIOR='artifacts/priority35b/b3/approved_seal_attempt_1791787594949980000'
base=chain.seal.original


def frozen(root):
    doc=base.integrity.read(root/FREEZE)
    for section in ('sources','inputs','eligible_missing_inputs'):
        base.manifest.verify_mapping(doc[section],lambda f:base.integrity.sha(root/f))
    if len(doc['eligible_missing_inputs'])!=13:raise ValueError('exact thirteen freezes required')
    return doc


def freeze_mappings(root,relative,eligible):
    path=root/relative;doc=base.integrity.read(path)
    if relative in eligible:
        if base.integrity.sha(path)!=eligible[relative] or 'inputs' in doc:
            raise ValueError('eligible source-only freeze changed')
        inputs={}
    else:inputs=doc['inputs']  # No default for any other document.
    sources=doc['sources']  # No default or relaxation of other fields.
    if not isinstance(inputs,dict) or not isinstance(sources,dict) or not sources:
        raise ValueError('strict source/input mapping schema')
    base.manifest.verify_mapping(sources,lambda f:base.integrity.sha(root/f))
    if inputs:base.manifest.verify_mapping(inputs,lambda f:base.integrity.sha(root/f))
    return sources,inputs


def seal(relative,root=base.prepare.ROOT):
    root=Path(root).resolve();out=root/'artifacts/priority35b/b3';p=base.prepare;i=base.integrity;m=base.manifest
    local=frozen(root);adapters=chain.report.labels.presentation.audit.verify_freeze()
    approved=dict(adapters['approved_infrastructure_failures']);approved.update(local['approved_seal_failure'])
    actual={str(f.relative_to(root)):i.sha(f) for pat in ('**/FAILURE.json','**/failure.json') for f in out.glob(pat)}
    chain.check(actual,approved)
    if (root/base.MANIFEST).exists() or (root/base.COMPLETE).exists():raise ValueError('preserve existing closure')
    path,published,prepared,snapshot,measurements=base.verify_publication(root,relative)
    prior=i.read(root/PRIOR/'START.json');base.render.require_actor_exit(p.postexit.process_snapshot(),prior['pid'])
    count=0
    for name in ('leakage_publish_report','leakage_final_seal','float64_verifier_adapter','scipy_verifier_adapter','approved_infrastructure_seal'):
        log=i.read(root/PRIOR/(name+'_tests.json'))
        expected=[str(root/'.venv-phase1/bin/python'),'-B','-m','unittest','tests.test_priority35b_'+name]
        if log['command']!=expected:raise ValueError('stored test command mismatch')
        count+=base.publication.validation.successful_suite(log['returncode'],log['stdout']+'\n'+log['stderr'])
    if i.read(root/PRIOR/'diff_check.json')['returncode']!=0:raise ValueError('stored diff check failed')
    names=base.render.wrapper.SEALS+(base.render.SEAL,'publication_contract_freeze.json',base.publication.validation.SEAL,base.publication.SEAL,base.SEAL)
    inputs=dict(prepared['inputs']);normalized=[]
    for name in names:
        relative_freeze='artifacts/priority35b/b3/'+name
        sources,declared=freeze_mappings(root,relative_freeze,local['eligible_missing_inputs'])
        inputs[relative_freeze]=i.sha(root/relative_freeze);inputs.update(sources);inputs.update(declared)
        if relative_freeze in local['eligible_missing_inputs']:normalized.append(relative_freeze)
    if set(normalized)!=set(local['eligible_missing_inputs']):raise ValueError('normalization scope mismatch')
    inputs[chain.report.labels.presentation.audit.FREEZE]=i.sha(root/chain.report.labels.presentation.audit.FREEZE)
    inputs.update(adapters['files'])
    mse=chain.report.labels.presentation.audit.mse.FREEZE
    inputs[mse]=i.sha(root/mse);inputs.update(i.read(root/mse)['files'])
    inputs[FREEZE]=i.sha(root/FREEZE);inputs.update(local['sources']);inputs.update(local['inputs'])
    directory=out/('seal_only_attempt_'+str(time.time_ns()));directory.mkdir()
    failure=str((directory/'FAILURE.json').relative_to(root))
    p.new_json(directory/'START.json',dict(at=p.now(),pid=os.getpid(),command=[sys.executable,*sys.argv],
        publication_receipt=relative,approved_failures=approved,tests_rerun=False,measurement_repeated=False,figure_rerun=False))
    try:
        py_compile.compile(str(root/'experiments/priority35b_seal_optional_inputs.py'),cfile=str(directory/'seal_adapter.pyc'),doraise=True)
        p.new_json(directory/'REUSED_VALIDATION.json',dict(status='HASH_VERIFIED_PRIOR_VALIDATION',extra_tests_passed=count,
            tests_rerun=False,compile_reused=True,prior=PRIOR,normalized_source_only_freezes=normalized))
        frozen(root);base.verify_sources(root);p.no_live(root)
        directories=['artifacts/priority35b/b3','artifacts/priority35b/figure'];excluded=[base.MANIFEST,base.COMPLETE,failure]
        outputs=m.inventory_files(root,directories,excluded);outputs[m.REPORT]=i.sha(root/m.REPORT)
        closure=dict(schema='P35B_B3_FULL_CLOSURE_V1',at=p.now(),inputs=inputs,outputs=outputs,
            output_directories=directories,excluded_after_exit_only=excluded,
            exclusion_reason='Only manifest self-reference, completion receipt and current unused failure path; all original failures included.',
            scientific_jobs=9984,scientific_files=49920,fixed_holm=144,measurement_repeated=False)
        p.new_json(root/base.MANIFEST,closure)
        audit=m.verify_closure(root,closure,set(measurements),snapshot)
        frozen(root);base.verify_sources(root);p.no_live(root)
        receipt=dict(at=p.now(),status='PRIORITY35B_VERIFIED_COMPLETE',pid=os.getpid(),
            report_sha256=outputs[m.REPORT],manifest_sha256=i.sha(root/base.MANIFEST),publication_receipt=relative,
            publication_sha256=i.sha(path),no_live_workers=True,postexit_seal=True,measurement_repeated=False,
            tests_rerun=False,figure_rerun=False,scientific_jobs=9984,scientific_files=49920,fixed_holm=144,
            seal_optional_inputs_freeze_sha256=i.sha(root/FREEZE),closure=audit)
        base.contract.final_receipt_checks(receipt,receipt['report_sha256'],receipt['manifest_sha256'])
        p.new_json(root/base.COMPLETE,receipt)
        return dict(status=receipt['status'],receipt=base.COMPLETE,requires_agent_postexit_recheck=True)
    except Exception:
        p.new_json(root/failure,dict(at=p.now(),traceback=traceback.format_exc(),measurement_repeated=False,requires_human_direction=True))
        raise


if __name__=='__main__':
    if '--stage' not in sys.argv or sys.argv[sys.argv.index('--stage')+1]!='seal':raise ValueError('seal step only')
    chain.seal.seal=seal;chain.main()
