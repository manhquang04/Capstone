"""Additive final seal preserving explicitly approved failed audit receipts."""
import json
import os
import py_compile
import subprocess
import sys
import time
import traceback
from pathlib import Path
from experiments import priority35b_leakage_final_seal as original
from experiments import priority35b_scipy_verifier_adapter as adapter


def verify_failure_allowlist(actual,approved):
    if actual!=approved:raise ValueError('new, missing or changed failure record; stop for direction')


def seal(relative,root=original.prepare.ROOT):
    root=Path(root).resolve();out=root/'artifacts/priority35b/b3'
    p=original.prepare;i=original.integrity;m=original.manifest
    doc=adapter.verify_freeze()
    actual={str(f.relative_to(root)):i.sha(f) for pattern in ('**/FAILURE.json','**/failure.json') for f in out.glob(pattern)}
    verify_failure_allowlist(actual,doc['approved_infrastructure_failures'])
    if i.child(root,original.MANIFEST).exists() or i.child(root,original.COMPLETE).exists():
        raise ValueError('preserve existing final closure; no automatic replay')
    path,published,prepared,snapshot,measurement_hashes=original.verify_publication(root,relative)
    directory=out/('approved_seal_attempt_'+str(time.time_ns()));directory.mkdir()
    p.new_json(directory/'START.json',dict(at=p.now(),pid=os.getpid(),command=[sys.executable,*sys.argv],
        publication_receipt=relative,publication_sha256=i.sha(path),approved_failures=actual,measurement_repeated=False))
    failure=str((directory/'FAILURE.json').relative_to(root))
    try:
        tests=('leakage_publish_report','leakage_final_seal','float64_verifier_adapter','scipy_verifier_adapter','approved_infrastructure_seal')
        for name in tests:
            command=[str(root/'.venv-phase1/bin/python'),'-B','-m','unittest','tests.test_priority35b_'+name]
            result=subprocess.run(command,cwd=root,capture_output=True,text=True,env=dict(os.environ,
                PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            p.new_json(directory/(name+'_tests.json'),dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
            original.publication.validation.successful_suite(result.returncode,result.stdout+'\n'+result.stderr)
        compiled=directory/'compiled';compiled.mkdir()
        sources={f for f in doc['files'] if f.endswith('.py')}
        sources.update('experiments/priority35b_'+n+'.py' for n in tests[:2])
        sources.update('tests/test_priority35b_'+n+'.py' for n in tests[:2])
        for index,f in enumerate(sorted(sources)):
            py_compile.compile(str(root/f),cfile=str(compiled/(str(index)+'.pyc')),doraise=True)
        result=subprocess.run(['git','diff','--check'],cwd=root,capture_output=True,text=True)
        p.new_json(directory/'diff_check.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:raise ValueError('final diff check failed')
        original.verify_publication(root,relative);p.no_live(root);adapter.verify_freeze()
        inputs=dict(prepared['inputs'])
        for name in original.render.wrapper.SEALS+(original.render.SEAL,'publication_contract_freeze.json',original.publication.validation.SEAL,original.publication.SEAL,original.SEAL):
            f=out/name;d=i.read(f);inputs[str(f.relative_to(root))]=i.sha(f);inputs.update(d['sources']);inputs.update(d['inputs'])
        inputs[adapter.FREEZE]=i.sha(root/adapter.FREEZE);inputs.update(doc['files'])
        mse_freeze=adapter.mse.FREEZE;inputs[mse_freeze]=i.sha(root/mse_freeze)
        inputs.update(i.read(root/mse_freeze)['files'])
        directories=['artifacts/priority35b/b3','artifacts/priority35b/figure']
        excluded=[original.MANIFEST,original.COMPLETE,failure]
        outputs=m.inventory_files(root,directories,excluded);outputs[m.REPORT]=i.sha(i.child(root,m.REPORT))
        closure=dict(schema='P35B_B3_FULL_CLOSURE_V1',at=p.now(),inputs=inputs,outputs=outputs,
            output_directories=directories,excluded_after_exit_only=excluded,
            exclusion_reason='Only manifest self-reference, completion receipt and current unused failure path; all preserved failures and scientific evidence included.',
            scientific_jobs=9984,scientific_files=49920,fixed_holm=144,measurement_repeated=False)
        p.new_json(i.child(root,original.MANIFEST),closure)
        audit=m.verify_closure(root,closure,set(measurement_hashes),snapshot)
        original.verify_sources(root);adapter.verify_freeze();p.no_live(root)
        receipt=dict(at=p.now(),status='PRIORITY35B_VERIFIED_COMPLETE',pid=os.getpid(),
            report_sha256=outputs[m.REPORT],manifest_sha256=i.sha(i.child(root,original.MANIFEST)),
            publication_receipt=relative,publication_sha256=i.sha(path),approved_infrastructure_freeze_sha256=i.sha(root/adapter.FREEZE),
            no_live_workers=True,postexit_seal=True,measurement_repeated=False,
            scientific_jobs=9984,scientific_files=49920,fixed_holm=144,closure=audit,
            next_action='Agent verifies sealer exit/closure then updates administrative progress; no Goal changes.')
        original.contract.final_receipt_checks(receipt,receipt['report_sha256'],receipt['manifest_sha256'])
        p.new_json(i.child(root,original.COMPLETE),receipt)
        return dict(status=receipt['status'],receipt=original.COMPLETE,requires_agent_postexit_recheck=True)
    except Exception:
        p.new_json(i.child(root,failure),dict(at=p.now(),traceback=traceback.format_exc(),measurement_repeated=False,requires_human_direction=True))
        raise
