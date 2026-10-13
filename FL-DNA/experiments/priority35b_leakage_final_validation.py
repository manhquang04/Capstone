"""Post-exit test/py_compile/diff runner, preserving every subprocess log.

Requires the independently verified preparation receipt. Each test suite uses a
fresh interpreter to avoid the preserved historical models import collision.
This performs synthetic tests only; no measurement or optimizer job is replayed.
"""
import argparse
import json
import os
import py_compile
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from experiments import priority35b_leakage_render_publication as render

prepare=render.prepare
integrity=render.integrity
SEAL='final_validation_freeze.json'
TESTS=('leakage','leakage_statistics','leakage_integrity','leakage_metrics_audit',
    'leakage_payload_audit','leakage_postexit','leakage_objectives_audit',
    'leakage_completion_audit','leakage_scientific_verify','leakage_tables',
    'leakage_report_guard','leakage_report_text','leakage_figure','leakage_data_bridge',
    'leakage_prepare_publication','leakage_manifest_audit','leakage_verified_preparation',
    'leakage_render_publication','leakage_publication_contract','leakage_final_validation')


def test_commands(python):
    return [[str(python),'-B','-m','unittest','tests.test_priority35b_'+name] for name in TESTS]


def successful_suite(returncode,output):
    match=re.search(r'Ran (\d+) tests? in ',output)
    if returncode!=0 or not match or int(match.group(1))<=0 or not re.search(r'^OK\s*$',output,re.M):
        raise ValueError('test suite failed or incomplete; preserve logs and ask direction')
    return int(match.group(1))


def verify_sources(root):
    render.wrapper.verify_sources(root)
    seals=(render.SEAL,'publication_contract_freeze.json',SEAL)
    prepare.scientific.verify_additive_sources(root,seals)
    for name in seals:
        for path,digest in integrity.read(Path(root)/'artifacts/priority35b/b3'/name)['inputs'].items():
            if integrity.sha(integrity.child(root,path))!=digest:raise ValueError('validation dependency drift')


def run(relative,root=prepare.ROOT):
    root=Path(root).resolve()
    attempt,ready_path=render.verify_ready(root,relative)
    verify_sources(root)
    python=root/'.venv-phase1/bin/python'
    if Path(sys.executable).resolve()!=python.resolve():raise ValueError('.venv-phase1 required')
    directory=attempt/('validation_attempt_'+str(time.time_ns()));directory.mkdir()
    commands=test_commands(python)
    prepare.new_json(directory/'START.json',dict(at=prepare.now(),pid=os.getpid(),commands=commands,
        ready_sha256=integrity.sha(ready_path),measurement_repeated=False))
    try:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
            env[key]='1'
        passed=0;logs={}
        for index,command in enumerate(commands):
            result=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True)
            log=directory/('suite_%02d.json'%index)
            prepare.new_json(log,dict(command=command,returncode=result.returncode,
                stdout=result.stdout,stderr=result.stderr))
            logs[str(log.relative_to(root))]=integrity.sha(log)
            passed+=successful_suite(result.returncode,result.stdout+'\n'+result.stderr)
        sources=set()
        for name in render.wrapper.SEALS+(render.SEAL,'publication_contract_freeze.json',SEAL):
            sources.update(p for p in integrity.read(root/'artifacts/priority35b/b3'/name)['sources'] if p.endswith('.py'))
        sources.add('experiments/priority35b_leakage.py');sources.add('tests/test_priority35b_leakage.py')
        compile_dir=directory/'compiled';compile_dir.mkdir()
        for index,path in enumerate(sorted(sources)):
            py_compile.compile(str(integrity.child(root,path)),cfile=str(compile_dir/('%03d.pyc'%index)),doraise=True)
        diff_command=['git','diff','--check']
        result=subprocess.run(diff_command,cwd=root,capture_output=True,text=True)
        diff=directory/'diff_check.json'
        prepare.new_json(diff,dict(command=diff_command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:raise ValueError('git diff --check failed')
        logs[str(diff.relative_to(root))]=integrity.sha(diff)
        verify_sources(root);render.verify_ready(root,relative);prepare.no_live(root)
        receipt=dict(at=prepare.now(),status='VALIDATION_PASS',pid=os.getpid(),commands=commands+[diff_command],
            tests_passed=passed,tests_failed=0,compile='PASS',git_diff_check='PASS',compiled_sources=sorted(sources),
            log_hashes=logs,ready_receipt=relative,ready_sha256=integrity.sha(ready_path),
            measurement_repeated=False,full_priority35b_complete=False)
        prepare.new_json(directory/'VALIDATION.json',receipt)
        return dict(receipt=str((directory/'VALIDATION.json').relative_to(root)),status=receipt['status'],full_priority35b_complete=False)
    except Exception:
        prepare.new_json(directory/'FAILURE.json',dict(at=prepare.now(),traceback=traceback.format_exc(),
            measurement_repeated=False,requires_human_direction=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ready',required=True)
    args=parser.parse_args();print(json.dumps(run(args.ready),allow_nan=False))
