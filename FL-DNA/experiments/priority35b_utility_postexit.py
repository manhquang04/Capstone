"""One blocking socket-exit wait, then one utility-only final audit; no polling."""
import argparse
import json
import os
import py_compile
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import priority35b_utility as u


def finish():
    if u.live(): raise ValueError('scientific supervisor/worker still alive')
    u.verify_freeze()
    receipt=u.read(u.OUT/'UTILITY_COMPLETE.json')
    if receipt['status']!='UTILITY_VERIFIED_COMPLETE': raise ValueError('utility incomplete')
    manifest=u.OUT/'utility_manifest.json'
    if u.sha(manifest)!=receipt['manifest_sha256']: raise ValueError('manifest digest mismatch')
    hashes=u.read(manifest)
    for relative,expected in hashes.items():
        if u.sha(ROOT/relative)!=expected: raise ValueError('manifest mismatch '+relative)
    checks=[]
    for command in ([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_priority35b*.py','-v'],
                    ['git','diff','--check']):
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        checks.append(dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:
            u.save(u.OUT/'POSTEXIT_FAILED_CHECKS.json',checks);raise ValueError('final checks failed')
    for path in list((ROOT/'experiments').glob('priority35b_*.py'))+list((ROOT/'tests').glob('test_priority35b*.py')):
        py_compile.compile(str(path),cfile=str(u.OUT/'compile_final'/(path.stem+'.pyc')),doraise=True)
    if u.live(): raise ValueError('live worker at final seal')
    u.stage('UTILITY_VERIFIED_COMPLETE',done=330,total=330,reused=297,failed=0,
            stop='utility only; no leakage; waiting for user')
    extra=[u.OUT/'detached_stdout.log',u.OUT/'detached_stderr.log',u.OUT/'supervisor.lock.json',
           ROOT/'artifacts/priority35b/progress.json',ROOT/'artifacts/priority35b/progress.log',
           ROOT/'artifacts/priority35b/checklist.json',Path(__file__).resolve()]
    u.save(u.OUT/'POSTEXIT_VERIFIED.json',dict(status='UTILITY_VERIFIED_COMPLETE',
        full_priority35b_complete=False,leakage_runs=0,live_scientific_workers=[],
        checked_manifest_entries=len(hashes),checks=checks,py_compile='PASS',
        manifest_sha256=u.sha(manifest),completion_receipt_sha256=u.sha(u.OUT/'UTILITY_COMPLETE.json'),
        administrative_hashes={str(p.relative_to(ROOT)):u.sha(p) for p in extra}))


def wait_exit():
    # Supervisor owns this loopback listener for its whole lifetime. A pending
    # connection receives EOF/reset when it closes, including abnormal exit.
    # No periodic reads, sleep loop, process mutation or scientific replay.
    with socket.socket() as connection:
        try:
            connection.connect(('127.0.0.1',u.PORT))
            connection.recv(1)
        except (ConnectionResetError,ConnectionRefusedError): pass
    finish()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait-exit',action='store_true')
    parser.add_argument('--launch-waiter',action='store_true');args=parser.parse_args()
    if args.launch_waiter:
        if list(u.OUT.glob('waiter_receipt_*.json')): raise SystemExit('waiter already launched')
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--wait-exit']
        with (u.OUT/'postexit_stdout.log').open('a') as so,(u.OUT/'postexit_stderr.log').open('a') as se:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=so,stderr=se,
                                     start_new_session=True,close_fds=True)
        u.save(u.OUT/f'waiter_receipt_{time.time_ns()}.json',dict(pid=process.pid,command=command,
            source_sha256=u.sha(Path(__file__).resolve()),at=u.local.p.now(),
            purpose='one native socket wait for supervisor exit then read-only audit; no measurements'))
        print(json.dumps(dict(waiter_pid=process.pid)),flush=True)
    else:
        try:
            if args.wait_exit: wait_exit()
            else: finish()
        except Exception:
            import traceback
            u.save(u.OUT/f'POSTEXIT_FAILURE_{time.time_ns()}.json',dict(traceback=traceback.format_exc()))
            raise
