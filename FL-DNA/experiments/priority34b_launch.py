"""Launch exactly one frozen P34B supervisor; explicit infrastructure resume."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import priority34b_core as b
from experiments.verify_priority34b import live_workers


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--resume',action='store_true'); args=parser.parse_args()
    b.verified_freeze()
    if live_workers():
        raise ValueError('existing scientific supervisor/workers alive; no duplicate')
    if (b.OUT/'REQUIRES_DIRECTION.json').exists() or list(b.OUT.rglob('*.failure.json')):
        raise ValueError('scientific failures need user direction; do not relaunch')
    if (b.OUT/'COMPLETE.json').exists():
        raise ValueError('already complete')
    receipts=list(b.OUT.glob('launch_receipt_*.json'))
    if receipts and not args.resume:
        raise ValueError('prior launch exists; inspect interruption before explicit --resume')
    if args.resume and not receipts:
        raise ValueError('nothing to resume')
    import fcntl
    with (b.OUT/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(lock,fcntl.LOCK_UN)
    command=[sys.executable,'-B','-u',str(ROOT/'experiments/supervise_priority34b.py'),'--supervise']
    with (b.OUT/'detached_stdout.log').open('a') as stdout, (b.OUT/'detached_stderr.log').open('a') as stderr:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                 start_new_session=True,close_fds=True)
    receipt=dict(pid=process.pid,command=command,at=b.p.now(),resume=args.resume,
                 start_new_session=True,stdout='detached_stdout.log',stderr='detached_stderr.log',
                 frozen_manifest_sha256=b.p.sha(b.OUT/'execution_freeze.json'))
    b.save(b.OUT/f'launch_receipt_{time.time_ns()}.json',receipt)
    print(receipt,flush=True)


if __name__=='__main__':
    main()
