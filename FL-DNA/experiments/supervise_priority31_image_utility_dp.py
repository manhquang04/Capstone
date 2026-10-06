"""One-shot P31 supervisor independent of the chat terminal's lifetime.

Only runs the frozen resumable runner and reporting finalizer. No retry on a
scientific gate or failure; no experimental setting is changed here.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/priority31_image_utility_dp'
os.chdir(ROOT)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'


def event(name, **values):
    with (OUT/'supervisor.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                    event=name, **values))+'\n')


if __name__ == '__main__':
    event('start', pid=os.getpid())
    for script, args in [('experiments/priority31_image_utility_dp.py', ['--stage','all','--workers','8']),
                         ('experiments/finalize_priority31_image_utility_dp.py', [])]:
        command=[sys.executable,'-B','-u',script,*args]
        event('command', argv=command)
        exit_code=subprocess.call(command)
        event('exit', argv=command, returncode=exit_code)
        if exit_code:
            sys.exit(exit_code)
    event('completed')
