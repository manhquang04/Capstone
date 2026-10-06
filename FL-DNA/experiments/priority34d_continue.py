"""One-shot stage continuation; no polling, sleeps, training duplicates or Goal changes."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/priority34d'
PYTHON=ROOT/'.venv-phase1/bin/python'


def live():
    if os.name=='nt':
        rows=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',
          'Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | ConvertTo-Json'],text=True))
        return [row for row in rows if 'priority34d_' in (row.get('CommandLine') or '')
                and any(flag in (row.get('CommandLine') or '') for flag in ('--supervise','--job'))]
    rows=subprocess.check_output(['ps','-axo','pid,command'],text=True).splitlines()
    return [row for row in rows if 'python' in row.lower() and 'priority34d_' in row
            and any(flag in row for flag in ('--supervise','--job'))
            and int(row.split()[0])!=os.getpid()]


def run(script,flag):
    subprocess.run([str(PYTHON),'-B','-u',str(ROOT/'experiments'/script),flag],cwd=ROOT,check=True)


def check(advance=False):
    markers=[str(path.relative_to(ROOT)) for path in OUT.rglob('*.json')
             if path.name.startswith('REQUIRES_DIRECTION') or path.name=='SUPERVISOR_FAILURE.json']
    processes=live()
    progress=json.loads((OUT/'progress.json').read_text())
    print(json.dumps(dict(progress=progress,live=processes,direction_markers=markers)))
    if markers:
        raise RuntimeError('preserve failures; inspect and request direction, never replay science')
    if processes or not advance:
        return
    utility=OUT/'image_utility/UTILITY_COMPLETE.json'
    bn=OUT/'bn/baf/BN_COMPLETE.json'
    image=OUT/'image_recovery/IMAGE_COMPLETE.json'
    if not utility.exists():
        raise RuntimeError('utility stopped without stage complete: inspect infrastructure, no automatic restart')
    if not bn.exists():
        run('priority34d_bn.py','--prepare')
        run('priority34d_bn.py','--launch')
        return
    if not image.exists():
        run('priority34d_image_recovery.py','--prepare')
        run('priority34d_image_recovery.py','--launch')
        return
    # A human-visible continuation turn must perform the independent final audit;
    # do not claim completion merely from stage markers.
    print(json.dumps(dict(status='READY_FOR_INDEPENDENT_FINAL_AUDIT',complete=False,
                          required='Independent BN payload/calibration verification, array scorer, fixed76/combined211, full manifest/report/compile/diff/no-live audit')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--advance',action='store_true')
    args=parser.parse_args(); check(args.advance)
