"""Detached P33c staged supervisor; no overlapping worker pools or scientific tuning."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / 'artifacts/priority33c'
PYTHON = str(ROOT / '.venv-phase1/bin/python')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def log(event, **details):
    with (OUT / 'supervisor.jsonl').open('a') as f:
        f.write(json.dumps(dict(at=time.time(), event=event, **details))+'\n')


def alive(pid, script):
    process = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
    return process.returncode == 0 and script in process.stdout


def checklist(stage, status, **details):
    path = OUT / 'checklist.json'
    doc = read(path)
    for row in doc['stages']:
        if row['stage'] == stage:
            row.update(status=status, **details)
    save(path, doc)


def run(script, receipt, stage, mark_completed=True):
    checklist(stage, 'in_progress')
    command = [PYTHON, '-B', '-u', str(ROOT / 'experiments' / script)]
    log('stage_invocation', command=command, stage=stage, receipt=str(receipt))
    started = time.time()
    with (OUT / (script+'.stdout.log')).open('a') as so, (OUT / (script+'.stderr.log')).open('a') as se:
        proc = subprocess.run(command, cwd=ROOT, stdout=so, stderr=se)
    if proc.returncode or not receipt.exists():
        raise RuntimeError('Stage failed '+script+' returncode='+str(proc.returncode))
    log('stage_completed', stage=stage, elapsed_seconds=time.time()-started, receipt_sha256=sha(receipt))
    if mark_completed:
        checklist(stage, 'completed', gate_receipt=str(receipt.relative_to(ROOT)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / 'supervisor.lock.json'
    if lock.exists():
        previous = read(lock)
        if alive(previous['pid'], 'supervise_priority33c.py'):
            raise RuntimeError('Existing supervisor is alive; no duplicate')
        log('infrastructure_interruption_resume', prior=previous,
            disclosure='same frozen configuration, completed results validate/skip; preserve all partial artifacts')
    save(lock, dict(pid=os.getpid(), started=time.time(), argv=sys.argv))
    paths = list((ROOT / 'experiments').glob('*priority33c*.py'))
    paths += list((ROOT / 'tests').glob('*priority33c*.py'))
    paths += list((ROOT / 'protocols/amendments').glob('2026-10-03_priority33c*.md'))
    fingerprint = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    freeze = OUT / 'execution_freeze.json'
    if freeze.exists():
        assert read(freeze)['source_hashes'] == fingerprint, 'frozen running sources changed'
    else:
        save(freeze, dict(at=time.time(), source_hashes=fingerprint, workers=4, CPU=True, torch_threads=1))
    log('supervisor_started', source_hashes=fingerprint)
    try:
        # Join the already-running utility process instead of launching another pool.
        launch = read(OUT / 'image_utility_launch.json')
        while alive(launch['pid'], 'priority33c_image_utility.py'):
            time.sleep(15)
        if not (OUT / 'image_calibration.json').exists():
            raise RuntimeError('image utility stopped without a calibration receipt; inspect preserved logs')
        checklist('C1_calibration_and_n39', 'in_progress', utility_receipt='artifacts/priority33c/image_calibration.json')
        run('priority33c_paysim_utility.py', OUT / 'paysim_calibration.json', 'D_per_tensor_utility_and_comparisons', False)
        run('priority33c_paysim_bn.py', OUT / 'PAYSIM_BN_COMPLETE.json', 'D_per_tensor_utility_and_comparisons', False)
        run('priority33c_image_recovery.py', OUT / 'IMAGE_RECOVERY_COMPLETE.json', 'C1_calibration_and_n39')
        checklist('C2_descriptive_n24', 'completed', gate_receipt='artifacts/priority33c/IMAGE_RECOVERY_COMPLETE.json')
        checklist('D_per_tensor_utility_and_comparisons', 'completed',
            gate_receipts=['artifacts/priority33c/PAYSIM_BN_COMPLETE.json', 'artifacts/priority33c/IMAGE_RECOVERY_COMPLETE.json'])
        run('analyze_priority33c.py', OUT / 'independent_statistics.json', 'independent_statistics_combined_summary')
        checklist('report_hashes_final_checks', 'in_progress')
        test = subprocess.run([PYTHON, '-B', '-m', 'unittest', 'tests.test_priority33c_image_utility',
            'tests.test_priority33c_image_recovery', 'tests.test_priority33c_statistics'], cwd=ROOT, capture_output=True, text=True)
        compile_result = subprocess.run([PYTHON, '-B', '-m', 'py_compile']+[str(p) for p in (ROOT / 'experiments').glob('*priority33c*.py')],
            cwd=ROOT, capture_output=True, text=True)
        whitespace = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
        checks = dict(tests=dict(returncode=test.returncode, output=test.stdout+test.stderr),
            py_compile=dict(returncode=compile_result.returncode, output=compile_result.stdout+compile_result.stderr),
            git_diff_check=dict(returncode=whitespace.returncode, output=whitespace.stdout+whitespace.stderr),
            source_hashes_unchanged={k: sha(ROOT / k) == v for k, v in fingerprint.items()})
        save(OUT / 'final_checks.json', checks)
        if test.returncode or compile_result.returncode or whitespace.returncode or not all(checks['source_hashes_unchanged'].values()):
            raise RuntimeError('Final verification failed')
        checklist('report_hashes_final_checks', 'completed', gate_receipt='artifacts/priority33c/final_checks.json')
        # Rebuild manifest after final checks; science arrays/results are not changed.
        with (OUT / 'sha256_manifest.csv').open('w', newline='') as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=['path', 'sha256']); writer.writeheader()
            files = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ['sha256_manifest.csv', 'COMPLETE.json']]
            files += paths+[ROOT / 'reports/priority33c_report.md']
            for p in sorted(set(files)):
                writer.writerow(dict(path=str(p.relative_to(ROOT)), sha256=sha(p)))
        save(OUT / 'COMPLETE.json', dict(status='scientific_stages_and_local_checks_resolved',
            at=time.time(), report='reports/priority33c_report.md', report_sha256=sha(ROOT / 'reports/priority33c_report.md'),
            independent_statistics=True, final_live_process_and_manifest_audit_required=True))
        # Human monitor must verify no live workers, final manifest, and report completion wording.
    except BaseException as error:
        log('supervisor_failed', error=repr(error))
        save(OUT / 'REQUIRES_DIRECTION_SUPERVISOR.json', dict(error=repr(error), at=time.time()))
        raise


if __name__ == '__main__':
    main()
