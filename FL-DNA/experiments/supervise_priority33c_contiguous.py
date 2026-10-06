"""Remaining-stage resume after the documented image layout interruption."""
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from experiments import supervise_priority33c as base

ROOT, OUT, PYTHON = base.ROOT, base.OUT, base.PYTHON


def main():
    # Never overlap any old/new priority33c scientific process.
    processes = subprocess.check_output(['ps', '-axo', 'pid,command'], text=True)
    for row in processes.splitlines():
        fields = row.strip().split(None, 1)
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) != os.getpid():
            if 'Python' in fields[1] and 'priority33c' in fields[1]:
                raise RuntimeError('Another P33c process is alive: ' + row)
    original = base.read(OUT / 'execution_freeze.json')['source_hashes']
    assert all(base.sha(ROOT / k) == v for k, v in original.items())
    paths = [Path(__file__), ROOT / 'experiments/priority33c_image_recovery_contiguous.py',
             ROOT / 'tests/test_priority33c_contiguous.py',
             ROOT / 'protocols/amendments/2026-10-03_priority33c_contiguous_input_repair.md']
    correction = {str(p.relative_to(ROOT)): base.sha(p) for p in paths}
    freeze = OUT / 'contiguous_correction_freeze.json'
    if freeze.exists():
        assert base.read(freeze)['source_hashes'] == correction
    else:
        base.save(freeze, dict(at=time.time(), source_hashes=correction, original_source_hashes=original))
    base.save(OUT / 'contiguous_supervisor.lock.json', dict(pid=os.getpid(), at=time.time()))
    base.log('technical_correction_resume', cause='noncontiguous image input before attack', source_hashes=correction)
    try:
        assert (OUT / 'paysim_calibration.json').exists() and (OUT / 'PAYSIM_BN_COMPLETE.json').exists()
        base.run('priority33c_image_recovery_contiguous.py', OUT / 'IMAGE_RECOVERY_COMPLETE.json', 'C1_calibration_and_n39')
        for stage in ['C2_descriptive_n24', 'D_per_tensor_utility_and_comparisons']:
            base.checklist(stage, 'completed', gate_receipt='artifacts/priority33c/IMAGE_RECOVERY_COMPLETE.json')
        base.run('analyze_priority33c.py', OUT / 'independent_statistics.json', 'independent_statistics_combined_summary')
        base.checklist('report_hashes_final_checks', 'in_progress')
        commands = {
            'tests': [PYTHON, '-B', '-m', 'unittest', 'tests.test_priority33c_image_utility', 'tests.test_priority33c_image_recovery', 'tests.test_priority33c_statistics', 'tests.test_priority33c_contiguous'],
            'py_compile': [PYTHON, '-B', '-m', 'py_compile'] + [str(ROOT / k) for k in {**original, **correction} if k.endswith('.py')],
            'git_diff_check': ['git', 'diff', '--check']}
        checks = {}
        for key, cmd in commands.items():
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            checks[key] = dict(returncode=p.returncode, output=p.stdout+p.stderr)
        checks['source_hashes_unchanged'] = {k: base.sha(ROOT / k) == v for k, v in {**original, **correction}.items()}
        base.save(OUT / 'final_checks.json', checks)
        assert all(checks[k]['returncode'] == 0 for k in commands) and all(checks['source_hashes_unchanged'].values())
        base.checklist('report_hashes_final_checks', 'completed', gate_receipt='artifacts/priority33c/final_checks.json')
        files = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ['sha256_manifest.csv', 'COMPLETE.json']]
        files += [ROOT / k for k in {**original, **correction}] + [ROOT / 'reports/priority33c_report.md']
        with (OUT / 'sha256_manifest.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['path', 'sha256']); writer.writeheader()
            for p in sorted(set(files)):
                writer.writerow(dict(path=str(p.relative_to(ROOT)), sha256=base.sha(p)))
        base.save(OUT / 'COMPLETE.json', dict(status='scientific_stages_and_local_checks_resolved', at=time.time(),
            report='reports/priority33c_report.md', report_sha256=base.sha(ROOT / 'reports/priority33c_report.md'),
            independent_statistics=True, technical_interruption='contiguous_input_repair', final_live_process_and_manifest_audit_required=True))
    except BaseException as error:
        base.save(OUT / 'REQUIRES_DIRECTION_CONTIGUOUS_SUPERVISOR.json', dict(at=time.time(), error=repr(error)))
        base.log('contiguous_supervisor_failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
