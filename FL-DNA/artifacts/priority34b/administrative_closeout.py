"""Post-completion preservation/hash audit; no scientific workload or key disclosure."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import py_compile
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/priority34b'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def main():
    receipt_path = OUT / 'administrative_verified_receipt.json'
    if receipt_path.exists():
        raise RuntimeError('Administrative seal already exists; preserve it.')
    frozen = read(OUT / 'execution_freeze.json')
    for section in ('sources', 'inputs'):
        for relative, digest in frozen[section].items():
            assert sha(ROOT / relative) == digest, relative
    manifest = read(OUT / 'sha256_manifest_final.json')
    report = 'reports/priority34b_report.md'
    relocated = OUT / 'report_post_exit_pre_admin.md'
    for relative, digest in manifest.items():
        path = relocated if relative == report else ROOT / relative
        assert sha(path) == digest, relative
    completed = read(OUT / 'COMPLETE.json')
    assert completed['status'] == 'PASS' and completed['jobs'] == 470
    assert completed['bit_exact_prediction_arrays'] == 1265
    assert completed['no_bn_assertions'] == 41250
    independent = read(OUT / 'analysis/independent_recomputation.json')
    assert independent['status'] == 'PASS'
    assert independent['paired_endpoints'] == 78
    assert independent['sign_tests'] == independent['holm_family'] == 24
    assert read(OUT / 'final_checks.json')['status'] == 'PASS'
    assert read(OUT / 'progress.json')['part'] == 'complete'
    assert all(v['status'] == 'complete' for v in read(OUT / 'checklist.json').values())
    assert not list(OUT.rglob('*.failure.json'))
    assert not list(OUT.glob('REQUIRES_DIRECTION*'))
    assert not list(OUT.glob('*INTERRUPTION*'))
    target = read(OUT / 'image_targets.json')
    assert len(set(target['targets'])) == 39
    assert not set(target['targets']).intersection(target['excluded'])
    scales = {}
    for job in frozen['jobs']:
        doc = read(OUT / job['result'])
        assert doc['status'] == 'COMPLETED' and doc['no_bn_transmitted']
        if job['stage'] == 'recovery':
            continue
        if job['method'] == 'baseline':
            continue
        audits = doc['dp_round_audit'] if job['stage'] == 'tabular' else doc['round_audit']
        assert len(audits) == 50
        for audit in audits:
            sensitivity = 2 * max(audit['weights']) * .01
            assert abs(audit['sensitivity'] - sensitivity) < 1e-14
            assert abs(audit['noise_sd'] - audit['sigma_sensitivity'] * sensitivity) < 1e-14
            epsilon = 50 * audit['alpha'] / (2 * audit['sigma_sensitivity'] ** 2)
            epsilon += math.log(1e5) / (audit['alpha'] - 1)
            assert abs(epsilon - job['epsilon']) < 1e-8
            key = f"{job['dataset']}/epsilon{job['epsilon']}"
            value = dict(sensitivity=sensitivity, noise_sd=audit['noise_sd'])
            assert key not in scales or scales[key] == value
            scales[key] = value
    # Inspect processes without starting, killing or resuming any workload.
    if os.name == 'nt':
        process_text = subprocess.check_output([
            'powershell', '-NoProfile', '-Command',
            'Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | ConvertTo-Json'
        ], text=True)
        rows = read_processes = json.loads(process_text)
        live = [r for r in rows if r.get('ProcessId') != os.getpid()
                and ('spawn_main' in (r.get('CommandLine') or '')
                     or 'supervise_priority34b.py --supervise' in (r.get('CommandLine') or ''))]
    else:
        live = []
        for line in subprocess.check_output(['ps', '-axo', 'pid=,ppid=,command='], text=True).splitlines():
            fields = line.strip().split(None, 2)
            if len(fields) != 3 or int(fields[0]) == os.getpid():
                continue
            command = fields[2]
            if 'python' in Path(command.split()[0]).name.lower():
                if 'spawn_main' in command or ('supervise_priority34b.py' in command and '--supervise' in command):
                    live.append(line)
    assert not live, 'Live workloads remain'
    subprocess.run(['git', 'diff', '--check'], cwd=ROOT, check=True)
    compile_path = OUT / 'administrative_closeout.pyc'
    assert not compile_path.exists()
    py_compile.compile(str(Path(__file__)), cfile=str(compile_path), doraise=True)
    files = {str(path.relative_to(ROOT)): sha(path) for path in OUT.iterdir()
             if path.is_file() and path.name != 'private_noise_seeds.json'}
    files[report] = sha(ROOT / report)
    files['../PROJECT.md'] = sha(ROOT.parent / 'PROJECT.md')
    receipt = dict(status='PASS', at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   sources_checked=len(frozen['sources']), inputs_checked=len(frozen['inputs']),
                   final_manifest_entries_checked=len(manifest), jobs=470,
                   bit_exact_prediction_arrays=1265, no_bn_assertions=41250,
                   paired_endpoints=78, fixed_holm_family=24, fresh_targets=39,
                   failures=0, live_workers=[], noise_scales=scales,
                   report_relocation='report_post_exit_pre_admin.md',
                   parent_project_preserved='PROJECT_pre_priority34b_completion.md',
                   py_compile='PASS', git_diff_check='PASS', files=files)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'files'}, indent=2))

if __name__ == '__main__':
    main()
