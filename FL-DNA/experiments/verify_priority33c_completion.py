"""Administrative final audit only; no training, attacks or scientific writes."""
import ast
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from experiments.analyze_priority33c import read, sha, save, verify_arrays

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / 'artifacts/priority33c'


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    live = []
    for row in subprocess.check_output(['ps', '-axo', 'pid,command'], text=True).splitlines():
        parts = row.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) == os.getpid():
            continue
        if ('Python' in parts[1] or 'python' in parts[1]) and ('priority33c' in parts[1] or 'spawn_main' in parts[1]):
            live.append(row)
    assert not live, live
    for row in rows(OUT / 'sha256_manifest.csv'):
        assert sha(ROOT / row['path']) == row['sha256'], row['path']
    hashes = {}
    for name in ['execution_freeze.json', 'contiguous_correction_freeze.json']:
        for key, value in read(OUT / name)['source_hashes'].items():
            assert sha(ROOT / key) == value, key
            hashes[key] = value
    for row in read(OUT / 'image_reused_utility.json'):
        assert sha(ROOT / row['path']) == row['sha256'], row['path']
        hashes[row['path']] = row['sha256']
    targets = read(OUT / 'image_targets.json')
    ids = targets['development'] + targets['C1'] + sum(targets['C2_batch4'], []) + targets['C2_trained']
    assert len(ids) == len(set(ids)) == 183 and not set(ids) & set(targets['excluded'])
    jobs = read(OUT / 'image_job_manifest.json')
    assert len(jobs) == 669
    for job in jobs:
        folder = OUT / 'image_recovery' / job['setting'] / job['arm'] / ('target_%03d' % job['target_id'])
        doc = read(folder / 'result.json')
        assert doc['job'] == job and doc['status'] == 'passed'
        assert doc['iterations'] == 4800 and doc['restarts'] == 1
        assert sha(folder / 'receipt.pt') == doc['receipt_sha256']
        for key, value in doc['source_hashes'].items():
            path = OUT / 'checkpoint_replay/final_state.pt' if key == 'trained_checkpoint' else ROOT / key
            assert sha(path) == value, key
    assert verify_arrays() == 669
    stats = read(OUT / 'independent_statistics.json')
    assert stats['agreement'] and stats['family_size'] == 20 and stats['combined_family_size'] == 135
    tests = stats['tests']
    for test in tests:
        if test['status'] != 'VALID':
            assert test['p_raw'] == 1
            continue
        identifier = test['hypothesis_id'].rsplit('::', 1)[0]
        paired = rows(OUT / (identifier.replace('::', '_') + '_paired.csv'))
        assert len(paired) == 39 and {int(r['target']) for r in paired} == set(range(39))
        for r in paired:
            assert ast.literal_eval(r['source_ids']) == targets['C1'][int(r['target']):int(r['target'])+1]
        a = np.array([float(r['dna']) for r in paired]); b = np.array([float(r['comparator']) for r in paired])
        diff = a-b
        wins = int(np.sum(diff < 0)) if test['direction'] == 'DNA_stronger' else int(np.sum(diff > 0))
        losses = int(np.sum(diff > 0)) if test['direction'] == 'DNA_stronger' else int(np.sum(diff < 0))
        assert wins == test['wins'] and losses == test['losses'] and 39-wins-losses == test['ties']
        assert abs(binomtest(wins, wins+losses, alternative='greater').pvalue-test['p_raw']) < 1e-14
        for key, value in [('median_dna', np.median(a)), ('median_comparator', np.median(b)),
                           ('median_paired_difference', np.median(diff)), ('rank13', np.sort(diff)[12]), ('rank27', np.sort(diff)[26])]:
            assert abs(test[key]-value) < 1e-12
    for table, field in [(tests, 'holm_p33c'), (rows(OUT / 'combined135_holm.csv'), 'holm_combined135')]:
        ordered = sorted(table, key=lambda r: float(r['p_raw']))
        independent = np.minimum(1, np.maximum.accumulate(np.array([float(r['p_raw']) for r in ordered])*np.arange(len(ordered), 0, -1)))
        assert np.allclose(independent, [float(r[field]) for r in ordered], atol=1e-14, rtol=0)
    calibration = read(OUT / 'image_calibration.json')
    assert calibration['baseline_passed'] and calibration['baseline_mean'] >= .4
    for match in calibration['matches'].values():
        assert match['bracketed'] and match['status'] == 'matched'
        assert match['grid_means'][str(match['sigma'])] >= match['threshold']
        assert match['grid_means'][str(match['next_sigma'])] < match['threshold']
    utility = list((OUT / 'image_utility').rglob('seed_*.json'))
    assert len(utility) == 160 and all(read(p)['status'] == 'passed' for p in utility)
    paysim = list((OUT / 'paysim_utility').rglob('result.json'))
    assert len(paysim) == 48
    assert sum(read(p)['status'] == 'passed' for p in paysim) == 16
    assert read(OUT / 'PAYSIM_BN_COMPLETE.json')['status'] == 'NOT_ASSESSABLE'
    checks = read(OUT / 'final_checks.json')
    assert all(checks[k]['returncode'] == 0 for k in ['tests', 'py_compile', 'git_diff_check'])
    assert all(checks['source_hashes_unchanged'].values())
    compile_check = subprocess.run([sys.executable, '-m', 'py_compile', str(Path(__file__))], capture_output=True, text=True)
    diff_check = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
    assert compile_check.returncode == diff_check.returncode == 0
    # Preserve the original report, COMPLETE and manifest; new final evidence only.
    hashes[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    for p in (ROOT / 'datasets/cifar10').rglob('*'):
        if p.is_file():
            hashes[str(p.relative_to(ROOT))] = sha(p)
    hashes['datasets/creditcard.csv'] = sha(ROOT / 'datasets/creditcard.csv')
    for key, expected in [('reports/priority33a_report.md', '8c9bf1890e5d0bf9b0f1f35a5c0252a6d0fa56c3f62e0e3f1a13642008dd1f61'),
                          ('reports/priority33b_report.md', '5e89fd68d91f0f414ca44f2722faa91f5b6e3bbd220af76549a27a66cc399670')]:
        assert sha(ROOT / key) == expected, key
        hashes[key] = expected
    if '--verify-only' in sys.argv:
        print('PASS: no live workers, original manifest, all sources/receipts, 669 arrays, pairing, calibration, independent 20/135 statistics and final checks')
        return
    receipt = dict(status='PASS', no_live_workers=True, verified_jobs=669, original_manifest_passed=True,
        source_hashes=hashes, pairing_and_fresh_source_disjointness=True, independently_recomputed_tests=20,
        independently_recomputed_combined_holm=135, verified_arrays=669, calibration_brackets=True,
        image_utility_jobs=160, paysim_jobs=48, paysim_gate_failures=32, no_scientific_replay=True,
        final_checks_passed=True, report_sha256=sha(ROOT / 'reports/priority33c_report.md'))
    save(OUT / 'final_verified_receipt.json', receipt)
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'sha256_manifest_final.csv']
    files += [ROOT / k for k in hashes] + [ROOT / 'reports/priority33c_report.md', ROOT / 'reports/priority33c_completion_addendum.md', ROOT.parent / 'PROJECT.md']
    with (OUT / 'sha256_manifest_final.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'sha256']); writer.writeheader()
        for p in sorted(set(files)):
            writer.writerow(dict(path=os.path.relpath(p, ROOT), sha256=sha(p)))
    for row in rows(OUT / 'sha256_manifest_final.csv'):
        assert sha(ROOT / row['path']) == row['sha256']
    print(json.dumps(dict(status='PASS', jobs=669, tests=20, combined=135, manifest_files=len(rows(OUT / 'sha256_manifest_final.csv')))))


if __name__ == '__main__':
    main()
