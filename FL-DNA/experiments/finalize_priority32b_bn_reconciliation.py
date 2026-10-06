"""Finalize only after the frozen diagnostic queue terminates; no training."""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'artifacts/priority32b_bn_reconciliation'
REPORT = ROOT / 'reports/priority32b_bn_variance_reconciliation.md'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, doc):
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n')


def main():
    import py_compile
    from experiments.analyze_priority32b_bn_reconciliation import analyze
    assert not (OUT / 'FINALIZED.json').exists(), 'Keep sealed artifacts; do not re-finalize'
    lock = json.loads((OUT / 'supervisor.json').read_text())
    try:
        os.kill(lock['pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError('Supervisor still alive; no finalization')
    snapshot = analyze(require_complete=True)
    dependencies = json.loads((OUT / 'transform_dependency_provenance.json').read_text())
    assert all(sha(ROOT / path) == expected for path, expected in dependencies['sha256'].items())
    registered = [row for row in snapshot['runs'] if row['job']['factor'] == 'registered']
    assert len(registered) == 6
    assert all(row.get('reproduction_gate') for row in registered), 'Report failed registered replay gate explicitly before finalizing'
    assert all(row['rounds_completed'] == 50 for row in snapshot['runs']), 'Review interrupted/failed diagnostics before finalizing'
    assert all(row['status'] == 'COMPLETED' for row in snapshot['runs']), 'Review every failed diagnostic before finalizing'
    # Failed metric/finiteness outcomes remain observed evidence, not retried jobs.
    backend = json.loads((OUT / 'backend_probe/result.json').read_text())
    activation = json.loads((OUT / 'activation_probe/result.json').read_text())
    assert len(backend['checkpoints']) == 6 and len(activation['traces']) == 12
    assert backend['cpu_rng_unchanged'] and activation['cpu_rng_unchanged']
    assert activation['relu_probe']['cpu']['finite'] == [False, True, True, True]
    assert activation['relu_probe']['mps']['values'] == [0., 0., 0., 1.]
    negative_final = [row for row in backend['checkpoints'] if any(row['negative_bn'].values())]
    assert len(negative_final) == 5
    assert all(row['devices']['cpu']['finite_logits'] == 0 and row['devices']['mps']['finite_logits'] == 1024 for row in negative_final)
    evaluations = []
    for row in registered:
        job = row['job']
        path = OUT / 'jobs/registered' / job['method'] / f"seed_{job['seed']}" / 'evaluations.jsonl'
        evaluations += [json.loads(line) for line in path.read_text().splitlines()]
    assert len(evaluations) == 600
    assert all(row['finite_logits'] and row['finite_probabilities'] for row in evaluations)
    cadence_checkpoint_equality = []
    for row in registered:
        job = row['job']
        base = OUT / 'jobs/p32_full' / job['method'] / f"seed_{job['seed']}" / 'final_state.pt'
        cadence = OUT / 'jobs/evaluation_cadence' / job['method'] / f"seed_{job['seed']}" / 'final_state.pt'
        cadence_checkpoint_equality.append({'method': job['method'], 'seed': job['seed'], 'equal': sha(base) == sha(cadence)})
    assert all(row['equal'] for row in cadence_checkpoint_equality)
    factors = sorted(set(row['job']['factor'] for row in snapshot['runs']))
    section = ['\n## Final diagnostic queue and one-factor reconciliation\n',
               f"All {len(snapshot['runs'])} frozen diagnostic jobs finished. No early stopping or scientific tuning was used.\n",
               '| Factor | Method | Seed | First negative BN round | Negative rounds | Final validation finite | Final test finite | Runtime seconds |',
               '|---|---|---:|---:|---:|---|---|---:|']
    for row in snapshot['runs']:
        job = row['job']
        values = [job['factor'], job['method'], str(job['seed']), str(row['first_negative_round']),
                  str(len(row['negative_rounds'])), str((row['final']['validation'] or {}).get('finite_probabilities')),
                  str((row['final']['test'] or {}).get('finite_probabilities')), f"{row['elapsed_seconds']:.3f}"]
        section.append('| ' + ' | '.join(values) + ' |')
    section += ['\nOne-factor contrasts are conditional on the six specified seed/method pairs, not a population causal estimate.\n',
                '| Factor | Completed jobs | Negative variance ever | Nonfinite final test | Changed negative-presence vs P32 |',
                '|---|---:|---:|---:|---:|']
    for factor in factors:
        rows = [r for r in snapshot['runs'] if r['job']['factor'] == factor]
        contrasts = [r for r in snapshot['one_factor_contrasts'] if r['job']['factor'] == factor]
        section.append('| ' + ' | '.join([factor, str(len(rows)), str(sum(r['first_negative_round'] is not None for r in rows)),
             str(sum(not (r['final']['test'] or {}).get('finite_probabilities', False) for r in rows)),
             str(sum(r['negative_presence_changed'] for r in contrasts)) if contrasts else 'Not a swap']) + ' |')
    section += ['\nThe source of invalid variance is the unconstrained full-floating-state transform of BN variance deltas before FedAvg; neither transform enforces a nonnegative reconstructed running_var. The registered pipeline ALSO has this failure, so a difference in dataset preprocessing is not required to explain its presence. Data/seed/trajectory may change its onset or final persistence. Backend inference probes isolate the finite-output discrepancy independently of those coupled training factors.\n',
        'Plain conclusion: the selected stored metrics are exactly reproducible, but successful jobs and finite endpoints masked invalid BN channels. The five final-negative selected registered models are not valid ordinary-BN utility evidence; MPS ReLU suppresses NaNs into zeros. All six selected intermediate evaluation histories were affected. The one nonnegative final checkpoint has valid-domain final inference in the cross-backend probe. This cannot certify or reject every unreplayed final endpoint across 21 v1 and 52 v2 runs. The original aggregate RQ2 claims must carry this numerical-validity caveat; the observed failure is not repaired by reinterpreting the separate trainable-only/raw-BN P32 variant as the registered experiment. No valid-BN counterfactual utility effect size is identified here.\n',
        'Every job command and result SHA is in runs.jsonl; every per-layer, per-round client/aggregate variance is in bn_rounds.jsonl. Full evaluation probability arrays and final checkpoints are retained. There were no diagnostic training interruptions or scientific reruns if all frozen round-count gates above pass. Additional inference-only probes were preregistered after observing finite-on-negative states, disclosed above, and did not modify frozen training or earlier artifacts.\n',
        'Evaluation cadence is excluded as a trajectory-changing cause on these six seeds: all six final_state.pt SHA-256 values are byte-identical to their P32 full-state counterparts. This is stronger than merely equal endpoint metrics.\n',
        'Finalization command: `.venv-phase1/bin/python -B experiments/finalize_priority32b_bn_reconciliation.py`.\n']
    report = REPORT.read_text().replace('Status: IN PROGRESS. Preliminary observations below are not final conclusions. P33 deferred.',
        'Status: COMPLETED. Interim sections are retained chronologically; final findings below govern. P33 was deferred throughout this reconciliation.')
    assert '## Final diagnostic queue and one-factor reconciliation' not in report
    REPORT.write_text(report + '\n'.join(section))
    sources = sorted(ROOT.glob('experiments/*priority32b*.py'))
    for source in sources:
        py_compile.compile(str(source), cfile=str(OUT / (source.name + 'c')), doraise=True)
    diff = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, text=True, capture_output=True)
    checks = {'at': datetime.now(timezone.utc).isoformat(), 'py_compile': 'PASS',
              'git_diff_check': 'PASS' if diff.returncode == 0 else 'FAIL', 'git_diff_output': diff.stdout + diff.stderr,
              'all_expected_jobs': True, 'registered_reproduction': '6/6 bit-exact', 'finite_registered_evaluations': '600/600',
              'independent_recomputation_agrees': True, 'earlier_hashes_unchanged': snapshot['earlier_hashes_unchanged'],
              'cadence_checkpoint_equality': cadence_checkpoint_equality,
              'inference_probe_checks': 'PASS', 'supervisor_stopped': True}
    write(OUT / 'final_checks.json', checks)
    assert diff.returncode == 0
    checklist = json.loads((OUT / 'checklist.json').read_text())
    for item in checklist.values():
        item['status'] = 'completed'
    write(OUT / 'checklist.json', checklist)
    progress = json.loads((OUT / 'progress.json').read_text())
    progress.update(stage='completed', last_update=datetime.now(timezone.utc).isoformat(), eta_minutes=0)
    write(OUT / 'progress.json', progress)
    with (OUT / 'progress.log').open('a') as log:
        log.write(json.dumps(progress) + '\n')
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name not in {'sha256_manifest.json', 'FINALIZED.json'}]
    files += sources + [REPORT] + sorted(ROOT.glob('protocols/amendments/*priority32b*.md'))
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}
    manifest.update(json.loads((OUT / 'source_provenance.json').read_text())['sha256'])
    manifest.update(dependencies['sha256'])
    write(OUT / 'sha256_manifest.json', manifest)
    assert all(sha(ROOT / path) == value for path, value in manifest.items())
    write(OUT / 'FINALIZED.json', {'at': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': sha(OUT / 'sha256_manifest.json'),
        'report_sha256': sha(REPORT), 'checks_sha256': sha(OUT / 'final_checks.json'), 'files': len(manifest)})
    print(json.dumps({'completed': True, 'jobs': len(snapshot['runs']), 'report': str(REPORT), 'manifest_files': len(manifest)}))


if __name__ == '__main__':
    main()
