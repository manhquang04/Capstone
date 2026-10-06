"""Independent, read-only analysis of completed P32b diagnostic artifacts."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'artifacts/priority32b_bn_reconciliation'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def summarize(folder):
    result = json.loads((folder / 'result.json').read_text())
    bn = lines(folder / 'bn_rounds.jsonl')
    evaluations = lines(folder / 'evaluations.jsonl')
    negative = [row['round'] for row in bn if any(v['negative'] for v in row['aggregate'].values())]
    assert result.get('first_negative_round') == (min(negative) if negative else None)
    finite_negative = [row for row in evaluations if row['round'] in negative and row.get('finite_probabilities')]
    final = {split: next((row for row in reversed(evaluations)
                         if row['round'] == 50 and row['split'] == split), None)
             for split in ('validation', 'test')}
    minima = {}
    for row in bn:
        for name, state in row['aggregate'].items():
            minima[name] = min(minima.get(name, float('inf')), state['min'])
    row = {'job': result['job'], 'status': result['status'], 'rounds_completed': len(bn),
           'first_negative_round': min(negative) if negative else None,
           'negative_rounds': negative, 'aggregate_layer_minima': minima,
           'client_negative_rounds': [r['round'] for r in bn if any(v['negative']
                                    for client in r['transmitted_clients'] for v in client.values())],
           'finite_evaluation_on_negative_rounds': len(finite_negative),
           'nonfinite_evaluation_rounds': sorted(set(r['round'] for r in evaluations
                                      if not r.get('finite_probabilities', False))),
           'final': final, 'native_device': result.get('native_device'),
           'elapsed_seconds': result.get('elapsed_seconds'), 'exception': result.get('exception')}
    if result['job']['factor'] == 'registered' and (folder / 'metrics.json').exists():
        method, seed = result['job']['method'], result['job']['seed']
        branch = 'rq2/confirmatory_run_20260913' if method == 'v1' else 'rq2_v2/confirmatory_20260916'
        label = 'dna_transform' if method == 'v1' else 'dna_transform_v2'
        original = json.loads((ROOT / 'artifacts' / branch / f'seed_{seed}' / label / 'metrics.json').read_text())
        replay = json.loads((folder / 'metrics.json').read_text())
        from experiments.priority32b_bn_reconcile import CORE
        errors, mismatches = [], []
        for before, after in zip(original['rounds'], replay['rounds']):
            for key in CORE:
                a, b = before[key], after[key]
                if a is None or b is None:
                    if a != b:
                        mismatches.append([before['round'], key, a, b])
                    continue
                difference = abs(float(a) - float(b))
                errors.append(difference)
                if difference > 1e-8 or (key in {'tn', 'fp', 'fn', 'tp'} and difference):
                    mismatches.append([before['round'], key, a, b])
        row['independent_reproduction'] = {
            'max_absolute_error': max(errors, default=0), 'mismatch_count': len(mismatches),
            'mismatches': mismatches, 'config_equal': original['config'] == replay['config'],
            'round_count_equal': len(original['rounds']) == len(replay['rounds'])}
        assert row['independent_reproduction']['max_absolute_error'] == result['reproduction']['max_absolute_error']
        assert mismatches == result['reproduction']['mismatches']
        row['reproduction_gate'] = (not mismatches and row['independent_reproduction']['config_equal']
                                    and len(original['rounds']) == len(replay['rounds']) == 50)
    return row


def analyze(require_complete=False):
    terminal = OUT / 'REPLAYS_COMPLETE.json'
    if require_complete and not terminal.exists():
        raise RuntimeError('Diagnostic queue not complete; do not finalize')
    frozen = json.loads((OUT / 'execution_freeze.json').read_text())
    assert digest(ROOT / 'experiments/priority32b_bn_reconcile.py') == frozen['runner_sha256']
    assert digest(ROOT / 'protocols/amendments/2026-10-02_priority32b_bn_variance_reconciliation.md') == frozen['amendment_sha256']
    provenance = json.loads((OUT / 'source_provenance.json').read_text())
    changed = [path for path, expected in provenance['sha256'].items() if digest(ROOT / path) != expected]
    assert not changed, 'Earlier inputs/source/artifacts changed: ' + repr(changed)
    rows = [summarize(path.parent) for path in sorted((OUT / 'jobs').glob('*/*/seed_*/result.json'))]
    expected = list(frozen['jobs'])
    if terminal.exists() and json.loads(terminal.read_text())['data_decomposition']:
        seeds = {method: [job['seed'] for job in frozen['jobs'] if job['factor'] == 'registered' and job['method'] == method]
                 for method in ('v1', 'v2')}
        expected += [{'method': m, 'seed': s, 'factor': factor} for factor in ('cap_before_fixed', 'cap_after_per_seed')
                     for m, values in seeds.items() for s in values]
    missing = [job for job in expected if job not in [row['job'] for row in rows]]
    if require_complete:
        assert not missing
        assert len(rows) == len(expected) == json.loads(terminal.read_text())['jobs']
    contrasts = []
    for row in rows:
        job = row['job']
        if job['factor'] in {'registered', 'p32_full'}:
            continue
        baseline = next((r for r in rows if r['job'] == {**job, 'factor': 'p32_full'}), None)
        if baseline is not None:
            contrasts.append({'job': job, 'p32_first_negative_round': baseline['first_negative_round'],
                'swap_first_negative_round': row['first_negative_round'],
                'negative_presence_changed': (baseline['first_negative_round'] is None) != (row['first_negative_round'] is None),
                'p32_final_test_finite': (baseline['final']['test'] or {}).get('finite_probabilities'),
                'swap_final_test_finite': (row['final']['test'] or {}).get('finite_probabilities')})
    snapshot = {'at': datetime.now(timezone.utc).isoformat(), 'complete': bool(terminal.exists() and not missing),
                'jobs_done': len(rows), 'jobs_expected': len(expected), 'missing': missing,
                'earlier_hashes_unchanged': not changed, 'runs': rows, 'one_factor_contrasts': contrasts}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target = OUT / 'analysis' / (stamp + '.json')
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'snapshot': str(target), 'done': len(rows), 'expected': len(expected), 'complete': snapshot['complete']}))
    return snapshot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    analyze(args.require_complete)
