"""Read-only B3 evidence integrity building block, not a COMPLETE finalizer.

Call only after all recorded measurement processes exit. Statistical, mechanism,
metric and figure verification remain separate requirements.
"""
import hashlib
import json
import math
import os
from pathlib import Path

EXPECTED_FILES = {'server_receipt.pt', 'private_truth.pt', 'reconstruction.pt'}
FREEZE_SHA = '51feaf5bbfb77e241c310e7fefe2c7782654fff1521b1e91c168fc08888d3eea'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def child(parent, name):
    """Reject escaping paths and symlinks, including dangling symlinks."""
    parent = Path(parent).resolve()
    candidate = parent / name
    if Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('escaping evidence path')
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(parent):
        raise ValueError('symlink or escaping evidence path')
    return candidate


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('non-finite serialized evidence')
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_tree(item)


def verify_result(directory, config, freeze_sha, target):
    directory = Path(directory)
    if list(directory.glob('attempt_*/failure.json')):
        raise ValueError('preserved failed measurement attempt')
    result = read(directory / 'result.json')
    finite_tree(result)
    if result['status'] != 'COMPLETED' or result['config'] != config:
        raise ValueError('measurement identity')
    if result['freeze_sha256'] != freeze_sha:
        raise ValueError('measurement freeze identity')
    if result['threads'] != 1 or result['inter_threads'] != 1:
        raise ValueError('one-thread contract')
    if result['source_ids'] != target['source_ids']:
        raise ValueError('source identity')
    if set(result['files']) != EXPECTED_FILES:
        raise ValueError('exact evidence bundle inventory')
    attempt = child(directory, result['attempt'])
    if not attempt.is_dir() or not result['attempt'].startswith('attempt_'):
        raise ValueError('attempt directory')
    if {p.name for p in attempt.glob('*.pt')} != EXPECTED_FILES:
        raise ValueError('unexpected or missing tensor bundle')
    for name, digest in result['files'].items():
        if sha(child(attempt, name)) != digest:
            raise ValueError('tensor bundle hash drift: ' + name)
    score = result['score']
    if not 0 <= score['feature_accuracy'] <= 1:
        raise ValueError('feature accuracy range')
    if score['standardized_mse'] < 0 or not 0 <= score['bounded_mse'] <= score['standardized_mse']:
        raise ValueError('MSE bounds')
    batch = len(target['source_ids'])
    if sorted(score['assignment']) != list(range(batch)):
        raise ValueError('assignment permutation')
    return result


def verify_targets(targets, exclusions):
    if len(targets) != 312 or sum(len(t['source_ids']) for t in targets) != 390:
        raise ValueError('312 targets / 390 records required')
    seen = set()
    identities = set()
    for target in targets:
        identity = (target['dataset'], target['cell'], target['index'])
        if identity in identities or target['index'] not in range(39):
            raise ValueError('duplicate or invalid target identity')
        identities.add(identity)
        ids = target['source_ids']
        batch = 2 if target['cell'].endswith('batch2') else 1
        if len(ids) != batch or len(target['indices']) != batch:
            raise ValueError('target batch cardinality')
        for source in ids:
            item = (target['dataset'], source)
            if item in seen or source in exclusions[target['dataset']]:
                raise ValueError('source-ID firewall')
            seen.add(item)
    return {tuple(t[k] for k in ('dataset', 'cell', 'index')): t for t in targets}


def verify_evidence(root, exited_pids):
    """Requires external no-live audit; never runs measurements or writes files."""
    root = Path(root)
    out = root / 'artifacts/priority35b/b3'
    freeze = out / 'execution_freeze.json'
    if sha(freeze) != FREEZE_SHA:
        raise ValueError('execution freeze digest')
    for name in ('REQUIRES_DIRECTION.json', 'SUPERVISOR_FAILURE.json'):
        if (out / name).exists():
            raise ValueError('preserved failure: ' + name)
    if list((out / 'jobs').glob('**/failure.json')):
        raise ValueError('preserved measurement failure')
    recorded = {read(out / 'launch.json')['pid'], read(out / 'supervisor.json')['pid']}
    recorded.update(read(p)['pid'] for p in (out / 'jobs').glob('**/process.json'))
    if not recorded <= set(exited_pids):
        raise ValueError('external post-exit audit does not cover every recorded PID')
    if read(root / 'artifacts/priority35b/progress.json')['stage'] != 'B3_MEASUREMENTS_FINISHED':
        raise ValueError('measurements not finished')
    doc = read(freeze)
    for section in ('sources', 'inputs'):
        for name, digest in doc[section].items():
            if sha(child(root, name)) != digest:
                raise ValueError('frozen input/source drift: ' + name)
    if os.name != 'nt' and (out / 'private_noise.json').stat().st_mode & 0o077:
        raise ValueError('private receipt permission')
    targets = verify_targets(read(out / 'targets.json')['targets'], read(out / 'historical_firewall.json')['exclusions'])
    jobs = doc['jobs']
    expected = {tuple(c[k] for k in ('dataset', 'cell', 'index', 'arm')) for c in jobs}
    if len(jobs) != 9984 or len(expected) != 9984:
        raise ValueError('exact 9984 unique jobs required')
    if len(list((out / 'jobs').glob('**/result.json'))) != 9984:
        raise ValueError('exact result inventory required')
    for c in jobs:
        directory = out / 'jobs' / c['dataset'] / c['cell'] / ('%03d' % c['index']) / c['arm']
        verify_result(directory, c, FREEZE_SHA, targets[tuple(c[k] for k in ('dataset', 'cell', 'index'))])
    return dict(status='EVIDENCE_INTEGRITY_PASS_ONLY',jobs=9984,targets=312,records=390,
                remaining='mechanisms, tensor contents, independent metrics/statistics, figure/report, final seal')
