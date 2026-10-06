"""Pre-freeze independent history/reservation check; no outcome arrays accessed."""
import hashlib
import json
import random
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIELDS = frozenset(('indices', 'target_indices', 'image_source_ids', 'cifar10_index',
    'decoy_cifar10_index', 'excluded', 'C1', 'C2_batch4', 'C2_trained',
    'development', 'targets'))
SETTINGS = ('init342042', 'init342043', 'init342044', 'trained_batch4')


def ids(document):
    found = set()
    def visit(value, selected=False):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, key in FIELDS)
        elif isinstance(value, list):
            for child in value:
                visit(child, selected)
        elif selected and type(value) is int and 0 <= value < 10000:
            found.add(value)
    visit(document)
    return found


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_history(root):
    excluded = set()
    evidence = {}
    for area in ('artifacts', 'results'):
        for path in sorted((root / area).rglob('*.json')):
            name = path.relative_to(root).as_posix()
            if name.startswith('artifacts/priority34d/') or path.name == 'split.json':
                continue
            if not any(tag in name.lower() for tag in ('image', 'cifar', 'priority29', 'priority30')):
                continue
            found = ids(json.loads(path.read_text()))
            if found:
                excluded |= found
                evidence[name] = dict(sha256=digest(path), count=len(found))
    for seed, count in ((29000801, 1000), (30400, 3100)):
        population = list(range(10000))
        random.Random(seed).shuffle(population)
        excluded.update(population[:count])
    for name in ('artifacts/priority33c/image_targets.json', 'artifacts/priority34b/image_targets.json'):
        path = root / name
        found = ids(json.loads(path.read_text()))
        if not found:
            raise ValueError('mandatory historical manifest not parsed: ' + name)
        excluded |= found
        evidence[name] = dict(sha256=digest(path), count=len(found))
    return excluded, evidence


def check_groups(groups, excluded):
    if set(groups) != set(SETTINGS):
        raise ValueError('reservation settings mismatch')
    flattened = []
    for setting in SETTINGS:
        rows = groups[setting]
        width = 4 if setting == 'trained_batch4' else 1
        if len(rows) != 39 or any(len(row) != width for row in rows):
            raise ValueError('reservation paired-unit shape mismatch')
        flattened.extend(item for row in rows for item in row)
    if any(type(item) is not int or not 0 <= item < 10000 for item in flattened):
        raise ValueError('invalid record ID')
    if len(set(flattened)) != 273 or set(flattened) & set(excluded):
        raise ValueError('duplicate or historical record ID')
    order = np.random.default_rng(342600).permutation(sorted(set(range(10000)) - set(excluded)))
    if flattened != order[:273].tolist():
        raise ValueError('reservation seed/order mismatch')
    return flattened


def audit():
    sys.path.insert(0, str(ROOT))
    from experiments import priority34d_image_recovery as driver
    from experiments.priority34d_continue import live
    if live() or driver.FREEZE.exists():
        raise RuntimeError('pre-freeze audit requires no live science and no recovery freeze')
    excluded, evidence = independent_history(ROOT)
    actual, history = driver.history()
    if actual != excluded or {row['path']: dict(sha256=row['sha256'], count=row['count']) for row in history} != evidence:
        raise ValueError('independent historical exclusion mismatch')
    candidate = driver.reserve(actual)
    check_groups(candidate['groups'], excluded)
    public = dict(stage='PASS', pre_freeze=True, total_records=273,
        excluded_count=len(excluded), groups=candidate['groups'], evidence=evidence,
        auditor_sha256=digest(Path(__file__)),
        recovery_source_sha256=digest(Path(driver.__file__)))
    destination = driver.OUT / 'audits/independent_reservation_preflight.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if json.loads(destination.read_text()) != public:
            raise ValueError('preserved reservation audit differs')
    else:
        destination.write_text(json.dumps(public, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(stage='PASS', total_records=273, excluded_count=len(excluded),
        receipt_sha256=digest(destination))))


if __name__ == '__main__':
    audit()
