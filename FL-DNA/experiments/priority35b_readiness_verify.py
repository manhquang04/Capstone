"""Read-only historical hash verification and additive preparation seal; no measurements."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'artifacts/priority35b'
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    frozen = json.loads((out / 'readiness_freeze.json').read_text())
    checked = 0
    for mapping in ('hashes', 'reference_hashes'):
        for relative, expected in frozen[mapping].items():
            assert digest(root / relative) == expected, relative
            checked += 1
    progress = json.loads((out / 'progress.json').read_text())
    assert progress['tests_passed'] == 15
    assert progress['measurement_runs'] == progress['new_real_target_allocations'] == 0
    assert not progress['measurement_authorized']
    snapshot = (out / 'PROJECT_pre_priority35b_preparation.md').read_bytes()
    assert (root.parent / 'PROJECT.md').read_bytes().startswith(snapshot)
    files = [p for p in out.rglob('*') if p.is_file() and 'bytecode' not in p.parts]
    files += list((root / 'experiments').glob('priority35b_*.py'))
    files += list((root / 'tests').glob('test_priority35b*.py'))
    files += [root / 'reports/priority35b_report.md', root / 'protocols/amendments/2026-10-10_priority35b_frontier_protocol.md']
    receipt = {
        'status': 'PREPARATION_VERIFIED_COMPLETE', 'full_priority35b_complete': False,
        'measurements_authorized': False, 'measurement_runs': 0,
        'targets_allocated': 0, 'verified_frozen_hashes': checked,
        'hashes': {str(p.relative_to(root)): digest(p) for p in sorted(set(files))},
        'project_snapshot_sha256': digest(out / 'PROJECT_pre_priority35b_preparation.md'),
        'project_after_administrative_append_sha256': digest(root.parent / 'PROJECT.md'),
    }
    with (out / 'PREPARATION_COMPLETE.json').open('x') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(f'PASS: {checked} frozen hashes; preparation only; no measurements')


if __name__ == '__main__':
    main()
