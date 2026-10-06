"""Approved additive preparation adapter; original scientific files untouched."""
import json
import os
import secrets
import sys
import time
from pathlib import Path


def normalize_keys(value):
    """Normalize only object keys, preserving values and exact comparison."""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if isinstance(key, str):
                normalized = key
            elif type(key) in (int, float):
                normalized = json.loads(json.dumps({key: None})).popitem()[0]
            else:
                raise ValueError('non-JSON comparison key')
            if normalized in result:
                raise ValueError('ambiguous normalized key collision')
            result[normalized] = normalize_keys(child)
        return result
    if isinstance(value, list):
        return [normalize_keys(child) for child in value]
    return value


def compare(left, right):
    if normalize_keys(left) != normalize_keys(right):
        raise ValueError('exact JSON-normalized preflight mismatch')


def prepare():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from experiments import priority34d_image_recovery as r
    from experiments.priority34d_continue import live
    from experiments import priority34d_reservation_audit as reservation
    amendment = root/'protocols/amendments/2026-10-06_priority34d_json_key_preflight_repair.md'
    tests = root/'tests/test_priority34d_json_preflight.py'
    failure = r.BASE/'PREFLIGHT_FAILURE.json'
    repair = r.OUT/'audits/json_key_preflight_repair.json'
    if live():
        raise RuntimeError('live science; no preparation')
    if r.FREEZE.exists():
        raise RuntimeError('existing recovery freeze; no replacement')
    r.u.verify()
    complete = r.read(r.u.BASE/'UTILITY_COMPLETE.json')
    assert complete['freeze_sha256'] == r.sha(r.u.FREEZE)
    assert complete['calibration_sha256'] == r.sha(r.u.BASE/'calibration.json')
    failed = r.read(failure)
    assert failed['science_started'] is False
    assert failed['error'] == "AssertionError('independent selection reload failed')"
    original_failure_hash = r.sha(failure)
    if repair.exists():
        raise RuntimeError('preserved adapter attempt; no automatic retry')
    try:
        calibration = r.read(r.u.BASE/'calibration.json')
        pairs = [tuple(p) for p in calibration['conditional_pairs']]
        replay = r.u.selection(pairs)
        assert len(calibration['matches']) == len(replay) == 16
        for cell in calibration['matches']:
            compare(calibration['matches'][cell], replay[cell])
        excluded, evidence = r.history()
        targets = r.reserve(excluded); targets['evidence'] = evidence
        independent = r.read(r.OUT/'audits/independent_reservation_preflight.json')
        assert independent['stage'] == 'PASS' and independent['pre_freeze']
        assert independent['recovery_source_sha256'] == r.sha(Path(r.__file__))
        independent_excluded, independent_evidence = reservation.independent_history(root)
        assert excluded == independent_excluded
        assert independent['evidence'] == independent_evidence
        compare(targets['groups'], independent['groups'])
        reservation.check_groups(targets['groups'], independent_excluded)
        if (r.BASE/'targets.json').exists():
            raise RuntimeError('partial target reservation; no overwrite')
        receipt = dict(classification='infrastructure-only', stage='PASS', matched_cells=16,
            failed_preflight_sha256=original_failure_hash, failed_preflight=failed,
            calibration_sha256=r.sha(r.u.BASE/'calibration.json'),
            amendment_sha256=r.sha(amendment), adapter_sha256=r.sha(Path(__file__)),
            tests_sha256=r.sha(tests), source_unchanged=True, tolerance='exact',
            target_reservation_sha256=r.sha(r.OUT/'audits/independent_reservation_preflight.json'),
            at=time.time())
        r.write(repair, receipt)
        r.write(r.BASE/'targets.json', targets)
        jobs, missing = r.build_jobs(targets, calibration['matches'])
        private = r.BASE/'private_noise_seeds.json'
        descriptor = os.open(str(private), os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            json.dump({job['id']: secrets.randbits(63) for job in jobs if job['mechanism'] is not None}, stream)
        sources = {Path(r.__file__).resolve(), r.ANNEX, root/'tests/test_priority34d_image_recovery.py',
            Path(__file__).resolve(), amendment, tests,
            root/'protocols/amendments/2026-10-05_priority34d_reservation_independent_preflight.md',
            root/'tests/test_priority34d_reservation_audit.py'}
        for module in list(sys.modules.values()):
            raw = getattr(module, '__file__', None)
            if raw:
                path = Path(raw).resolve()
                if path.is_relative_to(root) and '.venv' not in str(path) and path.suffix == '.py':
                    sources.add(path)
        sources.update((root/'external_defenses/invertinggradients').rglob('*.py'))
        source_map = {str(p.relative_to(root)): r.sha(p) for p in sorted(sources)}
        utility_freeze = r.read(r.u.FREEZE)
        source_map.update(utility_freeze['sources'])
        inputs = dict(utility_freeze['inputs'])
        for path in (r.BASE/'targets.json', private, r.u.FREEZE, r.u.BASE/'UTILITY_COMPLETE.json',
                     r.u.BASE/'calibration.json', failure, repair,
                     r.OUT/'audits/independent_reservation_preflight.json'):
            inputs[str(path.relative_to(root))] = r.sha(path)
        for row in evidence:
            inputs[row['path']] = row['sha256']
        assert r.sha(failure) == original_failure_hash
        assert r.sha(r.u.BASE/'calibration.json') == complete['calibration_sha256']
        r.write(r.FREEZE, dict(at=time.time(), sources=source_map, inputs=inputs,
            jobs=jobs, missing=missing, instrument=dict(r.native.IMAGE_CONFIG), maximum_jobs=1092))
        r.verify()
        print(json.dumps(dict(stage='PASS', matched_cells=16, jobs=len(jobs),
            missing=len(missing), targets=273, freeze_sha256=r.sha(r.FREEZE),
            repair_receipt_sha256=r.sha(repair), original_failure_preserved=True)))
    except Exception as error:
        destination = r.BASE/('PREFLIGHT_ADAPTER_FAILURE_'+str(time.time_ns())+'.json')
        r.write(destination, dict(error=repr(error), at=time.time(), science_started=False,
            original_failure_sha256=original_failure_hash, classification='infrastructure-only adapter attempt'))
        raise


if __name__ == '__main__':
    prepare()
