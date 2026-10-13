"""Pure append-only publication guards; no file writes or completion claims.

The publication caller must supply completed independent verification receipts.
These guards do not perform or replace those checks. Historical report bytes are
preserved exactly; completion is a separate final manifest/no-live operation.
"""
import hashlib

PREFIX_SHA = 'f168dbbc9207c0d4f5087e99b464b88adc3b5eae76d25b616dfab1eaaa66a633'
PREFIX_BYTES = 4793
SECTION = '## Priority 35B B3 — verified frontier results'
REQUIRED = frozenset(('scientific_bundles', 'csv_recomputation', 'utility_hashes',
                     'source_input_hashes', 'tests', 'compile', 'diff_check',
                     'figure_data', 'figure_visual_qa', 'no_live_workers'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def assert_prefix(data, digest=PREFIX_SHA, length=PREFIX_BYTES):
    if not isinstance(data, bytes) or len(data) < length or sha(data[:length]) != digest:
        raise ValueError('historical report prefix changed')
    return dict(prefix_sha256=digest, prefix_bytes=length)


def require_publication_checks(checks):
    if set(checks) != REQUIRED or any(value is not True for value in checks.values()):
        raise ValueError('publication requires every independent check; no partial completion')


def prepare_append(original, section, checks, digest=PREFIX_SHA, length=PREFIX_BYTES):
    """Return proposed bytes only, after strict preconditions; never overwrite."""
    require_publication_checks(checks)
    assert_prefix(original, digest, length)
    if len(original) != length:
        raise ValueError('report already appended; do not replace or duplicate text')
    if not isinstance(section, str) or not section.startswith(SECTION + '\n'):
        raise ValueError('new results section heading required')
    if SECTION in section[len(SECTION):]:
        raise ValueError('duplicate results heading')
    appended = original + b'\n\n' + section.encode('utf-8')
    assert_prefix(appended, digest, length)
    return appended


def verify_append(original, published, digest=PREFIX_SHA, length=PREFIX_BYTES):
    assert_prefix(original, digest, length)
    assert_prefix(published, digest, length)
    if len(original) != length or not published.startswith(original + b'\n\n' + SECTION.encode() + b'\n'):
        raise ValueError('append-only report structure')
    return dict(status='REPORT_PREFIX_ASSERTION_PASS_ONLY',
                prefix_sha256=digest, prefix_bytes=length,
                report_sha256=sha(published), full_priority35b_complete=False)
