"""Final closure hash/prefix building blocks, no writes or COMPLETE claims.

The publication caller must first pass independent scientific/CSV verification,
actual figure QA and no-live gates. Private audit receipts are hashed locally;
their contents are never exposed by these functions.
"""
from pathlib import Path
from experiments import priority35b_leakage_integrity as integrity
from experiments import priority35b_leakage_report_guard as guard

REPORT='reports/priority35b_report.md'


def verify_mapping(mapping, hasher):
    """All entries, never selected samples; callback permits synthetic tests."""
    if not mapping:raise ValueError('empty digest inventory')
    for name,digest in mapping.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or not isinstance(digest,str) or len(digest)!=64:
            raise ValueError('invalid manifest entry')
        try:int(digest,16)
        except ValueError:raise ValueError('invalid digest')
        if hasher(name)!=digest:raise ValueError('manifest discrepancy: '+name)
    return len(mapping)


def measurement_inventory(jobs, results):
    """All9984 result documents and their exact evidence bundle paths."""
    expected={tuple(c[k] for k in ('dataset','cell','index','arm')) for c in jobs}
    if len(jobs)!=9984 or len(expected)!=9984 or len(results)!=9984:
        raise ValueError('full scientific inventory required')
    seen=set();paths=set()
    for r in results:
        c=r['config'];key=tuple(c[k] for k in ('dataset','cell','index','arm'))
        if key in seen or key not in expected or r['status']!='COMPLETED':raise ValueError('result identity/inventory')
        seen.add(key)
        if set(r['files'])!=integrity.EXPECTED_FILES:raise ValueError('three evidence bundles required')
        attempt=r['attempt']
        if not attempt.startswith('attempt_') or len(Path(attempt).parts)!=1:raise ValueError('evidence attempt path')
        base=Path('artifacts/priority35b/b3/jobs')/c['dataset']/c['cell']/('%03d'%c['index'])/c['arm']
        paths.update(str(base/p) for p in ('result.json','process.json'))
        paths.update(str(base/attempt/p) for p in r['files'])
    if seen!=expected or len(paths)!=49920:raise ValueError('full49920 scientific file paths required')
    return paths


def verify_inputs(root, inputs, original_report):
    """Only the authorized report append changes an original input digest."""
    root=Path(root)
    if REPORT not in inputs or inputs[REPORT]!=guard.PREFIX_SHA:raise ValueError('original report digest required')
    if len(original_report)!=guard.PREFIX_BYTES:raise ValueError('original report snapshot length')
    guard.assert_prefix(original_report)
    current=integrity.child(root,REPORT).read_bytes()
    prefix=guard.verify_append(original_report,current)
    count=verify_mapping({p:h for p,h in inputs.items() if p!=REPORT},lambda p:integrity.sha(integrity.child(root,p)))
    return dict(status='INPUT_HASHES_AND_AUTHORIZED_REPORT_PREFIX_PASS_ONLY',
                unchanged_entries=count,authorized_append=prefix,full_priority35b_complete=False)


def inventory_files(root, directories, excluded):
    """Explicit closed scopes, with exclusions recorded in final manifest."""
    root=Path(root).resolve();files={};excluded=set(excluded)
    for name in directories:
        directory=integrity.child(root,name)
        if not directory.is_dir():raise ValueError('missing output scope')
        for path in sorted(directory.rglob('*')):
            if path.is_symlink():raise ValueError('output symlink')
            if not path.is_file():continue
            relative=str(path.relative_to(root))
            if relative not in excluded:files[relative]=integrity.sha(path)
    if not files:raise ValueError('empty output scopes')
    return files


def verify_closure(root, manifest, required_measurements, original_report):
    """Read-only final hash check, including prefix, not a statistical verifier."""
    if manifest['schema']!='P35B_B3_FULL_CLOSURE_V1':raise ValueError('manifest schema')
    if not set(required_measurements)<=set(manifest['outputs']):raise ValueError('scientific files omitted')
    forbidden=set(required_measurements)&set(manifest['excluded_after_exit_only'])
    if forbidden or REPORT in manifest['excluded_after_exit_only']:raise ValueError('scientific/report exclusions forbidden')
    if REPORT not in manifest['outputs']:raise ValueError('full current report hash required')
    inputs=verify_inputs(root,manifest['inputs'],original_report)
    outputs=verify_mapping(manifest['outputs'],lambda p:integrity.sha(integrity.child(root,p)))
    actual=inventory_files(root,manifest['output_directories'],manifest['excluded_after_exit_only'])
    # Report is explicitly authorized outside artifact output directories.
    actual[REPORT]=integrity.sha(integrity.child(root,REPORT))
    if actual!=manifest['outputs']:raise ValueError('extra/missing/changed output in closed scopes')
    return dict(status='FULL_CLOSURE_HASH_AUDIT_PASS_ONLY',outputs=outputs,inputs=inputs,
                full_priority35b_complete=False,
                remaining='actual post-publication process exit and final completion receipt')
