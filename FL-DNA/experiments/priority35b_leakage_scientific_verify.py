"""Post-exit B3 scientific verification coordinator, not publication finalizer.

No writes, training, target replacement or optimization. Caller preserves the
returned audit and any failure traceback outside immutable measurement evidence.
"""
from pathlib import Path
from experiments import priority35b_leakage_integrity as integrity
from experiments import priority35b_leakage_postexit as postexit

SEALS = ('statistics_audit_freeze.json','integrity_audit_freeze.json',
         'metrics_audit_freeze.json','payload_audit_freeze.json',
         'postexit_integration_freeze.json','objective_audit_freeze.json',
         'completion_audit_freeze.json')


def verify_additive_sources(root, seal_names=SEALS):
    root=Path(root);out=root/'artifacts/priority35b/b3'
    sources={}
    for name in seal_names:
        seal=integrity.read(out/name)
        if not seal.get('sources'):raise ValueError('empty audit source seal')
        for path,digest in seal['sources'].items():
            if path in sources and sources[path]!=digest:raise ValueError('conflicting audit source seal')
            if integrity.sha(integrity.child(root,path))!=digest:raise ValueError('additive audit source drift: '+path)
            sources[path]=digest
    return dict(source_count=len(sources),seal_count=len(seal_names),status='PASS')


def objective_sweep(driver, audit_saved):
    """Every frozen job audited; cache only the eight immutable data adapters."""
    freeze=driver.read(driver.FREEZE);jobs=freeze['jobs']
    if len(jobs)!=9984:raise ValueError('exact scientific inventory required')
    cache={};counts=dict(ratio_jobs=0,native_jobs=0,variants=0)
    for config in jobs:
        key=(config['dataset'],config['cell'])
        if key not in cache:cache[key]=driver.core.PreparedDataset(*key)
        receipt=audit_saved(driver,config,cache[key])
        if receipt['optimization_repeated'] is not False:raise ValueError('audit must not repeat optimization')
        counts['ratio_jobs' if receipt['ratio'] else 'native_jobs']+=1
        counts['variants']+=receipt['variants']
    if (counts['ratio_jobs'],counts['native_jobs'])!=(3744,6240):raise ValueError('ratio/native inventory required')
    return counts


def verify(root):
    """Only after measurements AND every recorded supervisor/worker have exited."""
    root=Path(root);out=root/'artifacts/priority35b/b3'
    # Before importing scientific helpers or reading ANY protected metric.
    postexit.require_exit(postexit.process_snapshot(),postexit.recorded_pids(out))
    postexit.port_free(43576)
    sources=verify_additive_sources(root)
    evidence=postexit.verify_all(root)
    from experiments import priority35b_leakage as driver
    from experiments import priority35b_leakage_completion_audit as complete
    driver.configure()
    counts=objective_sweep(driver,complete.audit_saved_objectives)
    # Recheck originals and audit sources AFTER the potentially lengthy sweep.
    driver.verify();verify_additive_sources(root)
    postexit.require_exit(postexit.process_snapshot(),postexit.recorded_pids(out));postexit.port_free(43576)
    return dict(status='SCIENTIFIC_BUNDLES_VERIFIED_PUBLICATION_PENDING',
                evidence=evidence,objective_and_pooling=counts,additive_sources=sources,
                full_priority35b_complete=False,
                required_before_complete=['statistics CSV independent recomputation','figure data and visual QA',
                                         'report prefix/hash assertion','full manifests and final no-live seal'])
