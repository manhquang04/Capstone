"""Pure receipt contracts for post-exit publication, never a COMPLETE finalizer.

These contracts consume independently verified receipts, not measurement data.
The caller must verify receipt/file hashes and actual process exit separately.
"""
from experiments import priority35b_leakage_report_guard as guard

VISUAL_CHECKS=frozenset(('pdf_render_inspected','png_inspected','single_column_readable',
    'three_shared_scale_panels','all_arms_and_intervals_visible','labels_readable',
    'data_free_lines_visible','different_configuration_caption','no_clipping_or_overlap'))
ASSETS=frozenset(('artifacts/priority35b/figure/frontier.pdf',
    'artifacts/priority35b/figure/frontier.png','artifacts/priority35b/figure/frontier_data.csv',
    'artifacts/priority35b/figure/rendered.png'))


def visual_qa(rendered,qa):
    if rendered.get('status')!='RENDERED_AWAITING_VISUAL_QA':raise ValueError('render receipt required')
    if rendered.get('measurement_repeated') is not False or rendered.get('full_priority35b_complete') is not False:
        raise ValueError('rendering cannot repeat measurements or complete Priority35B')
    assets=rendered.get('outputs',{})
    if set(assets)!=ASSETS or any(not isinstance(v,str) or len(v)!=64 or
        any(c not in '0123456789abcdef' for c in v) for v in assets.values()):
        raise ValueError('exact figure asset hash inventory required')
    if qa.get('status')!='VISUAL_QA_PASS' or qa.get('synthetic') is not False:
        raise ValueError('measured figure visual QA required, not synthetic layout QA')
    if not qa.get('at') or not qa.get('reviewer') or qa.get('outputs')!=assets:
        raise ValueError('visual inspection provenance and asset binding required')
    checks=qa.get('checks',{})
    if set(checks)!=VISUAL_CHECKS or any(v is not True for v in checks.values()):
        raise ValueError('every actual visual check required')
    return True


def publication_checks(scientific,csv,utility,figure_data,rendered,qa,validation,source_inputs,no_live):
    if scientific.get('status')!='SCIENTIFIC_BUNDLES_VERIFIED_PUBLICATION_PENDING':
        raise ValueError('full scientific recomputation required')
    if csv!=dict(status='CSV_RECOMPUTATION_PASS',per_target=9984,frontier=768,holm=144):
        raise ValueError('fixed complete CSV inventory required')
    if utility.get('status')!='PASS' or utility.get('entries')!=4739:
        raise ValueError('immutable utility manifest check required')
    if figure_data.get('status')!='FIGURE_DATA_RECOMPUTATION_PASS' or figure_data.get('rows')!=96:
        raise ValueError('all96 independently checked frontier figure points required')
    visual_qa(rendered,qa)
    if (validation.get('status')!='VALIDATION_PASS' or not validation.get('commands') or
        validation.get('tests_failed')!=0 or not isinstance(validation.get('tests_passed'),int) or
        validation['tests_passed']<=0 or validation.get('compile')!='PASS' or
        validation.get('git_diff_check')!='PASS' or validation.get('measurement_repeated') is not False):
        raise ValueError('actual test/compile/diff receipts required')
    if source_inputs is not True or no_live is not True:
        raise ValueError('actual source/input and no-live checks required')
    checks={name:True for name in guard.REQUIRED}
    guard.require_publication_checks(checks)
    return checks


def final_receipt_checks(receipt,report_sha,manifest_sha):
    """No writing; prevent a publication receipt masquerading as completion."""
    if (receipt.get('status')!='PRIORITY35B_VERIFIED_COMPLETE' or
        receipt.get('report_sha256')!=report_sha or receipt.get('manifest_sha256')!=manifest_sha or
        receipt.get('no_live_workers') is not True or receipt.get('postexit_seal') is not True or
        receipt.get('measurement_repeated') is not False):
        raise ValueError('post-exit full closure receipt required')
    for digest in (report_sha,manifest_sha):
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('SHA256 completion binding')
    return True
