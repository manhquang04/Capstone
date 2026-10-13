"""Pure report-section rendering from already independently verified tables.

No result reads, arithmetic inference, figure creation, writes or completion.
The publication caller supplies the actual commands and software provenance.
"""
import json
import math
from experiments import priority35b_leakage_report_guard as guard


def cell(value):
    if value is None:return 'NA'
    if isinstance(value, float):
        if not math.isfinite(value):raise ValueError('non-finite report value')
        return format(value, '.17g')
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, sort_keys=True, allow_nan=False)
    return str(value).replace('|', '&#124;').replace('\n', ' ')


def table(rows, fields):
    if not rows or not fields:raise ValueError('missing report table')
    out = ['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---'] * len(fields)) + ' |']
    for row in rows:
        if not set(fields) <= set(row):raise ValueError('missing report field')
        out.append('| ' + ' | '.join(cell(row[f]) for f in fields) + ' |')
    return '\n'.join(out)


def section(bundle, checks):
    """Render only completed evidence. Unit fixtures never publish this output."""
    guard.require_publication_checks(checks)
    required = {'commands', 'amendment_hashes', 'software', 'utility', 'decomposition',
                'matching', 'frontier', 'holm', 'selection', 'figure_png', 'figure_pdf',
                'figure_csv', 'verification'}
    if set(bundle) != required:raise ValueError('complete report inputs required')
    if not bundle['commands'] or not bundle['amendment_hashes'] or not bundle['software']:
        raise ValueError('missing actual provenance; never invent commands or versions')
    if len(bundle['frontier']) != 768 or len(bundle['holm']) != 144 or len(bundle['matching']) != 12:
        raise ValueError('frozen frontier/family/matching inventory')
    if len(bundle['utility']) != 288 or len(bundle['decomposition']) != 81 or len(bundle['selection']) != 3:
        raise ValueError('frozen utility/decomposition/selection inventory')
    v = bundle['verification']
    if v.get('scientific_jobs') != 9984 or v.get('controls') != 312 or v.get('csv_recomputation') != 'PASS':
        raise ValueError('independent scientific and CSV receipts required')
    chunks = [guard.SECTION, '',
        'This section is appended to the unchanged historical readiness report. '
        'B2b adds post-hoc clip norms and arms; it does not replace B2 results, matching rules or the Holm family.',
        '', '### Design, experimental units and statistics', '',
        'Utility uses Config-C local-BN Adam training: 50 rounds, K=3, 11 paired seeds. '
        'The client-average endpoint is computed within each seed; clients and rounds are not additional independent replicates. '
        'Utility deltas are arm minus baseline with descriptive two-sided 95% Student-t intervals (df=10). '
        'Leakage uses the different Priority 34C single-gradient configuration with 39 fresh row sets paired across all arms. '
        'The batch, not each feature or ensemble member, is the comparison unit. '
        'Leakage medians and paired differences use order-statistic ranks 13/27. '
        'Feature accuracy is higher for more recovery; standardized MSE and bounded min(MSE, MSE_prior) are lower for more recovery.',
        '', 'Exact one-sided sign tests are reported in both directions on transform-minus-comparator differences; '
        'ties are excluded from the effective sign-test n. The fixed Holm family has 144 entries, including '
        'NOT_ASSESSABLE reservations with p=1. No family shrinkage or outcome-dependent exclusions are used. '
        'Failure to reject is not evidence of equivalence or privacy.',
        '', '### Exact execution and provenance', '', '```text',
        '\n'.join(bundle['commands']), '```', '',
        table(bundle['amendment_hashes'], ('path', 'sha256')), '',
        'Software: ' + cell(bundle['software']), '',
        '### Complete utility results', '',
        table(bundle['utility'], tuple(bundle['utility'][0])), '',
        '### Clipping/noise decomposition (post hoc B2b)', '',
        'Decomposition contrasts are descriptive; they do not establish causality or introduce new tests. '
        'Clip-only arms have NO DP guarantee.', '',
        table(bundle['decomposition'], tuple(bundle['decomposition'][0])), '',
        '### Frozen matched-utility decisions', '',
        'Largest-noise passing point requires the next noisier point to fail the frozen mean-F1 threshold. '
        'MATCHED_CLIP_ONLY_NO_DP means a clip-only comparator, not a DP match. NOT_ASSESSABLE receives no leakage test.', '',
        table(bundle['matching'], tuple(bundle['matching'][0])), '',
        '### Complete leakage frontier (all qualified estimators)', '',
        table(bundle['frontier'], ('dataset', 'cell', 'arm', 'metric', 'n', 'median', 'interval', 'ranks')), '',
        '### Matched tests and Holm correction', '',
        'The summary contains median paired differences, ranks 13/27 intervals, lower/higher wins, ties and effective n. '
        'For feature accuracy, transform_lower favors lower recovery for the transform; for bounded MSE, '
        'transform_higher favors lower recovery. Both directions are retained.', '',
        table(bundle['holm'], ('dataset', 'cell', 'transform', 'clipping', 'metric', 'direction',
                               'status', 'summary', 'raw_p', 'holm_p', 'reason')), '',
        '### Figure and plain reading', '',
        'Panel estimator selection uses only all 39 unprotected targets: highest median feature accuracy, '
        'fixed cell-order tie break, before protected measurements are submitted.', '',
        table(bundle['selection'], tuple(bundle['selection'][0])), '',
        '![' + 'Privacy-utility frontier' + '](' + bundle['figure_png'] + ')', '',
        '[PDF](' + bundle['figure_pdf'] + '); [exact figure data CSV](' + bundle['figure_csv'] + ').', '',
        'Caption: x is utility loss (negative mean paired Delta-F1) from Config-C training, with 95% t intervals '
        'over 11 seeds; y is median feature accuracy from the Priority 34C single-gradient setting, with ranks '
        '13/27 over 39 row sets. These axes come from different configurations, not a single deployable-system experiment. '
        'Data-free lines are mean/mode and empirical marginal. DP joint lines are separate by clip norm; '
        'per-tensor DP and clip-only points are separate. Lower y means less recovery by the plotted estimator.', '',
        'Read the transforms relative to both utility error bars and leakage intervals, then consult the complete '
        'paired Holm table for inferential claims. Overlapping bars alone neither establish equivalence nor '
        'invalidate a paired contrast. A clip-only match cannot support superiority to formal DP at matched utility.', '',
        '### Limitations and independent verification', '',
        'This is a server-side analysis with seed access for v2. It is not metadata-free seed identification '
        'and does not establish a formal privacy guarantee for the transforms. DP accounting is replace-one '
        'client-upload accounting (50 full-participation releases, delta=1e-5), not a claimed record-level DP guarantee. '
        'P34C estimator qualification concerns its single-gradient setting, not realistic Config-C Adam uploads. '
        'The frontier is limited to the frozen grids, budgets and models; selection of the plotted estimator does '
        'not replace reporting every qualified estimator. Utility reuse and post-hoc B2b scope are explicit.', '',
        'Independent checks recompute scientific payloads, gradients, clipping/noise receipts, metrics, controls, '
        'saved objectives, selection/pooling, source-ID firewall, CSV medians/intervals/sign tests and Holm. '
        'The categorical decoder and transform/least-squares helper replays remain shared-source dependencies; '
        'their source hashes are audited, and they are not described as independent implementations. '
        'No optimization or measurement is repeated during verification.', '',
        'Verification receipt: ' + cell(v), '',
        'This rendered section alone is not a COMPLETE seal: the final full manifest, report-prefix assertion '
        'and fresh post-publication no-live audit are additionally required.', '']
    return '\n'.join(chunks)
