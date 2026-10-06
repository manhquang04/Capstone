"""P33c paired effects, fixed families, independent verification and full P33 summary."""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from skimage.metrics import structural_similarity

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / 'artifacts/priority33c'
REPORT = ROOT / 'reports/priority33c_report.md'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, allow_nan=False)+'\n')


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def sign(wins, losses):
    n = wins+losses
    p = sum(math.comb(n, k) for k in range(wins, n+1))/2**n if n else 1.
    independent = float(binomtest(wins, n, alternative='greater').pvalue) if n else 1.
    assert abs(p-independent) < 1e-14
    return p


def holm(rows, field):
    order = sorted(range(len(rows)), key=lambda i: rows[i]['p_raw'])
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(rows)-rank)*rows[i]['p_raw']))
        rows[i][field] = running
    # Independent vectorized implementation, no reuse of iterative result.
    p = np.asarray([rows[i]['p_raw'] for i in order])
    adjusted = np.minimum(1., np.maximum.accumulate(p*np.arange(len(rows), 0, -1)))
    assert np.allclose(adjusted, [rows[i][field] for i in order], atol=1e-15, rtol=0)


def effects(identifier, rows, quality, status='VALID', reason=None):
    if status != 'VALID':
        return [dict(hypothesis_id=identifier+'::'+direction, status=status, reason=reason,
                     direction=direction, p_raw=1.) for direction in ['DNA_stronger', 'comparator_stronger']]
    assert len(rows) == 39
    delta = np.asarray([r['dna']-r['comparator'] for r in rows])
    dna_wins = int((delta < 0).sum()) if quality == 'PSNR' else int((delta > 0).sum())
    losses = int((delta > 0).sum()) if quality == 'PSNR' else int((delta < 0).sum())
    ordered = np.sort(delta)
    data = dict(status='VALID', n=39, quality=quality,
        median_dna=float(np.median([r['dna'] for r in rows])),
        median_comparator=float(np.median([r['comparator'] for r in rows])),
        median_paired_difference=float(np.median(delta)), rank13=float(ordered[12]), rank27=float(ordered[26]),
        ties=39-dna_wins-losses)
    if quality == 'PSNR':
        data.update(median_dna_ssim=float(np.median([r['dna_ssim'] for r in rows])),
            median_comparator_ssim=float(np.median([r['comparator_ssim'] for r in rows])),
            median_gray=float(np.median([r['gray'] for r in rows])),
            median_cifar_mean=float(np.median([r['cifar_mean'] for r in rows])))
        data['both_at_reference_level'] = data['median_dna'] <= max(data['median_gray'], data['median_cifar_mean']) and data['median_comparator'] <= max(data['median_gray'], data['median_cifar_mean'])
    result = []
    for direction, w, l in [('DNA_stronger', dna_wins, losses), ('comparator_stronger', losses, dna_wins)]:
        result.append(dict(hypothesis_id=identifier+'::'+direction, direction=direction, wins=w, losses=l,
                           p_raw=sign(w, l), **data))
    csv_write(OUT / (identifier.replace('::', '_')+'_paired.csv'), rows)
    return result


def image_rows(setting, method, comparator):
    result = []
    for i in range(39):
        a = read(OUT / 'image_recovery' / setting / method / f'target_{i:03d}/result.json')
        b = read(OUT / 'image_recovery' / setting / comparator / f'target_{i:03d}/result.json')
        assert a['status'] == b['status'] == 'passed'
        assert a['job']['indices'] == b['job']['indices']
        result.append(dict(target=i, source_ids=a['job']['indices'], dna=a['metrics']['psnr_db'],
            comparator=b['metrics']['psnr_db'], dna_ssim=a['metrics']['ssim'],
            comparator_ssim=b['metrics']['ssim'], gray=a['references']['gray']['psnr_db'],
            cifar_mean=a['references']['cifar_mean']['psnr_db']))
    return result


def verify_arrays():
    count = 0
    for path in (OUT / 'image_recovery').rglob('result.json'):
        doc = read(path)
        assert sha(path.parent / 'arrays.npz') == doc['arrays_sha256']
        arrays = np.load(path.parent / 'arrays.npz')
        a, b = np.clip(arrays['truth'], 0, 1), np.clip(arrays['reconstruction'], 0, 1)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        psnr, ssim = [], []
        for x, y in zip(a, b):
            mse = float(np.mean((x-y)**2))
            psnr.append(float(-10*np.log10(mse)))
            ssim.append(float(structural_similarity(x.transpose(1,2,0), y.transpose(1,2,0), channel_axis=2, data_range=1.)))
        assert abs(np.mean(psnr)-doc['metrics']['psnr_db']) < 1e-6
        assert abs(np.mean(ssim)-doc['metrics']['ssim']) < 1e-6
        count += 1
    return count


def table(rows, fields):
    lines = ['| '+' | '.join(fields)+' |', '| '+' | '.join(['---']*len(fields))+' |']
    for row in rows:
        values = []
        for key in fields:
            value = row.get(key, '—')
            values.append(f'{value:.9g}' if isinstance(value, float) else str(value).replace('|', '/'))
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def main():
    assert (OUT / 'IMAGE_RECOVERY_COMPLETE.json').exists()
    assert (OUT / 'PAYSIM_BN_COMPLETE.json').exists()
    calibration = read(OUT / 'image_calibration.json')
    tests = []
    for method in ['dna_v1_medium', 'dna_v1_stronger']:
        for comparator in ['unprotected', 'distortion', 'single']:
            identifier = 'C1::'+method+'::'+comparator
            arm = 'unprotected' if comparator == 'unprotected' else 'dp_'+comparator+'_for_'+method
            match = calibration['matches'][method+'::single']
            if comparator == 'single' and (not calibration['baseline_passed'] or not match['bracketed']):
                tests += effects(identifier, [], 'PSNR', 'NOT_ASSESSABLE', json.dumps(match))
            else:
                tests += effects(identifier, image_rows('C1', method, arm), 'PSNR')
    for method in ['dna_v1_conservative', 'dna_v2_0p95']:
        identifier = 'D::image::'+method+'::per_tensor'
        match = calibration['matches'][method+'::per_tensor']
        if not calibration['baseline_passed'] or not match['bracketed']:
            tests += effects(identifier, [], 'PSNR', 'NOT_ASSESSABLE', json.dumps(match))
        else:
            tests += effects(identifier, image_rows('D', method, 'dp_per_tensor_for_'+method), 'PSNR')
        identifier = 'D::PaySim::'+method+'::per_tensor'
        bn = read(OUT / 'PAYSIM_BN_COMPLETE.json')
        if bn['status'] == 'NOT_ASSESSABLE':
            tests += effects(identifier, [], 'standardized_MSE', 'NOT_ASSESSABLE', bn['reason'])
        else:
            paths = [OUT / 'paysim_bn' / method / f'target_{i:03d}/result.json' for i in range(39)]
            docs = [read(p) for p in paths] if all(p.exists() for p in paths) else []
            if len(docs) != 39 or any(d['status'] != 'passed' for d in docs):
                tests += effects(identifier, [], 'standardized_MSE', 'NOT_ASSESSABLE', 'PaySim utility or per-target numerical gate failed; no target exclusions')
            else:
                rows = [dict(target=i, source_ids=d['source_ids'], dna=d['dna_mse'], comparator=d['dp_mse']) for i, d in enumerate(docs)]
                tests += effects(identifier, rows, 'standardized_MSE')
    assert len(tests) == 20
    holm(tests, 'holm_p33c')
    prior_path = ROOT / 'artifacts/priority31_image_utility_dp/expanded_holm.csv'
    with prior_path.open() as f:
        prior = [dict(hypothesis_id=r['hypothesis_id'], p_raw=float(r['p_value']), origin=r['origin']) for r in csv.DictReader(f)]
    assert len(prior) == 115
    combined = prior + [dict(r) for r in tests]
    holm(combined, 'holm_combined135')
    lookup = {r['hypothesis_id']: r['holm_combined135'] for r in combined}
    for row in tests:
        row['holm_combined135'] = lookup[row['hypothesis_id']]
    csv_write(OUT / 'paired_tests.csv', tests)
    csv_write(OUT / 'combined135_holm.csv', combined)
    n_verified = verify_arrays()
    sensitivity = []
    for setting in ['C2_batch4', 'C2_trained']:
        for folder in (OUT / 'image_recovery' / setting).iterdir():
            docs = [read(p) for p in folder.glob('*/result.json')]
            assert len(docs) == 24
            quality = np.array([d['metrics']['psnr_db'] for d in docs])
            sensitivity.append(dict(setting=setting, arm=folder.name, n=24,
                median_psnr=float(np.median(quality)), median_ssim=float(np.median([d['metrics']['ssim'] for d in docs])),
                psnr_min=float(quality.min()), psnr_max=float(quality.max()),
                utility_transfer='P31 untrained/batch256 development calibration; descriptive sensitivity'))
    csv_write(OUT / 'sensitivity.csv', sensitivity)
    all_p33 = []
    earlier = read(ROOT / 'artifacts/priority33a/independent_statistics.json')['tests']
    for r in earlier:
        if r['direction'] == 'DNA_better':
            all_p33.append(dict(priority='33a', dataset=r['dataset'], method=r['method'], comparator=r['comparator'],
                status=r['status'], metric='batch_mean_standardized_MSE', median_dna=r.get('median_dna_mse'),
                median_comparator=r.get('median_dp_mse'), delta=r.get('median_paired_difference'),
                interval=r.get('interval_rank13_27'), reason=r.get('reason')))
    earlier = read(ROOT / 'artifacts/priority33b/attempt2/independent_statistics.json')['tests']
    for r in earlier:
        if r['direction'] == 'DNA_stronger_protection':
            all_p33.append(dict(priority='33b', dataset=r['dataset'], method=r['method'], comparator=r['comparator'],
                status=r['status'], metric='record_feature_accuracy', reason=r.get('reason')))
    for r in tests:
        if r['direction'] == 'DNA_stronger':
            all_p33.append(dict(priority='33c', dataset='PaySim' if 'PaySim' in r['hypothesis_id'] else 'CIFAR10',
                method=r['hypothesis_id'], comparator=r['hypothesis_id'].split('::')[-2], status=r['status'],
                metric=r.get('quality'), median_dna=r.get('median_dna'), median_comparator=r.get('median_comparator'),
                delta=r.get('median_paired_difference'), interval=[r.get('rank13'), r.get('rank27')], reason=r.get('reason')))
    csv_write(OUT / 'all_priority33_summary.csv', all_p33)
    save(OUT / 'independent_statistics.json', dict(agreement=True, family_size=20,
        combined_family_size=135, verified_reconstruction_arrays=n_verified,
        existing115_source_sha256=sha(prior_path), tests=tests))
    lines = ['# Priority33c — CIFAR recovery coverage and per-tensor DP', '',
        'Scientific stages resolved; final checks and supervisor receipt determine completion.', '',
        'Earlier artifacts unchanged. CPU training/thread1; native image loops4800iterations/TV.01. '
        'No nonfinite/negative BN clamp or seed exclusion. Failed gates are NOT_ASSESSABLE, not privacy evidence.', '',
        'Protocols: 2026-10-03_priority33c_image_coverage.md; 2026-10-03_priority33c_execution_annex.md. '
        'Commands, launches, errors and resume skips: artifacts/priority33c/runs.jsonl and supervisor.jsonl.', '',
        'P31 checkpoint replay exactly reproduced validation46.1667% and test46.29%, seed51016; '
        'no checkpoint or learning-rate selection on test. C2 is two separate sensitivities, not a joint factorial.', '',
        'Per-tensor Gaussian std=sigma*C_l, joint update-level whitened sensitivity sqrt(L). '
        'This differs from P27 isotropic noise; epsilon is one release/delta1e-5, not record-level accounting.', '',
        '## Calibration', '', '```json', json.dumps(calibration, indent=2), '```', '',
        'PaySim full-state CPU utility (validation-F1 only):', '', '```json',
        json.dumps(read(OUT / 'paysim_calibration.json'), indent=2), '```', '',
        '## Paired primary tests', '',
        'PSNR DNA−comparator <0 favors DNA protection; BN-MSE difference >0 favors DNA. '
        'Intervals are one-based order ranks13/27 of39, not differences of marginal medians. '
        'NA hypotheses reserve p1 in fixed families20/135, no equivalence interpretation.', '',
        table(tests, ['hypothesis_id','status','wins','losses','ties','p_raw','holm_p33c','holm_combined135',
            'median_dna','median_comparator','median_dna_ssim','median_comparator_ssim','median_paired_difference','rank13','rank27']), '',
        '## C2 descriptive sensitivity', '', table(sensitivity, ['setting','arm','n','median_psnr','median_ssim','psnr_min','psnr_max']), '',
        'No confirmatory p-values or n39 intervals are attached to n24 sensitivity.', '',
        '## All Priority33 evidence', '',
        table(all_p33, ['priority','dataset','method','comparator','status','metric','median_dna','median_comparator','delta','interval','reason']), '',
        'BN recovery is a four-record batch mean, not individual records. TabLeak33b is single-gradient '
        'batch8, not multi-step Adam. Its failed two-reference qualification leaves BAF/IEEE record cells '
        'not assessable. Adult has no new P33 arm; historical results remain in existing115.', '',
        '## Independent checks and disclosures', '',
        f'SciPy sign tests and independent vectorized Holm agree; {n_verified} saved reconstruction arrays reloaded. '
        'Native payload identity/unknown-mode/identity-loss and recovery tests passed before attacks. '
        'P31 stored no checkpoint, so a separately hashed exact baseline replay was required.', '',
        'See per-target CSVs, image_gated_cells.json, calibration job failures and numerical logs. '
        'No failed target is dropped from an inferential comparison. Raw/checkpoint/code/output hashes '
        'are in sha256_manifest.csv; final_checks.json and COMPLETE.json are required for final completion.', '']
    REPORT.write_text('\n'.join(lines)+'\n')
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ['sha256_manifest.csv', 'COMPLETE.json']]
    files += [REPORT] + list((ROOT / 'experiments').glob('*priority33c*.py')) + list((ROOT / 'tests').glob('*priority33c*.py'))
    files += list((ROOT / 'protocols/amendments').glob('2026-10-03_priority33c*.md'))
    csv_write(OUT / 'sha256_manifest.csv', [dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in sorted(set(files))])


if __name__ == '__main__':
    main()
