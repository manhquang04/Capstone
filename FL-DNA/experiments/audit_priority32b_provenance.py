"""Read-only source/data/stored-output evidence; writes only new P32b JSON."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/priority32b_bn_reconciliation'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    import pandas as pd
    paths = [ROOT / name for name in (
        'datasets/creditcard.csv', 'data/load_creditcard.py',
        'models/fraud_mlp.py', 'experiments/fraud_fl_common.py',
        'experiments/run_fraud_fl_dna_transform.py',
        'experiments/run_fraud_fl_dna_transform_v2.py',
        'experiments/priority32_multidataset.py',
        'experiments/priority32b_bn_reconcile.py',
        'protocols/amendments/2026-10-02_priority32b_bn_variance_reconciliation.md')]
    paths += sorted((ROOT / 'privacy').glob('*.py'))
    paths += sorted((ROOT / 'artifacts/priority32_multidataset/prepared/paysim').glob('*'))
    stored = {}
    for method, branch, label in (
        ('v1', 'rq2/confirmatory_run_20260913', 'dna_transform'),
        ('v2', 'rq2_v2/confirmatory_20260916', 'dna_transform_v2')):
        files = sorted((ROOT / 'artifacts' / branch).glob('seed_*/' + label + '/metrics.json'))
        assert len(files) == (21 if method == 'v1' else 52)
        stored[method] = []
        for path in files:
            doc = json.loads(path.read_text())
            bad = [{k: row.get(k) for k in ('round', 'auc_roc', 'pr_auc')}
                   for row in doc['rounds'] if row.get('auc_roc') is None or row.get('pr_auc') is None]
            stored[method].append({'path': str(path.relative_to(ROOT)), 'rounds': len(doc['rounds']), 'null_endpoints': bad})
            paths.append(path)
    columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    missing = {name: 0 for name in columns}
    rows = 0
    for frame in pd.read_csv(ROOT / 'datasets/creditcard.csv', usecols=columns, chunksize=100000):
        rows += len(frame)
        for name, count in frame.isna().sum().items():
            missing[name] += int(count)
    result = {'at': datetime.now(timezone.utc).isoformat(), 'raw_rows': rows,
              'raw_numeric_missing': missing, 'stored_output_audit': stored,
              'sha256': {str(p.relative_to(ROOT)): digest(p) for p in paths if p.is_file()},
              'disclosure': 'Read-only audit after replay launch; immutable scientific runner untouched.'}
    target = OUT / 'source_provenance.json'
    assert not target.exists(), 'Preserve prior provenance'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'rows': rows, 'missing': missing, 'hashed_files': len(result['sha256'])}))


if __name__ == '__main__':
    main()
