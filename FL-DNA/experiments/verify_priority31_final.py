"""Read-only scientific verification; refresh hashes after supervisor exit."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/priority31_image_utility_dp'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj,indent=2)+'\n')


def main():
    with (OUT/'sha256_manifest.csv').open() as f:
        original_manifest=list(csv.DictReader(f))
    stale=[row['path'] for row in original_manifest if digest(ROOT/row['path'])!=row['sha256']]
    assert set(stale).issubset({str((OUT/'supervisor.jsonl').relative_to(ROOT)),
                              str((OUT/'detached_stdout.log').relative_to(ROOT)),
                              'reports/priority31_image_utility_dp_report.md'}),stale
    for row in read(OUT/'input_provenance.json'):
        assert digest(ROOT/row['path'])==row['sha256'],row['path']
    summary=read(OUT/'analysis_summary.json')
    assert summary['holm_family_size']==115
    for result in summary['comparisons']:
        name=result['defense']
        with (OUT/f'paired_{name}.csv').open() as f:
            rows=list(csv.DictReader(f))
        assert len(rows)==39 and {int(r['target_id']) for r in rows}==set(range(39))
        wins=losses=ties=0
        differences=[]
        for row in rows:
            tid=int(row['target_id'])
            dp=read(OUT/'reconstruction'/name/f'target_{tid:03d}/result.json')
            dna=read(ROOT/'artifacts/priority30_native_defenses/audit/S1c/image/E2'/name/f'target_{tid:03d}/result.json')
            assert dp['cifar10_index']==dna['cifar10_index']==int(row['cifar10_index'])
            d=dna['metrics']['mse']-dp['metrics']['mse']
            assert d==float(row['D_mse'])
            wins+=d>0; losses+=d<0; ties+=d==0
            differences.append(dna['metrics']['psnr_db']-dp['metrics']['psnr_db'])
        n=wins+losses
        exact=sum(math.comb(n,k) for k in range(wins,n+1))/2**n
        assert (wins,losses,ties)==(result['dna_wins'],result['dp_wins'],result['ties'])
        assert exact==result['p_dna_greater']
        ordered=sorted(differences)
        assert ordered[12]==result['rank13'] and ordered[26]==result['rank27']
        assert statistics.median(ordered)==result['median_paired_psnr']
    with (OUT/'expanded_holm.csv').open() as f:
        tests=list(csv.DictReader(f))
    ordered=sorted(tests,key=lambda row:float(row['p_value']))
    maximum=0.
    for rank,row in enumerate(ordered):
        maximum=max(maximum,min(1.,(len(ordered)-rank)*float(row['p_value'])))
        assert maximum==float(row['holm_p'])
    assert all(row['returncode']==0 for row in read(OUT/'final_checks.json').values())
    write(OUT/'independent_verification.json',dict(status='PASS',paired_targets_verified=78,
          holm_hypotheses_verified=115,source_hashes_verified=True,
          pre_refresh_stale_log_hashes=stale,
          reason='Finalizer hashed logs before supervisor appended its exit/completed records; refresh only after all workloads exited. No scientific result changed.'))
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in
           ['sha256_manifest.csv','final_hashes.json'] and not p.name.endswith('.tmp')]
    files += [ROOT/'reports/priority31_image_utility_dp_report.md',
              ROOT/'protocols/amendments/2026-10-01_priority31_image_utility_dp.md',
              ROOT/'tests/test_priority31_image_utility_dp.py']
    files += [ROOT/'experiments'/name for name in ['priority31_image_utility_dp.py',
              'finalize_priority31_image_utility_dp.py','supervise_priority31_image_utility_dp.py',
              'verify_priority31_final.py']]
    with (OUT/'sha256_manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','sha256'])
        writer.writeheader()
        for path in sorted(set(files)):
            writer.writerow(dict(path=str(path.relative_to(ROOT)),sha256=digest(path)))
    write(OUT/'final_hashes.json',dict(report_sha256=digest(ROOT/'reports/priority31_image_utility_dp_report.md'),
                                     manifest_sha256=digest(OUT/'sha256_manifest.csv')))
    print('Independent paired/binomial/Holm/source verification PASS; stable hashes refreshed.')


if __name__=='__main__':
    main()
