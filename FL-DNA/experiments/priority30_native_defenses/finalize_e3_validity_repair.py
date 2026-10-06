"""Create complete hashes/check record without touching original P30 files."""
from __future__ import annotations
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/priority30_native_defenses/audit/e3_repair_20261001'

def main():
    code=[ROOT/f'experiments/priority30_native_defenses/{name}.py' for name in ('repair_e3_validity','analyze_e3_validity_repair','finalize_e3_validity_repair')]
    tests=ROOT/'tests/test_priority30_e3_validity_repair.py'
    commands=[[sys.executable,'-B','-m','py_compile',*[str(p) for p in code],str(tests)],['git','diff','--check'],
        [sys.executable,'-B','-c','import runpy; t=runpy.run_path("tests/test_priority30_e3_validity_repair.py"); [fn() for name,fn in t.items() if name.startswith("test_")]; print("3 tests PASS")']]
    checks=[]
    for command in commands:
        r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        checks.append({'command':command,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
        if r.returncode:raise RuntimeError(checks[-1])
    expected_old={
        'artifacts/priority30_native_defenses/audit/S6/e3_sign_tests.csv':'4deb8c01a0c01f7c4857c5f02c461f0c3fc51380d670bbb934b7485afdb3bb8f',
        'artifacts/priority30_native_defenses/audit/S6/e3_utility_adult_sign_tests.csv':'0874b18f1d19c2760b9ca204de53f4b15c032a166b41b41cb8f23b4d1433391c'}
    for p,h in expected_old.items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    collision_command=['rg','-l',r'"seed": 420(00|01|1[6-9]|2[0-9]|3[01])\b','artifacts','--glob','*.json','--glob','!**/e3_repair_20261001/**']
    collision=subprocess.run(collision_command,cwd=ROOT,capture_output=True,text=True)
    assert collision.returncode==1 and not collision.stdout,collision.stdout
    checks.append({'command':collision_command,'returncode':collision.returncode,'meaning':'no earlier JSON seed-field collision','stdout':collision.stdout,'stderr':collision.stderr})
    amendment=ROOT/'protocols/amendments/2026-10-01_priority30_e3_validity_repair.md'
    stat=amendment.stat()
    first_soteria=min(p.stat().st_mtime for p in (OUT/'S3').rglob('result.json'))
    first_utility=min(p.stat().st_mtime for p in (OUT/'S4/lr').glob('*.json'))
    assert stat.st_birthtime<first_soteria and stat.st_mtime<first_utility
    checks.append({'meaning':'amendment creation before Soteria; utility clarification before utility',
        'amendment_creation_epoch':stat.st_birthtime,'utility_clarification_mtime_epoch':stat.st_mtime,
        'first_soteria_output_epoch':first_soteria,'first_utility_output_epoch':first_utility})
    ps=subprocess.run(['ps','-axo','pid,command'],capture_output=True,text=True,check=True)
    active=[line for line in ps.stdout.splitlines() if re.search(r'python.*\brepair_e3_validity\.py --stage',line)]
    assert not active,active
    checks.append({'meaning':'no repair experimental workload remains','active_main_processes':active})
    import numpy as np
    from scipy.stats import binom
    rows=list(csv.DictReader((OUT/'S6/tests_with_effects_and_references.csv').open()))
    observed=[];reported=[]
    for row in rows:
        w=int(row['defense_wins']);l=int(row['comparator_wins']);t=int(row['ties'])
        assert w+l+t==39
        if w+l:
            p=float(binom.sf(w-1,w+l,.5))
            assert abs(p-float(row['p_defense_greater_error']))<1e-14
            observed.append(p);reported.append(float(row['holm_p_defense_greater_error']))
            if row['evaluation']=='E3':
                p=float(binom.sf(l-1,w+l,.5))
                assert abs(p-float(row['p_comparator_greater_error']))<1e-14
                observed.append(p);reported.append(float(row['holm_p_comparator_greater_error']))
    order=np.argsort(observed,kind='stable');values=np.asarray(observed)
    adjusted=np.empty(len(values));adjusted[order]=np.minimum(1.,np.maximum.accumulate(values[order]*np.arange(len(values),0,-1)))
    assert np.allclose(adjusted,reported,rtol=1e-13,atol=1e-14)
    checks.append({'meaning':'independent SciPy exact-binomial and NumPy whole-family Holm cross-check','hypotheses':len(values),'passed':True})
    check_path=OUT/'checks.json'
    with check_path.open('x') as f:
        json.dump({'timestamp_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'old_summary_hashes_unchanged':expected_old},f,indent=2)
    paths=code+[tests,ROOT/'protocols/amendments/2026-10-01_priority30_e3_validity_repair.md',ROOT/'reports/priority30_e3_validity_repair_report_20261001.md']
    paths+=sorted(p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='sha256_manifest.csv')
    with (OUT/'sha256_manifest.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader()
        for p in paths:
            w.writerow({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    print(f'Checks PASS; {len(paths)} files hashed')

if __name__=='__main__':main()
