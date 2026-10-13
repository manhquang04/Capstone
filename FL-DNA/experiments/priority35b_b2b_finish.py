"""Additive post-exit CSV metadata adapter; frozen scientific verifier unchanged."""
import csv
from pathlib import Path
from experiments import priority35b_b2b as b
from experiments import priority35b_b2b_analysis as analysis

ORIGINAL_CSV=analysis.csv_save

def rectangular_csv(path,rows):
    if Path(path).name!='matching.csv': return ORIGINAL_CSV(path,rows)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def finish():
    f=b.u.read(b.OUT/'finalizer_infrastructure_freeze.json')
    for name,h in f['sources'].items():
        if b.u.sha(b.ROOT/name)!=h: raise ValueError('finalizer source drift')
    b.verify_freeze()
    if b.live() or b.u.live() or b.telemetry.live(): raise ValueError('scientific processes alive')
    analysis.csv_save=rectangular_csv
    try: analysis.finish()
    finally: analysis.csv_save=ORIGINAL_CSV

if __name__=='__main__':finish()
