"""Pure B3 table preparation and independent CSV-string recomputation.

No filesystem reads/writes, no partial-results analysis. The publication caller
must first pass scientific post-exit verification. CSVs are plain machine-readable
analysis intermediates, with unchanged stored values and explicit NA reservations.
"""
import csv
import io
import json
import math
import numpy as np
from experiments import priority35b_design as design
from experiments import priority35b_leakage_statistics as stats

METRICS=('feature_accuracy','standardized_mse','bounded_mse')


def index_records(records,arms):
    expected={(d,c,i,a['method']) for d,c in design.QUALIFIED for i in range(39) for a in arms[d]}
    if len(expected)!=9984 or any(len(arms[d])!=32 for d in design.DATASETS):raise ValueError('frozen32 arm inventory')
    indexed={}
    for r in records:
        key=tuple(r[f] for f in ('dataset','cell','index','arm'))
        if key in indexed or key not in expected:raise ValueError('duplicate/extra per-target record')
        values=[r[m] for m in METRICS]+[r['prior_mse']]
        if not all(math.isfinite(x) for x in values):raise ValueError('non-finite per-target record')
        if not 0<=r['feature_accuracy']<=1 or min(r['standardized_mse'],r['prior_mse'])<0:
            raise ValueError('metric range')
        if r['bounded_mse']!=min(r['standardized_mse'],r['prior_mse']):raise ValueError('bounded metric')
        indexed[key]=r
    if set(indexed)!=expected:raise ValueError('missing per-target record; never silently NA')
    return indexed


def pairs_from_records(indexed,matching):
    keys={(d,t,c) for d in design.DATASETS for t in design.TRANSFORMS for c in design.CLIPS}
    mapping={}
    for r in matching:
        key=tuple(r[f] for f in ('dataset','transform','clipping'))
        if key in mapping:raise ValueError('duplicate matching row')
        mapping[key]=r
    if set(mapping)!=keys:raise ValueError('12 matching entries required')
    pairs={};reasons={}
    for d,c,t,clip,metric in {tuple(k[:-1]) for k in design.family()}:
        key=(d,c,t,clip,metric);match=mapping[d,t,clip]
        if (d,c) not in design.QUALIFIED:reasons[key]='original P34C estimator not qualified'
        elif match['status']=='NOT_ASSESSABLE':reasons[key]='frozen utility match NOT_ASSESSABLE'
        elif match['status'] in ('MATCHED','MATCHED_CLIP_ONLY_NO_DP'):
            arm=match['matched_arm']
            if arm is None:raise ValueError('matched arm missing')
            pairs[key]=([indexed[d,c,i,t][metric] for i in range(39)],
                        [indexed[d,c,i,arm][metric] for i in range(39)])
        else:raise ValueError('unknown matching status')
    return pairs,reasons


def prepare(records,arms,matching):
    indexed=index_records(records,arms);summaries=[]
    for d,c in design.QUALIFIED:
        for arm in arms[d]:
            for metric in METRICS:
                s=design.interval39([indexed[d,c,i,arm['method']][metric] for i in range(39)])
                summaries.append(dict(dataset=d,cell=c,arm=arm['method'],metric=metric,**s))
    pairs,reasons=pairs_from_records(indexed,matching)
    holm=stats.evaluate(pairs,reasons)
    for row in holm:row.setdefault('reason',None)
    stats.independently_verify(holm,pairs,reasons)
    return dict(per_target=list(indexed.values()),frontier=summaries,holm=holm)


def csv_string(rows):
    """Serialize exact float repr, quoted JSON structures, and explicit null."""
    if not rows:raise ValueError('empty output table')
    fields=list(rows[0]);stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fields,lineterminator='\n')
    writer.writeheader()
    for row in rows:
        if set(row)!=set(fields):raise ValueError('heterogeneous table schema')
        writer.writerow({k:json.dumps(v,sort_keys=True,allow_nan=False,separators=(',',':')) for k,v in row.items()})
    return stream.getvalue()


def parse_csv(text):
    reader=csv.DictReader(io.StringIO(text));fields=reader.fieldnames
    if not fields or len(set(fields))!=len(fields):raise ValueError('duplicate/missing CSV header')
    rows=[]
    for r in reader:
        if set(r)!=set(fields) or None in r or any(v is None for v in r.values()):raise ValueError('CSV shape')
        rows.append({k:json.loads(v) for k,v in r.items()})
    if not rows:raise ValueError('empty CSV')
    return rows


def verify_csvs(per_target,frontier,holm,arms,matching):
    records=parse_csv(per_target);indexed=index_records(records,arms)
    expected={(d,c,a['method'],m) for d,c in design.QUALIFIED for a in arms[d] for m in METRICS}
    seen=set()
    for row in parse_csv(frontier):
        key=tuple(row[f] for f in ('dataset','cell','arm','metric'))
        if key in seen or key not in expected:raise ValueError('frontier inventory')
        seen.add(key);d,c,a,m=key
        values=np.array([indexed[d,c,i,a][m] for i in range(39)],dtype=np.float64)
        if row['n']!=39 or row['ranks']!=[13,27] or row['median']!=float(np.median(values)) or row['interval']!=np.sort(values)[[12,26]].tolist():
            raise ValueError('independent frontier median/ranks')
    if seen!=expected:raise ValueError('missing frontier summary')
    pairs,reasons=pairs_from_records(indexed,matching)
    stats.independently_verify(parse_csv(holm),pairs,reasons)
    return dict(status='CSV_RECOMPUTATION_PASS',per_target=9984,frontier=768,holm=144)
