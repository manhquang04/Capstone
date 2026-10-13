"""B3 fixed-144 family and independent serialized-statistics audit."""
from experiments import priority35b_design as design
import numpy as np
from scipy.stats import binomtest

def evaluate(pairs,reasons):
    expected={tuple(k[:-1]) for k in design.family()}
    if set(pairs)&set(reasons) or set(pairs)|set(reasons)!=expected:
        raise ValueError('complete disjoint fixed family required')
    summaries={k:design.paired(*v) for k,v in pairs.items()}
    rows=[]
    for dataset,cell,transform,clipping,metric,direction in design.family():
        k=(dataset,cell,transform,clipping,metric)
        row=dict(dataset=dataset,cell=cell,transform=transform,clipping=clipping,metric=metric,direction=direction)
        if k in reasons:row.update(status='NOT_ASSESSABLE',reason=reasons[k],raw_p=1.,summary=None)
        else:
            s=summaries[k];row.update(status='ASSESSABLE',summary=s,
               raw_p=s['raw_p_lower' if direction=='transform_lower' else 'raw_p_higher'])
        rows.append(row)
    for row,p in zip(rows,design.holm([r['raw_p'] for r in rows])):row['holm_p']=p
    return rows

def independently_verify(rows,pairs,reasons):
    if len(rows)!=144 or [tuple(r[k] for k in ('dataset','cell','transform','clipping','metric','direction')) for r in rows]!=design.family():
        raise ValueError('fixed144 ordered family')
    for r in rows:
        k=tuple(r[f] for f in ('dataset','cell','transform','clipping','metric'))
        if k in reasons:
            if r['raw_p']!=1 or r['status']!='NOT_ASSESSABLE':raise ValueError('NA reservation')
            continue
        a,b=(np.asarray(x,dtype=np.float64) for x in pairs[k])
        if a.shape!=(39,) or b.shape!=(39,) or not np.isfinite([a,b]).all():raise ValueError('complete finite pairs')
        delta=a-b;lo=int((delta<0).sum());hi=int((delta>0).sum());n=lo+hi
        w=lo if r['direction']=='transform_lower' else hi
        p=binomtest(w,n,.5,alternative='greater').pvalue if n else 1.
        if p!=r['raw_p']:raise ValueError('independent SciPy exact sign')
        s=r['summary']
        if s['median']!=float(np.median(delta)) or s['interval']!=np.sort(delta)[[12,26]].tolist():raise ValueError('independent paired ranks')
        if (s['lower_wins'],s['higher_wins'],s['ties'],s['effective_n'])!=(lo,hi,39-n,n):raise ValueError('integer counts')
    p=np.array([r['raw_p'] for r in rows]);order=np.argsort(p,kind='stable');adjusted=np.ones(144)
    adjusted[order]=np.minimum(1.,np.maximum.accumulate(p[order]*np.arange(144,0,-1)))
    if not np.array_equal(adjusted,np.array([r['holm_p'] for r in rows])):raise ValueError('independent NumPy Holm')
    return dict(status='PASS',fixed_family=144,sign_implementation='SciPy binomtest',holm_implementation='separate NumPy cumulative maximum')
