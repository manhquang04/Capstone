"""Approved audit-only SciPy tolerance; exact binomial probabilities unchanged."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from experiments import priority35b_leakage_statistics as stats
from experiments import priority35b_float64_verifier_adapter as mse

FREEZE='artifacts/priority35b/b3/scipy_verifier_infrastructure_freeze.json'
RTOL=1e-12
ATOL=1e-15
THRESHOLDS=(.05,)


def close(a,b):return abs(a-b)<=max(ATOL,RTOL*max(abs(a),abs(b)))


def check_margin(p):
    if not math.isfinite(p) or not 0<=p<=1:raise ValueError('finite p-value')
    if any(close(p,t) for t in THRESHOLDS):raise ValueError('p-value too close to decision threshold')


def independently_verify(rows,pairs,reasons):
    design=stats.design
    if len(rows)!=144 or [tuple(r[k] for k in ('dataset','cell','transform','clipping','metric','direction')) for r in rows]!=design.family():
        raise ValueError('fixed144 ordered family')
    max_error=0.;nonzero=0
    for r in rows:
        check_margin(r['raw_p']);check_margin(r['holm_p'])
        k=tuple(r[f] for f in ('dataset','cell','transform','clipping','metric'))
        if k in reasons:
            if r['raw_p']!=1 or r['status']!='NOT_ASSESSABLE':raise ValueError('NA reservation')
            continue
        a,b=(np.asarray(x,dtype=np.float64) for x in pairs[k])
        if a.shape!=(39,) or b.shape!=(39,) or not np.isfinite([a,b]).all():raise ValueError('complete finite pairs')
        delta=a-b;lo=int((delta<0).sum());hi=int((delta>0).sum());n=lo+hi
        w=lo if r['direction']=='transform_lower' else hi
        exact=sum(math.comb(n,i) for i in range(w,n+1))/2**n if n else 1.
        if exact!=r['raw_p']:raise ValueError('authoritative integer-binomial mismatch')
        p=float(binomtest(w,n,.5,alternative='greater').pvalue) if n else 1.
        if not close(p,exact):raise ValueError('SciPy cross-check exceeds float64 tolerance')
        if any((p<t)!=(exact<t) for t in THRESHOLDS):raise ValueError('SciPy threshold decision differs')
        error=abs(p-exact);max_error=max(max_error,error);nonzero+=int(error!=0)
        s=r['summary']
        if s['median']!=float(np.median(delta)) or s['interval']!=np.sort(delta)[[12,26]].tolist():raise ValueError('independent paired ranks')
        if (s['lower_wins'],s['higher_wins'],s['ties'],s['effective_n'])!=(lo,hi,39-n,n):raise ValueError('integer counts')
    p=np.array([r['raw_p'] for r in rows]);order=np.argsort(p,kind='stable');adjusted=np.ones(144)
    adjusted[order]=np.minimum(1.,np.maximum.accumulate(p[order]*np.arange(144,0,-1)))
    if not np.array_equal(adjusted,np.array([r['holm_p'] for r in rows])):raise ValueError('independent NumPy Holm')
    return dict(status='PASS',fixed_family=144,authoritative='exact integer-binomial',
                scipy_crosscheck_rtol=RTOL,scipy_crosscheck_atol=ATOL,maximum_scipy_difference=max_error,
                scipy_nonzero_differences=nonzero,threshold_margin='PASS',holm_implementation='separate NumPy cumulative maximum')


def verify_freeze():
    root=mse.ROOT;integrity=mse.wrapper.prepare.integrity
    doc=integrity.read(Path(root)/FREEZE)
    for p,digest in doc['files'].items():
        if integrity.sha(integrity.child(root,p))!=digest:raise ValueError('approved infrastructure drift: '+p)
    return doc


def install():
    verify_freeze();mse.install();stats.independently_verify=independently_verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','render','validate','publish','seal'],required=True)
    for key in ('ready','render','qa','validation','interpretation','published'):parser.add_argument('--'+key)
    a=parser.parse_args();install()
    if a.stage=='prepare':result=mse.wrapper.run(mse.ROOT)
    elif a.stage=='render':
        from experiments import priority35b_leakage_render_publication as mod
        result=mod.render(a.ready)
    elif a.stage=='validate':
        from experiments import priority35b_leakage_final_validation as mod
        result=mod.run(a.ready)
    elif a.stage=='publish':
        from experiments import priority35b_leakage_publish_report as mod
        result=mod.publish(a.ready,a.render,a.qa,a.validation,a.interpretation)
    else:
        from experiments import priority35b_approved_infrastructure_seal as mod
        result=mod.seal(a.published)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
