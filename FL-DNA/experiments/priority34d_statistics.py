"""Preregistered P34D fixed76 family, exact signs and median intervals."""
import math
import numpy as np

MODELS=('init342042','init342043','init342044')
DNA=('dna_v1_conservative','dna_v2_0p95')
BN_SEEDS=(342000,342001,342002)


def family():
    rows=[]
    for setting in MODELS+('pooled_initializations','trained_batch4'):
        for method in DNA:
            for comparator in ('unprotected','single','per_tensor'):
                for direction in ('dna_greater','dna_less'):
                    rows.append(dict(hypothesis_id='P34D::image::'+setting+'::'+method+'::'+comparator+'::'+direction,
                                     instrument='image',setting=setting,method=method,
                                     comparator=comparator,direction=direction))
    for setting in tuple(str(seed) for seed in BN_SEEDS)+('pooled_checkpoints',):
        for method in DNA:
            for direction in ('dna_greater','dna_less'):
                rows.append(dict(hypothesis_id='P34D::BN::'+setting+'::'+method+'::distortion::'+direction,
                                 instrument='BN',setting=setting,method=method,
                                 comparator='distortion',direction=direction))
    assert len(rows)==len({row['hypothesis_id'] for row in rows})==76
    return rows


def exact_sign(left,right,direction):
    left,right=np.asarray(left,dtype=float),np.asarray(right,dtype=float)
    if left.shape!=right.shape or left.ndim!=1 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('invalid paired observations')
    if direction not in ('dna_greater','dna_less'):
        raise ValueError('unknown direction')
    difference=left-right
    if direction=='dna_less':
        difference=-difference
    wins,losses=int((difference>0).sum()),int((difference<0).sum())
    n=wins+losses
    p=sum(math.comb(n,k) for k in range(wins,n+1))/2**n if n else 1.
    return dict(p_raw=float(p),wins=wins,losses=losses,ties=len(left)-n,n=len(left))


def median_interval(values):
    values=np.sort(np.asarray(values,dtype=float))
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('invalid median sample')
    n=len(values)
    eligible=[rank for rank in range(1,n//2+1)
              if sum(math.comb(n,k) for k in range(rank))/2**n<=.025]
    if not eligible:
        raise ValueError('sample too small for finite distribution-free 95% interval')
    low=max(eligible); high=n-low+1
    coverage=1-2*sum(math.comb(n,k) for k in range(low))/2**n
    return dict(median=float(np.median(values)),lower=float(values[low-1]),upper=float(values[high-1]),
                lower_rank=low,upper_rank=high,n=n,coverage=float(coverage))


def holm(pvalues):
    p=np.asarray(pvalues,dtype=float)
    if p.ndim!=1 or not len(p) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('invalid p-values')
    order=np.argsort(p,kind='stable'); adjusted=np.empty_like(p); running=0.
    for index,position in enumerate(order):
        running=max(running,float((len(p)-index)*p[position]))
        adjusted[position]=min(1.,running)
    return adjusted.tolist()


def comparison(spec,left=None,right=None,reason=None):
    row=dict(spec)
    if left is None or right is None:
        row.update(status='NOT_ASSESSABLE',p_raw=1.,reason=reason or 'gated comparison',n=0)
        return row
    if len(left) not in (39,117) or len(left)!=len(right):
        raise ValueError('require exactly39 or117 complete pairs; never analyze missing observations')
    row.update(status='ASSESSABLE',**exact_sign(left,right,spec['direction']))
    row['dna_interval']=median_interval(left)
    row['comparator_interval']=median_interval(right)
    row['paired_effect_interval']=median_interval(np.asarray(left)-np.asarray(right))
    return row


def finish(rows,prior135):
    registered=family()
    if {row['hypothesis_id'] for row in rows}!={row['hypothesis_id'] for row in registered} or len(rows)!=76:
        raise ValueError('fixed76 family changed')
    if len(prior135)!=135 or len({row['hypothesis_id'] for row in prior135})!=135:
        raise ValueError('prior135 family changed')
    prior_p=[float(row['p_raw']) for row in prior135]
    p=[float(row['p_raw']) for row in rows]
    own,combined=holm(p),holm(prior_p+p)
    for index,row in enumerate(rows):
        row['holm_p34d']=own[index]; row['holm_combined211']=combined[135+index]
    previous=[dict(row,holm_combined211=combined[index]) for index,row in enumerate(prior135)]
    return rows,previous+rows
