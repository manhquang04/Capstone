"""Pure, complete-only conversion of audited result/utility data for publication.

No file access, measurement, optimization, rendering, writes or COMPLETE claim.
Call only after scientific post-exit verification, never on partial results.
"""
import math
import statistics
import numpy as np
from experiments import priority35b_design as design
from experiments import priority35b_leakage_tables as tables
from experiments import priority35b_leakage_figure as figure


def result_rows(results, arms):
    indexed={}
    for r in results:
        c=r['config'];key=tuple(c[k] for k in ('dataset','cell','index','arm'))
        if key in indexed or r['status']!='COMPLETED':raise ValueError('duplicate/incomplete result')
        if r['threads']!=1 or r['inter_threads']!=1:raise ValueError('thread contract')
        indexed[key]=r
    expected={(d,c,i,a['method']) for d,c in design.QUALIFIED for i in range(39) for a in arms[d]}
    if set(indexed)!=expected or len(expected)!=9984:raise ValueError('all9984 results required')
    out=[]
    for key,r in indexed.items():
        d,c,i,a=key;u=indexed[d,c,i,'unprotected'];control=u['controls']
        if r['source_ids']!=u['source_ids'] or r['reconstruction_seed']!=u['reconstruction_seed']:
            raise ValueError('unpaired source IDs or reconstruction seed')
        prior=control['mean_mode_metric']['standardized_mse']
        score=r['score']
        row=dict(dataset=d,cell=c,index=i,arm=a,source_ids=r['source_ids'],
                 reconstruction_seed=r['reconstruction_seed'],selected_variant=r['selected_variant'],
                 feature_accuracy=score['feature_accuracy'],standardized_mse=score['standardized_mse'],
                 bounded_mse=score['bounded_mse'],prior_mse=prior,
                 mean_mode_accuracy=control['scores']['mean_mode']/100.,
                 empirical_marginal_accuracy=control['scores']['empirical_mean']/100.)
        if any(not math.isfinite(row[k]) or not 0<=row[k]<=1 for k in ('mean_mode_accuracy','empirical_marginal_accuracy')):
            raise ValueError('control units or nonfinite values')
        out.append(row)
    tables.index_records(out,arms)
    return out


def arm_style(a):
    method=a['method']
    if method=='unprotected':return 'unprotected',None
    if method==design.TRANSFORMS[0]:return 'v1',None
    if method==design.TRANSFORMS[1]:return 'v2',None
    if a['clipping']=='per_tensor':group='per_tensor_c001'
    elif method.startswith('local_dp_global_') or method.startswith('b2b_global_c001_'):group='joint_c001'
    elif method.startswith('b2b_global_c01_'):group='joint_c01'
    elif method.startswith('b2b_global_cmed_half_'):group='joint_half'
    elif method.startswith('b2b_global_cmed_'):group='joint_med'
    else:raise ValueError('unrecognized frozen arm name')
    return ('clip_only' if a['epsilon'] is None else 'dp'),group


def prepare_figure(records, arms, frontier, utility, selection):
    indexed=tables.index_records(records,arms)
    sf={}
    for r in frontier:
        key=tuple(r[k] for k in ('dataset','cell','arm','metric'))
        if key in sf:raise ValueError('duplicate frontier row')
        sf[key]=r
    expected={(d,c,a['method'],m) for d,c in design.QUALIFIED for a in arms[d] for m in tables.METRICS}
    if set(sf)!=expected:raise ValueError('all768 frontier summaries required')
    uf={}
    for r in utility:
        key=tuple(r[k] for k in ('dataset','method','endpoint'))
        if key in uf:raise ValueError('duplicate utility row')
        uf[key]=r
    ue={(d,('baseline' if a['method']=='unprotected' else a['method']),m)
        for d in design.DATASETS for a in arms[d] for m in ('f1','auc_roc','pr_auc')}
    if set(uf)!=ue:raise ValueError('all288 utility summaries required')
    panels={}
    for p in selection['panels']:
        if p['dataset'] in panels:raise ValueError('duplicate selection panel')
        panels[p['dataset']]=p
    if set(panels)!=set(design.DATASETS):raise ValueError('all3 selection panels required')
    rows=[];controls={}
    for d in design.DATASETS:
        cells=[c for dd,c in design.QUALIFIED if dd==d]
        medians={c:sf[d,c,'unprotected','feature_accuracy']['median'] for c in cells}
        chosen=max(cells,key=lambda c:medians[c])
        p=panels[d]
        if p['cell']!=chosen or p['unprotected_medians']!=medians or p['protected_results_inspected'] is not False:
            raise ValueError('preprotected estimator selection drift')
        controls[d]={name:statistics.median([indexed[d,chosen,i,'unprotected'][field] for i in range(39)])
                     for name,field in (('mean_mode','mean_mode_accuracy'),('empirical_marginal','empirical_marginal_accuracy'))}
        for a in arms[d]:
            method=a['method'];u=uf[d,'baseline' if method=='unprotected' else method,'f1']
            f=sf[d,chosen,method,'feature_accuracy'];kind,group=arm_style(a)
            if u['n']!=11 or u['df']!=10 or f['n']!=39 or f['ranks']!=[13,27]:raise ValueError('figure uncertainty units')
            rows.append(dict(dataset=d,cell=chosen,arm=method,kind=kind,group=group,
                epsilon=a['epsilon'],C=a['C'],noise_sd=a['noise_sd'],
                x=-u['mean_delta'],x_low=-u['ci_high'],x_high=-u['ci_low'],
                y=f['median'],y_low=f['interval'][0],y_high=f['interval'][1],
                utility_n=11,leakage_n=39,ranks=[13,27]))
    figure.validate(rows,controls)
    return rows,controls


def independently_verify_figure(rows, controls, records, utility):
    """Direct NumPy recomputation from39 raw batches, not frontier summaries."""
    uf={(r['dataset'],r['method']):r for r in utility if r['endpoint']=='f1'}
    indexed={(r['dataset'],r['cell'],r['index'],r['arm']):r for r in records}
    for r in rows:
        d,c,a=r['dataset'],r['cell'],r['arm']
        values=np.array([indexed[d,c,i,a]['feature_accuracy'] for i in range(39)],dtype=np.float64)
        if (r['y'],r['y_low'],r['y_high'])!=(float(np.median(values)),float(np.sort(values)[12]),float(np.sort(values)[26])):
            raise ValueError('independent figure median/ranks')
        u=uf[d,'baseline' if a=='unprotected' else a]
        if (r['x'],r['x_low'],r['x_high'])!=(-u['mean_delta'],-u['ci_high'],-u['ci_low']):
            raise ValueError('independent figure utility sign/interval')
        for name,field in (('mean_mode','mean_mode_accuracy'),('empirical_marginal','empirical_marginal_accuracy')):
            expected=float(np.median([indexed[d,c,i,'unprotected'][field] for i in range(39)]))
            if controls[d][name]!=expected:raise ValueError('independent data-free line')
    figure.validate(rows,controls)
    return dict(status='FIGURE_DATA_RECOMPUTATION_PASS',rows=96,full_priority35b_complete=False)
