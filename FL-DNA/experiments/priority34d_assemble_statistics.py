"""Pure synthesis of independently audited endpoints; no disk/model access."""
import numpy as np
from scipy.stats import binomtest, kruskal
from experiments import priority34d_statistics as s


def image_registry(endpoints):
    lookup = {}
    ids = {}
    for row in endpoints:
        setting, arm, target = row['setting'], row['arm'], row['target']
        if setting not in s.MODELS+('trained_batch4',) or type(target) is not int or not 0<=target<39:
            raise ValueError('unregistered image setting/target')
        key = (setting,arm,target)
        if key in lookup:
            raise ValueError('duplicate image endpoint')
        metrics = row['metrics']
        if set(metrics)!={'psnr_db','ssim','mse'} or not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError('invalid image metrics')
        source = tuple(row['indices'])
        if len(source)!=(4 if setting=='trained_batch4' else 1) or len(set(source))!=len(source):
            raise ValueError('invalid recovery batch IDs')
        pair = (setting,target)
        if pair in ids and ids[pair]!=source:
            raise ValueError('unpaired image source IDs')
        ids[pair] = source; lookup[key] = metrics
    for setting,arm in {(k[0],k[1]) for k in lookup}:
        if {k[2] for k in lookup if k[:2]==(setting,arm)}!=set(range(39)):
            raise ValueError('incomplete scheduled image arm')
    if len(ids)!=156:
        raise ValueError('missing image setting/target')
    flattened = [index for source in ids.values() for index in source]
    if len(flattened)!=len(set(flattened)):
        raise ValueError('image source overlap')
    return lookup


def bn_registry(endpoints, qualified):
    if len(qualified)!=len(set(qualified)) or not set(qualified)<=set(s.BN_SEEDS):
        raise ValueError('invalid qualified checkpoint registry')
    lookup = {}; all_ids = []
    expected_arms = {'unprotected',*s.DNA,*['dp_'+a for a in s.DNA]}
    for row in endpoints:
        seed,index = row['checkpoint'],row['index']
        if seed not in qualified or type(index) is not int or not 0<=index<39 or (seed,index) in lookup:
            raise ValueError('unregistered/duplicate BN endpoint')
        if set(row['metrics'])!=expected_arms or not all(np.isfinite(v) and v>=0 for v in row['metrics'].values()):
            raise ValueError('invalid BN metrics')
        if len(row['source_ids'])!=4:
            raise ValueError('invalid BN source batch')
        all_ids.extend(row['source_ids']); lookup[(seed,index)] = row['metrics']
    if set(lookup)!={(seed,index) for seed in qualified for index in range(39)}:
        raise ValueError('incomplete scheduled BN pairs')
    if len(all_ids)!=len(set(all_ids)):
        raise ValueError('BN source overlap')
    return lookup


def independent_holm(values):
    p = np.asarray(values,dtype=float)
    if p.ndim!=1 or not len(p) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('invalid independent p values')
    order = np.argsort(p,kind='stable')
    sorted_adjusted = np.minimum(1.,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
    inverse = np.argsort(order)
    return sorted_adjusted[inverse]


def verify_statistics(rows, combined):
    for row in rows:
        if row['status']=='NOT_ASSESSABLE':
            if row['p_raw']!=1. or row['n']!=0:
                raise ValueError('invalid gated reservation')
            continue
        trials = row['wins']+row['losses']
        expected = binomtest(row['wins'],trials,.5,alternative='greater').pvalue if trials else 1.
        if not np.isclose(row['p_raw'],expected,rtol=1e-12,atol=1e-15):
            raise ValueError('independent sign mismatch')
    own = independent_holm([r['p_raw'] for r in rows])
    full = independent_holm([float(r['p_raw']) for r in combined])
    if not np.allclose(own,[r['holm_p34d'] for r in rows],rtol=0,atol=1e-15):
        raise ValueError('independent Holm76 mismatch')
    if not np.allclose(full,[float(r['holm_combined211']) for r in combined],rtol=0,atol=1e-15):
        raise ValueError('independent Holm211 mismatch')


def assemble(image_endpoints, bn_endpoints, qualified, matches, prior135):
    images = image_registry(image_endpoints); bn = bn_registry(bn_endpoints,qualified)
    rows=[]; paired=[]; summaries=[]; heterogeneity=[]
    def image_values(setting,arm,metric='psnr_db'):
        return np.array([images[(setting,arm,index)][metric] for index in range(39)])
    def eligible(setting,method,comparator):
        if comparator=='unprotected': return True
        key=setting+'::'+method+'::'+comparator
        if key not in matches: raise ValueError('missing calibration decision')
        if matches[key]['status'] not in ('MATCHED','NOT_ASSESSABLE'): raise ValueError('invalid calibration status')
        return matches[key]['status']=='MATCHED'
    for setting in s.MODELS+('trained_batch4',):
        required={'unprotected',*s.DNA}
        for method in s.DNA:
            for comparator in ('single','per_tensor'):
                if eligible(setting,method,comparator): required.add('dp_'+comparator+'_for_'+method)
        actual={k[1] for k in images if k[0]==setting}
        if actual!=required: raise ValueError('scheduled image arm registry mismatch')
        for arm in sorted(actual):
            summaries.append(dict(instrument='image',setting=setting,arm=arm,
                                  metrics={metric:s.median_interval(image_values(setting,arm,metric))
                                           for metric in ('psnr_db','ssim','mse')}))
    for spec in s.family():
        settings = s.MODELS if spec['setting']=='pooled_initializations' else (spec['setting'],)
        left=right=None; reason=None
        if spec['instrument']=='image':
            if all(eligible(setting,spec['method'],spec['comparator']) for setting in settings):
                arm='unprotected' if spec['comparator']=='unprotected' else 'dp_'+spec['comparator']+'_for_'+spec['method']
                left=np.concatenate([image_values(setting,spec['method']) for setting in settings])
                right=np.concatenate([image_values(setting,arm) for setting in settings])
            else: reason='utility bracket NOT_ASSESSABLE in at least one required stratum'
        else:
            seeds=s.BN_SEEDS if spec['setting']=='pooled_checkpoints' else (int(spec['setting']),)
            if all(seed in qualified for seed in seeds):
                left=np.array([bn[(seed,index)][spec['method']] for seed in seeds for index in range(39)])
                right=np.array([bn[(seed,index)]['dp_'+spec['method']] for seed in seeds for index in range(39)])
            else: reason='BN qualification NOT_ASSESSABLE in at least one required stratum'
        row=s.comparison(spec,left,right,reason); rows.append(row)
        if left is not None and spec['direction']=='dna_greater':
            paired.append(dict(hypothesis_base=spec['hypothesis_id'].rsplit('::',1)[0],
                               left=left.tolist(),right=right.tolist(),differences=(left-right).tolist()))
    for method in s.DNA:
        for comparator in ('unprotected','single','per_tensor'):
            if not all(eligible(setting,method,comparator) for setting in s.MODELS):
                heterogeneity.append(dict(method=method,comparator=comparator,status='NOT_ASSESSABLE'))
                continue
            arm='unprotected' if comparator=='unprotected' else 'dp_'+comparator+'_for_'+method
            groups=[image_values(setting,method)-image_values(setting,arm) for setting in s.MODELS]
            if np.ptp(np.concatenate(groups))==0:
                result=dict(status='UNDEFINED_CONSTANT_EFFECTS')
            else:
                test=kruskal(*groups); result=dict(status='DESCRIPTIVE_ONLY',statistic=float(test.statistic),p=float(test.pvalue))
                if not all(np.isfinite(v) for v in (test.statistic,test.pvalue)): raise ValueError('nonfinite heterogeneity')
            heterogeneity.append(dict(method=method,comparator=comparator,outside_holm_family=True,**result))
    rows,combined=s.finish(rows,prior135); verify_statistics(rows,combined)
    return dict(rows=rows,combined=combined,paired=paired,summaries=summaries,
                heterogeneity=heterogeneity,independent_sign_holm_verified=True)
