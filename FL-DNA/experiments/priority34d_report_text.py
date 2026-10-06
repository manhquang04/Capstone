"""Pure P34D Markdown rendering, no report writes or completion mutation."""
import math
from experiments import priority34d_statistics as s
from experiments.priority34d_independent_metrics import gaussian_epsilon


def number(value):
    if not math.isfinite(float(value)):
        raise ValueError('nonfinite report number')
    return format(float(value),'.8g')


def interval(doc):
    return number(doc['median'])+' ['+number(doc['lower'])+', '+number(doc['upper'])+'] (ranks '+str(doc['lower_rank'])+'/'+str(doc['upper_rank'])+')'


def table(headers, rows):
    def cells(row):
        return '| '+' | '.join(str(v).replace('|','\\|').replace('\n',' ') for v in row)+' |'
    return '\n'.join([cells(headers),cells(['---']*len(headers))]+[cells(row) for row in rows])


def render(bundle, utility, clips, bn_endpoints, qualified, bn_context, hashes, verified=False):
    if len(bundle['rows'])!=76 or len(bundle['combined'])!=211 or not bundle['independent_sign_holm_verified']:
        raise ValueError('incomplete statistics report context')
    if len(hashes)==0 or any(len(value)!=64 or any(c not in '0123456789abcdef' for c in value) for value in hashes.values()):
        raise ValueError('invalid report hash registry')
    chunks=['# Priority34D — checkpoint robustness and harder image recovery',
            'Status: '+('verified final audit PASS.' if verified else 'DRAFT — final audit pending; not COMPLETE.'),
            '## Scope and paired units',
            'Three independently initialized LeNet-Zhu models (342042/342043/342044), 39 fresh single-image targets each. '
            'The fixed trained P31/P33C checkpoint uses 39 fresh four-image recovery batches. '
            'Batch 4 is recovery only: utility training remains SGD lr0.1/momentum0, batch256, 100epochs, '
            '16 paired order seeds51016–51031. Warm-start utility calibration is computed for this checkpoint; '
            'no sigma is transferred from another initialization. The recovery budget is4800 iterations, '
            'one restart, known labels, TV0.01. Three additional BAF checkpoints use seeds342000/342001/342002. '
            'BAF BN recovery is a batch-mean statistic instrument, not individual-record reconstruction.',
            '## Utility calibration and DP scope']
    utility_rows=[]
    for key,row in sorted(utility['matches'].items()):
        setting,dna,mechanism=key.split('::')
        clip=clips[setting]
        limits=number(clip['C']) if mechanism=='single' else ', '.join(number(c) for c in clip['clips'])
        sigma=row['sigma']
        eps='NOT_ASSESSABLE'
        if row['status']=='MATCHED':
            if sigma is None: raise ValueError('matched sigma missing')
            ratio=1. if mechanism=='single' else math.sqrt(len(clip['clips']))
            eps=number(gaussian_epsilon(sigma,ratio=ratio))
        utility_rows.append([setting,dna,mechanism,row['status'],number(row['baseline_mean']),
                             number(row['target_delta']),number(row['threshold']),
                             'none' if sigma is None else number(sigma),
                             'none' if row['next_sigma'] is None else number(row['next_sigma']),limits,eps])
    chunks.append(table(['Setting','DNA','Mechanism','Gate','Baseline validation','DNA delta','Threshold','Sigma','Next bracket','Clip(s)','One-release epsilon'],utility_rows))
    chunks.append('Matching: baseline validation >=0.40; largest passing sigma with paired mean delta '
                  '>=DNA delta−0.005 and the next higher grid point below threshold. Missing brackets are '
                  'NOT_ASSESSABLE, never privacy evidence. Initial grid0.0001/0.0003/0.001/0.003/0.01/0.03/0.1/0.3; '
                  'conditional1/3/10 only when preregistered top-grid trigger passes. '
                  'Epsilon above is an analytic Gaussian RDP bound for one bounded-update release, delta1e-5, '
                  'under the original add/remove convention, not record-level or whole-training DP. '
                  'Per-tensor composition uses sqrt(number of tensors) sensitivity ratio. Noise is applied to '
                  'the same individual client gradient as each transform; no noisy aggregate comparator.')
    chunks.append('## Image endpoint medians and intervals')
    chunks.append(table(['Setting','Arm','PSNR dB','SSIM','MSE'],[
        [r['setting'],r['arm'],*[interval(r['metrics'][m]) for m in ('psnr_db','ssim','mse')]]
        for r in bundle['summaries']]))
    chunks.append('Intervals are distribution-free two-sided >=95% median intervals. For n39 use ranks13/27. '
                  'Image batch4 metrics use post-optimization Hungarian assignment and the mean across four images; '
                  'the paired unit is the batch, not156 independent records. PSNR is the primary endpoint; '
                  'SSIM and MSE are descriptive.')
    chunks.append('## BAF qualification, calibration and endpoint medians')
    chunks.append(table(['Checkpoint','Qualification status','Development / distortion context'],[
        [seed,'QUALIFIED' if seed in qualified else 'NOT_ASSESSABLE',bn_context[str(seed)]] for seed in s.BN_SEEDS]))
    bn_rows=[]
    for setting,seeds in [(str(seed),(seed,)) for seed in qualified]+([('pooled_checkpoints',s.BN_SEEDS)] if set(qualified)==set(s.BN_SEEDS) else []):
        for arm in ('unprotected',*s.DNA,*['dp_'+dna for dna in s.DNA]):
            values=[r['metrics'][arm] for seed in seeds for r in sorted(bn_endpoints,key=lambda x:x['index']) if r['checkpoint']==seed]
            if len(values)!=39*len(seeds): raise ValueError('incomplete BN report endpoints')
            bn_rows.append([setting,arm,interval(s.median_interval(values))])
    chunks.append(table(['Checkpoint','Arm','Standardized batch-mean MSE'],bn_rows))
    chunks.append('Fresh qualification n8 then n24 must beat both training-prior and cyclic-decoy controls '
                  'with exact one-sided p<0.05. Failed checkpoints are retained without replacement. '
                  'Development n24 calibrates C=1.01×maximum raw norm and a median Gaussian-norm distortion match. '
                  'All scheduled confirmatory pairs must be complete and finite; no seed exclusion or clamping.')
    chunks.append('## Fixed confirmatory statistics — all76 reservations')
    stats=[]
    for row in bundle['rows']:
        assessed=row['status']=='ASSESSABLE'
        stats.append([row['hypothesis_id'],row['status'],row['n'],
                      interval(row['dna_interval']) if assessed else 'NA',
                      interval(row['comparator_interval']) if assessed else 'NA',
                      interval(row['paired_effect_interval']) if assessed else row['reason'],
                      number(row['p_raw']),number(row['holm_p34d']),number(row['holm_combined211'])])
    chunks.append(table(['Hypothesis','Status','n','DNA median','Comparator median','Paired DNA−comparator','Raw p','Holm76','Holm211'],stats))
    chunks.append('Directions are numeric DNA greater/less, tested separately. Lower PSNR means less image '
                  'recovery; higher BN MSE means less BN-statistic recovery. Exact signs omit exact ties; '
                  'all ties give p1. Gated tests reserve p1. Holm76 is P34D only; Holm211 combines existing '
                  '135 raw tests with76 new tests, never shrinking the family. Pooled117 comparisons require '
                  'all three complete eligible strata and use their exact binomial order-statistic ranks.')
    chunks.append('## Descriptive heterogeneity (outside Holm families)')
    chunks.append(table(['DNA','Comparator','Status','Kruskal-Wallis statistic','Exploratory p'],[
        [r['method'],r['comparator'],r['status'],number(r['statistic']) if 'statistic' in r else 'NA',
         number(r['p']) if 'p' in r else 'NA'] for r in bundle['heterogeneity']]))
    chunks.extend(['## Independent checks and disclosures',
                   'Independent metric reload/Hungarian pairing, public BN decoding, complete source-ID pairs, '
                   'checkpoint validation inference, receipt/source/input hashes, calibration reload, SciPy exact '
                   'binomial tests and separate vectorized NumPy Holm verification are required. '
                   'Probe arrays and quantiles are hash-verified; this utility auditor does not regenerate probe gradients. '
                   'All code runs CPU/thread1. Numerical failures and negative BN variance fail closed; no clamp, '
                   'tune, replacement or outcome-based design change is allowed. '
                   'This is conditional known-label client-gradient recovery with public model and known v2 key, '
                   'not final-model inversion or cryptographic key secrecy. v1 candidate selection uses observable '
                   'objective only; v2 uses sketch-space matching/proper least squares. Private truth/captures/noise '
                   'seeds are audit-only bundles outside the DP release and never provided to the attacker. '
                   'P34B central-DP results are excluded from individual-gradient comparisons. Earlier artifacts, '
                   'Latex and external defenses are preserved.',
                   '## SHA-256 evidence',table(['File / evidence','SHA-256'],sorted(hashes.items()))])
    return '\n\n'.join(chunks)+'\n'
