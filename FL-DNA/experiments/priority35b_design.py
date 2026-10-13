"""P35B prospective constants and pure accounting/matching/statistics; no runner."""
from __future__ import annotations
import itertools
import math
import statistics

DATASETS = ('baf', 'ieee_cis', 'paysim')
CELLS = ('ratio_batch1', 'tableak_batch1', 'tableak_batch2')
SEEDS = tuple(range(321000, 321011))
EPSILONS = (1, 3, 10, 30, 100, 1000, 10000, 1000000)
CLIPS = ('global', 'per_tensor')
TRANSFORMS = ('dna_v1_conservative', 'dna_v2_0p95')
METRICS = ('feature_accuracy', 'bounded_mse')
DIRECTIONS = ('transform_lower', 'transform_higher')
QUALIFIED = tuple((d, c) for d in DATASETS for c in CELLS
                  if (d, c) != ('ieee_cis', 'tableak_batch2'))
FIGURE = dict(width_cm=8.8, height_cm=18, dpi=300, min_font_pt=7,
              panel_order=DATASETS, x_sign=-1, y_limits=(0, 100),
              selection='highest unprotected median; fixed cell-order ties',
              configurations=('Config-C local-BN Adam', 'P34C single gradient'))


def finite(values):
    values = list(values)
    if not values or any(isinstance(x, bool) or not math.isfinite(x) for x in values):
        raise ValueError('missing/nonfinite numerical input')
    return values


def account(epsilon):
    if isinstance(epsilon, bool) or epsilon not in EPSILONS:
        raise ValueError('epsilon outside frozen grid')
    d = math.log(1 / 1e-5)
    rho = (epsilon / (math.sqrt(d + epsilon) + math.sqrt(d))) ** 2
    sigma = math.sqrt(50 / (2 * rho))
    alpha = 1 + math.sqrt(d / rho)
    bound = 50 * alpha / (2 * sigma * sigma) + d / (alpha - 1)
    if abs(bound - epsilon) > 16 * math.ulp(float(epsilon)):
        raise ValueError('RDP inversion inconsistent')
    return dict(epsilon=epsilon, rounds=50, delta=1e-5, q=1,
                C=.01, sensitivity=.02, rho=rho, alpha=alpha,
                certified_epsilon=bound, sigma_sensitivity=sigma,
                noise_sd=.02 * sigma, record_level_dp=False)


def f1_values(values):
    if set(values) != set(SEEDS) or any(type(s) is not int for s in values):
        raise ValueError('require exact eleven paired seeds')
    result = finite(values[s] for s in SEEDS)
    if any(x < 0 or x > 1 for x in result):
        raise ValueError('invalid F1')
    return result


def match(transform, grid):
    """Full grid, exact mean threshold; largest noise with a failing neighbor."""
    target = statistics.fmean(f1_values(transform)) - .005
    if set(grid) != set(EPSILONS):
        raise ValueError('missing/extra utility grid point')
    points = [dict(epsilon=e, noise_sd=account(e)['noise_sd'],
                   mean_f1=statistics.fmean(f1_values(grid[e]))) for e in EPSILONS]
    points.sort(key=lambda r: -r['noise_sd'])
    for r in points:
        r['passes'] = r['mean_f1'] >= target
    selected = next((i for i, r in enumerate(points) if r['passes']), None)
    result = dict(threshold=target, points=points,
                  nonmonotonic=any(points[i]['mean_f1'] > points[i+1]['mean_f1']
                                   for i in range(len(points)-1)))
    if selected is None or selected == 0:
        return dict(result, status='NOT_ASSESSABLE',
                    reason='no passing point' if selected is None else 'no noisier neighbor',
                    epsilon=None, next_noisier_epsilon=None)
    assert not points[selected-1]['passes']
    return dict(result, status='MATCHED', epsilon=points[selected]['epsilon'],
                next_noisier_epsilon=points[selected-1]['epsilon'])


def interval39(values):
    x = sorted(finite(values))
    if len(x) != 39:
        raise ValueError('require all 39 finite row sets')
    return dict(n=39, median=statistics.median(x), interval=[x[12], x[26]], ranks=[13, 27])


def exact_tail(wins, losses):
    if any(type(x) is not int or x < 0 for x in (wins, losses)):
        raise ValueError('invalid integer sign counts')
    n = wins + losses
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n if n else 1.


def paired(a, b):
    a, b = finite(a), finite(b)
    if len(a) != 39 or len(b) != 39:
        raise ValueError('missing paired row sets')
    differences = [x-y for x, y in zip(a, b)]
    low, high = sum(x < 0 for x in differences), sum(x > 0 for x in differences)
    return dict(interval39(differences), lower_wins=low, higher_wins=high,
                ties=39-low-high, effective_n=low+high,
                raw_p_lower=exact_tail(low, high), raw_p_higher=exact_tail(high, low))


def bounded_mse(mse, prior_mse):
    finite([mse, prior_mse])
    if min(mse, prior_mse) < 0:
        raise ValueError('negative MSE')
    return min(mse, prior_mse)


def family():
    return list(itertools.product(DATASETS, CELLS, TRANSFORMS, CLIPS, METRICS, DIRECTIONS))


def holm(pvalues):
    pvalues = finite(pvalues)
    if len(pvalues) != len(family()) or any(x < 0 or x > 1 for x in pvalues):
        raise ValueError('require fixed 144 valid p-values including NA reservations')
    order = sorted(range(len(pvalues)), key=lambda i: (pvalues[i], i))
    result, previous = [0.] * len(pvalues), 0.
    for rank, i in enumerate(order):
        previous = max(previous, min(1., (len(pvalues)-rank)*pvalues[i]))
        result[i] = previous
    return result


def select_estimator(dataset, unprotected):
    expected = [c for d, c in QUALIFIED if d == dataset]
    if not expected or set(unprotected) != set(expected):
        raise ValueError('qualified unprotected panel coverage mismatch')
    if any(not 0 <= x <= 1 for c in expected for x in finite(unprotected[c])):
        raise ValueError('feature accuracy must be a fraction')
    medians = {c: interval39(unprotected[c])['median'] for c in expected}
    chosen = max(expected, key=lambda c: medians[c])  # first in fixed order wins ties
    return dict(dataset=dataset, cell=chosen, unprotected_medians=medians,
                protected_results_inspected=False)


def utility_interval(method, baseline):
    from scipy.stats import t
    a, b = f1_values(method), f1_values(baseline)
    delta = [x-y for x, y in zip(a, b)]
    mean = statistics.fmean(delta)
    half = float(t.ppf(.975, 10))*statistics.stdev(delta)/math.sqrt(11)
    return dict(n=11, df=10, delta_mean=mean, delta_interval=[mean-half, mean+half],
                figure_loss_mean=-mean, figure_loss_interval=[-mean-half, -mean+half])
