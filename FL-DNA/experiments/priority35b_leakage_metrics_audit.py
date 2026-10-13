"""Independent NumPy/SciPy reproduction of the frozen native scoring path.

Includes the native detailed-score denominator convention (a missing feature
type still contributes one to the denominator); does not repair/change it.
Dataset decoding is supplied by the frozen adapter, not reimplemented here.
"""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment


def decoded_costs(truth, estimate, tolerances):
    truth = np.asarray(truth, dtype=object)
    estimate = np.asarray(estimate, dtype=object)
    if truth.ndim != 2 or truth.shape != estimate.shape or truth.shape[1] != len(tolerances):
        raise ValueError('complete equal decoded batch shapes')
    if not truth.shape[0] or not truth.shape[1]:
        raise ValueError('empty decoded batch')
    cats = sum(t == 'cat' for t in tolerances)
    conts = len(tolerances) - cats
    denominator = max(cats, 1) + max(conts, 1)
    cost = np.zeros((len(truth), len(estimate)), dtype=np.float64)
    for i, row in enumerate(truth):
        for j, other in enumerate(estimate):
            errors = 0
            for x, y, tolerance in zip(row, other, tolerances):
                if tolerance == 'cat':
                    errors += str(x) != str(y)
                else:
                    if isinstance(tolerance, str):
                        raise ValueError('unknown feature tolerance')
                    x, y, tolerance = float(x), float(y), float(tolerance)
                    if not all(map(math.isfinite, (x, y, tolerance))) or tolerance < 0:
                        raise ValueError('non-finite or negative numerical tolerance')
                    errors += not x - tolerance <= y <= x + tolerance
            cost[i, j] = errors / denominator
    return cost


def independently_score(data, truth, recovered):
    """No native matching, metric or postprocessing helper is used."""
    a = truth.detach().cpu().numpy()
    rec = recovered.detach().cpu().numpy().copy()
    if a.shape != rec.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(rec).all():
        raise ValueError('finite equal encoded batches')
    # Clip continuous components in their ORIGINAL dtype, as the native helper.
    bounds = data.standardized_continuous_bounds if data.standardized else data.continuous_bounds
    for name, (kind, indices) in data.train_feature_index_map.items():
        if kind == 'cont':
            low, high = bounds[name]
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError('invalid continuous bounds')
            rec[:, indices] = np.clip(rec[:, indices], low, high)
    # Decoder accepts tensors, including the clamped clone made below.
    import torch
    decoded_a = data.decode_batch(truth)
    decoded_b = data.decode_batch(torch.from_numpy(rec))
    costs = decoded_costs(decoded_a, decoded_b, data.create_tolerance_map())
    rows, cols = linear_sum_assignment(costs)
    mse = float(np.mean((rec[cols].astype(np.float64) - a[rows].astype(np.float64)) ** 2))
    return dict(feature_accuracy=1-float(np.mean(costs[rows, cols])),
                standardized_mse=mse, assignment=cols.tolist())


def verify_score(stored, recomputed, prior_mse, mse_tolerance=1e-12):
    """Accuracy/assignment exact; float64 MSE reduction absolute <=1e-12.

    Reduction tolerance is audit-only, fixed before this audit sees results.
    Measurement scores, ranks, ties and tests always use frozen stored values.
    """
    for key in ('feature_accuracy', 'assignment'):
        if stored[key] != recomputed[key]:
            raise ValueError('independent metric mismatch: ' + key)
    error = abs(stored['standardized_mse'] - recomputed['standardized_mse'])
    if not math.isfinite(error) or error > mse_tolerance:
        raise ValueError('independent float64 MSE reduction mismatch')
    if stored['bounded_mse'] != min(stored['standardized_mse'], prior_mse):
        raise ValueError('bounded score mismatch')
    return error
