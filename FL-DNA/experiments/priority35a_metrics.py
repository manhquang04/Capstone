"""Independent TabLeak-compatible batch metric and standardized MSE."""
from __future__ import annotations
import math
import numbers
import numpy as np
from scipy.optimize import linear_sum_assignment

TOLERANCE_SD = 0.319


def tolerance_map(training_std, categorical):
    if len(training_std) != len(categorical) or not len(training_std):
        raise ValueError("invalid logical feature schema")
    result = []
    for std, is_cat in zip(training_std, categorical):
        if not isinstance(is_cat, (bool, np.bool_)):
            raise ValueError("categorical indicator must be boolean")
        if not isinstance(std, numbers.Real) or not math.isfinite(std) or std < 0:
            raise ValueError("invalid training SD")
        result.append("cat" if is_cat else TOLERANCE_SD * float(std))
    return result


def aligned_feature_accuracy(truth, guess, tolerances):
    truth, guess = np.asarray(truth, dtype=object), np.asarray(guess, dtype=object)
    if truth.ndim != 2 or truth.shape != guess.shape or not truth.shape[0]:
        raise ValueError("missing/unequal two-dimensional target batches")
    if truth.shape[1] != len(tolerances) or not len(tolerances):
        raise ValueError("metric schema mismatch")
    categories = [t == "cat" if isinstance(t, str) else False for t in tolerances]
    for t, cat in zip(tolerances, categories):
        if not cat and (not isinstance(t, numbers.Real) or not math.isfinite(t) or t < 0):
            raise ValueError("invalid tolerance")
    for array in (truth, guess):
        for column, cat in enumerate(categories):
            if cat:
                if any(v is None or isinstance(v, numbers.Real) and not math.isfinite(v)
                       for v in array[:, column]):
                    raise ValueError("missing/non-finite category")
            else:
                if not np.isfinite(np.asarray(array[:, column], dtype=float)).all():
                    raise ValueError("non-finite continuous coordinate")
    n_cat, n_cont = sum(categories), len(categories) - sum(categories)
    # Reproduce native detailed matching's absent-type denominator convention.
    denominator = max(n_cat, 1) + max(n_cont, 1)
    cost = np.zeros((len(truth), len(guess)), dtype=float)
    for i, row in enumerate(truth):
        for j, candidate in enumerate(guess):
            errors = 0
            for a, b, tol, cat in zip(row, candidate, tolerances, categories):
                errors += (str(a) != str(b)) if cat else not (float(a)-tol <= float(b) <= float(a)+tol)
            cost[i, j] = errors / denominator
    rows, columns = linear_sum_assignment(cost)
    errors = cost[rows, columns]
    return dict(feature_accuracy=1-float(errors.mean()), row_indices=rows,
                guess_indices=columns, aligned_guess=guess[columns],
                per_row_error=errors, absent_type_denominator=denominator)


def standardized_mse(truth, aligned_guess, training_std):
    truth, guess = np.asarray(truth, dtype=float), np.asarray(aligned_guess, dtype=float)
    std = np.asarray(training_std, dtype=float)
    if truth.ndim != 2 or truth.shape != guess.shape or not truth.size or std.shape != (truth.shape[1],):
        raise ValueError("MSE batch/schema mismatch")
    if not all(np.isfinite(v).all() for v in (truth, guess, std)) or (std < 0).any():
        raise ValueError("non-finite/negative MSE input")
    scale = np.where(std == 0, 1., std)
    value = float(np.mean(np.square((guess-truth)/scale)))
    if not math.isfinite(value):
        raise ValueError("non-finite MSE result")
    return value


def bounded_score(mse, prior_mse):
    if any(not isinstance(v, numbers.Real) or not math.isfinite(v) or v < 0 for v in (mse, prior_mse)):
        raise ValueError("invalid bounded-score inputs")
    return float(min(mse, prior_mse))
