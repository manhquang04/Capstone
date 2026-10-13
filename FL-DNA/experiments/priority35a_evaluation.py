"""Evaluator-only training references, blind matching and qualification gates."""
import math
import numpy as np
from experiments.priority35a_estimator import schema_blocks, public_seed
from experiments.priority35a_metrics import aligned_feature_accuracy, standardized_mse, bounded_score


def logical(array, blocks):
    array = np.asarray(array)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError("invalid encoded batch")
    columns = []
    for block in blocks:
        values = array[:, block["indices"]]
        if block["categories"] is None:
            columns.append(values[:, 0].astype(object))
        else:
            columns.append(np.array(block["categories"], dtype=object)[values.argmax(axis=1)])
    return np.stack(columns, axis=1)


def reference_statistics(training, meta):
    blocks = schema_blocks(meta)
    # Float64 accumulation, population SD with the locked ddof=1.
    std = np.std(training, axis=0, dtype=np.float64, ddof=1)
    mean = np.mean(training, axis=0, dtype=np.float64)
    if not np.isfinite(std).all() or not np.isfinite(mean).all() or (std < 0).any():
        raise ValueError("nonfinite evaluator population statistics")
    prior = mean.copy()
    tolerances = []
    for block in blocks:
        if block["categories"] is None:
            tolerances.append(.319 * float(std[block["indices"][0]]))
        else:
            idx = block["indices"]
            prior[idx] = 0
            prior[idx[int(mean[idx].argmax())]] = 1
            tolerances.append("cat")
    return dict(blocks=blocks, std=std.tolist(), prior=prior.tolist(), tolerances=tolerances,
                frequency_encoded_as_continuous=[b["name"] for b in blocks if b["name"].endswith(":frequency")])


def marginal(training, stats, batch, ordinal):
    rng = np.random.default_rng(public_seed("empirical-marginal-reference", ordinal))
    result = np.empty((batch, training.shape[1]), dtype=np.float64)
    # Independent row draws per logical coordinate/category, never a full row.
    for block in stats["blocks"]:
        rows = rng.integers(0, len(training), size=batch)
        idx = block["indices"]
        result[:, idx] = training[rows][:, idx]
    if not np.isfinite(result).all():
        raise ValueError("nonfinite empirical marginal")
    return result


def quality(truth, guess, stats, prior_mse=None):
    result = aligned_feature_accuracy(logical(truth, stats["blocks"]), logical(guess, stats["blocks"]), stats["tolerances"])
    aligned = np.asarray(guess)[result["guess_indices"]]
    mse = standardized_mse(truth, aligned, stats["std"])
    return dict(feature_accuracy=result["feature_accuracy"], standardized_mse=mse,
                bounded_score=bounded_score(mse, mse if prior_mse is None else prior_mse),
                assignment=result["guess_indices"].tolist(),
                absent_type_denominator=result["absent_type_denominator"])


def sign_test(differences):
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("missing/nonfinite sign-test units")
    wins, losses, ties = int((values > 0).sum()), int((values < 0).sum()), int((values == 0).sum())
    n = wins + losses
    probability = sum(math.comb(n, i) for i in range(wins, n + 1)) / 2**n if n else 1.
    return dict(n=len(values), wins=wins, losses=losses, ties=ties, p_one_sided=probability,
                median_difference=float(np.median(values)))


def gate(rows, expected):
    if len(rows) != expected or sorted(row["index"] for row in rows) != list(range(expected)):
        raise ValueError("missing/duplicate gate target")
    tests = {control: sign_test([row["estimator"]["feature_accuracy"] - row[control]["feature_accuracy"] for row in rows])
             for control in ("mean_mode", "empirical_marginal", "decoy")}
    return dict(n=expected, tests=tests, passed=all(t["p_one_sided"] < .05 for t in tests.values()),
                worst_p=max(t["p_one_sided"] for t in tests.values()))
