"""Fixed 72-test P34C family and paired order-statistic summaries."""
from __future__ import annotations
import math
import numpy as np

DATASETS = ("paysim", "ieee_cis", "baf")
CELLS = ("ratio_batch1", "tableak_batch1", "tableak_batch2")
DNA = ("dna_v1_conservative", "dna_v2_0p95")
COMPARATORS = ("distortion_matched", "local_dp_eps10")
DIRECTIONS = ("dna_lower_recovery", "dna_higher_recovery")


def family():
    return [(d, c, arm, comparator, direction) for d in DATASETS for c in CELLS
            for arm in DNA for comparator in COMPARATORS for direction in DIRECTIONS]


def exact_tail(wins, losses):
    if not isinstance(wins, int) or not isinstance(losses, int) or wins < 0 or losses < 0:
        raise ValueError("invalid integer sign counts")
    n = wins + losses
    return sum(math.comb(n, k) for k in range(wins, n+1)) / 2**n if n else 1.


def paired_summary(dna, dp):
    a, b = np.asarray(dna, dtype=np.float64), np.asarray(dp, dtype=np.float64)
    if a.shape != (39,) or b.shape != (39,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("require 39 complete finite paired observations; no missing-output analysis")
    diff = a-b
    ordered = np.sort(diff)
    lower, higher = int(np.sum(diff < 0)), int(np.sum(diff > 0))
    return dict(dna_median=float(np.median(a)), dp_median=float(np.median(b)),
                dna_order_interval=np.sort(a)[[12, 26]].tolist(), dp_order_interval=np.sort(b)[[12, 26]].tolist(),
                paired_effect_median=float(np.median(diff)),
                paired_effect_order_interval=[float(ordered[12]), float(ordered[26])],
                interval_ranks=[13, 27], n=39, ties=39-lower-higher, effective_n=lower+higher,
                dna_lower_wins=lower, dna_higher_wins=higher,
                raw_p_lower=exact_tail(lower, higher), raw_p_higher=exact_tail(higher, lower))


def fixed_holm(raw):
    if len(raw) != 72:
        raise ValueError("P34C requires the complete fixed72 family")
    p = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("invalid p values")
    order = sorted(range(72), key=lambda i: (p[i], i))
    adjusted, previous = [0.] * 72, 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., float((72-rank)*p[index])))
        adjusted[index] = previous
    return adjusted


def evaluate_family(pairs, not_assessable):
    expected = {(d, c, a, comparator) for d in DATASETS for c in CELLS for a in DNA for comparator in COMPARATORS}
    if set(pairs) & set(not_assessable) or set(pairs) | set(not_assessable) != expected:
        raise ValueError("family coverage must be complete, disjoint and explicitly gated")
    summaries = {key: paired_summary(*values) for key, values in pairs.items()}
    rows = []
    for dataset, cell, arm, comparator, direction in family():
        key = (dataset, cell, arm, comparator)
        row = dict(dataset=dataset, cell=cell, defense=arm, comparator=comparator, direction=direction)
        if key in not_assessable:
            row.update(status="NOT_ASSESSABLE", reason=not_assessable[key], raw_p=1., summary=None)
        else:
            summary = summaries[key]
            row.update(status="ASSESSABLE", raw_p=summary["raw_p_lower" if direction == DIRECTIONS[0] else "raw_p_higher"], summary=summary)
        rows.append(row)
    for row, adjusted in zip(rows, fixed_holm([row["raw_p"] for row in rows])):
        row["holm_p"] = adjusted
    return rows
