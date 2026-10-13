"""Observable-objective label-count rule, adapted to binary focal loss.

The caller supplies each count's blinded, fixed-budget update-matching score.
This rule never accepts true labels, true records or evaluator recovery scores.
"""
from __future__ import annotations
import math
import numbers
import torch


def canonical_labels(batch_size, fraud_count):
    if any(isinstance(v, bool) or not isinstance(v, numbers.Integral)
           for v in (batch_size, fraud_count)):
        raise ValueError("batch size/count must be integers")
    if batch_size < 1 or not 0 <= fraud_count <= batch_size:
        raise ValueError("invalid batch size/count")
    return torch.cat((torch.zeros(batch_size - fraud_count), torch.ones(fraud_count)))[:, None]


def estimate_label_count(batch_size, observable_score):
    canonical_labels(batch_size, 0)
    rows = []
    for count in range(batch_size + 1):
        value = observable_score(canonical_labels(batch_size, count))
        if torch.is_tensor(value):
            if value.device.type != "cpu" or value.numel() != 1:
                raise ValueError("expected one CPU observable score")
            value = value.detach().item()
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError("non-finite/invalid label candidate objective")
        rows.append(dict(fraud_count=count, observable_objective=float(value)))
    best = min(rows, key=lambda r: (r["observable_objective"], r["fraud_count"]))
    tied = [r["fraud_count"] for r in rows if r["observable_objective"] == best["observable_objective"]]
    return canonical_labels(batch_size, best["fraud_count"]), dict(
        rule="binary_focal_observable_count_enumeration", candidates=rows,
        estimated_fraud_count=best["fraud_count"], exact_tied_counts=tied,
        tie_rule="smallest count", synthetic_success_is_not_identifiability_proof=True)
