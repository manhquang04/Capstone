"""Pure target allocator: no records, model outputs, or artifact writes.

Historical manifest discovery/provenance verification is a separate required
audit. This allocator must not be used to claim that discovery is complete.
"""
from __future__ import annotations
import hashlib
import json
import numbers
import numpy as np

STAGES = (("n8", 8), ("n24", 24), ("n39", 39))
ROUNDS = (10, 30)
BATCHES = (4, 8, 16)
DATASETS = ("baf", "ieee_cis", "paysim")


def integer_ids(values):
    values = list(values)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, numbers.Integral) for i in values):
        raise ValueError("source IDs must be integers, not coerced values")
    return [int(i) for i in values]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pools(partitions, historical):
    if len(partitions) != 3:
        raise ValueError("exactly three client partitions required")
    partitions = [integer_ids(part) for part in partitions]
    flat = [i for part in partitions for i in part]
    if len(flat) != len(set(flat)):
        raise ValueError("duplicate source ID within/across client partitions")
    excluded = set(integer_ids(historical))
    return partitions, excluded


def reserve(dataset, partitions, historical, seed):
    if dataset not in DATASETS or isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError("invalid dataset/seed")
    partitions, excluded = pools(partitions, historical)
    rng = np.random.default_rng(int(seed))
    queues = [list(map(int, rng.permutation(sorted(set(part) - excluded)))) for part in partitions]
    positions = [0, 0, 0]
    targets = []
    for checkpoint in ROUNDS:
        for batch in BATCHES:
            for stage, count in STAGES:
                for index in range(count):
                    client = index % 3
                    start = positions[client]
                    ids = queues[client][start:start + batch]
                    if len(ids) != batch:
                        raise ValueError("insufficient fresh client sources; no replacement")
                    positions[client] += batch
                    targets.append(dict(checkpoint=checkpoint, batch=batch, stage=stage,
                                        index=index, client=client, source_ids=ids))
    doc = dict(dataset=dataset, seed=int(seed), targets=targets,
               partition_sha256=digest([sorted(p) for p in partitions]),
               historical_sha256=digest(sorted(excluded)))
    validate(doc, partitions, historical)
    return doc


def validate(doc, partitions, historical):
    partitions, excluded = pools(partitions, historical)
    if doc["dataset"] not in DATASETS:
        raise ValueError("unknown dataset")
    if doc["partition_sha256"] != digest([sorted(p) for p in partitions]) or doc["historical_sha256"] != digest(sorted(excluded)):
        raise ValueError("firewall input drift")
    expected = [(r, b, s, i, i % 3) for r in ROUNDS for b in BATCHES
                for s, n in STAGES for i in range(n)]
    if len(doc["targets"]) != len(expected):
        raise ValueError("missing/extra targets")
    seen = set()
    for target, identity in zip(doc["targets"], expected):
        if tuple(target[k] for k in ("checkpoint", "batch", "stage", "index", "client")) != identity:
            raise ValueError("target schema/order drift")
        ids = integer_ids(target["source_ids"])
        if len(ids) != target["batch"] or len(ids) != len(set(ids)):
            raise ValueError("batch size/duplicate source violation")
        selected = set(ids)
        if selected & (seen | excluded) or not selected <= set(partitions[target["client"]]):
            raise ValueError("historical overlap, target reuse or wrong client")
        seen.update(selected)
    return True
