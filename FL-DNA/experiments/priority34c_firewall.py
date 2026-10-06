"""Pure source-ID reservations; no target records or outcomes accepted."""
from __future__ import annotations
import hashlib
import json
import numbers
import numpy as np

SEEDS = {"paysim": 340320, "ieee_cis": 340321, "baf": 340322}
CELLS = (("ratio_batch1", 1), ("tableak_batch1", 1), ("tableak_batch2", 2))
STAGES = (("n8", 8), ("n24", 24), ("development24", 24), ("n39", 39))


def integer_ids(values):
    values = list(values)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, numbers.Integral) for i in values):
        raise ValueError("source IDs must be integers")
    return [int(i) for i in values]


def digest(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def reserve(dataset, source_ids, historical_ids):
    if dataset not in SEEDS:
        raise ValueError("unknown dataset")
    source = integer_ids(source_ids)
    if len(source) != len(set(source)):
        raise ValueError("duplicate source IDs in training pool")
    excluded = set(integer_ids(historical_ids))
    pool = sorted(set(source) - excluded)
    required = sum(batch * sum(n for _, n in STAGES) for _, batch in CELLS)
    if len(pool) < required:
        raise ValueError("insufficient disjoint source IDs; no replacement")
    permutation = np.random.default_rng(SEEDS[dataset]).permutation(len(pool))[:required]
    chosen = [pool[int(i)] for i in permutation]
    cells, offset = {}, 0
    for cell, batch in CELLS:
        groups = {}
        for stage, n in STAGES:
            groups[stage] = [chosen[offset + i * batch:offset + (i + 1) * batch] for i in range(n)]
            offset += n * batch
        cells[cell] = groups
    doc = dict(dataset=dataset, seed=SEEDS[dataset], source_pool_sha256=digest(sorted(source)),
               historical_ids_sha256=digest(sorted(excluded)), excluded_count=len(excluded),
               total_reserved=required, source_ids=cells)
    validate(doc, source, excluded)
    return doc


def validate(doc, source_ids, historical_ids):
    source = integer_ids(source_ids)
    if len(source) != len(set(source)):
        raise ValueError("duplicate source pool")
    excluded = set(integer_ids(historical_ids))
    dataset = doc["dataset"]
    if dataset not in SEEDS or doc["seed"] != SEEDS[dataset]:
        raise ValueError("dataset seed drift")
    if doc["source_pool_sha256"] != digest(sorted(source)) or doc["historical_ids_sha256"] != digest(sorted(excluded)):
        raise ValueError("firewall input drift")
    if set(doc["source_ids"]) != {c for c, _ in CELLS}:
        raise ValueError("cell schema drift")
    flattened = []
    for cell, batch in CELLS:
        groups = doc["source_ids"][cell]
        if set(groups) != {s for s, _ in STAGES}:
            raise ValueError("stage schema drift")
        for stage, n in STAGES:
            if len(groups[stage]) != n or any(len(g) != batch for g in groups[stage]):
                raise ValueError("target budget drift")
            flattened.extend(i for group in groups[stage] for i in integer_ids(group))
    if len(flattened) != doc["total_reserved"] or len(flattened) != len(set(flattened)):
        raise ValueError("targets overlap or count drift")
    if set(flattened) & excluded or not set(flattened) <= set(source):
        raise ValueError("historical overlap or unknown source")
    return True
