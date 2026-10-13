"""Prospective Config-C server-side estimators; public inputs only.

No evaluator data, source IDs, client BN tensors, or dropout RNG are accepted.
Smaller budgets are available only through the explicitly synthetic test API.
"""
from collections import OrderedDict
import hashlib
import json
import time
import numpy as np
import torch
from experiments.priority35a_adam import fresh_adam_parameters, parameter_difference, require_finite_cpu
from experiments.priority35a_nuisance import estimator_model
from experiments.priority35a_labels import canonical_labels, estimate_label_count
from experiments.priority34a_local_bn import p


def public_seed(*parts):
    value = json.dumps(["priority35a-public-estimator-v1", *parts], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big")


def schema_blocks(meta):
    names = meta["feature_names"]
    if len(names) != len(set(names)):
        raise ValueError("duplicate encoded features")
    categories = {f"{key}={level}": key for key, levels in meta["one_hot_levels"].items() for level in levels}
    blocks = []
    for i, name in enumerate(names):
        if name in categories:
            parent = categories[name]
            levels = meta["one_hot_levels"][parent]
            if name != f"{parent}={levels[0]}":
                continue
            indices = [names.index(f"{parent}={level}") for level in levels]
            if indices != list(range(i, i + len(levels))):
                raise ValueError("noncontiguous categorical block")
            blocks.append(dict(indices=indices, categories=levels, name=parent))
        else:
            blocks.append(dict(indices=[i], categories=None, name=name))
    if sorted(i for block in blocks for i in block["indices"]) != list(range(len(names))):
        raise ValueError("schema does not cover encoded features exactly")
    return blocks


def candidate_rows(raw, blocks, hard=False):
    columns = []
    for block in blocks:
        value = raw[:, block["indices"]]
        if block["categories"] is not None:
            value = (torch.nn.functional.one_hot(value.argmax(dim=1), len(block["indices"])).to(raw.dtype)
                     if hard else value.softmax(dim=1))
        columns.append(value)
    return torch.cat(columns, dim=1)


def flatten(state):
    return torch.cat([value.reshape(-1) for value in state.values()])


def observable(model, rows, labels, public):
    params = OrderedDict((name, param) for name, param in model.named_parameters() if name in public)
    logits = model(rows)
    require_finite_cpu(model.state_dict(), "surrogate forward")
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite surrogate logits")
    loss = p.BinaryFocalLoss(alpha=.95, gamma=2.)(logits, labels)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite surrogate focal loss")
    values = torch.autograd.grad(loss, tuple(params.values()), create_graph=True)
    gradients = OrderedDict(zip(params, values))
    after = fresh_adam_parameters(params, gradients)
    return flatten(parameter_difference(params, after))


def objective(candidate, observed, instrument):
    if instrument not in ("tableak", "wan") or candidate.shape != observed.shape:
        raise ValueError("invalid observable objective")
    if not torch.isfinite(candidate).all() or not torch.isfinite(observed).all():
        raise ValueError("nonfinite observable")
    result = 1 - torch.nn.functional.cosine_similarity(candidate, observed, dim=0, eps=1e-12)
    if instrument == "tableak":
        result = result + .1 * (candidate - observed).square().sum() / (observed.square().sum() + 1e-12)
    if not torch.isfinite(result):
        raise ValueError("nonfinite objective")
    return result


def _fit(public, observed, blocks, labels, instrument, seed, iterations, anneal):
    torch.set_num_threads(1)
    model, nuisance = estimator_model(public, sum(len(b["indices"]) for b in blocks), seed)
    generator = torch.Generator().manual_seed(seed)
    raw = torch.nn.Parameter(torch.randn(len(labels), model.network[0].in_features, generator=generator))
    variables = [raw, *nuisance.values()]
    optimizer = torch.optim.Adam(variables, lr=.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=0.) if anneal else None
    trace, zero_norm_count = [], 0
    for iteration in range(iterations):
        optimizer.zero_grad()
        model.zero_grad()
        candidate = observable(model, candidate_rows(raw, blocks), labels, public)
        zero_norm_count += int(float(candidate.detach().norm()) == 0)
        value = objective(candidate, observed, instrument)
        value.backward()
        if any(v.grad is None or not torch.isfinite(v.grad).all() for v in variables):
            raise ValueError("missing/nonfinite reconstruction gradient")
        if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):
            raise ValueError("nonfinite surrogate parameter gradient")
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        require_finite_cpu(model.state_dict(), "surrogate optimized state")
        require_finite_cpu({"candidate": raw}, "optimized candidate")
        if iteration == 0 or (iteration + 1) % 500 == 0 or iteration + 1 == iterations:
            trace.append(dict(iteration=iteration + 1, before_step_objective=float(value.detach())))
    final = objective(observable(model, candidate_rows(raw, blocks), labels, public), observed, instrument)
    # The non-BN parameters are differentiable inputs but never optimizer variables.
    if any(not torch.equal(dict(model.named_parameters())[name].detach(), value) for name, value in public.items()):
        raise ValueError("fixed public non-BN parameter changed")
    return dict(rows=candidate_rows(raw.detach(), blocks, hard=True), raw=raw.detach().clone(),
                nuisance=OrderedDict((name, value.detach().clone()) for name, value in nuisance.items()),
                final_objective=float(final.detach()), trace=trace, zero_candidate_norms=zero_norm_count)


def reconstruct(public, observed, meta, batch, label_setting, instrument, ordinal, known_count=None, *, synthetic=False, test_iterations=None):
    if batch not in (4, 8, 16) or label_setting not in ("given", "estimated"):
        raise ValueError("Config-C batch/label-setting contract")
    if test_iterations is not None and not synthetic:
        raise ValueError("real reconstruction budgets cannot be overridden")
    require_finite_cpu(public, "public shared state")
    require_finite_cpu(observed, "public measured upload")
    if tuple(public) != tuple(observed):
        raise ValueError("public upload key mismatch")
    for name in public:
        if public[name].shape != observed[name].shape or public[name].dtype != observed[name].dtype:
            raise ValueError("public upload metadata mismatch")
    blocks = schema_blocks(meta)
    flat_observed = flatten(observed).detach()
    started = time.time()
    label_receipt = None
    identity = [ordinal, batch, label_setting, instrument]
    if label_setting == "given":
        if known_count is None:
            raise ValueError("missing unordered known-label multiset")
        labels = canonical_labels(batch, known_count)
    else:
        if known_count is not None:
            raise ValueError("unknown-label estimator cannot receive true count")
        count_seed = public_seed(*identity, "label-enumeration")
        labels, label_receipt = estimate_label_count(batch, lambda candidate_labels: _fit(
            public, flat_observed, blocks, candidate_labels, instrument, count_seed,
            test_iterations if synthetic and test_iterations is not None else 64, False)["final_objective"])
    starts = [_fit(public, flat_observed, blocks, labels, instrument, public_seed(*identity, "start", i),
                   test_iterations if synthetic and test_iterations is not None else 4000, True) for i in range(2)]
    chosen = min(range(2), key=lambda i: (starts[i]["final_objective"], i))
    return dict(**starts[chosen], chosen_start=chosen, start_objectives=[s["final_objective"] for s in starts],
                start_traces=[s["trace"] for s in starts], labels=labels, label_receipt=label_receipt,
                torch_threads=torch.get_num_threads(), elapsed_seconds=time.time()-started,
                observed_zero_norm=bool(float(flat_observed.norm()) == 0), synthetic=synthetic)
