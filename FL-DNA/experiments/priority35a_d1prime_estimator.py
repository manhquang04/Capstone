"""Privileged descriptive known-mask reconstruction; no qualification changes."""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import time
from collections import OrderedDict
import torch
from experiments import priority34a_local_bn as old
from experiments.priority35a_adam import require_finite_cpu
from experiments.priority35a_diagnostic_estimator import oracle_model, operator
from experiments.priority35a_estimator import candidate_rows, schema_blocks, public_seed, flatten, objective

BUDGETS = {"a": (2, 4000), "b": (30, 1500)}
MASK_LAYOUT = {"network.3": (128, .3), "network.7": (64, .2), "network.11": (32, .1)}


class FixedDropout(torch.nn.Module):
    def __init__(self, multiplier, probability):
        super().__init__()
        require_finite_cpu({"mask": multiplier}, "dropout mask")
        scale = torch.tensor(1. / (1. - probability), dtype=torch.float32)
        if multiplier.dtype != torch.float32 or not ((multiplier == 0) | (multiplier == scale)).all():
            raise ValueError("invalid native scaled Bernoulli mask")
        self.register_buffer("multiplier", multiplier.detach().clone())

    def forward(self, rows):
        if rows.shape != self.multiplier.shape:
            raise ValueError("fixed dropout row/width mismatch")
        return rows * self.multiplier


def mask_model(shared, bn, dimension, seed, masks):
    if set(masks) != set(MASK_LAYOUT):
        raise ValueError("missing/extra dropout layer masks")
    model = oracle_model(shared, bn, dimension, seed)
    for prefix, (width, probability) in MASK_LAYOUT.items():
        if masks[prefix].shape != (4, width):
            raise ValueError("dropout shape contract")
        parent, index = prefix.rsplit(".", 1)
        setattr(model.get_submodule(parent), index, FixedDropout(masks[prefix], probability))
    model.train()
    return model


def capture_masks(public, private):
    """Evaluator-only capture: no input rows or private RNG enter solver payload."""
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        model = old.p.FraudMLP(private["features"].shape[1]).cpu()
        state = model.state_dict()
        state.update(private["client_bn"])
        state.update(public["shared"])
        model.load_state_dict(state, strict=True)
        model.train()
        require_finite_cpu(model.state_dict(), "native mask capture state")
        masks, handles = OrderedDict(), []
        def hook(prefix):
            def save(module, inputs, output):
                # PyTorch CPU dropout forward stores the actual scaled mask for backward.
                if not isinstance(module, torch.nn.Dropout) or not hasattr(output.grad_fn, "_saved_other"):
                    raise ValueError("native dropout mask unavailable; no fallback")
                multiplier = output.grad_fn._saved_other.detach().clone()
                if not torch.equal(output.detach(), inputs[0].detach() * multiplier):
                    raise ValueError("mask forward is not bit-exact")
                masks[prefix] = multiplier
            return save
        for prefix in MASK_LAYOUT:
            handles.append(model.get_submodule(prefix).register_forward_hook(hook(prefix)))
        torch.manual_seed(private["dropout_seed"])
        try:
            logits = model(private["features"])
            loss = old.p.BinaryFocalLoss(alpha=.95, gamma=2.)(logits, private["labels"])
            params = OrderedDict((k, p) for k, p in model.named_parameters() if k in public["shared"])
            grads = OrderedDict(zip(params, [g.detach() for g in torch.autograd.grad(loss, tuple(params.values()))]))
        finally:
            for handle in handles:
                handle.remove()
        require_finite_cpu(dict(logits=logits, **grads), "native captured gradient")
        replay = mask_model(public["shared"], private["client_bn"], private["features"].shape[1], 0, masks)
        replay_logits = replay(private["features"])
        if not torch.equal(logits, replay_logits):
            raise ValueError("known-mask logits differ bit-for-bit")
        replay_grad = operator(replay, private["features"], private["labels"], public["shared"], "D1").detach()
        if not torch.equal(replay_grad, flatten(grads)):
            raise ValueError("known-mask gradients differ bit-for-bit")
    return masks, grads


def fit(shared, observed, bn, masks, blocks, labels, instrument, seed, iterations):
    model = mask_model(shared, bn, sum(len(b["indices"]) for b in blocks), seed, masks)
    raw = torch.nn.Parameter(torch.randn(4, model.network[0].in_features,
                                       generator=torch.Generator().manual_seed(seed)))
    optimizer = torch.optim.Adam([raw], lr=.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=0.)
    trace, zeros = [], 0
    for iteration in range(iterations):
        optimizer.zero_grad()
        model.zero_grad()
        candidate = operator(model, candidate_rows(raw, blocks), labels, shared, "D1")
        zeros += int(float(candidate.detach().norm()) == 0)
        value = objective(candidate, observed, instrument)
        value.backward()
        if raw.grad is None or not torch.isfinite(raw.grad).all():
            raise ValueError("nonfinite/missing candidate derivative")
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("nonfinite model derivative")
        optimizer.step()
        scheduler.step()
        require_finite_cpu(model.state_dict(), "known-mask optimized model")
        require_finite_cpu({"raw": raw}, "known-mask optimized candidate")
        if iteration == 0 or (iteration+1) % 500 == 0 or iteration+1 == iterations:
            trace.append(dict(iteration=iteration+1, before_step_objective=float(value.detach())))
    final = objective(operator(model, candidate_rows(raw, blocks), labels, shared, "D1"), observed, instrument)
    params = dict(model.named_parameters())
    if any(not torch.equal(params[k].detach(), v) for k, v in shared.items()):
        raise ValueError("fixed shared parameters changed")
    affine = OrderedDict((k, p.detach().clone()) for k, p in params.items() if k in bn)
    if any(params[k].requires_grad or not torch.equal(v, bn[k]) for k, v in affine.items()):
        raise ValueError("true BN affine changed")
    return dict(raw=raw.detach().clone(), final_objective=float(final.detach()), trace=trace,
                oracle_affine=affine, zero_candidate_norms=zeros)


def aggregate(members, blocks, condition):
    if condition == "a":
        chosen = min(range(len(members)), key=lambda i: (members[i]["final_objective"], i))
        soft = candidate_rows(members[chosen]["raw"], blocks)
        rows = candidate_rows(members[chosen]["raw"], blocks, hard=True)
    elif condition == "b":
        chosen = None
        soft = torch.stack([candidate_rows(m["raw"], blocks) for m in members]).median(dim=0).values
        columns = []
        for block in blocks:
            v = soft[:, block["indices"]]
            columns.append(v if block["categories"] is None else
                           torch.nn.functional.one_hot(v.argmax(1), len(block["indices"])).to(v.dtype))
        rows = torch.cat(columns, dim=1)
    else:
        raise ValueError("unauthorized D1-prime condition")
    require_finite_cpu(dict(soft=soft, rows=rows), "pooled candidates")
    return chosen, soft.detach(), rows.detach()


def reconstruct(payload, instrument, condition, *, synthetic=False, test_iterations=None):
    torch.set_num_threads(1)
    if set(payload) != {"shared", "gradient", "bn", "masks", "labels", "meta", "ordinal"}:
        raise ValueError("privileged known-mask knowledge whitelist")
    if condition not in BUDGETS or (test_iterations is not None and not synthetic):
        raise ValueError("fixed scientific budget")
    labels = payload["labels"]
    require_finite_cpu({"labels": labels}, "ordered given labels")
    if labels.shape != (4, 1) or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("binary ordered label contract")
    require_finite_cpu(payload["shared"], "shared")
    require_finite_cpu(payload["gradient"], "observed gradient")
    if tuple(payload["shared"]) != tuple(payload["gradient"]) or any(
            payload["shared"][k].shape != v.shape or payload["shared"][k].dtype != v.dtype
            for k, v in payload["gradient"].items()):
        raise ValueError("gradient tensor contract")
    starts, iterations = BUDGETS[condition]
    iterations = test_iterations if synthetic and test_iterations is not None else iterations
    began = time.time()
    blocks = schema_blocks(payload["meta"])
    observed = flatten(payload["gradient"]).detach()
    members = [fit(payload["shared"], observed, payload["bn"], payload["masks"], blocks, labels,
                   instrument, public_seed(payload["ordinal"], 4, "given", instrument, "start", i), iterations)
               for i in range(starts)]
    chosen, soft, rows = aggregate(members, blocks, condition)
    model = mask_model(payload["shared"], payload["bn"], soft.shape[1], 0, payload["masks"])
    final = float(objective(operator(model, soft, labels, payload["shared"], "D1"), observed, instrument).detach())
    return dict(rows=rows, pooled_soft=soft, members=members, chosen_member=chosen, final_objective=final,
                labels=labels.clone(), condition=condition, starts=starts, iterations=iterations,
                dropout_masks_supplied=True, ordered_labels_supplied=True, torch_threads=torch.get_num_threads(),
                synthetic=synthetic, elapsed_seconds=time.time()-began)
