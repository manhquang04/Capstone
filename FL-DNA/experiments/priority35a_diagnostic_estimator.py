"""Post-hoc privileged BN diagnostic; never a Config-C qualification instrument."""
from collections import OrderedDict
import time
import torch
from experiments.priority35a_adam import require_finite_cpu, fresh_adam_parameters, parameter_difference
from experiments.priority35a_estimator import flatten, objective, candidate_rows, schema_blocks, public_seed
from experiments.priority35a_nuisance import estimator_model
from experiments.priority35a_labels import canonical_labels
from experiments import priority34a_local_bn as old


def oracle_model(public, bn_state, dimension, seed):
    model, affine = estimator_model(public, dimension, seed)
    bn, _ = old.domains(model)
    require_finite_cpu(bn_state, "authorized diagnostic BN")
    if set(bn_state) != set(bn):
        raise ValueError("oracle BN schema")
    state = model.state_dict()
    for key in bn_state:
        if state[key].shape != bn_state[key].shape or state[key].dtype != bn_state[key].dtype:
            raise ValueError("oracle BN metadata")
        state[key] = bn_state[key].clone()
    model.load_state_dict(state, strict=True)
    for value in affine.values():
        value.requires_grad_(False)
    if not model.training or any(isinstance(m, torch.nn.Dropout) for m in model.modules()):
        raise ValueError("fixed train-BN/dropout identity contract")
    return model


def operator(model, rows, labels, public, condition):
    if condition not in ("D1", "D2"):
        raise ValueError("diagnostic condition")
    params = OrderedDict((k, p) for k, p in model.named_parameters() if k in public)
    logits = model(rows)
    require_finite_cpu(model.state_dict(), "diagnostic surrogate forward")
    loss = old.p.BinaryFocalLoss(alpha=.95, gamma=2.)(logits, labels)
    if not torch.isfinite(loss) or not torch.isfinite(logits).all():
        raise ValueError("nonfinite diagnostic forward")
    values = torch.autograd.grad(loss, tuple(params.values()), create_graph=True)
    gradients = OrderedDict(zip(params, values))
    require_finite_cpu(gradients, "diagnostic surrogate gradients")
    if condition == "D1":
        return flatten(gradients)
    return flatten(parameter_difference(params, fresh_adam_parameters(params, gradients)))


def observed_gradient(public, private):
    """Evaluator-only original forward; private RNG never enters reconstruct()."""
    torch.set_num_threads(1)
    shared = public["shared"]
    require_finite_cpu(shared, "shared checkpoint")
    require_finite_cpu(private["client_bn"], "actual client BN")
    require_finite_cpu({k: private[k] for k in ("features", "labels")}, "actual batch")
    with torch.random.fork_rng(devices=[]):
        model = old.p.FraudMLP(private["features"].shape[1]).cpu()
        bn, allowed = old.domains(model)
        if set(private["client_bn"]) != set(bn) or tuple(shared) != allowed:
            raise ValueError("actual state schema")
        state = model.state_dict()
        state.update(private["client_bn"])
        model.load_state_dict(state, strict=True)
        old.broadcast(model, shared, bn, allowed)
        model.train()
        old.guard(model, "diagnostic original before")
        torch.manual_seed(private["dropout_seed"])
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        optimizer.zero_grad()
        logits = model(private["features"])
        loss = old.p.BinaryFocalLoss(alpha=.95, gamma=2.)(logits, private["labels"])
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise ValueError("nonfinite actual forward")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("nonfinite/missing actual gradient")
        gradients = OrderedDict((k, p.grad.detach().clone()) for k, p in model.named_parameters() if k in shared)
        optimizer.step()
        old.guard(model, "diagnostic original after")
        after = OrderedDict((k, model.state_dict()[k].clone()) for k in allowed)
        delta = parameter_difference(shared, after)
        if any(not torch.equal(delta[k], public["upload"][k]) for k in shared):
            raise ValueError("stored Adam upload differs bit-for-bit")
    return gradients


def sign_cosines(gradients, upload):
    gradient = flatten(gradients).detach()
    delta = flatten(upload).detach()
    sign = gradient.sign()
    require_finite_cpu({"g": gradient, "delta": delta}, "sign-like diagnostic")
    cosine = lambda value: float(torch.nn.functional.cosine_similarity(value, sign, dim=0, eps=1e-12))
    return dict(gradient_vs_sign=cosine(gradient), adam_vs_sign=cosine(delta),
                descent_aligned_adam_vs_sign=cosine(-delta),
                zero_gradient_norm=bool(gradient.norm() == 0), zero_upload_norm=bool(delta.norm() == 0))


def fit(public, observed, bn_state, blocks, labels, instrument, seed, condition, iterations):
    model = oracle_model(public, bn_state, sum(len(b["indices"]) for b in blocks), seed)
    raw = torch.nn.Parameter(torch.randn(len(labels), model.network[0].in_features,
                                       generator=torch.Generator().manual_seed(seed)))
    optimizer = torch.optim.Adam([raw], lr=.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=0.)
    trace, zeros = [], 0
    for iteration in range(iterations):
        optimizer.zero_grad()
        model.zero_grad()
        candidate = operator(model, candidate_rows(raw, blocks), labels, public, condition)
        zeros += int(float(candidate.detach().norm()) == 0)
        value = objective(candidate, observed, instrument)
        value.backward()
        if raw.grad is None or not torch.isfinite(raw.grad).all():
            raise ValueError("nonfinite/missing candidate derivative")
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("nonfinite model derivative")
        optimizer.step()
        scheduler.step()
        require_finite_cpu(model.state_dict(), "diagnostic optimized surrogate")
        require_finite_cpu({"candidate": raw}, "diagnostic optimized candidate")
        if iteration == 0 or (iteration+1) % 500 == 0 or iteration+1 == iterations:
            trace.append(dict(iteration=iteration+1, before_step_objective=float(value.detach())))
    final = objective(operator(model, candidate_rows(raw, blocks), labels, public, condition), observed, instrument)
    params = dict(model.named_parameters())
    if any(not torch.equal(params[k].detach(), v) for k, v in public.items()):
        raise ValueError("public parameters changed")
    affine = OrderedDict((k, p.detach().clone()) for k, p in params.items() if k in bn_state)
    if any(params[k].requires_grad or not torch.equal(v, bn_state[k]) for k, v in affine.items()):
        raise ValueError("known BN affine changed")
    return dict(rows=candidate_rows(raw.detach(), blocks, hard=True), raw=raw.detach().clone(),
                oracle_affine=affine, final_objective=float(final.detach()), trace=trace, zero_candidate_norms=zeros)


def reconstruct(shared, observed, bn_state, meta, batch, instrument, ordinal, known_count,
                condition, *, synthetic=False, test_iterations=None):
    torch.set_num_threads(1)
    if batch != 4 or condition not in ("D1", "D2"):
        raise ValueError("diagnostic design scope")
    if test_iterations is not None and not synthetic:
        raise ValueError("real diagnostic budgets immutable")
    require_finite_cpu(shared, "diagnostic shared")
    require_finite_cpu(observed, "diagnostic observed")
    if tuple(shared) != tuple(observed) or any(shared[k].shape != observed[k].shape or
            shared[k].dtype != observed[k].dtype for k in shared):
        raise ValueError("observed tensor contract")
    labels = canonical_labels(batch, known_count)
    blocks = schema_blocks(meta)
    started = time.time()
    iterations = test_iterations if synthetic and test_iterations is not None else 4000
    starts = [fit(shared, flatten(observed).detach(), bn_state, blocks, labels, instrument,
                  public_seed(ordinal, batch, "given", instrument, "start", i), condition, iterations) for i in range(2)]
    chosen = min(range(2), key=lambda i: (starts[i]["final_objective"], i))
    return dict(**starts[chosen], chosen_start=chosen, start_objectives=[s["final_objective"] for s in starts],
                start_traces=[s["trace"] for s in starts], labels=labels, condition=condition,
                torch_threads=torch.get_num_threads(), elapsed_seconds=time.time()-started, synthetic=synthetic,
                observed_zero_norm=bool(flatten(observed).norm() == 0), dropout_masks_supplied=False)
