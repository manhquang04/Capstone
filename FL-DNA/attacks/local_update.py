"""CPU local Adam replay for bounded, known-order update inversion."""
import torch
from contextlib import contextmanager
from torch.func import functional_call


@contextmanager
def _batchnorm_state(model, initial):
    state = dict(initial)
    handles = []
    def make_hook(name):
        def hook(module, inputs):
            if not module.training or not module.track_running_stats:
                return
            value = inputs[0]
            key = name + '.num_batches_tracked'
            state[key] = state[key] + 1
            momentum = module.momentum if module.momentum is not None else 1.0 / float(state[key])
            axes = (0,) + tuple(range(2, value.ndim))
            for suffix, statistic in [('running_mean', value.mean(axes)),
                                      ('running_var', value.var(axes, unbiased=True))]:
                key = name + '.' + suffix
                state[key] = (1-momentum)*state[key] + momentum*statistic
        return hook
    try:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                handles.append(module.register_forward_pre_hook(make_hook(name)))
        yield state
    finally:
        for handle in handles:
            handle.remove()


def simulate(model, criterion, x, y, batches, rng_state, lr=1e-3, create_graph=True):
    """Fresh Adam state, train mode, fixed dropout realization; return full state delta."""
    params = dict(model.named_parameters())
    initial = {k: v.detach().clone() for k, v in params.items()}
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    initial_buffers = {k: v.clone() for k, v in buffers.items()}
    first = {k: torch.zeros_like(v) for k, v in params.items()}
    second = {k: torch.zeros_like(v) for k, v in params.items()}
    with _batchnorm_state(model, initial_buffers) as tracked, torch.random.fork_rng(devices=[]):
        torch.set_rng_state(rng_state)
        for step, ids in enumerate(batches, 1):
            logits = functional_call(model, (params, buffers), (x[ids],))
            grads = torch.autograd.grad(criterion(logits, y[ids]), tuple(params.values()), create_graph=create_graph)
            updated = {}
            for (name, value), grad in zip(params.items(), grads):
                first[name] = torch.lerp(first[name], grad, .1)
                second[name] = torch.addcmul(second[name] * .999, grad, grad, value=.001)
                variance = second[name]
                # At exactly zero variance Adam has no useful square-root derivative.
                # Preserve sqrt(0)=0 forward and assign derivative zero at that point.
                root = torch.where(variance > 0, variance.clamp_min(torch.finfo(variance.dtype).tiny).sqrt(), torch.zeros_like(variance))
                denominator = root / (1 - .999 ** step) ** .5 + 1e-8
                updated[name] = torch.addcdiv(value, first[name], denominator, value=-lr / (1 - .9 ** step))
            params = updated
    delta = {k: v - initial[k] for k, v in params.items()}
    delta.update({k: v - initial_buffers[k] for k, v in tracked.items()})
    return delta


def objective(delta, observed, names):
    return torch.stack([(delta[k] - observed[k]).square().mean() for k in names]).mean()


def simulate_sgd(model, criterion, x, y, batches, rng_state, lr=1e-3, create_graph=True):
    """Differentiable train-mode SGD replay used only for bounded Level-1 validation."""
    params = dict(model.named_parameters())
    initial = {k: v.detach().clone() for k, v in params.items()}
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    initial_buffers = {k: v.clone() for k, v in buffers.items()}
    differentiable_buffers = dict(initial_buffers)
    handles = []
    def capture_stats(name):
        def hook(module, inputs):
            value = inputs[0]
            if not module.training or not module.track_running_stats:
                return
            count_key = name + '.num_batches_tracked'
            count = differentiable_buffers[count_key] + 1
            differentiable_buffers[count_key] = count
            factor = module.momentum if module.momentum is not None else 1.0 / float(count)
            axes = (0,) + tuple(range(2, value.ndim))
            for suffix, statistic in [('running_mean', value.mean(axes)),
                                      ('running_var', value.var(axes, unbiased=True))]:
                key = name + '.' + suffix
                differentiable_buffers[key] = (1-factor)*differentiable_buffers[key] + factor*statistic
        return hook
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            handles.append(module.register_forward_pre_hook(capture_stats(name)))
    try:
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(rng_state)
            for ids in batches:
                logits = functional_call(model, (params, buffers), (x[ids],))
                grads = torch.autograd.grad(criterion(logits, y[ids]), tuple(params.values()), create_graph=create_graph)
                params = {name: value - lr * grad for (name, value), grad in zip(params.items(), grads)}
    finally:
        for handle in handles:
            handle.remove()
    delta = {k: v - initial[k] for k, v in params.items()}
    delta.update({k: v - initial_buffers[k] for k, v in differentiable_buffers.items()})
    return delta
