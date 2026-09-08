"""CPU local Adam replay for bounded, known-order update inversion."""
import torch
from torch.func import functional_call


def simulate(model, criterion, x, y, batches, rng_state, lr=1e-3, create_graph=True):
    """Fresh Adam state, train mode, fixed dropout realization; return full state delta."""
    params = dict(model.named_parameters())
    initial = {k: v.detach().clone() for k, v in params.items()}
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    initial_buffers = {k: v.clone() for k, v in buffers.items()}
    first = {k: torch.zeros_like(v) for k, v in params.items()}
    second = {k: torch.zeros_like(v) for k, v in params.items()}
    with torch.random.fork_rng(devices=[]):
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
    delta.update({k: v - initial_buffers[k] for k, v in buffers.items()})
    return delta


def objective(delta, observed, names):
    return torch.stack([(delta[k] - observed[k]).square().mean() for k in names]).mean()
