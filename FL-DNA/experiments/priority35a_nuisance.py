"""Build the declared server surrogate from non-BN public tensors only."""
from collections import OrderedDict
import torch
from experiments.priority34a_local_bn import p, domains
from experiments.priority35a_adam import require_finite_cpu


def estimator_model(public_non_bn, input_dimension, public_initialization_seed):
    # This function intentionally has no client BN, record, dropout-mask or
    # private RNG input. Reject extra tensor names rather than ignoring them.
    require_finite_cpu(public_non_bn, "public non-BN input")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(public_initialization_seed)
        model = p.FraudMLP(input_dimension).cpu()
    bn, allowed = domains(model)
    if tuple(public_non_bn) != allowed or set(public_non_bn) & bn:
        raise ValueError("estimator accepts the exact non-BN upload allowlist only")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in public_non_bn:
                value = public_non_bn[name]
                if value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError("public parameter metadata mismatch")
                parameter.copy_(value)
    # Dropout-only identity approximation; BN remains in training mode.
    for prefix, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Dropout):
            parent_name, child_name = prefix.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, torch.nn.Identity())
    model.train()
    require_finite_cpu(model.state_dict(), "server surrogate initialization")
    nuisance = OrderedDict((name, parameter) for name, parameter in model.named_parameters() if name in bn)
    if not nuisance or any(not parameter.requires_grad for parameter in nuisance.values()):
        raise ValueError("BN affine nuisance variables must be free")
    return model, nuisance
