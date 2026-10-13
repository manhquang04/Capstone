"""Synthetic no-BN correctness reference; NOT_APPLICABLE to Config-C."""
import torch


def synthetic_single_record_ratio(weight_gradient, bias_gradient):
    """Recover a linear-layer input from gradients, never Adam differences."""
    if weight_gradient.device.type != "cpu" or bias_gradient.device.type != "cpu":
        raise ValueError("CPU tensors required")
    if weight_gradient.ndim != 2 or bias_gradient.ndim != 1:
        raise ValueError("expected matrix weight and vector bias gradients")
    if weight_gradient.shape[0] != bias_gradient.shape[0]:
        raise ValueError("gradient dimensions differ")
    if not torch.isfinite(weight_gradient).all() or not torch.isfinite(bias_gradient).all():
        raise ValueError("non-finite gradient")
    usable = torch.nonzero(bias_gradient != 0, as_tuple=False).flatten()
    if not len(usable):
        raise ValueError("ratio undefined: all bias gradients zero")
    result = weight_gradient[usable] / bias_gradient[usable, None]
    if not torch.isfinite(result).all():
        raise ValueError("non-finite ratio")
    return result
