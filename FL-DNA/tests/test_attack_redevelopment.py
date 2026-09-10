from types import SimpleNamespace

import torch

from attacks.adaptive_dna import transform_state_bpda
from attacks.tabular_parameterization import PaySimManifold, update_matching_objective
from dna_encoder.transform_defense import DNATransformConfig, transform_update_array


def _metadata():
    return SimpleNamespace(
        numeric_center=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.0, 0.0],
        numeric_scale=[2.0, 4.0, 5.0, 10.0, 20.0, 25.0, 2.0, 5.0],
        type_categories=list("ABCDE"),
    )


def test_manifold_derives_balance_features_and_is_differentiable():
    manifold = PaySimManifold.from_metadata(_metadata(), dtype=torch.float64)
    latent = torch.zeros((2, manifold.latent_dim), dtype=torch.float64, requires_grad=True)
    decoded = manifold.decode(latent)
    # raw orig difference = 30 - 40; raw dest difference = 60 - 50
    torch.testing.assert_close(decoded[:, 6], torch.full((2,), -5.0, dtype=torch.float64))
    torch.testing.assert_close(decoded[:, 7], torch.full((2,), 2.0, dtype=torch.float64))
    torch.testing.assert_close(decoded[:, 8:].sum(1), torch.ones(2, dtype=torch.float64))
    decoded.sum().backward()
    assert torch.isfinite(latent.grad).all()


def test_balanced_objective_prefers_exact_update():
    observed = {"weight": torch.tensor([1.0, -2.0]), "running_mean": torch.tensor([0.5])}
    exact = update_matching_objective(observed, observed, list(observed), mode="balanced_bn", bn_weight=3.0)
    wrong = update_matching_objective(
        {key: torch.zeros_like(value) for key, value in observed.items()},
        observed,
        list(observed),
        mode="balanced_bn",
        bn_weight=3.0,
    )
    assert exact.item() < wrong.item()


def test_zero_signal_objective_does_not_read_a_reference_update():
    candidate = {"weight": torch.tensor([1.0, 2.0], requires_grad=True)}
    zero = {"weight": torch.zeros(2)}
    loss = update_matching_objective(candidate, zero, ["weight"], mode="balanced_bn")
    torch.testing.assert_close(loss, torch.tensor(2.5))


def test_dna_bpda_uses_exact_forward_and_identity_backward():
    value = torch.tensor([0.2, -0.4, 0.1, 0.9], requires_grad=True)
    config = DNATransformConfig(block_size=4, mix_ratio=0.08, keep_ratio=0.88, shrink_factor=0.45, seed=123)
    transformed = transform_state_bpda({"weight": value}, config)["weight"]
    expected, _ = transform_update_array(value.detach().numpy(), config, tensor_index=0)
    torch.testing.assert_close(transformed, torch.from_numpy(expected))
    transformed.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))
