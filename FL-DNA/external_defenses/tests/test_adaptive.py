"""Unit tests for the external defense implementations.

Run from FL-DNA with:
PYTHONPATH=. external_defenses/.venv/bin/python -m unittest external_defenses.tests.test_adaptive
"""

from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from experiments.harness.interfaces import DefenseContext, IdentityDefense, ServerKnowledge
from external_defenses.adaptive import (
    ATSDefense,
    CountSketchDefense,
    GradientMatchingAttack,
    GradientPruningDefense,
    MaskAwareGradientAttack,
    KnownTransformAttack,
    OmitGradientAttack,
    PRECODEDefense,
    SketchGradientAttack,
    SoteriaDefense,
    StochasticNoiseAttack,
)


def context(update):
    return DefenseContext(0, 0, tuple(update), ServerKnowledge())


def update_for(model, x, label):
    loss = F.mse_loss(model(x), label)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    return {name: gradient.detach() for (name, _), gradient in zip(model.named_parameters(), gradients)}


class OneRecordLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        with torch.no_grad():
            self.linear.weight.fill_(0.7)
            self.linear.bias.fill_(0.2)

    def forward(self, x):
        return self.linear(x)


class AdaptiveDefenseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(3)
        self.update = {"linear.weight": torch.tensor([[1.0, -2.0]]), "linear.bias": torch.tensor([0.5])}

    def assert_identity(self, defense):
        transmitted = defense.encode(self.update, context(self.update))
        self.assertEqual(set(transmitted.tensors), set(self.update))
        for name, tensor in self.update.items():
            self.assertTrue(torch.equal(tensor, transmitted.tensors[name]), name)

    def test_identity_paths(self):
        self.assert_identity(GradientPruningDefense(0.0))
        self.assert_identity(SoteriaDefense("linear.weight", torch.ones(2), 0.0))
        self.assert_identity(PRECODEDefense())
        self.assert_identity(ATSDefense())
        self.assert_identity(CountSketchDefense(rows=0))
        self.assert_identity(IdentityDefense())

    def test_pruning_is_deterministic_and_masks_lowest_coordinates(self):
        defense = GradientPruningDefense(0.5)
        first = defense.encode(self.update, context(self.update))
        second = defense.encode(self.update, context(self.update))
        self.assertTrue(torch.equal(first.tensors["linear.weight"], second.tensors["linear.weight"]))
        self.assertEqual(int((first.tensors["linear.weight"] == 0).sum()), 1)

    def test_gradient_matching_recovers_trivial_undefended_linear_input(self):
        model = OneRecordLinear()
        x = torch.tensor([[0.4]])
        label = torch.tensor([[1.3]])
        update = update_for(model, x, label)
        view = IdentityDefense().server_view(IdentityDefense().encode(update, context(update)), ServerKnowledge())
        attacker = GradientMatchingAttack((1, 1), label, steps=800, lr=0.05, seed=9, loss_fn=F.mse_loss)
        result = attacker.reconstruct(view, model, ServerKnowledge())
        self.assertLess(float((result.estimate - x).abs().max()), 1e-3)

    def test_mask_aware_attack_is_seeded(self):
        model = OneRecordLinear()
        x = torch.tensor([[0.4]])
        label = torch.tensor([[1.3]])
        raw = update_for(model, x, label)
        transmitted = GradientPruningDefense(0.25).encode(raw, context(raw))
        view = GradientPruningDefense(0.25).server_view(transmitted, ServerKnowledge())
        attack = MaskAwareGradientAttack((1, 1), label, steps=50, lr=0.05, seed=11, loss_fn=F.mse_loss, masks=transmitted.metadata["mask"])
        first = attack.reconstruct(view, model, ServerKnowledge())
        second = attack.reconstruct(view, model, ServerKnowledge())
        self.assertTrue(torch.equal(first.estimate, second.estimate))
        self.assertEqual(first.objective_value, second.objective_value)

    def test_omit_gradient_and_known_transform_recover_trivial_identity_case(self):
        model = OneRecordLinear()
        x = torch.tensor([[0.4]])
        label = torch.tensor([[1.3]])
        raw = update_for(model, x, label)
        view = IdentityDefense().server_view(IdentityDefense().encode(raw, context(raw)), ServerKnowledge())
        omit = OmitGradientAttack((1, 1), label, steps=800, lr=0.05, seed=9, loss_fn=F.mse_loss, omit_names=())
        known = KnownTransformAttack((1, 1), label, steps=800, lr=0.05, seed=9, loss_fn=F.mse_loss, transform=lambda value: value)
        self.assertLess(float((omit.reconstruct(view, model, ServerKnowledge()).estimate - x).abs().max()), 1e-3)
        self.assertLess(float((known.reconstruct(view, model, ServerKnowledge()).estimate - x).abs().max()), 1e-3)

    def test_known_hash_sketch_attack_recovers_trivial_linear_input(self):
        model = OneRecordLinear()
        x = torch.tensor([[0.4]])
        label = torch.tensor([[1.3]])
        raw = update_for(model, x, label)
        defense = CountSketchDefense(rows=3, columns=23, seed=5, reveal_hashes=True)
        transmitted = defense.encode(raw, context(raw))
        view = defense.server_view(transmitted, ServerKnowledge(seeds=frozenset({"sketch_hashes"})))
        result = SketchGradientAttack((1, 1), label, steps=800, lr=0.05, seed=9, loss_fn=F.mse_loss).reconstruct(view, model, view.server_knowledge)
        self.assertLess(float((result.estimate - x).abs().max()), 1e-3)

    def test_every_adaptive_attack_recovers_in_lossless_identity_setting(self):
        """Sanity check: defenses/noise off must not conceal a weak attacker."""
        model = OneRecordLinear()
        x = torch.tensor([[0.4]])
        label = torch.tensor([[1.3]])
        raw = update_for(model, x, label)
        view = IdentityDefense().server_view(IdentityDefense().encode(raw, context(raw)), ServerKnowledge())
        attacks = [
            OmitGradientAttack((1, 1), label, steps=800, lr=0.05, seed=9, loss_fn=F.mse_loss),
            MaskAwareGradientAttack((1, 1), label, steps=800, lr=0.05, seed=9, loss_fn=F.mse_loss, masks={name: torch.ones_like(value, dtype=torch.bool) for name, value in raw.items()}),
            KnownTransformAttack((1, 1), label, steps=800, lr=0.05, seed=9, loss_fn=F.mse_loss, transform=lambda value: value),
            StochasticNoiseAttack((1, 1), label, steps=800, lr=0.05, seed=9, loss_fn=F.mse_loss, noise_shape=(1,), noise_setter=lambda noise: None),
        ]
        for attack in attacks:
            result = attack.reconstruct(view, model, ServerKnowledge())
            self.assertLess(float((result.estimate - x).abs().max()), 1e-3, attack.name)

    def test_unknown_hash_sketch_attack_is_explicitly_unavailable(self):
        model = OneRecordLinear()
        raw = update_for(model, torch.tensor([[0.4]]), torch.tensor([[1.3]]))
        defense = CountSketchDefense(rows=3, columns=23, seed=5, reveal_hashes=False)
        view = defense.server_view(defense.encode(raw, context(raw)), ServerKnowledge())
        result = SketchGradientAttack((1, 1), torch.tensor([[1.3]]), steps=2, loss_fn=F.mse_loss).reconstruct(view, model, ServerKnowledge())
        self.assertEqual(result.metadata["status"], "unavailable_without_sketch_hashes")


if __name__ == "__main__":
    unittest.main(verbosity=2)
