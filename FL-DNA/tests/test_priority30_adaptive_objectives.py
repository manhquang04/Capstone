import torch

from experiments.priority30_native_defenses.run_audit import (
    AdaptiveImageReconstructor,
    cosine_loss_lists,
    count_sketch_loss,
    dna_v1_debias,
    v2_sketch_loss,
)


def test_mask_aware_identity_matches_undefended_cosine() -> None:
    left = [torch.tensor([1.0, 2.0]), torch.tensor([-1.0])]
    right = [torch.tensor([1.5, 2.5]), torch.tensor([-0.5])]
    masks = [torch.ones_like(t, dtype=torch.bool) for t in left]
    assert torch.allclose(cosine_loss_lists(left, right, masks), cosine_loss_lists(left, right))


def test_v2_sketch_identity_matches_undefended_cosine() -> None:
    grad = [torch.tensor([1.0, 2.0, -3.0])]
    matrix = [torch.eye(3)]
    sketch = [grad[0].clone()]
    assert torch.allclose(v2_sketch_loss(grad, sketch, matrix), cosine_loss_lists(grad, sketch))


def test_count_sketch_identity_matches_undefended_cosine() -> None:
    grad = [torch.tensor([1.0, -2.0, 3.0])]
    table = grad[0].reshape(1, -1)
    buckets = torch.arange(3, dtype=torch.long).reshape(1, -1)
    signs = torch.ones_like(table)
    assert torch.allclose(count_sketch_loss(grad, table, buckets, signs), cosine_loss_lists(grad, [table.reshape(-1)]))


def test_dna_v1_debias_identity_mix_zero_is_noop() -> None:
    update = {"w": torch.tensor([1.0, 3.0, 5.0])}
    assert torch.equal(dna_v1_debias(update, mix=0.0)["w"], update["w"])


def test_adaptive_image_unknown_mode_fails_closed() -> None:
    reconstructor = AdaptiveImageReconstructor.__new__(AdaptiveImageReconstructor)
    reconstructor.adaptive_mode = "gradient_pruning_mask_aware_e2"
    reconstructor.adaptive_aux = {}
    reconstructor.config = {"cost_fn": "sim", "indices": "def", "weights": "equal"}
    try:
        reconstructor._adaptive_loss([torch.tensor([1.0])], [torch.tensor([1.0])])
    except ValueError as exc:
        assert "unknown adaptive image mode" in str(exc)
    else:
        raise AssertionError("unknown adaptive mode silently fell back instead of failing closed")


def test_v2_sketch_loss_ignores_raw_gradient_payload() -> None:
    grad = [torch.tensor([1.0, 0.0])]
    matrix = [torch.eye(2)]
    sketch = [torch.tensor([1.0, 0.0])]
    raw_gradient_that_must_not_be_used = [torch.tensor([-1.0, 0.0])]
    assert torch.allclose(v2_sketch_loss(grad, sketch, matrix), torch.tensor(0.0))
    assert not torch.allclose(cosine_loss_lists(grad, raw_gradient_that_must_not_be_used), torch.tensor(0.0))
