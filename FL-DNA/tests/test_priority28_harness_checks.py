from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.priority28.harness_checks import defense_changes_update, least_squares_decode


def test_least_squares_decode_is_not_naive_transpose_lift() -> None:
    matrix = np.asarray([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]], dtype=np.float64)
    truth = np.asarray([0.25, -0.75], dtype=np.float64)
    sketch = matrix @ truth

    recovered = least_squares_decode(matrix, sketch)
    naive = matrix.T @ sketch

    assert np.allclose(recovered, truth, atol=1e-7)
    assert not np.allclose(naive, truth, atol=1e-3)


def test_least_squares_decode_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="sketch length"):
        least_squares_decode(np.ones((2, 3)), np.ones(3))


def test_defense_changes_update_detects_identity_and_change() -> None:
    original = {
        "w": torch.tensor([1.0, 2.0]),
        "counter": torch.tensor([3], dtype=torch.int64),
    }
    identity = {"w": torch.tensor([1.0, 2.0]), "counter": torch.tensor([8], dtype=torch.int64)}
    changed = {"w": torch.tensor([1.0, 2.5]), "counter": torch.tensor([3], dtype=torch.int64)}

    is_changed, effect = defense_changes_update(original, identity)
    assert not is_changed
    assert effect == pytest.approx(0.0)

    is_changed, effect = defense_changes_update(original, changed)
    assert is_changed
    assert effect == pytest.approx(0.5)

