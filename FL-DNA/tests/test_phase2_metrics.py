"""Known-answer tests for the Phase 2 PaySim reconstruction evaluator."""

import numpy as np
import pytest

from attacks.tabular_metrics import paysim_reconstruction_metrics


FEATURES = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_diff_orig",
    "balance_diff_dest",
    "type=A",
    "type=B",
]


def test_valid_exact_reconstruction() -> None:
    vector = np.array([10, 4, 3, 8, 6, 5, 1, 0], dtype=np.float32)
    metrics = paysim_reconstruction_metrics(vector, vector.copy(), FEATURES, [0] * 6, [1] * 6)
    assert metrics["feature_mse"] == 0.0
    assert metrics["numeric_mae_raw"] == 0.0
    assert metrics["categorical_accuracy"] == 1.0
    assert metrics["one_hot_valid_before_decode"] is True
    assert metrics["balance_diff_orig_consistency_error_raw"] == 0.0
    assert metrics["balance_diff_dest_consistency_error_raw"] == 0.0


def test_invalid_one_hot_is_decoded_but_flagged() -> None:
    original = np.array([10, 4, 3, 8, 6, 5, 1, 0], dtype=np.float32)
    reconstructed = original.copy()
    reconstructed[-2:] = [2.0, 1.0]
    metrics = paysim_reconstruction_metrics(original, reconstructed, FEATURES, [0] * 6, [1] * 6)
    assert metrics["categorical_accuracy"] == 1.0
    assert metrics["one_hot_valid_before_decode"] is False


def test_inconsistent_derived_feature_is_reported() -> None:
    original = np.array([10, 4, 3, 8, 6, 5, 1, 0], dtype=np.float32)
    reconstructed = original.copy()
    reconstructed[4] = 99
    metrics = paysim_reconstruction_metrics(original, reconstructed, FEATURES, [0] * 6, [1] * 6)
    assert metrics["balance_diff_orig_consistency_error_raw"] == 93.0


def test_scaler_metadata_must_match_numeric_features() -> None:
    vector = np.zeros(len(FEATURES), dtype=np.float32)
    with pytest.raises(ValueError):
        paysim_reconstruction_metrics(vector, vector, FEATURES, [0], [1])


def test_soft_simplex_is_not_one_hot():
    vector = np.array([10, 4, 3, 8, 6, 5, .5, .5], dtype=np.float32)
    assert not paysim_reconstruction_metrics(vector, vector, FEATURES, [0]*6, [1]*6)['one_hot_valid_before_decode']


def test_source_rows_replay_dataframe_split():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from experiments.phase2_provenance import validation_source_rows
    y = np.tile([0, 0, 0, 1], 100)
    frame = pd.DataFrame({'isFraud': y, 'source': np.arange(len(y))})
    _, subset = train_test_split(frame, test_size=200, random_state=31, stratify=y)
    _, temp = train_test_split(subset, test_size=.15+.2, random_state=31, stratify=subset.isFraud)
    val, _ = train_test_split(temp, test_size=.2/(.15+.2), random_state=31, stratify=temp.isFraud)
    np.testing.assert_array_equal(validation_source_rows(y, 200, 31), val.source.to_numpy())


def test_attack_history_and_best_components():
    import torch
    from attacks.gradient_inversion import gradient_inversion_attack, parameter_gradients, GradientInversionConfig
    model = torch.nn.Linear(2, 1)
    criterion = torch.nn.MSELoss()
    label = torch.ones(1, 1)
    observed = parameter_gradients(model, criterion, torch.ones(1, 2), label)
    result = gradient_inversion_attack(model, criterion, observed, label, 2, GradientInversionConfig(iterations=3))
    assert len(result.component_history) == 4
    assert result.best_loss == min(row['total_objective'] for row in result.component_history)
    for row in result.component_history:
        assert row['total_objective'] == pytest.approx(row['gradient_match_loss'] + row['regularization_loss'])
