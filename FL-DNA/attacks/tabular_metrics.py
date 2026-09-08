"""PaySim-aware reconstruction metrics for gradient inversion evaluation."""

from __future__ import annotations

import numpy as np

from attacks.inversion_metrics import reconstruction_metrics


def paysim_reconstruction_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    feature_names: list[str],
    numeric_center: list[float],
    numeric_scale: list[float],
) -> dict[str, object]:
    """Evaluate normalized, raw-numeric, categorical, and derived-feature errors."""
    original = np.asarray(original, dtype=np.float32).reshape(-1)
    reconstructed = np.asarray(reconstructed, dtype=np.float32).reshape(-1)
    if original.size != len(feature_names) or reconstructed.size != len(feature_names):
        raise ValueError("feature vector length must match feature_names")

    metrics: dict[str, object] = reconstruction_metrics(original, reconstructed)
    absolute_error = np.abs(reconstructed - original)
    metrics["per_feature_abs_error_normalized"] = {
        name: float(error) for name, error in zip(feature_names, absolute_error)
    }

    numeric_indices = [index for index, name in enumerate(feature_names) if not name.startswith("type=")]
    if len(numeric_indices) != len(numeric_center) or len(numeric_indices) != len(numeric_scale):
        raise ValueError("numeric scaler metadata does not match numeric feature count")
    center = np.asarray(numeric_center, dtype=np.float32)
    scale = np.asarray(numeric_scale, dtype=np.float32)
    original_raw = original[numeric_indices] * scale + center
    reconstructed_raw = reconstructed[numeric_indices] * scale + center
    raw_error = np.abs(reconstructed_raw - original_raw)
    numeric_names = [feature_names[index] for index in numeric_indices]
    metrics["numeric_mae_raw"] = float(np.mean(raw_error))
    metrics["per_numeric_feature_abs_error_raw"] = {
        name: float(error) for name, error in zip(numeric_names, raw_error)
    }

    type_indices = [index for index, name in enumerate(feature_names) if name.startswith("type=")]
    if type_indices:
        original_type = original[type_indices]
        reconstructed_type = reconstructed[type_indices]
        original_choice = int(np.argmax(original_type))
        reconstructed_choice = int(np.argmax(reconstructed_type))
        metrics.update(
            {
                "categorical_accuracy": float(original_choice == reconstructed_choice),
                "original_type": feature_names[type_indices[original_choice]].split("=", 1)[1],
                "reconstructed_type": feature_names[type_indices[reconstructed_choice]].split("=", 1)[1],
                "one_hot_valid_before_decode": bool(
                    np.all(np.isclose(reconstructed_type, 0, atol=1e-6, rtol=0)
                           | np.isclose(reconstructed_type, 1, atol=1e-6, rtol=0))
                    and np.count_nonzero(np.isclose(reconstructed_type, 1, atol=1e-6, rtol=0)) == 1
                ),
                "one_hot_sum_before_decode": float(np.sum(reconstructed_type)),
            }
        )
    else:
        metrics.update(
            {
                "categorical_accuracy": None,
                "original_type": None,
                "reconstructed_type": None,
                "one_hot_valid_before_decode": None,
                "one_hot_sum_before_decode": None,
            }
        )

    raw_by_name = {name: float(value) for name, value in zip(numeric_names, reconstructed_raw)}
    metrics["balance_diff_orig_consistency_error_raw"] = _consistency_error(
        raw_by_name, "balance_diff_orig", "oldbalanceOrg", "newbalanceOrig", destination=False
    )
    metrics["balance_diff_dest_consistency_error_raw"] = _consistency_error(
        raw_by_name, "balance_diff_dest", "oldbalanceDest", "newbalanceDest", destination=True
    )
    return metrics


def _consistency_error(values, derived, old_balance, new_balance, destination):
    if not all(name in values for name in (derived, old_balance, new_balance)):
        return None
    expected = values[new_balance] - values[old_balance] if destination else values[old_balance] - values[new_balance]
    return float(abs(values[derived] - expected))
