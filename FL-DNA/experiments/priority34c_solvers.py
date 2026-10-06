"""Payload-only P34C ratio and known-key v2 least-squares primitives."""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr
from dna_encoder.transform_defense_v2 import _fwht_normalized, _signs


def finite(value, label):
    if not np.isfinite(value).all():
        raise FloatingPointError("nonfinite " + label)


def projection_operator(metadata):
    d, n, k = metadata.original_size, metadata.padded_size, metadata.sketch_size
    indices = np.asarray(metadata.sampled_indices, dtype=np.int64)
    if not (0 < d <= n and n & (n - 1) == 0 and 0 < k <= n):
        raise ValueError("invalid projection dimensions")
    if indices.shape != (k,) or len(np.unique(indices)) != k:
        raise ValueError("invalid sampled indices")
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("sampled index outside projection")
    signs = _signs(n, metadata.seed)
    scale = float(np.sqrt(n / k))

    def forward(value):
        value = np.asarray(value, dtype=np.float64).reshape(d)
        finite(value, "projection input")
        padded = np.zeros(n, dtype=np.float64)
        padded[:d] = value
        return scale * _fwht_normalized(padded * signs)[indices]

    def adjoint(value):
        value = np.asarray(value, dtype=np.float64).reshape(k)
        finite(value, "adjoint input")
        lifted = np.zeros(n, dtype=np.float64)
        lifted[indices] = scale * value
        return (_fwht_normalized(lifted) * signs)[:d]

    return LinearOperator((k, d), matvec=forward, rmatvec=adjoint, dtype=np.float64)


def known_key_least_squares(sketch, metadata):
    q = np.asarray(sketch, dtype=np.float64).reshape(-1)
    if q.size != metadata.sketch_size:
        raise ValueError("sketch size mismatch")
    finite(q, "sketch")
    operator = projection_operator(metadata)
    result = lsmr(operator, q, damp=0., atol=1e-10, btol=1e-10,
                  conlim=1e12, maxiter=4000)
    x, code, iterations, residual, normal_residual, norm_a, condition, norm_x = result
    finite(x, "least-squares output")
    finite([residual, normal_residual, norm_a, condition, norm_x], "LS diagnostics")
    if code not in (0, 1, 2, 4, 5):
        raise RuntimeError("frozen LSMR failed: code=" + str(code))
    receipt = dict(solver="LSMR", stop_code=int(code), iterations=int(iterations),
                   residual=float(residual), normal_residual=float(normal_residual),
                   operator_norm_estimate=float(norm_a), condition_estimate=float(condition),
                   solution_norm=float(norm_x), original_size=metadata.original_size,
                   padded_size=metadata.padded_size, sketch_size=metadata.sketch_size,
                   rank_upper_bound=min(metadata.original_size, metadata.sketch_size),
                   atol=1e-10, btol=1e-10, conlim=1e12, maxiter=4000)
    return x.reshape(metadata.original_shape), receipt


def ratio_record(weight_gradient, bias_gradient):
    weights = np.asarray(weight_gradient, dtype=np.float64)
    bias = np.asarray(bias_gradient, dtype=np.float64)
    if weights.ndim != 2 or bias.ndim != 1 or weights.shape[0] != bias.size or not bias.size:
        raise ValueError("invalid first-layer gradient pair")
    finite(weights, "weight gradient")
    finite(bias, "bias gradient")
    row = int(np.argmax(np.abs(bias)))
    if bias[row] == 0:
        raise ArithmeticError("unresolved zero bias gradient; no clamp or target replacement")
    recovered = weights[row] / bias[row]
    finite(recovered, "ratio output")
    return recovered, dict(selected_row=row, observed_denominator=float(bias[row]))
