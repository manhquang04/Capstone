# Amendment: DNA Transform v2 Step 1 mathematical design

**Date:** 2026-09-16  
**Supervisor:** Manh Quang  
**Status:** STEP-1 DESIGN ONLY — NO CODE AUTHORIZED BY THIS FILE  
**Scope:** DNA Transform v2 mathematical feasibility assessment.

## Naming lock

From this point forward:

- `DNA Transform v1` means the existing mechanism implemented in
  `dna_encoder/transform_defense.py`: DNA-seeded permutation, selective
  attenuation and residual mixing.  Conservative, medium and stronger are
  parameter variants of v1.
- `DNA Transform v2` means a new mechanism with real information loss:
  low-dimensional DNA-seeded projection plus quantization.

No v1 conclusion may be reused as evidence for v2.  v2 requires its own
mechanical audit, utility test, RQ1 calibration and confirmatory evidence.

## Proposed v2 object

For one client update vector:

```text
u ∈ R^d
```

v2 uses a DNA-seeded random sketch:

```text
R_s ∈ R^(k × d), k < d
```

where `s` is derived from DNA-style randomness and public metadata such as
round, client id and tensor id.  The preferred construction for implementation
is a structured random orthogonal sketch:

```text
R_s = sqrt(d/k) · P_s · H · D_s
```

Definitions:

- `D_s` is a diagonal sign matrix with entries `±1` from the DNA-seeded PRNG.
- `H` is an orthonormal transform, for example a normalized Hadamard transform
  when the vector is padded to the next supported power of two, or a blockwise
  orthonormal transform if padding is too expensive.
- `P_s` selects `k` coordinates without replacement using the DNA-seeded PRNG.
- The scale `sqrt(d/k)` makes the sketch unbiased after server-side lifting.

The client computes:

```text
y = R_s u ∈ R^k
```

Then the client quantizes each coordinate with unbiased stochastic rounding:

```text
q_j = Δ · floor(y_j / Δ)       with probability 1 - ρ_j
q_j = Δ · (floor(y_j / Δ) + 1) with probability ρ_j
ρ_j = y_j / Δ - floor(y_j / Δ)
```

For signed values, apply the same rule after shifting to the adjacent lower
grid point.  This gives:

```text
E[q | y] = y
Var(q_j | y_j) ≤ Δ² / 4
```

The transmitted payload contains:

```text
q, d, k, Δ, seed metadata, original tensor shapes
```

It does not contain `u`.

## Server-side approximate reconstruction

The server regenerates `R_s` from metadata and computes:

```text
u_hat = R_s^T q
```

If quantization is ignored:

```text
u_hat = R_s^T R_s u
```

Because `P_s` samples rows uniformly from an orthonormal basis and
`R_s = sqrt(d/k) P_s H D_s`:

```text
E_s[R_s^T R_s] = I_d
```

With stochastic quantization:

```text
E[u_hat | u] = E_s[ R_s^T E[q | R_s u] ] = E_s[R_s^T R_s u] = u
```

Thus the reconstruction is not exact for one realization, but it is unbiased
over the DNA-seeded sketch and quantizer randomness.

## FedAvg compatibility

Let client `i` have update `u_i` and FedAvg weight `w_i`, with
`Σ_i w_i = 1`.  Each client uses its own seed `s_i` and sends `q_i`.

The server reconstructs each approximate update:

```text
u_hat_i = R_{s_i}^T q_i
```

and aggregates:

```text
U_hat = Σ_i w_i u_hat_i
```

Then:

```text
E[U_hat | {u_i}] = Σ_i w_i E[u_hat_i | u_i] = Σ_i w_i u_i
```

So FedAvg is compatible in expectation if and only if the server lifts each
client update back to the original parameter space before averaging.

The server must not average incompatible low-dimensional sketches directly:

```text
wrong if seeds differ:  Σ_i w_i q_i
```

because `q_i` live in different random coordinate systems.

A common sketch shared by all clients would make low-dimensional averaging
possible:

```text
q_bar = Σ_i w_i q_i
u_hat = R^T q_bar
```

but this estimates `R^T R (Σ_i w_i u_i)`, not the full average update, and is
biased for the missing nullspace unless error-feedback is added.  Therefore
the default v2 design uses per-client lift-then-average.

## Expected information loss

For the unquantized sketch, the single-client lifted estimator has approximate
relative mean-square error:

```text
E[ ||R_s^T R_s u - u||² ] / ||u||² ≈ d/k - 1
```

This is the intended information loss: `rank(R_s) ≤ k < d`, so the transform is
not invertible even if the seed is known.

Illustrative relative RMSE from projection only:

| k/d | Relative RMSE approx |
| ---: | ---: |
| 0.95 | 0.229 |
| 0.90 | 0.333 |
| 0.80 | 0.500 |
| 0.75 | 0.577 |
| 0.50 | 1.000 |
| 0.25 | 1.732 |

Quantization adds:

```text
E[ ||R_s^T(q - y)||² | R_s ] ≤ (d · Δ²) / 4
```

So a conservative quantization rule should choose:

```text
Δ = η · ||u|| / sqrt(d)
```

with small `η`, giving added relative MSE at most `η²/4`.

## Utility risk estimate

The projection loss is large unless `k/d` is close to 1.  For example, `k/d=0.5`
implies projection RMSE approximately equal to the update norm for a single
client update.  That is likely to damage model utility severely.

Therefore the initial feasible v2 parameter range is deliberately conservative:

```text
k/d ∈ {0.90, 0.95}
η ∈ {0.005, 0.01, 0.02}
```

These values still create true rank loss but limit the first-order update
distortion enough to justify a small utility smoke test.

If later Step 4 shows severe utility collapse even at `k/d ≥ 0.90` and
`η ≤ 0.02`, v2 should stop before any RQ1 security calibration.

## Error-feedback option

If stateless v2 has unacceptable utility but the design is otherwise promising,
the only mathematically clean extension is client-side error feedback:

```text
v_{i,t} = u_{i,t} + e_{i,t}
q_{i,t} = Q(R_{i,t} v_{i,t})
u_hat_{i,t} = R_{i,t}^T q_{i,t}
e_{i,t+1} = v_{i,t} - u_hat_{i,t}
```

This preserves the residual locally and can reduce long-run bias/utility loss.
It changes the FL state machine and must be a separately named v2 sub-variant,
not silently added after seeing bad results.

## Step-1 feasibility conclusion

This is not a design dead-end at the mathematical level, provided all of the
following constraints are obeyed:

1. The server reconstructs each client update back to full parameter space
   before FedAvg aggregation.
2. The sketch is scaled so that `E[R_s^T R_s] = I_d`.
3. Quantization uses stochastic unbiased rounding.
4. Initial compression is conservative (`k/d ≥ 0.90`) because theoretical
   projection error grows quickly for smaller `k/d`.
5. v2 is evaluated from scratch and is not described using v1 evidence.

The design is high risk for utility, but it is mathematically compatible with
FedAvg in expectation.  Step 2 may implement a separate v2 module and unit
tests for the stateless variant only.  No RQ1/RQ2/RQ3 claim is authorized yet.
