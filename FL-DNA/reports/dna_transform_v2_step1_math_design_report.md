# DNA Transform v2 — Step 1 mathematical design report

**Date:** 2026-09-16  
**Status:** STEP 1 COMPLETE — FEASIBLE WITH STRICT CONDITIONS  
**Code written for v2:** none  
**Data/post-hoc artifacts touched:** none

## 1. Đã làm gì

This report completes Step 1 only: mathematical design before implementation.
No `dna_encoder/transform_defense_v2.py` file was created, no v1 code was
modified, and no experiment or dataset was run.

The naming rule is now fixed:

- `DNA Transform v1`: the existing linear mechanism, including conservative,
  medium and stronger parameter variants.
- `DNA Transform v2`: the proposed lossy mechanism based on low-dimensional
  DNA-seeded projection plus quantization.

## 2. Proposed v2 transform

Flatten a model update tensor collection into:

```text
u ∈ R^d
```

Generate a DNA-seeded sketch matrix:

```text
R_s = sqrt(d/k) · P_s · H · D_s,   R_s ∈ R^(k × d), k < d
```

where:

- `D_s` is a seeded random sign flip matrix;
- `H` is an orthonormal transform;
- `P_s` samples `k` coordinates;
- `sqrt(d/k)` makes the lifted estimator unbiased.

The client sends:

```text
q = Q_Δ(R_s u)
```

where `Q_Δ` is stochastic unbiased quantization.

The server reconstructs:

```text
u_hat = R_s^T q
```

This is lossy because `rank(R_s) ≤ k < d`.

## 3. FedAvg compatibility

For client `i`:

```text
u_hat_i = R_i^T Q(R_i u_i)
```

With the proposed scaling and unbiased quantization:

```text
E[u_hat_i | u_i] = u_i
```

FedAvg then computes:

```text
U_hat = Σ_i w_i u_hat_i
```

and:

```text
E[U_hat | {u_i}] = Σ_i w_i u_i
```

So v2 is compatible with FedAvg in expectation if the server reconstructs every
client update into full parameter space before averaging.

Important negative result:

```text
The server must not average low-dimensional q_i directly when clients use
different projection seeds.
```

Those sketches live in different random coordinate systems.  Direct
low-dimensional averaging would be mathematically invalid.

## 4. Expected information loss

Projection-only relative MSE is approximately:

```text
E[||R_s^T R_s u - u||²] / ||u||² ≈ d/k - 1
```

Projection-only relative RMSE:

| k/d | Approx relative RMSE |
| ---: | ---: |
| 0.95 | 0.229 |
| 0.90 | 0.333 |
| 0.80 | 0.500 |
| 0.75 | 0.577 |
| 0.50 | 1.000 |
| 0.25 | 1.732 |

Quantization adds at most:

```text
relative MSE ≤ η² / 4
```

if the quantization step is chosen as:

```text
Δ = η · ||u|| / sqrt(d)
```

## 5. Utility feasibility assessment

The design is mathematically feasible but high-risk:

- `k/d ≤ 0.5` is likely too destructive for utility.
- Initial implementation should test only conservative lossy settings:

```text
k/d ∈ {0.90, 0.95}
η ∈ {0.005, 0.01, 0.02}
```

Even these settings may fail utility.  That is why Step 4 must test utility
before any RQ1 v2 security calibration.

## 6. Step-1 decision

Decision:

```text
FEASIBLE WITH STRICT CONDITIONS
```

Not a design dead-end, because the lift-then-average construction is unbiased
for FedAvg.

But v2 is not yet “working.”  The only authorized next step is Step 2:

- create a separate module such as `dna_encoder/transform_defense_v2.py`;
- do not modify v1;
- implement only the stateless design described here;
- write unit tests for projection shape, deterministic seeding, rank loss,
  unbiased quantization behavior and reconstruction-error bounds.

No utility, privacy or RQ claim is authorized until Steps 2–4 produce their own
evidence.

## 7. Gate đạt/chưa đạt

| Gate | Status | Reason |
| --- | --- | --- |
| v1/v2 naming separation | PASS | Defined v1 and v2 explicitly |
| Concrete projection formula | PASS | `R_s = sqrt(d/k) P_s H D_s` |
| Quantization formula | PASS | stochastic unbiased quantization |
| Server reconstruction formula | PASS | `u_hat = R_s^T q` |
| FedAvg compatibility | PASS WITH CONDITION | server must lift each client before averaging |
| Direct low-dimensional FedAvg | FAIL / FORBIDDEN | incompatible when seeds differ |
| Utility precheck | HIGH RISK BUT NOT DEAD-END | conservative initial `k/d ≥ 0.90` recommended |
| Permission to code Step 2 | ALLOWED NEXT, NOT DONE | Step 1 is complete; implementation still separate |

## 8. Bước tiếp theo được phép

Step 2 may begin only as a separate implementation task:

```text
dna_encoder/transform_defense_v2.py
tests/test_transform_defense_v2.py
```

Do not run FL utility, attacks, RQ1 calibration or confirmatory experiments
until the v2 unit tests and mechanical audit steps are complete.
