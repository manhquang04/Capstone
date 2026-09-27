# Amendment: DNA Transform v2 structured-attacker development

**Timestamp:** 2026-09-16, before any new v2 structured-attacker result is
generated.  
**Status:** DEVELOPMENT ONLY — no confirmatory target generation authorized.  
**Time-box:** maximum 4-5 days of experimentation.  
**Applies to:** DNA Transform v2 `v2_ratio0p95_eta0p01`.

## 1. Motivation

The previous Step-5 attacker was a raw-style attacker against the server-lifted
full update:

```text
argmin_x L_balanced(delta(x), lift(Q(R_s u_raw)))
```

It was not the v1 `M_r` realization-search attacker, but it was also not a
v2-structured sketch/JL/quantization attacker. Therefore its failure must not be
interpreted as privacy evidence.

This amendment authorizes development of a v2-appropriate attacker on
development data only.

## 2. Development data

Use only:

```text
artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/development_gate_targets.pt
```

Do not use any post-hoc, confirmatory, replication or Priority-2 target set.

## 3. Frozen v2 defense configuration

```text
compression_ratio = 0.95
quantization_eta = 0.01
base_seed = 20260916
```

The already matched development DP comparator is:

```text
DP_DISTORTION_MATCHED_V2 noise_multiplier = 0.00105
```

This amendment does not authorize running DP attacks or confirmatory comparisons.

## 4. Attacker plan and priority order

### Priority A — sketch-space STE attacker

Optimize candidate records `x` by matching the observed v2 sketch directly:

```text
q_obs = Q_delta(R_s u_raw)
u_cand = delta(x)
q_cand = STE_Q_delta(R_s u_cand)
objective = L_balanced(q_cand, q_obs)
```

Quantization handling:

```text
STE_Q(z) = z + (Q(z) - z).detach()
```

so the forward pass sees quantized values while the backward pass treats
quantization as identity.

Initial development grid:

```text
learning_rate ∈ {0.03, 0.1}
iterations ∈ {600}
restarts = 4
init_mode = standard
nonnegative_lambda = 0.001
sketch_loss_weight = 1.0
```

### Priority B — sketch-space + prior regularization

Only if Priority A fails to pass both controls, try one regularized variant:

```text
objective =
  L_balanced(STE_Q(R_s delta(x)), q_obs)
  + nonnegative_lambda * nonnegative_penalty
  + l1_update_lambda * mean_abs(delta(x))
```

with:

```text
l1_update_lambda ∈ {1e-5, 1e-4}
```

This is a compressed-sensing-style prior in spirit: prefer sparse/small
candidate updates among candidates with similar sketch fit. It is still a
development heuristic, not a proof of optimal compressed sensing recovery.

## 5. Gate and interpretation

Use the same development gate against Prior and Zero-update controls:

```text
mean difference < 0
median difference < 0
one-sided sign-test p < 0.05
```

on all 8 development groups.

Possible outcomes:

1. **PASS both controls:** v2 has a validated development attacker; a separate
   confirmatory protocol may be drafted later.
2. **FAIL after Priority A/B within time-box:** report that no effective
   v2-structured attacker was found in the attempted scope. This is not a proof
   of absolute security and must be worded cautiously.
3. **Time-box expires:** stop and report partial progress; do not extend
   without a new supervisor decision.

## 6. Prohibitions

- Do not create confirmatory targets.
- Do not use post-hoc data.
- Do not modify v1 code.
- Do not modify `torch.set_num_threads(1)`.
- Do not relabel attacker failure as a privacy guarantee.
- Do not tune using any future confirmatory target.

## 7. Reporting

Write a separate attacker-development report including:

- exact variants attempted;
- commands run;
- per-control gate results;
- whether a validated v2 attacker exists;
- whether v2 RQ1 remains unresolved.
