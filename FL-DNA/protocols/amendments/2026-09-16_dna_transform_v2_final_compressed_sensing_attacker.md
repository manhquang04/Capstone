# Amendment: DNA Transform v2 final compressed-sensing attacker

**Timestamp:** 2026-09-16, before final compressed-sensing attacker results are
generated.  
**Status:** FINAL DEVELOPMENT ATTACKER ATTEMPT — no confirmatory target
generation authorized.  
**Time-box:** maximum 4-5 days; this is the last attacker-generation attempt for
v2 unless supervisor opens a new research direction later.

## 1. Motivation

Three prior v2 attacker families have failed development controls:

1. raw-style attacker against the lifted full update;
2. sketch-space straight-through-estimator attacker;
3. sketch-space STE with a weak update-L1 prior.

Supervisor authorizes one final attacker family using compressed-sensing-style
update reconstruction.

## 2. Threat model and known measurement matrix

The attacker knows the v2 algorithm and seed contract under Level-1-style
public-algorithm assumptions. Therefore the sensing matrix is known:

```text
R_s = sqrt(n/k) P_s H D_s
q_obs = Q_delta(R_s u_raw)
```

where `H` is the normalized Hadamard transform, `D_s` is the seeded sign flip,
and `P_s` samples coordinates.

## 3. Quantization model

The quantizer is modeled as additive bounded noise:

```text
q_obs = R_s u_raw + e_q
```

with scale determined by the logged v2 quantization delta. The attacker does
not backpropagate through `Q`; instead it first reconstructs an approximate
full update from the noisy sketch.

## 4. Compressed-sensing reconstruction

Use Iterative Hard Thresholding (IHT) independently per floating tensor:

```text
u_0 = 0
u_{t+1/2} = u_t + step_size * R_s^T (q_obs - R_s u_t)
u_{t+1} = H_K(u_{t+1/2})
```

where `H_K` keeps the top-`K` coordinates by absolute value and zeros the rest.

This uses a general sparsity prior for model updates. It does not use oracle
labels and does not use any target-specific hidden information.

## 5. Data inversion after IHT

After reconstructing a sparse full update `u_iht`, run the existing hard-diff
data inversion objective against `u_iht`:

```text
argmin_x L_balanced(delta(x), u_iht)
```

The zero-update control uses a zero signal of the same shape. Prior is the
existing initialization prior.

## 6. Development variants

Run at most three variants:

```text
sparsity_fraction ∈ {0.05, 0.10, 0.20}
iht_iterations = 80
iht_step_size = 1.0
attack_lr = 0.1
attack_iterations = 600
restarts = 4
nonnegative_lambda = 0.001
init_mode = standard
```

If one variant passes both controls, stop and report the validated attacker.
If no variant passes, stop the v2 attacker line.

## 7. Final interpretation rule

If this final compressed-sensing attacker fails development controls, the
official v2 privacy status becomes:

```text
After three controlled attacker generations (raw-lift, sketch-space STE, and
compressed-sensing/IHT), no effective v2 attacker was found on the development
pilot. This is meaningful evidence that v2 is harder to reconstruct within the
tested scope, but it is not a formal proof of security and does not authorize a
confirmatory privacy claim.
```

Do not create v2 confirmatory targets unless this final attacker passes both
controls.

## 8. Prohibitions

- Do not use post-hoc or confirmatory target data.
- Do not modify `torch.set_num_threads(1)`.
- Do not modify v1 code.
- Do not extend beyond the three variants without a new supervisor decision.
- Do not phrase failure as absolute security.
