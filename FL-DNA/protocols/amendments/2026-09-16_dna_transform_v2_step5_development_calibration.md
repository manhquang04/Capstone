# Amendment: DNA Transform v2 Step 5 development calibration

**Timestamp:** 2026-09-16, before any v2 RQ1 development-calibration result is
generated or inspected.  
**Status:** DEVELOPMENT ONLY — no confirmatory target generation authorized.  
**Applies to:** DNA Transform v2 only.

## 1. Background

DNA Transform v2 Step 1-4 established:

1. mathematical feasibility with strict conditions;
2. a separate v2 implementation, with v1 untouched;
3. mechanical lossiness even when the v2 seed/projection is known;
4. no obvious utility collapse in a small development smoke for
   `compression_ratio=0.95`, `quantization_eta=0.01`.

This amendment authorizes the next development step: v2-specific RQ1
calibration. It does not authorize any v2 confirmatory target set or any privacy
claim.

## 2. Frozen v2 configuration for Step 5

The primary v2 configuration for Step 5 is:

```text
v2_ratio0p95_eta0p01
compression_ratio = 0.95
quantization_eta = 0.01
base_seed = 20260916
```

The `0.90/0.01` configuration from Step 4 is not promoted to RQ1 calibration in
this amendment because it had weaker smoke utility.

## 3. Development data

Use only the Phase-4 development target pool:

```text
artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/development_gate_targets.pt
```

and its corresponding development raw-update artifacts. Do not use any
post-hoc, confirmatory, replication, or clean Priority-2 target set.

## 4. DP distortion calibration

Calibrate a v2-specific `DP_DISTORTION_MATCHED_V2` comparator on development
data only.

Statistic:

```text
median_relative_l2_delta(update, defended_update)
```

DNA target:

```text
median relative-L2 between raw observed update and v2 lifted update
```

DP mechanism:

```text
full-client-update clipping/noise
clip_norm = 100.0
noise_multiplier grid =
  [0.00025, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
calibration_seed_list = [11, 22, 33, 44, 55]
```

Selection rule:

1. choose the multiplier with minimum absolute distance to the v2 DNA target
   median relative-L2;
2. tie-break by smaller noise multiplier;
3. mark as matched only if relative distance is `<= 5%`;
4. if no point is within tolerance, report `NOT_MATCHED` and do not silently
   widen the grid after seeing attack outcomes. A grid amendment must be written
   before any additional calibration run.

## 5. Attacker development gate

Because v2 is lossy and lifted back to full update space before FedAvg, the
development attacker observes a full defended update:

```text
u_v2 = lift(Q(R_s u_raw))
```

The first development gate uses a conservative raw-style hard-diff attacker
against `u_v2`:

```text
argmin_x L_balanced(delta(x), u_v2)
```

This is attacker-favorable in the sense that it directly optimizes against the
observed defended full update rather than requiring a differentiable v2 inverse.
It is also only a development gate. If this gate fails, v2 RQ1 calibration stops
until a separate amendment justifies a different attacker.

Development gate budget:

```text
groups = all development groups in development_gate_targets.pt
restarts = 4
iterations = 600
attack_lr = 0.1
nonnegative_lambda = 0.001
init_mode = standard
```

Gate rule is the existing Phase-4 branch rule against both Prior and Zero-update
controls:

```text
mean difference < 0
median difference < 0
one-sided sign-test p < 0.05
```

## 6. Stop rules

- If v2 attacker development gate fails, do not create any v2 confirmatory
  target set.
- If DP distortion matching fails, do not create any v2 confirmatory target set
  unless a new amendment freezes a valid next calibration step before further
  results are inspected.
- Do not tune attacker parameters after seeing v2 development-gate outcomes.
- Do not modify `torch.set_num_threads(1)`.
- Do not modify v1 code.

## 7. Reporting

Step 5 must report separately:

1. v2 DNA distortion target;
2. full DP grid and selected candidate;
3. v2 attacker development-gate result;
4. whether Step 5 permits drafting a later confirmatory protocol.

Even if Step 5 passes, no confirmatory execution is authorized by this
amendment alone.
