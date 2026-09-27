# Amendment: RQ1-v2 confirmatory protocol

**Timestamp:** 2026-09-16, before creating any RQ1-v2 confirmatory target set.  
**Status:** CONFIRMATORY PROTOCOL FROZEN — target generation authorized only
after this amendment and its machine-readable config are created.  
**Scope:** DNA Transform v2 at `compression_ratio=0.95`, `quantization_eta=0.01`
on the bounded PaySim Phase-4 scope.

## 1. Frozen mechanism and attacker

DNA Transform v2:

```text
compression_ratio = 0.95
quantization_eta = 0.01
base_seed = 20260916
```

This is the primary v2 configuration that passed the Step-4 development utility
smoke gate.

Validated development attacker:

```text
attacker = compressed-sensing/IHT
sparsity_fraction = 0.20
iht_iterations = 80
iht_step_size = 1.0
restarts = 4
attack_iterations = 600
attack_lr = 0.1
nonnegative_lambda = 0.001
init_mode = standard
```

This is the exact IHT-0.20 configuration that passed the independent n=24
development gate:

```text
Prior: 20/24, p = 0.00077194
Zero-update: 20/24, p = 0.00077194
```

No attacker tuning is allowed after confirmatory target generation.

## 2. Comparator

Use the v2-specific distortion-matched comparator:

```text
DP_DISTORTION_MATCHED_V2 noise_multiplier = 0.00105
clip_norm = 100.0
mc_noise_samples = 100
defense_seed = 314159265
```

This value was selected before v2 confirmatory work in the local-grid
interpolation amendment. It matched the v2 development median relative-L2
distortion within tolerance:

```text
DNA v2 median relative-L2 = 0.3312435936
DP v2 median relative-L2 = 0.3279535493
relative distance = 0.993%
```

The comparator remains a weak-accounting DP-style clipping/noise comparator,
not a strong record-level DP claim.

## 3. Primary contrast

Primary metric:

```text
feature-MSE of reconstructed data
```

Primary paired difference:

```text
D_i = feature_mse(DNA_TRANSFORM_V2_i) - feature_mse(DP_DISTORTION_MATCHED_V2_i)
```

A DNA-v2 win is:

```text
D_i > tie_threshold_mse
```

Primary hypothesis:

```text
H0: P(D_i > tie_threshold_mse | non-tie) <= 0.50
H1: P(D_i > tie_threshold_mse | non-tie) > 0.50
```

Use an exact one-sided sign test at alpha `0.05`.

## 4. V2-specific power analysis

Do not reuse v1 `p1=0.65` or `p1=0.70` by default.

For v2, the minimum practically meaningful win probability is frozen as:

```text
p1 = 0.60
```

Rationale:

- v2 is a new lossy mechanism and should not require a large observed advantage
  before the study is sensitive to it;
- a 60% non-tied win probability is the smallest effect the project will treat
  as practically worth detecting for this v2 privacy comparison;
- the observed 83% development win rate is not used as the expected effect size.

Exact-binomial power calculation:

```text
p0 = 0.50
p1 = 0.60
alpha = 0.05
target_power = 0.80
required_non_tied_n = 158
rejection_minimum_wins = 90
actual_alpha = 0.04723726761096572
actual_power = 0.8056549358366878
```

Planning inflation:

```text
planned_tie_rate = 0.05
planned_dropout_rate = 0.05
initial_target_draw = 176
```

This intentionally exceeds the older v1 target-count ceiling of 150 because
that ceiling was tied to earlier resource assumptions and to a larger effect
size. For v2, the confirmatory design prioritizes sufficient power for the
smaller, predeclared practical effect.

## 5. V2-specific tie threshold

A separate v2 numerical-stability study was run on development-only v2 IHT
artifacts:

```text
artifacts_checked = 192
observed_max_absolute_replay_discrepancy = 0.001953125
safety_factor = 10
tie_threshold_mse = 0.01953125
```

Do not reuse v1 `tie_threshold_mse=0.0390625`.

## 6. Target generation

Create exactly one RQ1-v2 confirmatory target set:

```text
groups = 176
records_per_group = 4
fraud_records_per_group = 1
```

It must be source-disjoint with every existing target set, including:

- original v2 n=8 development target;
- v2 n=24 expanded development target;
- all v1 development/post-hoc/confirmatory/replication targets;
- Priority-2 clean confirmatory targets;
- all other `*targets.pt` artifacts under `artifacts/`.

Old/post-hoc targets may be read only for source-ID overlap checking.

## 7. Confirmatory execution

Run exactly three branches:

1. `raw`;
2. `dna_v2`;
3. `dp_v2`.

Each branch has its own Prior and Zero-update control gate:

```text
mean difference < 0
median difference < 0
one-sided sign-test p < 0.05
```

The primary DNA-v2-vs-DP-v2 contrast is interpreted only if the DNA-v2 and
DP-v2 branch gates pass. Raw gate is reported as a branch validity/context gate.

This is the only confirmatory execution authorized for this v2 configuration.
Do not redraw targets, rerun with modified parameters, or add replications based
on unfavorable outcomes.

## 8. Prohibitions

- Do not touch post-hoc target outcomes.
- Do not modify `torch.set_num_threads(1)`.
- Do not modify v1 code.
- Do not tune attacker, v2 transform, DP comparator, tie threshold, p1, or target
  count after target generation.
- Do not run theoretical-limit analysis as a substitute for this confirmatory
  result; that branch was closed when IHT-0.20 passed the n=24 development gate.
