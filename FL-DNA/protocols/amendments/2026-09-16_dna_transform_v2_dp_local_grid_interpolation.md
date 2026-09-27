# Amendment: DNA Transform v2 DP local-grid interpolation

**Timestamp:** 2026-09-16, before any local-grid interpolation result is
generated.  
**Status:** DEVELOPMENT ONLY — distortion calibration only.  
**Applies to:** DNA Transform v2 `v2_ratio0p95_eta0p01`.

## 1. Reason

The initial Step-5 v2 DP distortion calibration used the frozen grid:

```text
[0.00025, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
```

The nearest candidate was:

```text
noise_multiplier = 0.001
DP median relative-L2 = 0.3123367286
DNA median relative-L2 = 0.3312435936
relative distance = 5.71%
```

This narrowly missed the frozen 5% tolerance. Following the same local
interpolation logic previously used for v1 medium/stronger calibration, this
amendment authorizes a small predeclared local grid around `0.001`.

No privacy/attack result is used to choose this grid. The only evidence used is
development-only distortion calibration.

## 2. Local grid

Run only the following additional multipliers:

```text
[0.00105, 0.00110, 0.00115]
```

Keep all other values unchanged:

```text
clip_norm = 100.0
calibration_seed_list = [11, 22, 33, 44, 55]
target_statistic = median_relative_l2
matching_tolerance = 5%
selection_rule = minimum absolute distance; smallest multiplier tie-break
```

## 3. Interpretation

If one local-grid point falls within tolerance, it may be labeled
`DP_DISTORTION_MATCHED_V2` for development planning only.

This amendment does not authorize:

- any v2 confirmatory target generation;
- any v2 privacy claim;
- any attacker rerun or attacker parameter change;
- any statement that v2 is more secure than v1.

The attacker-development-gate issue remains separate and unresolved unless a
v2-appropriate attacker amendment is written and executed later.
