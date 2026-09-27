# Amendment — RQ1 medium/stronger and one-time utility-grid extension

**Amendment ID:** `2026-09-13-RQ1-VARIANTS-UTILITY-EXTENSION`  
**Timestamp:** 2026-09-13, before variant calibration or new targets  
**Approved by:** Manh Quang via the Group 2 execution instruction  

## Scope

This amendment prospectively defines the bounded RQ1 extensions for the
existing `medium` and `stronger` DNA Transform configurations, the realization-
known Level-2 analysis of the already frozen conservative targets, and a
single reopened development-only utility-matching search. It does not alter or
rerun the completed conservative confirmatory experiment.

## DNA Transform variants

| ID | block | mix | keep | shrink |
| --- | ---: | ---: | ---: | ---: |
| `DNA-TRANSFORM-MEDIUM-V1` | 256 | 0.10 | 0.85 | 0.40 |
| `DNA-TRANSFORM-STRONGER-V1` | 256 | 0.12 | 0.82 | 0.35 |

These are existing project configurations; they were not selected from the
completed conservative privacy outcome.

## Distortion calibration freeze

Each variant is calibrated separately on `development_gate_targets.pt`. The
full multiplier grid is
`[0.0001, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005,
0.00075, 0.001, 0.005, 0.01]`, with clip norm 100, seeds
`[11,22,33,44,55]`, median relative-L2, 5% relative tolerance and
smallest-multiplier tie break. The denser interval is fixed from the monotonic
distortion scale and stronger transform definitions, without consulting a
reconstruction/privacy result.

If no point is within 5%, the affected variant stops before confirmatory target
generation unless a new pre-target amendment is approved. The full grid is
retained.

## Per-variant power and target contract

Each variant is a separate confirmatory family with the original exact
one-sided sign-test design: `p0=0.50`, `p1=0.70`, `alpha=0.05`, power `0.80`,
expected ties `10%`, technical dropout `5%`, tie threshold MSE `0.0390625`.
This requires 37 non-tied observations, rejection at 24 wins, actual alpha
`0.0494359`, actual power `0.807096`, and an initial draw of 44 targets. The
maximum remains 150 targets per study.

Medium and stronger each receive exactly one new 44-group target set, generated
after their comparator is selected. Each set must be source-disjoint from every
historical/development/post-hoc/confirmatory target and from the other variant.
Raw, DNA Level-1 and the variant-specific DP distortion comparator run on the
same set. Branch gates and analysis are independent per variant. No target set
may be redrawn because of an unfavorable outcome.

## Conservative Level 2

Level 2 uses the exact existing conservative target SHA-256
`caee0024c318c2874037382996b843ead8024b02b132185e502e245d3266267f`
and DNA seed `1184685071`. It discloses the exact transform realization, applies
direct linear inversion, and evaluates the already generated raw-attack
candidates against the recovered signal. It is an attacker-favorable threat-
model sensitivity analysis, not a new Level-1 confirmatory run and not a reason
to alter the completed primary result.

## One-time DP utility-grid extension

The supervisor explicitly reopens the previously closed development search
once. It uses only the original eight development seeds and existing paired
baseline/conservative-DNA results. New DP multipliers are
`[0.00001, 0.000025, 0.00005, 0.000075]`, chosen between the baseline/no-noise
endpoint and the prior nearest candidate `0.0001`. No privacy reconstruction
metric is used.

The original strict tolerances remain primary: absolute F1 deviation `<=0.01`
and AUC deviation `<=0.0025`, with max normalized distance and the smallest-
multiplier tie break. A separately labeled sensitivity rule also evaluates
F1 `<=0.02` and AUC `<=0.005`; a candidate found only there is
`UTILITY_MATCHED_RELAXED_SENSITIVITY`, not the strict comparator. Selection
priority is strict eligibility first, then relaxed sensitivity. If neither
finds a candidate, status returns to terminal `NOT_FOUND`. The search closes
permanently after this grid; no adaptive point or tolerance expansion is
permitted.

