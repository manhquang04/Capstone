# Amendment: Priority 2 RQ1 clean confirmatory redesign

**Date:** 2026-09-16  
**Supervisor:** Manh Quang  
**Status:** APPROVED FOR PRIORITY-2 DEVELOPMENT DESIGN AND POWER LOCK  
**Scope:** RQ1 only.  This amendment prepares a cleaner, single-draw
confirmatory-quality RQ1 analysis with a smaller practically meaningful effect
than the original `p1=0.70` design.  It does not authorize viewing any new
confirmatory attack result before the Pareto development sweep is reported and
the final representative configuration is frozen.

## Motivation

The original RQ1 conservative confirmatory run and three later conservative
replications produced unstable but informative evidence:

- original Group 2 conservative: `19/39`, `p=0.6254`;
- replication 1: `26/39`, `p=0.02663`;
- replication 2: `26/39`, `p=0.02663`;
- replication 3: `25/39`, `p=0.05406`;
- pooled exploratory/post-hoc total: `96/156`, `p=0.002456`, CI95
  `[0.534, 0.692]`.

The pooled estimate suggests that a large win probability such as `p1=0.70` is
not the right practical target for a cleaner follow-up.  Priority 2 therefore
locks a more sensitive design before drawing any new confirmatory targets.

Priority 3 DP accounting also changed how RQ1 must be worded.  The existing
distortion-matched clipping/noise comparator has finite but extremely large
epsilon under the approved update-level Gaussian/RDP accountant.  For example,
the conservative comparator with `clip_norm=100` and
`noise_multiplier=0.00025` has epsilon approximately `8.019e6` for
`delta=1e-5`, add/remove neighboring updates and a single release.  Therefore
Priority 2 must describe the comparator as:

```text
DP-style full-client-update clipping/noise with weak formal accounting under
the stated update-level assumptions
```

It must not be described as strong record-level differential privacy or as a
meaningful DP privacy baseline.

## Locked statistical design for the clean RQ1 confirmation

Primary test:

```text
one-sided exact binomial/sign test
H0: win probability <= p0 = 0.50
H1 planning point: p1 = 0.65
alpha = 0.05
target power = 0.80
```

The win definition remains unchanged from the existing RQ1 design:

```text
DNA protection is counted as a win on a non-tied target if it produces larger
reconstruction error than the DP-style comparator on that same target.
```

Ties are still removed according to the locked RQ1 tie rule.

Exact-binomial power analysis with `p0=0.50`, `p1=0.65`, `alpha=0.05`,
`power=0.80`, planned tie rate `0.10`, dropout rate `0.05` and
`maximum_feasible_target_count=150` gives:

```text
required effective non-tied n = 69
rejection threshold = at least 42 wins
actual alpha = 0.04559324528350576
actual power at p1 = 0.8020563639664955
initial draw count = 81
```

The power output is stored at:

```text
artifacts/rq1/priority2_power_p1_0p65_20260916/rq1_power_analysis.json
artifacts/rq1/priority2_power_p1_0p65_20260916/rq1_power_sensitivity.csv
```

This design is inside the previously approved non-hour resource ceiling:

```text
maximum_feasible_target_count = 150
maximum_artifact_gb = 500
```

The previously removed `maximum_machine_hours` ceiling remains removed and is
not reinstated by this amendment.

## p1=0.60 sensitivity note

The supervisor requested that `p1=0.60` must not be used unless a separate
amendment explicitly raises `maximum_feasible_target_count`.  The exact
calculation confirms why:

```text
p1 = 0.60
required effective non-tied n = 158
initial draw count at 10% ties and 5% dropout = 185
```

This exceeds the current target-count ceiling of `150`.  Priority 2 therefore
uses `p1=0.65` as the approved main design.

## Scope held fixed

Unless another pre-result amendment explicitly says otherwise, the clean RQ1
confirmation keeps the existing bounded scope:

- dataset: PaySim credit-card/fraud workflow already used by RQ1;
- records per group: `4`;
- fraud records per group: `1`;
- local update: `1` Adam step;
- raw branch and controls: same gate structure as the existing RQ1 protocol;
- branch gates: interpreted separately per branch;
- primary metric: feature-MSE reconstruction resistance;
- secondary metrics: pseudo-image PSNR/SSIM where the required reconstruction
  vectors and normalization metadata are available;
- no post-hoc target set may be used except for source-disjointness checks.

The conservative DNA Transform remains the default reference configuration
until the Priority 2 Pareto development sweep is completed.  If the Pareto
sweep identifies a different DNA configuration as the representative point for
confirmation, that choice must be frozen in a later pre-target amendment before
any confirmatory target set is created.

## Development-only Pareto sweep rule

Before creating the new confirmatory target set, run a development-only Pareto
sweep across multiple DNA and DP-style settings:

- DNA grid: at least five to six points spanning weaker than, comparable to,
  and stronger than the existing `conservative`, `medium`, and `stronger`
  configurations.
- DP-style grid: at least five to six noise multipliers, including the already
  used distortion-matched values and any previously examined
  utility-extension value.
- Metrics per point:
  - reconstruction resistance relative to raw where available;
  - utility cost in F1/AUC delta if available;
  - otherwise relative-L2 update distortion as a utility/distortion proxy;
  - epsilon for each DP-style point using the Priority 3 accountant.

The sweep may use development data only.  It must not use new confirmatory
targets, Group 2/3 targets, replication targets, `level1_fresh_final_targets.pt`
or any other historical/post-hoc target outcomes for selecting the
representative point.

The Pareto sweep must be reported separately before the final confirmatory
configuration is frozen.  The supervisor may then approve one or two
representative DNA points for the clean confirmation.

## Confirmatory execution rule

After the Pareto development report and supervisor selection:

1. write a separate pre-target freeze amendment/config identifying the selected
   DNA point(s), the DP-style comparator, epsilon wording, and target count;
2. create exactly one new confirmatory target set, source-disjoint with every
   previous development, post-hoc, confirmatory and replication target set;
3. run the full raw/DNA/DP-style branches and controls once;
4. do not rerun or redraw because the result is favorable or unfavorable.

The new clean confirmation may be used as the primary Priority 2 RQ1 evidence.
The older Group 2/3 pooled results must remain reported as historical
exploratory/post-hoc evidence and must not be silently overwritten.

## Reporting wording required for RQ1

All Priority 2 RQ1 reports, and any notes appended to older RQ1 reports, must
include the following limitation:

```text
All DNA-vs-DP-style comparisons in this project use clipping/noise
comparators whose formal epsilon values are extremely large under the approved
update-level Gaussian/RDP accountant.  Therefore the RQ1 conclusion is only a
comparison against weak-accounting DP-style clipping/noise at comparable update
distortion, not a comparison against strong record-level differential privacy.
```

## Non-tuning constraints

- Do not modify `torch.set_num_threads(1)`.
- Do not tune after seeing development or confirmatory outcomes; if an
  unexpected technical problem requires a change, write a timestamped amendment
  before continuing.
- Do not use privacy outcomes from post-hoc targets to choose Pareto points.
- Do not exceed `maximum_feasible_target_count=150` unless a separate
  supervisor-approved amendment raises that ceiling before target creation.
- Do not run a second clean confirmatory draw for the same comparison.
