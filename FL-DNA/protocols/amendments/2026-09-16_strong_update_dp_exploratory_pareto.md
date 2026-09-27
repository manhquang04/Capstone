# Amendment: exploratory strong-DP update-level Pareto study

**Date:** 2026-09-16  
**Supervisor:** Manh Quang  
**Status:** AUTHORIZED — DEVELOPMENT / EXPLORATORY ONLY

## Purpose and scope

This amendment addresses the missing comparison against clipping/noise settings
with a small *update-level* `(epsilon, delta)` value.  It is not a new
confirmatory RQ1 experiment and cannot modify the conclusions of any frozen
v1/v2 confirmatory run.

The existing utility-matched extension (`noise_multiplier=0.00001`) has
epsilon approximately `5.000e9` at `delta=1e-5`, add/remove, one release.
It is therefore weaker even than the conservative distortion-matched setting
(`8.019e6` at `noise_multiplier=0.00025`).  Comparable observed utility at
that point must not be described as meaningful formal privacy.

The implementation still clips a whole client update, rather than individual
record gradients.  Thus every epsilon reported here is a Gaussian-mechanism
diagnostic for the explicitly stated client/update adjacency, **not** a
record-level DP guarantee for PaySim transactions.

## Frozen accounting convention and candidate budgets

The primary convention is exactly the one used by Priority 3:

- add/remove adjacency, sensitivity ratio `1`;
- `delta=1e-5`;
- one released clipped client update (no composition);
- `clip_norm=100`;
- Gaussian RDP accountant and the existing deterministic alpha grid.

`replace_one` is additionally reported as a sensitivity analysis, but does
not select the candidate multipliers.  The predeclared target values are
`epsilon in {100, 50, 10, 1}`.  These are illustrative privacy-budget
landmarks spanning moderate to stringent update-level privacy, not a claim
that they are universally appropriate.  They are motivated by the public
Opacus examples at epsilon 50 and lower epsilon 5/7.5, and by its guidance to
target a small constant epsilon; sources are logged in the report.

For each target, the multiplier is found by deterministic bisection against
the existing RDP accountant.  The selected value is the smallest multiplier
whose computed epsilon is at most the target.  The script records both the
requested and attained epsilon; there is no tuning against utility or attack
outcomes.

## Frozen development execution

### Utility

Run exactly the four selected strong-DP multipliers and paired Baseline on
PaySim development scale:

- seeds `[101, 202, 303]`;
- `MAX_ROWS=100000`, three clients, one local epoch, ten rounds;
- focal loss (`alpha=0.95`, `gamma=2.0`);
- `torch.set_num_threads(1)` remains unchanged inside the invoked pipeline.

The observed F1/AUC deltas versus the paired baseline are descriptive only.
No candidate is dropped, expanded, or rerun because its utility is favourable
or unfavourable.

### Reconstruction resistance and gates

Create one new eight-group development target set (four records and one fraud
record per group) source-disjoint from every existing target artifact.  Run:

1. raw attacker against that set as the raw-pipeline gate;
2. the existing clipping/noise Monte-Carlo attacker, separately for each of
   the four frozen multipliers, with 8 restarts, the inherited iterations and
   learning rate, 100 MC samples, and defense seed `314159265`.

Each DP branch is interpreted only if its own attacker passes both Prior and
Zero-update controls.  A failed gate means reconstruction resistance for that
candidate is **unresolved**, not that the DP mechanism is proven protective.
No target is confirmatory and no result selects a subsequent comparator.

## Data and integrity restrictions

Do not read any post-hoc target for attack, calibration, or utility selection,
including all old Group 2/3 targets and `level1_fresh_final_targets.pt`.
The only permitted access to old target artifacts is fingerprint comparison to
prove source disjointness.  Do not change DNA code, attacker parameters,
`torch.set_num_threads(1)`, clipping norm, or any frozen v1/v2 configuration.
All four DP levels are run and reported even if an earlier one fails its gate.

## Deliverables

Produce four separately labelled development reports: inverse accounting,
utility, attack/gates, and a final three-axis Pareto table.  The final table
must include the existing DNA v1/v2 and utility-matched context, each clearly
labelled with its weak update-level accounting and without implying a
record-level DP claim.
