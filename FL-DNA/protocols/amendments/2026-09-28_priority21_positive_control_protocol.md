# 2026-09-28 — Priority 21 positive-control protocol

Status: FROZEN / AUTHORIZED BEFORE EXECUTION

Supervisor context: Priority 21 audits whether the tabular DP-vs-transform head-to-head
metric is measuring input reconstruction privacy or a defended-update residual that can
be dominated by mechanical defense noise.  Steps 1--3 are re-analyses of existing
artifacts only.  This amendment freezes the only new experimental work in Priority 21:
an undefended positive-control branch.

## Frozen vector definition

All update-space objectives and measurements in this positive-control run use only
trainable parameters, i.e. the names returned by `model.named_parameters()`.  Model
buffers, including BatchNorm `running_mean`, `running_var`, and `num_batches_tracked`,
are excluded.

## Target pool

The positive control reuses the already-frozen Priority 16 v1-medium head-to-head
confirmatory target artifact:

- `artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_targets_20260927/paysim_priority16_v1_medium_confirmatory_targets.pt`

This is permitted because the positive control is an instrument-validity check on the
same targets, not a new scientific comparison requiring source-disjoint targets.
No target is dropped or replaced.

## Branches

The positive-control branch `none` is defined as:

- observed signal: the undefended trainable local update;
- candidate signal: the simulated trainable local update from the attacker's candidate
  input;
- objective: the same `balanced_tensor` update objective used by the Priority 14/16
  head-to-head harness;
- iterations/restarts/init distribution: same values as the Priority 16 confirmatory
  harness (`iterations=600`, `restarts=4`, same hard-diff initialization family and
  nonnegative penalty).

For the required controls:

- `Prior`: the input reconstruction before optimization for the selected `none` restart;
- `Zero-update`: the same attacker optimized against a zero trainable-update signal,
  with the same objective/iterations/restarts/init distribution.  Its selected restart
  is the restart with the smallest attacker-observable zero-update objective.

## Gate and interpretation

The input-space metric is `feature_metrics.mean_mse`, evaluated with the same PaySim
alignment/scoring function used by the head-to-head artifacts.

The pre-registered gate uses the v1-family tie band:

- tie band: `0.0390625`;
- one-sided exact sign test against `p0=0.5`;
- `none` wins against a control when `control_input_mse - none_input_mse > 0.0390625`.

Pre-registered interpretation:

- If `none` does not pass its gate against `Prior` at `n=39`, the attacker is not a
  valid instrument for interpreting tabular DNA-vs-DP privacy comparisons under this
  bounded 4-record setting.
- If `none` passes against `Prior` but not against `Zero-update`, the result is reported
  as partial instrument validity only.

No early stopping, target dropping, or post-result tuning is allowed.

## Scope not run in this amendment

The Priority 14 v2 target pool is not included in this first positive-control run.  It
requires a separate amendment if the supervisor decides that the extra runtime is needed
after seeing the Priority 16 instrument-validity result.
