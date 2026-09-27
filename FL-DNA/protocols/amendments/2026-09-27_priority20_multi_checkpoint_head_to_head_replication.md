# Priority 20 amendment — multi-checkpoint replication of DP-vs-transform head-to-head

**Date:** 2026-09-27  
**Status:** FROZEN BEFORE EXECUTION  
**Scope:** new pre-registered experiment addressing the effective-sample-size / checkpoint-sharing concern raised by Priority 18 for the tabular DP-vs-transform head-to-head result.

## Motivation

Priority 18 showed that the original DP-vs-transform head-to-head confirmatory
run used one shared tabular pre-local model checkpoint within a run. Although
the targetwise result was unanimous, a maximally conservative treatment could
regard one run as one independent checkpoint-level observation. This amendment
therefore freezes a multi-checkpoint replication before any new Priority 20
checkpoint, target, or attack execution is created.

## Frozen comparator cell

Only one comparator cell is authorized:

- Transform: DNA Transform v1 medium
  - `block_size = 256`
  - `mix_ratio = 0.10`
  - `keep_ratio = 0.85`
  - `shrink_factor = 0.40`
  - `seed = 681958327`
- DP comparator:
  - `clip_norm = 100.0`
  - `noise_multiplier = 0.000315`
  - `mc_noise_samples = 100`
  - `defense_seed = 314159265`
- Attacker / head-to-head harness:
  - same clean-vector raw-lift / balanced-tensor objective used in Priority 16;
  - vector scope is trainable parameters only;
  - non-trainable buffers remain excluded.

Rationale: v1-medium is the configuration used for the project's RQ1/RQ2
configuration-alignment story and has the smallest absolute mean `D` among the
reported head-to-head cells, so it is the most conservative single cell for
testing checkpoint-level replication.

## Checkpoint replication design

The experiment uses **5 independently initialized and independently warmup-trained checkpoints**.

Checkpoint seeds are frozen as:

| checkpoint_id | checkpoint_seed |
|---|---:|
| ckpt_0 | 2026092801 |
| ckpt_1 | 2026092802 |
| ckpt_2 | 2026092803 |
| ckpt_3 | 2026092804 |
| ckpt_4 | 2026092805 |

Each checkpoint is created by:

1. loading the same PaySim preprocessing/split convention used by the original
   Phase 3/Phase 4 tabular pipeline (`max_rows=500000`, split seed
   `20260907`) so feature coordinates remain compatible with the existing
   target-generation code;
2. initializing `FraudMLP` from the checkpoint-specific seed;
3. running one FedAvg warmup round with three clients and Adam local training,
   using checkpoint-specific warmup seeds;
4. saving one independent `pre_local.pt` for that checkpoint.

The fixed preprocessing/split convention is intentional: this experiment
isolates the model-checkpoint effect without changing the feature coordinate
system or target-generation protocol. The scientific independence unit is the
checkpoint initialization/warmup realization, not a new dataset definition.

## Targets per checkpoint

Each checkpoint uses a fresh target set of **8 groups**, with:

- `records_per_group = 4`
- `fraud_per_group = 1`
- source-disjointness checked against all prior project target pools and
  against the other Priority 20 target sets.

Target seeds are frozen as:

| checkpoint_id | target_seed |
|---|---:|
| ckpt_0 | 2026092811 |
| ckpt_1 | 2026092812 |
| ckpt_2 | 2026092813 |
| ckpt_3 | 2026092814 |
| ckpt_4 | 2026092815 |

Run seeds for attacker/defense execution are frozen as:

| checkpoint_id | run_seed |
|---|---:|
| ckpt_0 | 2026092821 |
| ckpt_1 | 2026092822 |
| ckpt_2 | 2026092823 |
| ckpt_3 | 2026092824 |
| ckpt_4 | 2026092825 |

Each checkpoint run uses `restarts = 4` and `workers <= 8`.

## Win/tie rule

For each target:

```text
D_i = MSE(attacker, DNA-defended update)_i - MSE(attacker, DP-defended update)_i
```

- DP win: `D_i < 0`
- DNA win: `D_i > 0`
- Tie: `D_i == 0`

The threshold is strict zero-threshold, matching the original Priority
14/Priority 16 head-to-head design. No legacy wide tie band is used for the
primary Priority 20 analysis.

## Frozen statistical summaries

Per checkpoint, report:

- target count;
- DP wins, DNA wins, exact ties;
- mean `D`, SD(`D`), median `D`;
- whether DP wins every target;
- whether DP wins a strict majority of non-tied targets.

Across checkpoints, report:

1. checkpoint-majority sign count: number of checkpoints where DP wins a strict
   majority of targets, with exact one-sided binomial p-value under p0 = 0.5;
2. checkpoint-mean sign count: number of checkpoints where mean `D < 0`, with
   exact one-sided binomial p-value under p0 = 0.5;
3. whether DP wins every target in every checkpoint;
4. aggregate targetwise sign count as descriptive only, not as a replacement
   for the checkpoint-level analysis.

With 5 checkpoints, unanimous checkpoint-level agreement corresponds to
one-sided exact p = `0.03125`.

## Non-negotiables

- Do not stop early if an intermediate checkpoint looks favorable or
  unfavorable.
- Do not add, drop, or replace checkpoints after seeing partial results.
- Report all 5 checkpoint runs regardless of outcome.
- Do not alter the transform, DP comparator, attacker objective, restart count,
  target count, or statistical rule after execution starts.
- Rerun is permitted only for demonstrable infrastructure failure before a
  scientific result exists, and must be documented as a technical replay.
- No file under `Latex/` may be edited for this task.
