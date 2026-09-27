# Amendment: Priority 16 v1-medium DP comparator utility gap

**Date:** 2026-09-27
**Status:** CONFIRMATORY FROZEN / AUTHORIZED
**Scope:** utility-only follow-up for the exact Priority 16 v1-medium distortion-matched DP comparator.

## Rationale

Priority 16 added a third Priority-14-style DNA-vs-DP head-to-head row for
Transform v1-medium:

- DNA Transform v1-medium: `mix_ratio=0.10`, `keep_ratio=0.85`,
  `shrink_factor=0.40`;
- distortion-matched DP comparator: `DP_CLIP_NORM=100.0`,
  `DP_NOISE_MULTIPLIER=0.000315`.

The existing Priority 14 DP utility-gap follow-up covered only:

- `dp_v1_stronger_0p0004`;
- `dp_v2_ratio0p95_0p00105`.

This amendment freezes the missing RQ2-style utility non-inferiority test for
the exact v1-medium DP comparator before running any new DP utility jobs.

## Frozen hypotheses and decision rule

This is a paired non-inferiority utility test, matching the existing RQ2 and
Priority 14 DP utility framework:

- alpha: `0.05`;
- confidence level: `95%`;
- F1 non-inferiority margin: `0.02`;
- AUC-ROC non-inferiority margin: `0.005`;
- endpoint rule: both F1 and AUC-ROC must pass;
- paired seed contract: the same `FL_RUN_SEED` controls data subsampling/split,
  client partition, model initialization, and loader order.

The contrast is:

```text
FL_DP_V1_MEDIUM_COMPARATOR_MINUS_FL_BASELINE
```

The lower bound of the 95% paired CI must be at least `-0.02` for F1 and
at least `-0.005` for AUC-ROC.

## Frozen sample size and seed list

Use the already frozen RQ2-v2 confirmatory sample-size discipline, identical to
the Priority 14 DP utility-gap follow-up:

- required seed count: `52`;
- seed list: exactly the 52 seeds from
  `protocols/config/rq2_v2_confirmatory.json`;
- no smaller pilot is used, because this is a missing confirmatory utility
  measurement for an already selected head-to-head DP comparator, not a new
  method-selection exercise.

## Baseline reuse

The unprotected baseline for these 52 seeds already exists from the frozen
RQ2-v2 confirmatory run:

```text
artifacts/rq2_v2/confirmatory_20260916/
```

That run used the same seed contract and training configuration required here
(`500000` rows, `50` rounds, `3` clients, one local epoch, focal loss with
alpha `0.95` and gamma `2.0`).  This amendment authorizes reusing those
baseline metrics rather than rerunning the baseline, while running fresh DP
jobs for the Priority 16 v1-medium DP comparator.

## Frozen DP method

Run exactly this DP-style full-client-update clipping/noise comparator:

| Method ID | Clip norm | Noise multiplier | Source |
| --- | ---: | ---: | --- |
| `dp_v1_medium_0p000315` | `100.0` | `0.000315` | Priority 16 v1-medium vs DP 0.000315 |

The DP implementation is the existing `experiments/run_fraud_fl_dp.py`.  No
model, optimizer, DP, threshold-selection, seed, margin, or stopping rule may
be changed after seeing outcomes.

## Failure and rerun policy

Run the DP job once per seed.  Rerun only for demonstrable infrastructure
failure before a scientific result is available, retaining failed job records.
Do not replace low-performing seeds, relax margins, or remove observations.

This amendment does not modify any frozen RQ1/RQ2/RQ3 conclusion.  It supplies
the missing utility evidence needed to interpret the Priority 16 v1-medium DP
comparator.
