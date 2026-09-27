# Amendment: Priority 14 DP comparator utility gap

**Date:** 2026-09-27
**Status:** CONFIRMATORY FROZEN / AUTHORIZED
**Scope:** utility-only follow-up for the exact Priority 14 distortion-matched DP comparators.

## Rationale

The Priority 14 clean-vector DNA-vs-DP head-to-head comparison showed that the
distortion-matched DP-style comparator won 39/39 paired targets against both:

- DNA Transform v1 stronger vs `DP_CLIP_NORM=100.0`, `DP_NOISE_MULTIPLIER=0.0004`;
- DNA Transform v2 ratio `0.95`, eta `0.01` vs `DP_CLIP_NORM=100.0`,
  `DP_NOISE_MULTIPLIER=0.00105`.

The paper currently lacks an RQ2-style utility test for those exact DP
comparators against the unprotected FL baseline.  This amendment freezes that
missing utility test before running any new DP utility jobs.

## Frozen hypotheses and decision rule

This is a paired non-inferiority utility test, matching the existing RQ2
framework:

- alpha: `0.05`;
- confidence level: `95%`;
- F1 non-inferiority margin: `0.02`;
- AUC-ROC non-inferiority margin: `0.005`;
- endpoint rule: both F1 and AUC-ROC must pass;
- paired seed contract: the same `FL_RUN_SEED` controls data subsampling/split,
  client partition, model initialization, and loader order.

For each DP comparator, the contrast is:

```text
FL_DP_COMPARATOR_MINUS_FL_BASELINE
```

The lower bound of the 95% paired CI must be at least `-0.02` for F1 and
at least `-0.005` for AUC-ROC.

## Frozen sample size and seed list

Use the already frozen RQ2-v2 confirmatory sample-size discipline:

- required seed count: `52`;
- maximum feasible seed count precedent: `60`;
- seed list: exactly the 52 seeds from
  `protocols/config/rq2_v2_confirmatory.json`.

Rationale: RQ2-v2 is the most conservative existing utility precedent after
the machine-hour ceiling was removed and variance was observed to be higher
than the original v1 utility runs.  No smaller pilot is used because this is a
known missing confirmatory utility measurement for already selected Priority
14 comparators, not a new method-selection exercise.

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
jobs for both Priority 14 DP comparators.

## Frozen DP methods

Run exactly these two DP-style full-client-update clipping/noise comparators:

| Method ID | Clip norm | Noise multiplier | Priority 14 source |
| --- | ---: | ---: | --- |
| `dp_v1_stronger_0p0004` | `100.0` | `0.0004` | v1 stronger vs DP 0.0004 |
| `dp_v2_ratio0p95_0p00105` | `100.0` | `0.00105` | v2 ratio0.95/eta0.01 vs DP 0.00105 |

The DP implementation is the existing `experiments/run_fraud_fl_dp.py`.  No
model, optimizer, DP, threshold-selection, seed, margin, or stopping rule may
be changed after seeing outcomes.

## Failure and rerun policy

Run each DP job once.  Rerun only for demonstrable infrastructure failure
before a scientific result is available, retaining failed job records.  Do not
replace low-performing seeds, relax margins, or remove observations.

This amendment does not modify any frozen RQ1/RQ2/RQ3 conclusion.  It supplies
the missing utility evidence needed to interpret the Priority 14 DP comparator.
