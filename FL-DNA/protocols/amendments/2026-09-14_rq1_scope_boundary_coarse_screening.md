# Amendment: RQ1 attacker scope-boundary coarse screening

**Date:** 2026-09-14  
**Supervisor:** Manh Quang  
**Status:** APPROVED FOR DEVELOPMENT SCOPE-BOUNDARY SCREENING  
**Scope:** RQ1 raw-branch validity boundary only; this amendment does not alter
any prior confirmatory conclusion.

## Purpose

This amendment probes the valid scope boundary of the already-frozen RQ1
attacker pipeline by controlled escalation.  It is not an attacker-improvement
study.  The only experimental variable changed from the confirmed 4-record
scope is:

```text
records_per_group
```

The diagnostic hypothesis is that adding non-fraud records dilutes the fraud
signal that the current attacker can exploit.

## Locked screening design

- Levels tested in order: `records_per_group = 8, 12, 16`.
- `fraud_records_per_group = 1` at every level.
- `local_steps = 1` at every level, inherited from the existing Phase 4 target
  construction / baseline gate contract.
- Target groups per screening level: `10`.
- Branches run during coarse screening: `raw` only.
- Raw attack budget: `4` restarts per group, standard initialization,
  `nonnegative_lambda = 0.001`.
- Maximum concurrent processes: `9`.

Pre-committed target seeds:

| records_per_group | target seed |
| ---: | ---: |
| 8 | `1940948331` |
| 12 | `1563092417` |
| 16 | `377561827` |

## Source-disjointness rule

Each screening target set must be source-disjoint with all previous target
artifacts, including development, confirmatory, replication, and post-hoc target
sets.  Historical/post-hoc targets may be inspected only for source identifiers
to verify disjointness; no prior outcomes or reconstructed values may be used
for tuning.

## Gate and stopping rule

For each level, raw branch validity is judged against the same two controls used
in the existing RQ1 gate structure:

1. objective minus prior fraud-MSE;
2. objective minus zero-update fraud-MSE.

A control gate passes only if all are true:

```text
mean difference < 0
median difference < 0
one-sided exact sign-test p < 0.05
```

The raw branch passes a level only if both control gates pass.

The levels must be run in ascending order.  If raw fails at any level, screening
stops immediately and no larger level may be tested.  The scope-boundary result
is then reported as:

```text
the current method is supported up to the previous passing level and fails at
the first failing level
```

If raw passes all three levels, screening stops at `records_per_group = 16`.
No additional level may be added under this amendment.

## Confirmatory escalation rule

Full raw/DNA/DP confirmatory execution is not authorized by this amendment.
If coarse screening identifies a largest passing level, a separate amendment
must be written before any confirmatory escalation.  That later amendment must
lock:

- the single level selected for confirmatory escalation;
- a separate DP distortion calibration plan for that level;
- power analysis assumptions and target count;
- a new source-disjoint confirmatory target set;
- unchanged attacker/DNA/DP settings except where explicitly calibrated before
  seeing confirmatory outcomes.

If `records_per_group = 8` fails, no confirmatory escalation is needed; the
boundary finding `4 < scope <= 8` is sufficient for reporting.

## Non-tuning constraints

- Do not modify `torch.set_num_threads(1)`.
- Do not change objective, parameterization, restarts, initialization mode, or
  other attacker parameters after seeing a screening result.
- Do not rerun a failed level with altered settings to rescue the gate.
- Do not use this exploratory/development-style screening to overturn any prior
  confirmatory RQ1 conclusion.
