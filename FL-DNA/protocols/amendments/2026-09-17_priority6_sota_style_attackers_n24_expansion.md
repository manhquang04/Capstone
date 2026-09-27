# Amendment: Priority 6 SOTA-Style Attackers n=24 Development Expansion

Status: PRE-RUN FROZEN
Timestamp: 2026-09-17T00:00:00+07:00
Scope: Development only; no confirmatory execution authorized

## Rationale

The pre-registered n=8 development gate for Priority 6 found two attacker/defense/dataset cells that passed both the Prior and Zero-update controls under the exact one-sided sign-test gate:

- PaySim, DNA Transform v1 medium, GEN_IDLG_STYLE: Prior 7/8, p=0.03515625; Zero 8/8, p=0.00390625.
- IEEE-CIS, DNA Transform v2 ratio0.95 eta0.01, GEN_COSINE_TV: Prior 7/8, p=0.03515625; Zero 7/8, p=0.03515625.

Per the frozen Priority 6 design amendment, a generation that clearly passes the minimal n=8 gate may be expanded to an independent n=24 development pilot before any confirmatory decision. This amendment freezes that expansion before creating or inspecting any n=24 targets.

## Frozen Expansion Cells

Only the following cells are authorized for the n=24 development expansion:

1. PaySim / v1_medium / GEN_IDLG_STYLE.
2. IEEE-CIS / v2_ratio0p95_eta0p01 / GEN_COSINE_TV.

All other Priority 6 n=8 cells remain development-gate FAIL and are not authorized for n=24 expansion under this amendment.

## Target and Seed Rules

- Each expanded cell uses a fresh n=24 development target set.
- All new source rows must be disjoint from every known development, post-hoc, and confirmatory target pool, including the Priority 6 n=8 targets created on 2026-09-16.
- The PaySim replay seed is frozen as 2026091761.
- The IEEE-CIS target seed is frozen as 2026091762.
- The PaySim attack seed is frozen as 2026091763.
- The IEEE-CIS attack seed is frozen as 2026091764.
- The attack hyperparameters remain unchanged from the n=8 gate: 4 restarts per group; PaySim uses 600 optimization steps with lr=0.1; IEEE-CIS uses 300 optimization steps with lr=0.05.

## Gate Rule

For each expanded cell, run the same group-level selection and exact one-sided sign test used in the n=8 Priority 6 gate. A cell passes the n=24 development expansion only if it passes both controls:

- baseline reconstruction beats Prior with mean difference < 0, median difference < 0, and exact one-sided p < 0.05;
- baseline reconstruction beats Zero-update with mean difference < 0, median difference < 0, and exact one-sided p < 0.05.

This remains a development pilot. Passing n=24 does not authorize any confirmatory run by itself. Any confirmatory escalation requires a separate pre-run amendment with a fresh target set and power analysis.

## Non-Tuning Commitment

No attacker parameters, defenses, target composition rules, or gate criteria may be changed after inspecting the n=24 results. Technical replay is allowed only for implementation/runtime failures and must be documented in a separate technical replay amendment before rerun.
