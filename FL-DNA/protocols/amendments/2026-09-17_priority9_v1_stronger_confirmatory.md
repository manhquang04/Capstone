# Amendment: Priority 9 v1-stronger Confirmatory Escalation

**Timestamp:** 2026-09-17, after the precommitted n=24 development pilot passed
and before any confirmatory target generation.  
**Status:** CONFIRMATORY AUTHORIZED FOR ONE RUN ONLY.  
**Applies to:** PaySim `v1_stronger__GEN_IDLG_STYLE`.

## Triggering evidence

The parent amendment
`protocols/amendments/2026-09-17_priority9_v1_config_family_sensitivity.md`
authorized confirmatory escalation only if the independent n=24 development
pilot passed both controls.

The `v1_stronger__GEN_IDLG_STYLE` n=24 pilot met that rule:

- Prior control: 20/24 wins, one-sided exact sign-test p=0.000771939754486084.
- Zero-update control: 19/24 wins, one-sided exact sign-test p=0.003305375576019287.

This amendment does not refit the effect size from those observed pilot win
rates.

## Frozen confirmatory design

- Dataset: PaySim (`datasets/creditcard.csv`).
- Scope: 4 records/group, 1 fraud record/group, 1 client step.
- Defense: DNA Transform v1 stronger.
  - `mix_ratio=0.12`
  - `keep_ratio=0.82`
  - `shrink_factor=0.35`
- Attacker generation: `GEN_IDLG_STYLE`.
- Controls: Prior and Zero-update.
- Statistical test: exact one-sided binomial/sign test.
- Alpha: 0.05.
- Minimum meaningful win probability: `p1=0.70`, inherited from the existing
  RQ1 escalation rule; not refit from pilot data.
- Required non-tied count: 37.
- Draw count: 39, preserving the same two-target buffer used in Priority 6/7.
- Confirmatory execution: exactly one run.

## Data firewall

The confirmatory target set must be newly generated and source-disjoint from all
PaySim development, post-hoc, confirmatory, Priority 6, Priority 8, and Priority
9 n=8/n=24 target pools detected by the source-disjoint target-generation
script.

## Stopping rule

Regardless of outcome, no second confirmatory run is authorized for
`v1_stronger__GEN_IDLG_STYLE` under this amendment.
