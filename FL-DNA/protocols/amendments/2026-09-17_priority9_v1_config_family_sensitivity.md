# Amendment: Priority 9 DNA Transform v1 Config-Family Sensitivity

**Timestamp:** 2026-09-17, before target generation or execution.  
**Status:** DEVELOPMENT-GATE ONLY until a config independently passes escalation.  
**Applies to:** PaySim parameter-sensitivity checks for DNA Transform v1.

## Purpose

`GEN_IDLG_STYLE` has a confirmed attack against DNA Transform v1 medium on
PaySim. This amendment tests whether that vulnerability extends to the other two
pre-existing v1 configurations:

1. `v1_conservative`
   - `mix_ratio=0.08`
   - `keep_ratio=0.88`
   - `shrink_factor=0.45`
2. `v1_stronger`
   - `mix_ratio=0.12`
   - `keep_ratio=0.82`
   - `shrink_factor=0.35`

This is a v1 parameter-sensitivity check, symmetric in spirit with Priority 8
for v2. It does not alter any frozen RQ1/RQ2/RQ3 conclusion.

## Order

Run sequentially:

1. `v1_conservative__GEN_IDLG_STYLE`;
2. `v1_stronger__GEN_IDLG_STYLE`.

Do not start the stronger branch until the conservative branch has reached its
protocol-defined stopping point.

## Initial gate for each config

For each config:

- dataset: PaySim (`datasets/creditcard.csv`);
- scope: 4 records/group, 1 fraud record/group, 1 client step;
- groups: 8;
- attacker generation: `GEN_IDLG_STYLE`;
- controls: Prior and Zero-update;
- test: exact one-sided sign test, `alpha=0.05`;
- pass condition: must beat both controls.

Each target set must be newly generated and source-disjoint from all prior
PaySim development, post-hoc, confirmatory, Priority 6, Priority 8, and other
target pools detected by the source-disjoint target-generation script.

## Escalation rule

If n=8 passes, escalate to an independent n=24 development pilot with a new
source-disjoint target set.

If n=24 also passes strongly, write a separate confirmatory amendment before
any confirmatory target generation. Confirmatory must use:

- `p1=0.70`;
- required non-tied `n=37`;
- draw `n=39`;
- exact one-sided binomial/sign-test;
- exactly one confirmatory run.

If a config fails at n=8 or n=24, stop that config and report the boundary. Do
not interpret failure as safety; it only means this attacker/gate did not meet
the precommitted threshold at that setting.

## Out of scope

- No IEEE-CIS run is authorized.
- No CIFAR-10 run is authorized.
- No v1-medium rerun is authorized; use the existing Priority 6 confirmatory
  result for the medium row in the final summary table.

