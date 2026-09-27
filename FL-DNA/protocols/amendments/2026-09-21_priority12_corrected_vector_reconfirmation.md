# Amendment: Priority 12 corrected-vector reconfirmation

**Date written:** 2026-09-21  
**Status:** AUTHORIZED FOR DEVELOPMENT GATES ONLY; escalation follows the rule below.  
**Applies to:** tabular RQ1 attacker reconfirmation after Priority 11 contamination audit.

## Triggering issue

Priority 11 found that the tabular bounded-update RQ1 pipeline used full model
state deltas from `attacks.local_update.simulate()`. Those deltas included
floating BatchNorm running-stat buffers (`running_mean`, `running_var`) and the
attacker/defense code matched all floating keys. The resulting vector was not the
intended trainable-parameter update vector.

This amendment starts a fresh corrected-vector reconfirmation for the three
highest-value cells that previously passed under the contaminated vector:

1. PaySim / DNA Transform v1 medium / `GEN_IDLG_STYLE`
2. PaySim / DNA Transform v1 stronger / `GEN_IDLG_STYLE`
3. IEEE-CIS / DNA Transform v2 `ratio=0.95, eta=0.01` / `GEN_COSINE_TV`

## Corrected update-vector definition

For this amendment, the update vector is defined as:

> the ordered set of trainable model parameters only, equivalent to
> `model.named_parameters()` / `model.parameters()` with `requires_grad=True`.

All non-trainable buffers are excluded before any DNA transform, sketching,
defense operation, candidate transform, or attacker matching loss. This excludes
BatchNorm `running_mean`, `running_var`, `num_batches_tracked`, and any other
buffer found in `model.named_buffers()`.

This matches the vector-scope logic used by the Priority 7 CIFAR-10 runner, which
constructed updates from `named_parameters()` gradients only.

The mathematical DNA Transform v1/v2 formulas and attacker loss families are not
changed. Only the domain of the vector supplied to those formulas is corrected.

## Data firewall

All Priority 6/8/9 targets and any target/artifact inspected during Priority 10/11
are treated as post-hoc for this corrected-vector reconfirmation. New target sets
must be source-disjoint with every existing target pool.

## Execution rule

For each of the three cells, execute sequentially:

1. Development gate, n=8, fresh target set.
2. If and only if n=8 passes both Prior and Zero-update controls, escalate to n=24
   pilot with a fresh disjoint target set.
3. If and only if n=24 remains strong, write a separate confirmatory amendment,
   use the existing RQ1 exact-binomial method with `p1=0.70` (not refit from pilot),
   create a fresh disjoint confirmatory target set, and run exactly one
   confirmatory execution.
4. If a cell fails at n=8, stop that cell. Do not tune and do not escalate.

## Invariants

- `torch.set_num_threads(1)` remains unchanged.
- No attacker parameter is tuned after seeing a gate result.
- No old target set is reused.
- This amendment does not alter any frozen historical RQ1 conclusion.

