# Technical replay amendment: trainable-key selection objective

**Date:** 2026-09-21  
**Status:** TECHNICAL REPLAY AUTHORIZED  
**Scope:** attacker candidate-selection objective in the Phase-4 tabular attack
backends used by RQ1-v2 and strong update-DP exploratory gates.

## Issue found

A follow-up code audit found a related but distinct vector-scope issue from the
earlier BatchNorm-buffer contamination bug.

The following runner scripts built the attacker's matching/selection objective
key list with:

```python
keys = [key for key, value in observed.items() if value.is_floating_point()]
```

or the equivalent `_floating_keys(update)` helper:

- `experiments/run_phase4_harddiff_reparam_for_misselected.py`
- `experiments/run_phase4_dna_v2_iht_attack.py`
- `experiments/run_phase4_simple_defense_attack.py`

This excluded only integer buffers such as `num_batches_tracked`. It still
included floating BatchNorm buffers such as `running_mean` and `running_var`.
Those keys were then passed to `update_objective(..., mode="balanced_tensor")`,
so the optimizer's candidate-selection loss was computed over a
buffer-contaminated vector.

## Exact fix

The affected runners now construct the matching key list from trainable model
parameters only:

```python
def _trainable_keys(model):
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
```

and use:

```python
keys = _trainable_keys(model)
```

For `run_phase4_simple_defense_attack.py`, observed/candidate defense
application now applies the defense only to keys in this trainable set and
clones all other entries unchanged. This is required because the defense plan is
now defined only over trainable parameters; it does not alter the DNA/DP
scientific parameters or attacker hyperparameters.

## Why this is a technical replay

This amendment restores the already intended and repeatedly documented vector
definition:

- trainable parameters only;
- all model buffers excluded from the attack matching objective;
- no scientific parameter change;
- no target-set reuse;
- no seed, attacker generation, optimizer, restart count, iteration budget,
  DNA setting, DP setting, or statistical threshold change.

The replay is required because the corrected selection objective can change
candidate choice and therefore can change downstream results in either
direction.

## Authorized replay chain

Run fresh source-disjoint target pools for both affected result families:

1. RQ1-v2 DNA-v2 vs DP-v2 comparison.
2. Strong update-DP utility-ceiling attacker gates.

For each family:

- Stage 1: n=8 development probe.
- Stage 2: n=24 pilot if Stage 1 does not fail decisively.
- Stage 3: confirmatory only after the pilot, using the frozen statistical
  framework for that family.

For RQ1-v2, the pre-registered v2 protocol specifies `groups = 176` for
confirmatory execution; this value must be used instead of the generic
Priority-6 `draw n = 39` if the chain reaches Stage 3.

For strong update-DP, use the generic exact one-sided sign-test framework:

- p0 = 0.5;
- p1 = 0.70;
- alpha = 0.05;
- power = 0.80;
- required non-tied n = 37;
- draw n = 39 unless a later pre-run amendment documents a different
  pre-existing protocol requirement.

At every stage, report raw win/loss/tie counts and exact one-sided p-values for
both controls. Do not skip a stage. Do not interpret a small-sample pass as final
without the later pilot/confirmatory stage.

## Invariants

- Do not change `torch.set_num_threads(1)`.
- Do not reuse old target pools, including old RQ1-v2 n=176 or strong-DP target
  sets.
- Do not tune after seeing replay outcomes.
- Technical rerun is allowed only for clear infrastructure/runtime failure
  before obtaining a scientific result.
