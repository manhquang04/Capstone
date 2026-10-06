# 2026-09-28 — Priority 22 corrected positive-control protocol

Status: FROZEN / AUTHORIZED BEFORE EXECUTION

## Reason for this amendment

Priority 21 Step 4 is invalid as a positive control because it did not match the
Priority 16 head-to-head harness: it used `init_mode="plausible"` and
`Adam(lr=0.08)`, while `experiments/priority16_v1_medium_dna_vs_dp_probe.py`
uses `init_mode="standard"` and `Adam(lr=0.1)`.  This amendment freezes a
corrected rerun and does not edit or reinterpret the Priority 21 amendment.

## Target pool

Use the existing frozen Priority 16 confirmatory target set:

- path: `artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_targets_20260927/paysim_priority16_v1_medium_confirmatory_targets.pt`
- SHA-256: `645fe7ff62403e0bce81a3fc6cbed4d857306d36719cc2e3bad740acb73297fe`
- groups: 39

This is an instrument-validity check on the exact Priority 16 setting, not a new
source-disjoint scientific comparison.

## Attacker definition

The corrected positive control must match `experiments/priority16_v1_medium_dna_vs_dp_probe.py::_optimize_branch`:

- initialization:
  `_initial(original.shape, derive_seed(seed, "priority14-init", cell, branch, group, restart), "standard", distribution)`;
- optimizer: `torch.optim.Adam([latent], lr=0.1)`;
- evaluations: 601 (`step = 0..600`), with 600 optimizer updates;
- objective: `update_objective(candidate, signal, list(signal), reference=signal, mode="balanced_tensor")`;
- penalty: `0.001 * _nonnegative_penalty(latent, meta)`;
- restarts: 4;
- selection: minimum attacker-observable `best_objective`.

Before using the custom positive-control implementation, run an automated
equivalence check: rerun the Priority 16 DNA branch for two target groups using
the same code path and verify that the reproduced `clean_vector_mse` equals the
saved Priority 16 artifact value bit-for-bit.  If this check fails, stop and do
not interpret the positive control.

## Branches

- `none`: observed signal is the undefended trainable update, candidate signal is
  the simulated trainable update with no defense.
- `zero_update`: observed signal is a zero trainable-update dictionary,
  candidate signal is the simulated trainable update with no defense.

## Controls and gate

- Prior is the selected branch's own initial decoded and aligned to the target.
- Tie band: `0.0390625`.
- Exact one-sided sign test, `p0=0.5`, `n=39`.
- `none` wins against a control if `control_input_mse - none_input_mse > 0.0390625`.

## Pre-registered interpretation

- If `none` beats Prior, the attacker is a working instrument.  Since Priority 21
  re-analysis showed DNA and DP head-to-head branches do not beat their own
  priors in input space, report that both defenses defeat this attacker and that
  neither DNA nor DP is distinguishable from the other in input space.
- If `none` does not beat Prior, the attacker is not a valid instrument in this
  bounded tabular setting, and no tabular DNA-vs-DP privacy conclusion can be
  drawn from it.

No early stopping, target dropping, parameter tuning, or post-result branch
addition is allowed.
