# Amendment: Priority 6 SOTA-style gradient-inversion attacker variants

Status: `DEVELOPMENT_GATE_AUTHORIZED`

Written at: 2026-09-16T16:06:42Z, before creating any new Priority 6 target set or running any Priority 6 development gate.

Scope: supplemental development-only attacker evaluation. This amendment does not change any frozen conclusion for RQ1/RQ2/RQ3, and does not authorize confirmatory execution.

## Motivation

The existing project attacker implementations include DLG-style gradient/update matching based primarily on squared/L2 discrepancy. To reduce the risk that the evaluated attacker is perceived as a strawman, Priority 6 adds two literature-inspired variants while preserving the existing development-gate discipline.

Primary references:

- Geiping, Bauermeister, Dröge, and Moeller, "Inverting Gradients -- How easy is it to break privacy in federated learning?", NeurIPS 2020 / arXiv:2003.14053. This motivates a magnitude-invariant gradient objective based on gradient direction/cosine similarity.
- Zhao, Mopuri, and Bilen, "iDLG: Improved Deep Leakage from Gradients", arXiv:2001.02610, 2020. This motivates fixing labels inferred from gradients instead of optimizing dummy labels jointly.

## New generation names

The following names are reserved for Priority 6 and must not rename any existing generation:

- `GEN_COSINE_TV`
- `GEN_IDLG_STYLE`

Existing names such as `GEN_RAW_LIFT`, `GEN_SKETCH_STE`, and `GEN_IHT` remain unchanged.

## Supervisor quick-nod decision (recorded before all Priority 6 gates)

Decision recorded at: 2026-09-16T16:41:21Z.

The supervisor rejected the proposed public-feature-graph smoothness term.
`GEN_COSINE_TV` therefore uses no feature graph, no adjacency set `E`, and no
edge weights `w_jk`.  The frozen objective is range-only:

```text
L_total = L_cos + lambda_range * L_range
lambda_tabular_tv = 0
```

The tabular-TV term is removed from the executable objective, rather than kept
with a small weight.  `L_range` is limited to the public validity/range
constraints already present in the corresponding dataset attacker/evaluator.
This decision was made before creating any Priority 6 target or observing any
Priority 6 gate result.  No further quick nod is required for
`GEN_IDLG_STYLE`.

## GEN_COSINE_TV loss

For candidate update/gradient tensors `g_hat(x)` and observed defended or raw signal `g`, define the matching loss as:

```text
L_cos(g_hat, g) = 1 - <flatten(g_hat), flatten(g)> / (||flatten(g_hat)||_2 ||flatten(g)||_2 + eps)
```

The frozen total objective is:

```text
L_total = L_cos + lambda_range * L_range
```

where `L_range` is the same public-validity/range penalty already used by the
corresponding dataset attacker.  There is no graph-smoothness or tabular-TV
term.

For PaySim, `L_range` is the existing nonnegative amount/balance penalty with
weight `0.001`; the hard balance-difference parameterization and categorical
simplex remain unchanged.  For IEEE-CIS, numeric quantile bounds and
categorical simplexes are enforced by the existing decoder, so the additional
soft range penalty is zero.  These are existing public constraints, not new
data-dependent regularizers.

## GEN_IDLG_STYLE loss

The `GEN_IDLG_STYLE` attacker freezes labels before optimizing inputs:

- When the scenario already supplies labels under the project threat model, use those known labels directly.
- Otherwise, infer labels from the sign/direction of the final classification-layer gradient in the iDLG spirit, then keep them fixed throughout optimization.

The input objective may use the existing squared or balanced update loss unless paired with `GEN_COSINE_TV`; the defining change for this generation is fixed labels rather than jointly optimized soft labels.

No oracle label information may be used in a dataset/scope where labels are not allowed by the threat model.

For the PaySim bounded scope, the already-declared known-label threat model is
retained.  For IEEE-CIS, the attacker uses only the public frozen batch
composition (exactly one fraud record among four) and fixes a canonical label
multiset `[1, 0, 0, 0]`.  Because the single local step consumes the whole
batch and evaluation is permutation-aligned, the canonical row position does
not reveal the true target-row membership.  The observed last-layer update
direction is stored as an iDLG-style diagnostic, but true per-row target labels
are not read to construct the attacker labels.

## Development gate plan

After the quick nod is recorded:

1. Create new target sets for PaySim and IEEE-CIS at the minimal four-record scope, each source-disjoint from all existing development, post-hoc, and confirmatory pools.
2. Run `GEN_COSINE_TV` and `GEN_IDLG_STYLE` against:
   - PaySim, DNA v1 medium configuration.
   - PaySim, DNA v2 ratio `0.95`, eta `0.01`.
   - IEEE-CIS, DNA v1 medium configuration.
   - IEEE-CIS, DNA v2 ratio `0.95`, eta `0.01`.
3. For each cell, evaluate against Prior and Zero-update controls with the same one-sided exact sign-test gate used previously.
4. Record exact p-values, win counts, non-tie counts, and data-firewall overlap evidence in `reports/priority6_sota_style_attackers_report_20260916.md`.

The development gate is frozen at `n=8` groups per dataset, four records and
one fraud record per group.  Each attack uses four restarts.  The existing
dataset-specific development budgets are retained: PaySim uses 600 iterations
and learning rate 0.1; IEEE-CIS uses 300 iterations and learning rate 0.05.
The exact one-sided sign-test alpha is 0.05; a branch passes only when its mean
and median MSE differences are negative and its exact p-value is below 0.05
against both Prior and Zero-update controls.  No hyperparameter selection is
authorized after gate results are observed.

Pre-run seeds are frozen as follows: PaySim target generation `2026091661`,
IEEE-CIS target/preprocessing generation `2026091662`, PaySim attack
initialization/local-update derivation `2026091663`, and IEEE-CIS attack
initialization/local-update derivation `2026091664`.  DNA v1 medium retains its
previously frozen base seed `681958327`; DNA v2 retains base seed `20260916`.

## Escalation rule

If a generation clearly passes the development gate at minimal scope, a separate new amendment must be written before any scope expansion or n=24 pilot. Target/seed pools for the expansion must be new and source-disjoint.

If both new generations fail, the result only strengthens the statement that no effective attacker was found under the tested development conditions. It must not be stated as proof that DNA v1/v2 is private.
