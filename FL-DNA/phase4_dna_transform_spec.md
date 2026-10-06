# Phase 4 DNA Transform Specification

This document locks the DNA Transform behavior before paired Phase 4 DNA
evaluation. It is based on the current implementation in
`dna_encoder/transform_defense.py`, `experiments/run_fraud_fl_dna_transform.py`,
and `attacks/adaptive_dna.py`.

## Scope

The current Phase 4 baseline gate is open only for:

```text
4 records per group / 1 local Adam step / fraud-focused evaluation /
known labels, order, RNG / no class-decomposed oracle update
```

DNA Transform evaluation must stay in this exact scope until a larger-scope
baseline gate passes.

## Transform Configuration

The TransformDefense configuration is:

| Parameter | Current default | Conservative variant |
| --- | ---: | ---: |
| `block_size` | 256 | 256 |
| `mix_ratio` | 0.05 | 0.08 |
| `keep_ratio` | 0.90 | 0.88 |
| `shrink_factor` | 0.50 | 0.45 |

The conservative variant is the first candidate for paired Phase 4 testing
because it was previously used as the utility-friendlier TransformDefense
setting.

## Seed Derivation

The transform requires `DNATransformConfig.seed`; the code raises an error if it
is missing. In FL utility runs, the run seed is generated from secure randomness
unless `DNA_TRANSFORM_RUN_SEED` is explicitly supplied for reproduction.

Client-round seeds are derived as:

```text
derive_seed(DNA_TRANSFORM_RUN_SEED, "dna_transform", round_number, client_id)
```

The seed strategy is reproducible because `DNA_TRANSFORM_RUN_SEED` and derived
client-round seeds are logged in metrics. It is not a cryptographic security
claim.

For the bounded Phase 4 attack scope, the same rule must be reduced to a
target-group realization and written to the paired DNA artifact before the
attack runs. The report must state whether the attacker is:

- seed-unknown: knows algorithm and hyperparameters, not the exact seed;
- seed-known: knows the exact base seed, but not the unknown raw-update-derived
  block realization;
- realization-known: knows the exact permutation and attenuation mask for each
  transformed block.

## Per-Block Operation

For each floating tensor update:

1. Flatten the tensor.
2. Split into blocks of at most `block_size` elements.
3. Convert each float32 block to a binary string, then to a DNA string.
4. Derive `block_seed` from:
   - base transform seed;
   - tensor index;
   - block index;
   - DNA symbol counts;
   - a rolling hash over the first 512 DNA symbols.
5. Use `block_seed` to generate a permutation.
6. Permute the block.
7. Compute the magnitude quantile threshold from the permuted block.
8. Multiply low-energy elements by `shrink_factor`.
9. Mix the transformed block back with the original block:

```text
transformed = (1 - mix_ratio) * original + mix_ratio * attenuated_permuted
```

Non-floating state entries are copied unchanged.

## Lossless vs TransformDefense

Lossless DNA encoding and DNA TransformDefense must be kept separate:

- lossless DNA is a representation/transport baseline;
- DNA TransformDefense intentionally changes the numeric update.

Lossless DNA should not be presented as a gradient-inversion defense if the
decoded update is numerically identical to the raw update.

## Linearity and Inversion

The actual forward transform is input-dependent. The block seed depends on the
raw update values after binary/DNA conversion, and the attenuation mask depends
on the magnitude quantile of the permuted block. Therefore the full mapping is
not a fixed global linear transform from arbitrary input updates to transformed
updates.

If the exact block realization is fixed, the block map becomes linear:

```text
y = ((1 - mix_ratio) I + mix_ratio D P) x
```

where `P` is the permutation matrix and `D` is the diagonal attenuation matrix.

A mechanics check was run on fresh Phase 4 group 0 using the conservative
configuration:

```bash
python -m experiments.analyze_dna_transform_mechanics \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --target-file fresh_final_targets.pt \
  --group-id 0 \
  --mix-ratio 0.08 \
  --keep-ratio 0.88 \
  --shrink-factor 0.45 \
  --block-size 256
```

| Quantity | Value |
| --- | ---: |
| Fixed-realization blocks | 64 |
| Full-rank blocks | 64/64 |
| Max condition number | 1.1905 |
| Median condition number | 1.1905 |
| Mean relative L2 transform delta | 0.1049 |
| Mean cosine similarity | 0.9966 |

This means direct inversion is straightforward if the attacker receives the
exact block realization. Knowing only the algorithm and base seed is not
equivalent to knowing the realization, because the realization is derived from
the unknown raw update block.

A full realization-known direct inversion check was then run on all 12 fresh
Phase 4 target groups:

| Quantity | Value |
| --- | ---: |
| DNA run seed | 1776371525 |
| Floating tensors recovered | 240 |
| Blocks inverted | 768 |
| Full-rank blocks | 768/768 |
| Max condition number | 1.1905 |
| Mean relative L2 recovery error | 3.3163e-08 |
| Max relative L2 recovery error | 8.3623e-08 |
| Max absolute recovery error | 3.8147e-06 |

Under this attacker-favorable realization-known condition, the transformed
update can be inverted back to the raw update up to float32-level numerical
error. This result should not be reported as a seed-unknown attack result.

The Level 1 seed-space audit gives the following constraints:

- the run seed has a 31-bit space when generated normally;
- child seeds are BLAKE2b-derived from the run seed and public parts;
- each block seed also depends on DNA counts and a rolling hash computed from
  the unknown raw block;
- therefore brute-force over a small public seed space is not available in the
  current implementation.

A small surrogate-realization check with eight random candidate realizations per
fresh group gave oracle-selected mean relative L2 recovery error `0.0883` and
mean cosine to the raw update `0.9961`. This is not a Level 1 attack result
because the oracle row selects the best surrogate with raw-update error. The
first inverse-pool selection formula was also deprecated because it compared
candidate updates to `M_r^{-1}` decoded signals while using that decoded signal
as its own reference.

The corrected attacker-visible forward-transmitted rule is:

```text
argmin_r,c L_balanced(M_r delta(c), transmitted_update;
                      reference = transmitted_update)
```

It does not use raw-update error or a self-referential decoded-signal reference
to select `r`, but it still reuses a saved baseline candidate pool rather than
optimizing a fresh DNA-aware attack. With 8 candidates/group on fresh targets,
this rule gave mean relative L2 error `0.0965` and mean cosine `0.9954`. A
candidate-count sensitivity check increased the surrogate pool from 8 to 32.
On fresh groups, oracle mean relative L2 error improved from `0.0883` to
`0.0841`, while forward-transmitted selection changed from `0.0965` to
`0.0945`. This supports treating realization selection as the unresolved
Level 1 problem.

A fresh-optimized Level 1 surrogate-forward runner was then implemented with the
same forward-transmitted objective, hard balance-diff parameterization, and
non-negative penalty. On the four development groups with four surrogate
realizations, two restarts, and 600 iterations, it beat prior in 2/4 groups
and zero-update in 3/4 groups. The development gate remains closed, so this
two-restart runner is not a final Level 1 evaluation path.

A targeted budget diagnostic then kept the same objective, four surrogate
realizations, 600 iterations, learning rate `0.1`, non-negative lambda `0.001`,
and DNA run seed `477885591`, but increased restarts from 2 to 8 per surrogate
realization. On the same four development groups, it beat both prior and
zero-update in 4/4 groups:

| Control | Groups | Wins | Mean delta MSE | Median delta MSE | One-sided p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prior | 4 | 4/4 | -475.1981 | -21.0412 | 0.0625 |
| Zero-update | 4 | 4/4 | -294.5972 | -129.8088 | 0.0625 |

This is a positive budget diagnostic, not a formal statistical gate: with only
four groups, the best possible one-sided sign-test p-value is `0.0625`, above
the locked `p < 0.05` threshold. It shows that the earlier Level 1 run was
under-budgeted.

A larger source-disjoint development gate was then created with 8 groups and
zero overlap with all existing target artifacts. With the same locked
C4/R8/I600 configuration, the Level 1 surrogate-forward attack beat prior in
7/8 groups and zero-update in 7/8 groups; both controls passed with one-sided
sign-test `p = 0.03515625`.

A new source-disjoint fresh final Level 1 set was then created because the older
`fresh_final_targets.pt` had already been used in surrogate diagnostics. On this
new 12-group single-shot set, the locked Level 1 attack beat prior in 10/12
groups (`p = 0.019287109375`) and zero-update in 11/12 groups
(`p = 0.003173828125`). This opens the Level 1 surrogate-forward gate for the
bounded scope only.

A paired raw-update branch was then run on the same fresh targets with the same
hard balance-diff parameterization, non-negative lambda `0.001`, 600 iterations,
learning rate `0.1`, and 8 restarts. The raw branch passed controls with prior
10/12 (`p = 0.019287109375`) and zero-update 12/12
(`p = 0.000244140625`). The paired DNA-minus-raw fraud-MSE comparison found DNA
harder in 6/12 groups, with mean delta `245.7226`, median delta `1.4513`, and
one-sided sign-test `p = 0.61279296875`. This does not support a claim that DNA
Transform conservative adds measurable Level 1 protection over the raw update in
the bounded 4-record/1-step scope.

The largest paired conservative outliers were checked by reloading selected raw
and DNA artifacts. The diagnostic found no missing artifact, replay crash, or
final-step selection error. The selected Adam histories are non-monotone, but
selection uses the stored best objective as intended. The official paired
statistic is therefore kept unchanged.

The second allowed DNA candidate configuration, `stronger`
(`mix_ratio=0.12`, `keep_ratio=0.82`, `shrink_factor=0.35`), was also evaluated
on the same fresh target set and budget. It passed no-update controls, but its
paired DNA-minus-raw result was DNA harder in only 4/12 groups, with mean delta
`342.8169`, median delta `-9.3844`, and one-sided sign-test
`p = 0.927001953125`. This also fails to show added Level 1 protection over raw.

## Adaptive Attack Implications

Phase 4 should evaluate DNA Transform in levels:

| Level | Attacker knowledge | What it tests |
| --- | --- | --- |
| Level 0 | Does not model DNA | Non-adaptive reference only |
| Level 1 | Knows algorithm/config, seed or realization unknown | Whether transform helps against an algorithm-aware but realization-unknown attacker |
| Level 2 | Knows exact realization | Stronger upper-bound setting; direct fixed-realization inversion is the first check |

Level 0 failure is not evidence of protection. Level 2 failure is only
meaningful if the same-scope baseline gate is still open and the attacker has
been checked against paired controls.

## Reporting Constraints

Paired DNA evaluation must report:

- target file and source-row overlap;
- transform config;
- transform seed strategy and exact logged seeds;
- whether attacker seed knowledge is unknown, seed-known, or realization-known;
- baseline and DNA results under the same target groups and initialization
  seeds;
- fraud and non-fraud metrics separately;
- all failed jobs.

Do not claim formal cryptographic security or formal differential privacy from
this transform.
