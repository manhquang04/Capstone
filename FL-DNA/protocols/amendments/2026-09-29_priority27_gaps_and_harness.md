# Amendment — Priority 27 gaps and reusable defense-agnostic harness

Date written: 2026-09-29  
Status: FROZEN / AUTHORIZED  
Scope: RQ1 extension analysis and reusable evaluation harness.  This amendment does not modify, replace, or re-score any earlier Priority 24/25/25b/26 result.

## Motivation

The conference paper keeps the original RQ1 wording and currently answers it "no" using the qualified BN-statistics instrument and related post-hoc analyses.  Review feedback identified remaining experimental gaps:

1. no gradient-only attacker qualified in the fraud domain;
2. utility-matched DP used a single full-state clip norm dominated by BN buffers;
3. v1 had not been attacked adaptively;
4. only 4-record, one-step targets were used.

Priority 27 addresses these gaps as an extension analysis and builds a defense-agnostic harness for later use on published defenses.  It does not replace the registered answer already reported for the paper.

## Compute estimate and pre-declared reduction

The unreduced request would require:

- Part H harness implementation and tests;
- Part C1: 2 local-update settings × 3 batch sizes × development and pilot gates with expensive TabLeak-style optimization;
- Part C2: 3 DP variants × 2 transforms × utility grids × 16 paired full RQ2-style training replicates;
- Part C3: confirmatory n=39 for every qualified instrument × every DP variant × every matching arm;
- Part C4: RQ2-style full-update BN-channel descriptive runs;
- Part C5: IEEE-CIS plus K=10-client PaySim repeats.

This exceeds the practical ~4-day compute budget once C4/C5 are added to the calibration and gradient-attack workload.  The reduction is frozen before execution, in the required order:

1. **Drop C4** (realistic full RQ2-style BN channel descriptive analysis).
2. **Drop C5** (second dataset and K=10 clients).
3. **Shrink C2 grids** from 6 initial points to 5 initial points, with one pre-declared 2-point extension.

Executed scope is therefore: Part H, C1, C2, and C3 only.

## Part H — defense-agnostic harness

Create `experiments/harness/` with the following API:

- Defense interface:
  - `encode(update, context) -> transmitted`
  - `server_view(transmitted, server_knowledge) -> view`
  - each defense must declare required server knowledge: keys, seeds, metadata, clip scope, and DP parameters if any.
- Attack interface:
  - `reconstruct(server_view, model, knowledge) -> input estimate`
  - built-in attack types:
    1. `BNClosedFormAttack`;
    2. `TabLeakGradientAttack`;
    3. `ImageCosineTVAttack`;
    4. `AdaptiveAttackHook`.
- Required checks, enforced by `HarnessResultValidator`:
  1. positive-control evidence contains n=8 and n=24 undefended gates beating Prior and decoy;
  2. scoring is input-space: per-feature standardized MSE for tabular, MSE/PSNR/SSIM for images;
  3. server-knowledge declaration includes every secret used by decoding;
  4. DP comparator specification records clip scope, C, sigma, epsilon, delta, releases, and accounting method;
  5. decoy reconstruction is scored against the current target, not the decoy source.

Unit tests must cover every item above.  The report generator must refuse to emit a result when validation fails.

## Part C1 — gradient-only instrument qualification

### C1a protocol setting

- Domain: PaySim tabular.
- Local update: one local epoch of Adam, matching the RQ2 optimizer family.
- Batch sizes: 4, 8, 16 records per target, one fraud record when feasible.
- Attacker: harness `TabLeakGradientAttack`.
  - cosine objective on trainable-parameter update only;
  - softmax relaxation for categorical transaction type;
  - known labels;
  - median-ensemble pooling over restarts;
  - budget frozen here: 8 restarts, 2,000 Adam optimizer steps, attacker learning rate 0.05.
- Gates:
  - development n=8 fresh source-disjoint targets;
  - pilot n=24 fresh source-disjoint targets if development gate qualifies;
  - exact one-sided sign test at alpha 0.05, input-space standardized MSE;
  - must beat both Prior and decoy.

### C1b secondary FedSGD setting

- Same domain, batch sizes, attacker, scoring, and gates as C1a.
- Local update is one SGD gradient step with learning rate 0.01.
- This is explicitly secondary and differs from the RQ2 protocol.

### Adam-sign diagnostic

Before interpreting Adam-gate failures, report analytically that the first Adam step from fresh optimizer state is approximately `lr * g / (|g| + eps)`, i.e. a sign update for coordinates with `|g| >> eps`.  Cite Balles and Hennig, ICML 2018.  On the C1 development captures, also report cosine similarity, sign agreement, and correlation between `|step|` and `|g|`.

Cells that qualify at both n=8 and n=24 are eligible for C3.  Failed cells stop and are reported as not qualified; no unregistered retries are allowed.

## Part C2 — competent DP comparators

Implement and calibrate three DP variants:

1. `full_state_single_clip`: current mechanism; one clip norm over every transmitted floating tensor, including BN buffers.
2. `per_tensor_clip`: one clip norm per floating tensor, each at that tensor's 95th percentile update norm; total sensitivity is accounted for by root-sum-square of per-tensor clips.
3. `fedbn_trainable_only`: BN running statistics are kept local and not transmitted; DP is applied only to trainable parameters.

For each variant and transform (`v1_conservative`, `v2_0p95_eta0p01`), utility-match by the Priority 25b rule:

- measure clip norms on two development training replicates;
- use 16 paired RQ2-style training replicates for the utility grid;
- transform targets:
  - v1 conservative mean ΔF1 target: `+0.0014`, threshold `-0.0036`;
  - v2 mean ΔF1 target: `-0.0117`, threshold `-0.0167`;
- initial sigma grid: `[1e-6, 1e-5, 3e-5, 1e-4, 3e-4]`;
- matching rule: choose the largest sigma whose mean ΔF1 is at least the transform threshold;
- bracketing requirement: selected sigma is not the largest grid point, and the next larger grid point has mean ΔF1 below threshold;
- one extension if needed: `[1e-3, 3e-3]`;
- if still not bracketed, report "not bracketed" and do not use that comparator in C3.

Report ΔF1, ΔAUC, ΔPR-AUC, SDs, clip norms, selected sigma, and RDP epsilon at delta `1e-5` for one release and 50 releases.  Accounting convention is update-level add/remove, matching earlier project reports.

## Part C3 — RQ1-style retest with adaptive attackers

Eligible instruments:

- the BN closed-form instrument (T1), always included;
- any C1 gradient-only cell that qualifies at both development and pilot gates.

Targets:

- fresh source-disjoint confirmatory targets, n=39, for every eligible instrument × DP variant × transform/matching cell.
- target source-ID overlap must be checked in code against every existing target pool, not only trusted from bundle metadata.

Defenses and attackers:

- v2: key-holder least squares through `R_s`; server must hold the key for this branch.
- v1 adaptive Level-1 attacker set:
  1. direct BN/statistic recovery on transmitted statistic;
  2. debias estimator `(T - m * mean(T)) / (1 - m)`;
  3. seed-invariant block-sum estimator;
  4. surrogate-permutation least-squares estimator when the attacked statistic has enough dimension.
  Report the best-performing pre-registered estimator for v1.
- DP: Bayes-sensible unbiased estimator from the noisy statistic, using the known DP noise scale and clipping metadata.

Tests:

- DNA vs DP in both directions using exact one-sided sign tests on input-space reconstruction MSE.
- Tie band is obtained from replay on development targets; if replay is bit-exact, tie band is 0.
- Holm correction across the whole executed C3 family, separately for DNA>DP and DP>DNA.

Pre-registered descriptive labels per instrument × DP variant:

- `transform_better`: DNA>DP significant after Holm and DP>DNA not significant;
- `dp_better`: DP>DNA significant after Holm;
- `no_difference`: neither direction significant after Holm.

Priority 27 is an extension analysis; these labels do not replace the paper's registered RQ1 answer.

## Dropped parts

Part C4 and C5 are not executed in Priority 27 due to the frozen reduction above.  They may be registered later as a separate priority if needed.

## Disclosures and invariants

- No edits under `Latex/`.
- Existing artifacts and reports are not modified.
- Every run, including interrupted, resumed, discarded, or replayed runs, is disclosed.
- No early stopping and no tuning after confirmatory data.
- Parallel workers may be used, but every worker must set `torch.set_num_threads(1)`.
- Per-target CSVs are produced for every executed confirmatory test.

