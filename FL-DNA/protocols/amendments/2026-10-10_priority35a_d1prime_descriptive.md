# Priority35A D1-prime: prospective execution plan for a post-hoc descriptive diagnostic

Date: 2026-10-10. Post hoc to completed Priority35A, but written and frozen before any D1-prime measurement. Original qualification 0/72, all six D1 failures, interpretation, Holm6048, original COMPLETE receipt, manifests and report prefix remain unchanged. Only a new report section will be appended after independent verification; the old report digest remains valid for the preserved byte prefix. No confirmatory or transformed/DP measurement is authorized here.

## Scope and knowledge

Six existing D1 cells: BAF, IEEE-CIS and PaySim; round10, b4, labels given; TabLeak-style and Wan-style estimators. Reuse exactly their eight n8 targets, shared parameters, true local BN, observed plain non-BN gradient, metric, mean/mode and empirical-marginal controls. Decoy remains the next target's reconstruction under the same condition and cell, cyclic over eight. No new rows, seeds for existing starts, calibration or gates. All original firewall and finite/negative-BN checks remain mandatory.

D1-prime-a supplies actual per-layer dropout masks (including their inverted-dropout scale), fixed at the actual client's forward, as well as true local BN affine/statistics. BN remains train mode; affine is fixed in this privileged diagnostic. Estimator receives no true feature rows, source IDs or private RNG state. Masks are replayed as exact float32 elementwise multipliers, not regenerated with an estimator RNG. Capture from native dropout's saved backward multiplier; verify bit-exact native logits and gradients on evaluator-only true rows. If exact capture/replay fails, stop, preserve logs, do not substitute masks.

Direct user approval additionally permits ordered given labels corresponding to the mask row positions. Original D1 had an unordered label multiset; this extra privileged information is disclosed. Existing counts are homogeneous for all BAF/PaySim targets and six IEEE-CIS targets; two IEEE-CIS targets are mixed. No reconstruction outcome has been generated for this diagnostic. In mixed-label batches an improvement identifies the joint masks-and-label-position condition, not dropout alone. Original D1 results will never be relabelled or rerun.

## Fixed budgets and ensemble

The official TabLeak experiment46 and example specify post_selection=30, max_iterations=1500 and median+softmax pooling (local immutable reference commit 6b8c1c82ffe85bc6fdd96938ac42b6c9b5685300; https://github.com/eth-sri/tableak). Only this budget/ensemble is adapted: objectives remain existing TabLeak-style cosine+.1 normalized L2 or Wan-style cosine, raw normal initialization, Adam .03, cosine annealing to zero, native categorical softmax and final hard projection, no new regularizer, tuning or truth-based selection.

- D1-prime-a: two starts x4000 iterations, choose minimum observable final objective, tie by smallest member index. Same public_seed(ordinal,4,given,instrument,start,index) as D1.
- D1-prime-b: 30 members x1500 iterations, same seed rule (first two identical starts); total45000 versus8000 steps, 5.625x. Pool continuous candidates and once-softmaxed categorical candidates using torch's coordinate lower median, then hard-project categories. Masks and ordered labels index physical rows, so ensembles are already row-aligned; do not perform a new truth-based or feature-based permutation that would break fixed mask associations. Save all30 raw candidates/objectives/traces and pooled soft result. This is a position-aware adaptation of reference median+softmax, not a full reproduction of its hyperparameters or a pure iterations-only intervention; the ensemble aggregation also changes.
- Optional D1-prime-c is NOT_RUN in this execution, to keep the authorized diagnostic bounded; no outcome-dependent decision to launch it.

96 reconstruction jobs (6cells x8batches x2conditions); 24 distinct masks/gradient captures shared between estimators/conditions. Serial per-job computation, optional four-process spawn pool across jobs; CPU only, torch thread1 and BLAS thread1 in each process. Explicit per-member RNG and a synthetic serial-vs-spawn exact-result test before freeze. No measurement exclusions/replacements, no early stopping, no scientific replay on failure. Detached supervisor with exclusive lock, attempt-preserving receipts and hash-validated infrastructure-only resume after no-live audit.

## Descriptive analysis and locked interpretation

Unit n=8 original paired batches per cell/condition, not rows, features, starts or ensemble members. Report median feature accuracy, paired estimator-minus-reference median accuracy differences, W/L/T with exact equality ties, exact one-sided sign p against each of mean/mode, empirical marginal and cyclic same-condition decoy. Descriptive 'beats all' requires all three p<.05, unchanged from D1; ties excluded only from binomial trial count. No qualification update, n24, family shrink/expansion or Holm6048 change. These post-hoc unadjusted classifications are outside the original family and exploratory, not confirmatory claims.

- If a beats all three controls: estimators work in the privileged known-randomness condition; original D1 failure is consistent with unknown dropout as an obstacle. For the two mixed-label IEEE-CIS batches, ordered-label knowledge is an additional confound, so do not attribute the improvement solely to dropout.
- If a fails but b passes: the larger budget/ensemble was limiting under the known-mask/ordered-label condition; not proof of a universal causal budget effect, because ensemble aggregation changes and these are small selected samples.
- If both fail: estimators remain inadequate for these models under this fixed evaluation; state that plainly. Original negative qualification still says nothing about privacy.

Publish every cell/condition, failures if any and all per-target metrics; no selective claims. Compare with existing D1 descriptively, never revise its interpretation. Independent native mask recapture, gradient/objective/budget/knowledge/tensor validation; independent TabLeak/Hungarian metric recomputation and SciPy exact tests from CSV; hashes and original report-prefix check before diagnostic COMPLETE. Append-only new section and separate diagnostic manifest/COMPLETE, preserving all earlier immutable manifests and report text.

## Commands and tracking

`.venv-phase1/bin/python -B experiments/priority35a_d1prime.py --prepare`, `--launch` (detached), and `--finish` only after supervisor/workers exit. Resume is `--launch --resume` only for verified infrastructure interruption, never a scientific error. Progress and checklist gain diagnostic-specific fields while original Priority35A COMPLETE evidence remains intact. Synthetic tests, py_compile and git diff --check precede freeze. No commit, no edits outside allowed Priority35A paths.
