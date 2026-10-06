# Priority33b: native TabLeak individual-record recovery

FROZEN before scientific runs. P33a COMPLETE.json and its report must verify.
This priority is separate from BN batch-mean recovery. No earlier artifacts,
datasets, Latex/ or external_defenses/ may be edited. New outputs only under
artifacts/priority33b; report reports/priority33b_report.md. CPU only, torch
threads1, maximum4 worker processes, detached supervisor, per-job atomic JSON,
validated resume skipping completed jobs. Record all errors, never replace seeds,
clamp nonfinite values, tune after gates, or shrink attack budgets.

## Checklist and sequence

0. Protocol, P33a completion and source audit.
1. P32 representation adapters, train-only IEEE feature selection, synthetic
   payload/identity/scoring/finite tests; freeze source and inputs before runs.
2. Unprotected qualification n8 then fresh n24, independently per dataset.
3. Development distortion and validation-F1 utility calibration for passing
   datasets, freeze comparators before confirmatory access.
4. Paired n39 record recovery for all assessable arms.
5. Independent statistics, report, hashes, py_compile, git diff --check.
Track per-job progress.json/progress.log/checklist.json and commands/runs.jsonl.
No stage is complete before its gates pass; gated branches explicitly resolved
NOT_ASSESSABLE. Qualifying one dataset does not require the other to pass.

## Representation, model, seeds, data firewall

Use unchanged P32 prepared train/validation/test arrays and preprocessing.json;
verify hashes against audit.json. Group complete one-hot blocks into logical
categorical features. Numeric, missing-indicator and frequency-encoded coordinates
retain their P32 representation; frequency coordinates are scored as continuous
surrogates, not recovery of their lost original categories. No new imputation or
sentinel repair. Missing indicators remain numeric as P32 classified them.
For TabLeak, train-only additional mean/std standardization and continuous bounds;
zero/near-zero std<1e-12 becomes1. No test/validation statistics for these priors.
Decode continuous values without integer rounding: P32 robust-scaled continuous
coordinates are fractional, unlike Adult's integer fields. Categorical decoding
and Hungarian record alignment use official utilities. Tolerance0.319*train std,
exact categorical equality; accuracy equally weighted over LOGICAL features.

BAF uses all58 encoded coordinates. IEEE selects complete logical feature blocks
with at most60 encoded coordinates, never partial categories: rank by maximum
absolute training-only Pearson correlation with isFraud over a block; constant
columns score0, ties original feature order. Greedy include fitting blocks then
restore original order. Save every score, selected/excluded list and SHA before
target generation. This describes recovery only of the selected IEEE features.

Use official TabLeak FullyConnected(input_dim,[100,100,2]), CrossEntropyLoss,
fixed untrained initialization seed333042, exactly P29/P30 model family. No
architecture/checkpoint selection based on reconstruction. Threat: a single
FedSGD batch8 gradient, known labels/model and P32 training priors. This differs
from P32 multi-step local Adam FL utility training, and measures individual
records up to permutation rather than BN means.

Use P32 TRAIN records as simulated client targets, matching P29 Adult's train
target convention. Fresh source-disjoint groups8, no label stratification or
replacement. Target shuffle seeds333320 IEEE /333321 BAF. Reserve n8/n24/n39
disjoint groups, source-ID assertions per stage and against P33a and historical
IEEE manifests/bundles (TransactionID domain), P33b never uses earlier targets.
Attack seeds333400+dataset_index*1000+stage_offset+target_id, offsets0/100/200;
paired arms use identical attack seeds. Empirical-control seed333500 plus same
index offsets. Defense/noise seeds derive333600 in distinct dataset/method/target
namespaces. Log actual seeds and input hashes. Do not inspect confirmatory records
or compute their gradients before qualification/calibration have resolved.

## Official native attack budget, controls, server knowledge

Reuse P29 official experiment46: cosine_sim, uniform init, lr.06,1500iterations,
naive optimization,30-member ensemble, median+softmax pooling, sign trick,
softmax categorical and sigmoid continuous bounds, no priors, labels known,
perfect_pooling=False. Return all candidates/losses for audit; no early stopping.
Use existing P29 nonleaf sigmoid compatibility wrapper, never edit official code.
Ground truth only used AFTER reconstruction for scoring; official true_data
argument receives an empty shape-only tensor, never truth.
Fail closed at model logits, inversion objective, returned candidates/ensemble/
losses and scoring. Official categorical projection/continuous bound operations
are part of the frozen attack, not a repair of nonfinite numerical outputs.

Mean/mode baseline: continuous training means, categorical train modes, repeated
8 times. Empirical-marginal baseline: sample each logical feature independently
from train,30 independent batches; primary control accuracy is their mean,
matching P29; also disclose the first single-guess score and dispersion.
At n8 then n24, unprotected accuracy must beat BOTH controls by exact one-sided
sign test p<.05 (ties excluded, no epsilon band). Failure marks that dataset
NOT_ASSESSABLE, skips later stages, never tune on confirmatory.

v1 conservative block256/mix.08/keep.88/shrink.45 key681958327, server lacks key.
Reuse P30 plain and structure-debiased matching `(T-.08*block_mean(T))/.92`,
two fixed30-member runs from identical seeds. Select only by the minimum
attacker-observable official objective, never ground truth. Save both scores.
v2 ratio.95/eta.01 key20260916: known-key SRHT sketch-space cosine objective
inside the official TabLeak loop, using P30 differentiable projection plan.
No naive transpose lift and no undefended gradient passed to evaluator.
DP uses the same unprotected native cosine objective on ONLY noised gradient.
Synthetic tests assert actual invert_grad true_grad is the transmitted payload;
unknown adaptive modes raise, identity losses match undefended losses.

## Distortion and utility comparator calibration

Separate distortion/utility DP arms for EACH transform. Distortion measured on
the whole gradient, v2 server-decoded gradient used ONLY for defender calibration,
not adaptive evaluator input. On n24 development targets: C1.01*max gradient norm;
calibrate sigma from median defense L2 distortion/(C*median seeded standard
Gaussian norm) over the same24 gradients; save target-level deviations, require
median relative distortion error<=5%, otherwise NOT_ASSESSABLE, no post-hoc grid.
DP clip+Gaussian independently seeded per target, unchanged known-label attack.
Account RDP one vector release, delta1e-5, add/remove update-level, not record DP.

There is no P32 DP utility calibration. Its BN-invalid original transform utility
cannot be reused, and its raw-BN repaired variant cannot silently substitute.
For qualified datasets, calibrate the SAME selected-representation FC model
family (no BN), P32 non-IID3clients/prepared splits,50rounds,local Adam epoch1,
lr.001,batch1024, CrossEntropyLoss (native two-logit model). This is a disclosed
representation/model variant, not a replacement RQ2 test. Apply full-state
transforms before FedAvg using original modules and fixed transform configs.
16paired development seeds333100..333115; baseline/v1/v2 required finite.
Validation threshold maximizes F1 as P32, never select on test. Raw update C is
1.01*p95 norms from first2 baseline jobs.8sigma grid
[1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,.001,.003]; once-only extension[.01,.03].
Largest VALID sigma whose mean paired validation F1 delta>=transform delta-.005,
with next larger VALID point below tolerance; otherwise NOT_BRACKETED and
utility arm NOT_ASSESSABLE. Missing/nonfinite required replicates=>NA, no
exclude/clamp/substitute. Save selected C/sigma/one-release epsilon and all jobs.
Utility sigma is descriptive multi-step-Adam calibration applied to single
gradient; disclose update/model/settings mismatch, not claim record-level DP.

## Confirmatory analysis and completion

Passing datasets: same39 fresh batch8 targets paired across v1/v2, their distinct
distortion-DP and utility-DP arms. Each attack saves result.json immediately,
receipt, model, record IDs, reconstruction/ensemble/objectives/config hashes.
Fixed16-test Holm family=2datasets*2transforms*2comparators*2directions. Lower
mixed-feature accuracy means stronger protection. Report accuracy medians in
percent, paired DNA-minus-DP median and order-statistic interval ranks13/27of39,
wins/losses/ties/raw and Holm p. Gated absent contrasts reserve p1 and statusNA;
do not shrink family. Gate tests outside primary family. Report references
descriptively; low accuracy/failed instrument is not proof of privacy. Independent
integer binomial tails and NumPy/SciPy Holm recomputation mandatory. Final report,
SHA256 manifest, py_compile and git diff --check before goal completion.
