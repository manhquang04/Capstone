# Priority34D — staged protocol BEFORE scientific runs

Date 2026-10-04, Asia/Ho_Chi_Minh. P34C independently COMPLETE PASS before
P34D. Earlier artifacts, datasets, Latex/ and external_defenses/ are immutable.
New namespace artifacts/priority34d, report reports/priority34d_report.md.
CPU only, torch intra/inter threads1, at most4 concurrent processes. Detached
jobs, per-job receipts/logs, atomic progress.json and append progress.log after
each job. Resume only matching source/config/output hashes after disclosed
infrastructure interruption; no duplicate launch, clamp, seed exclusion or tune.
Nonfinite values or negative BN running_var fail closed; preserve all attempts.

## Execution scope of this initial amendment

ONLY the three Part3 unprotected BAF checkpoint-training jobs are execution
authorized here. No recovery/target drawing, CIFAR utility training, calibration
or confirmatory job may run before an additive execution annex seals its driver,
inputs, source exclusions, targets and complete schedule. Part2 utility-training
batch interpretation is awaiting user direction; it must not be inferred from
the different reconstruction batch size. The sections below register the
intended statistical design, not a claim that those drivers are implemented.

## Part1 — additional image initializations

LeNet-Zhu init seeds342042,342043,342044 (three additional independently
initialized models, not selected using recovery). Official P30/P33C native
4800iterations, one restart, TV.01, signed/boxed Adam.1, cosine, known labels.
v1 conservative .08/.88/.45 block256 with observable plain/debiased candidate
selection; v2 .95/.01 key-known sketch-space objective, never transpose lift.
Fresh source-disjoint n39 paired targets per model; shared within model across
unprotected/v1/v2 and transform-specific single/per-tensor utility-DP arms.
DP is added to the same individual gradient before server observation.
Utility calibration per new initialization: P31 split/SGD.1 momentum0,
100epochs,batch256, paired order seeds51016..51031. Recompute initial clip probes
for each initialization; p95 global/per-tensor norms, same P31 probe seeds.
Initial sigma grid[.0001,.0003,.001,.003,.01,.03,.1,.3], one extension[1,3,10]
only if necessary. Largest finite sigma meeting transform's paired validation
accuracy delta minus.005, with next higher finite point below threshold; baseline
mean>=.40. No bracket/numeric failure=>NOT_ASSESSABLE, not privacy evidence.
These are weak update-level utility-matched DP comparators, not epsilon10
comparators or record-level DP. No central noisy-aggregate substitution.

## Part2 — trained checkpoint plus batch4 recovery

Use immutable P31 baseline seed51016 replay checkpoint from P33C:
artifacts/priority33c/checkpoint_replay/final_state.pt, SHA256
6d390aaf2bbb52aab0bc5991a5c682bf1cd88cf3d170957edfe8fbbd834d36c5.
Fresh source-disjoint n39 batches of4; Hungarian matching only after observable
candidate selection; per-batch mean PSNR/SSIM is the independent paired unit.
Unprotected/v1/v2/transform-specific single/per-tensor utility-DP arms. All
utility targets, clips and sigma brackets must be NEW for the trained starting
checkpoint; no P33C sigma transfer. Training minibatch256 versus4 is unresolved
and must be sealed by direction annex before this part runs. Keep100epochs,
SGD.1/momentum0,16paired order seeds, frozen grid/bracket rule; disclose warm-start
training as additional training, not reconstruction-time optimization.

## Part3 — three additional independently trained BAF checkpoints

Seeds342000,342001,342002; unprotected baseline only, all retained irrespective
of utility/recovery scores. Same P32 prepared data/preprocessing/FraudMLP,
3 seed-specific client partitions, Adam.001, focalalpha.95/gamma2,50rounds,
epoch1 per round,batch1024, final round checkpoint. Source/P32 training settings
remain immutable. No new dataset preparation, model or training-budget change.
Use P33A strict guards on every actual forward/local/upload/aggregate/final
state; validate finite logits/probabilities and nonnegative BN; save all50 round
BN receipts and final_state.pt. Evaluate validation/test descriptively; no
quality-based replacement, no requirement to reproduce seed321000 metrics.
This stage collects3 checkpoints, not a completed P34D privacy result.

Later recovery annex must reuse the unchanged P33A four-record BN batch-mean
method, v1 conservative and key-known v2 least squares THROUGH sketch.
Qualification on fresh n8 then n24 per checkpoint: beat both train Prior and
cyclic decoy using exact one-sided sign p<.05. A failed checkpoint is retained
and dependent tests NOT_ASSESSABLE, never replaced. Fresh n24 development and
n39 confirmatory groups source-disjoint across checkpoints/stages/history.
Distortion matching per checkpoint/method uses P33A C1.01maxBNnorm and median
distortion/(C*median Gaussian norm), with no utility-DP arm in this part.
Offline transformed BN-mean vector is not loaded as a model running_var.
Score per-feature train-standardized batch-mean MSE; not individual records.

## Statistics registered before target access

Images PSNR primary, SSIM/MSE descriptive; BN standardized MSE primary.
Two exact sign directions, ties excluded/all-ties p1. P34D fixed76 tests:
Part1 per-model36 plus pooled12; Part2 twelve; Part3 per-checkpoint12 plus
pooled4. Absent qualified/matched cells reserve p1; never shrink family.
Combined211 tests: unchanged135 raw tests from P33C combined135_holm.csv plus
all76 P34D tests. Do not add P34C or local-DP families to this requested135 base.
Every n39 arm/effect median with one-based order ranks13/27. Pooled n117
requires all3 complete eligible model cells, report exact binomial order interval
for median (not reuse ranks13/27); otherwise pooled NOT_ASSESSABLE.
Heterogeneity descriptive: model-specific effects/range and a labeled exploratory
distribution comparison, never a new confirmatory superiority claim.
Independent SciPy sign/separate Holm and reload metrics must agree. Full source,
input/checkpoint/target/per-job hashes, paired CSVs, interruption/failure/gate
disclosures, py_compile/git diff --check/no-live workers before COMPLETE/report.
