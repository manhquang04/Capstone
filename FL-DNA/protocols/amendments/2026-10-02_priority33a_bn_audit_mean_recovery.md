# Priority33a: historical DP BN validity and two-dataset BN mean recovery

FROZEN before any scientific run. P32 repaired report and P32b report exist and
are complete. Only latest requested A1/A2 are in scope, not the rest of P33.
Earlier artifacts, datasets, Latex and external_defenses remain immutable.
New namespace artifacts/priority33a; report reports/priority33a_report.md.
Every job: detached/resumable result JSON, progress JSON/log, explicit command,
SHA256, torch threads1, maximum4 parallel job processes. No installations.

## A1 fixed historical audit jobs

Replay original entry points/settings/native MPS, without changing RNG or tensor
contents; this explicit historical-replay exception is NOT a new CPU training
experiment. Read-only hooks record every client and aggregate running_var per
layer/round, logits/probability finite flags, final state and actual final inputs.
Negative variance is recorded without changing the historical loop; all50 rounds
continue unless the original code errors. Audit returncodes alone are insufficient.

P25: run_fraud_fl_dp.py, C259.0841131896973, sigmas1e-5 and3e-5,
seeds2501101/2501102/2501103, original utility_grid_20260929 metrics.
P25b: same entry/C/sigmas, first3 NEW manifest seeds2501201/2501202/2501203,
utility_grid_initial_20260929. This gives distinct3job samples, not duplicate P25
jobs relabeled P25b; disclose seed rule explicitly.
P27: run_fraud_fl_dp_priority27.py, seeds270201/270202/270203:
full_state_single_clip sigma3e-5 C283.364730834961;
per_tensor_clip sigma3e-5 original clip dictionary from frozen config;
fedbn_trainable_only sigma1e-6 original clip1.3000768184661866, as a selected
but NOT_BRACKETED control. This variant does NOT noise BN buffers; disclose it.
All selected methods frozen from final stored summary JSONs;21 audit jobs total.
Common original settings: cap500k per-seed before split, K3,50rounds,epoch1,
Adam.001,batch1024,focalalpha.95gamma2, native device,4persistent loaderworkers.
Compare13stored core metrics in every50round (tolerance1e-8, counts exact),
including null patterns. Replay gate failures are reported, never tuned/retried.
Evaluate each unchanged final checkpoint on its actual heldout test inputs on
CPU AND MPS, eval mode,4096blocks, no buffer/parameter corrections. Record finite
counts and BN-invalid channel paths; read-only constructor RNG restoration.
Numbers are numerically ordinary-BN valid only if all relevant model BN states
are nonnegative and recorded outputs/replay metrics/CPU final outputs pass.
Distinguish negative intermediate from negative final states; do not extrapolate
three audited seeds to every unobserved replicate. Genuine reproduced MPS scores
may still be invalid-domain/backend-specific evidence, as P32b established.

## A2 model/checkpoint/data/targets

Unchanged P32 prepared IEEE-CIS476feature and BAF58feature data; same FraudMLP,
loss/partition/Adam50round .001 setup. No saved P32 checkpoints exist. Therefore
replay baseline seed321000 on CPU per dataset into NEW namespace, save final
checkpoint, compare stored P32 baseline metrics tolerance1e-8/counts exact.
Freeze this checkpoint before qualification; do not select it by attack outcomes.
Every NEW training job checks local/global BN variance and actual logits after
every forward/state release; negative variance/nonfinite output raises a recorded
GATE_FAILED, never clamp/exclude/retry the seed. No silent raw-BN repair.

One local Adam step (.001/focal.95gamma2) on4records, known model and pre-update
BN state. Capture first-layer running_mean delta; P24 recover_mean unchanged
defines the plain attack. Prior/population std from P32 TRAIN arrays, std<1e-12
set1 as original method. Decoy = next target's recovered mean, cyclic shift.
Use heldout TEST source records for simulated client batches, seed330320 for
IEEE and330321 forBAF, independently shuffled. Disjoint groups of4, no replacement:
n8 qualification then fresh n24 qualification then fresh n39 confirmatory.
All defenses/comparators share these39targets for pairing. Prior/control targets
never reused as confirmatory; code asserts dataset-qualified source IDs disjoint
across these stages and all earlier reconstruction-target provenance for the
same dataset. P32 classification test evaluation is not reconstruction targeting.
Qualification is exact one-sided sign test recovered MSE<prior/decoy at BOTH
n8/n24, p<.05 each; ties excluded, tie band0. Failure=>dataset NOT_ASSESSABLE;
no tuning or confirmatory access. IEEE 128x476 system is underdetermined; no
feature reduction or architecture change to force qualification.

v1 conservative block256/mix.08/keep.88/shrink.45 key681958327; plain transformed
BN-mean channel recovery (same P24 methodology), with P27 debias descriptive.
v2 ratio.95/eta.01 key20260916, key-holder least squares in SKETCH SPACE:
solve Rs(momentum*W)x = q - Rs(momentum*(bias-pre_running_mean)); never transpose
lift. Build Rs from original transform metadata/key, tensor index equal original
state_dict index for BN_KEY. Identity/lossless algebra and server-payload-only
receipt unit gates before defense use. Score standardized per-feature batch-mean
MSE, not individual-record recovery. Offline channel updates are not loaded into
the model; no running_var state is transformed as part of this BN-mean measurement.

Distortion DP: clip BN vector at1.01*max norm on n24 DEVELOPMENT captures,
sigma = median defended-vs-raw L2 / (C*median deterministic Gaussian-vector L2),
same P24 calibration, fixed noise seeds derive330330/dataset/method/target.
Freeze calibration before confirmatory; RDP one release delta1e-5, report C/sigma.

Utility DP: no P32 utility grid exists (its DP cost arm is not calibration).
New CPU full-state update calibration follows P32 validation F1 and P25b bracket
rule, NOT the raw-BN repaired variant:16paired dev seeds330100..330115 per dataset,
methods baseline/v1/v2;50rounds same P32 hyperparameters. C1.01*p95 raw full-state
client update norms from first2dev baseline seeds, fixed before DP grid.
8sigmas[1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,.001,.003], once-only extension[.01,.03]
if needed. Largest VALID sigma with mean validation-F1 delta >= transform paired
delta-.005, next larger VALID point must be below tolerance. Test metrics never
used for selection. Any missing/BN-invalid/nonfinite required transform replicate
=>that utility arm NOT_ASSESSABLE; do not use an invalid DP grid point as a quality
bracket. If a dataset fails reconstruction qualification, no expensive utility
grid for that dataset. Strict gates may prevent utility matching; report this,
do not substitute a different BN scope or call failure privacy protection.

## Statistics/answer rule/report

Fixed A2 Holm family16tests =2datasets*2defenses*2comparators*2directions;
gated absent cells reserve p1 and statusNA, so family does not shrink post hoc.
Exact sign tests higher MSE means stronger protection; ties removed without
epsilon bands. Report each-arm median MSE and paired DNA-minus-DP median with
order-statistic interval ranks13/27of39, sign wins/losses/ties/raw/Holm p.
Gates outside primary family; no combined115family requested in this narrowed
P33a task. No assessability=>no privacy superiority verdict. Independent sign/
Holm recomputation, commands/hashes/checks/report mandatory before completion.
Progress checklist: protocol+sourceaudit;A1replays+backend;A2baselinecheckpoint;
A2gates;adapters/calibration;confirmatory;independentstatistics/report/checks.
Initial estimates A1~30–90min;A2checkpoint/gates~10–30min; utility if qualified
and BN-valid up to severalhours. No silent reduction; gates dictate skip states.
