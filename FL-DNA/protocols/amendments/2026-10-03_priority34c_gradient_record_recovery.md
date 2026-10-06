# Priority34C — gradient-only fraud record recovery

Status: DRAFT / REQUIRES_DIRECTION before implementation freeze or any scientific
run. Written 2026-10-03, after verified P34B completion. No targets have been
drawn and no model, calibration, qualification or confirmatory job has run.

## Prerequisite and immutable evidence

P34B final_verified_receipt.json and administrative_verified_receipt.json must
both pass, with the administrative report digest matching the current report.
Preserve every earlier artifact and frozen source. Never edit Latex/,
external_defenses/ or datasets/. New evidence belongs in artifacts/priority34c
and the final report in reports/priority34c_report.md.

## Required direction: the epsilon10 release boundary

P34B's actual frozen mechanism is trusted-central client-level DP: clip
non-BN client updates at C=.01, weight-average three clients, then add one
Gaussian with sensitivity S=2*max(w)*C. Its epsilon10 sigma/sensitivity is
4.0156365196350094, accounting 50 releases, q=1, delta1e-5, fixed-slot replace-one
client adjacency. Raw uploads received by the trusted server are NOT DP releases.
See experiments/priority34b_core.py:account/aggregate and P34B's amendment.

Thus a raw individual gradient cannot be described as protected by that epsilon.
Choose before execution:

A (recommended): reuse P34B's conditional noisy-aggregate instrument. Fix the
other two contributions known zero, release w_j*clip(g_j)+noise and rescale by
the public target weight w_j. The observed vector has noise SD
sigma*2*max(w)*C/w_j. Use P34B's actual public tabular weights and explicitly
record target client/weight, not silently assume balanced weights. This measures
single-round information conditional on side information, not raw-upload
protection, final-model inversion, or record-level DP. Transferring the gradient
instead of the Adam-update query is a disclosed query variant with the same
clipping bound/accountant, not a replay of a P34B training job.

B: genuinely client-side clipping/noise before transmitting each gradient.
This is a NEW mechanism/release model, requiring its own sensitivity/accounting
amendment; it must not be called the unchanged P34B comparator.

No option has been selected by the agent. No expensive qualification or
confirmatory execution is authorized by this draft until this choice resolves.

## Planned instruments and representation (pre-outcome)

Nine qualification cells: three datasets (PaySim, IEEE-CIS, BAF) crossed with
closed-form batch1, native TabLeak batch1, native TabLeak batch2.

Closed form uses the unchanged P32 FraudMLP and all prepared coordinates
(13/476/58 respectively). Use fixed public untrained initialization and eval
mode: BatchNorm reads fixed nonnegative running statistics and dropout is off.
This is required to make batch1 meaningful; training-mode BatchNorm1d with a
single record is not a valid native P32 forward. BN parameters/statistics stay
local and no BN tensor is uploaded. The attacker is explicitly given the fixed
public architecture/checkpoint, including these initial BN values; no learned
private BN checkpoint is silently disclosed. Capture one FedSGD gradient of
the remaining trainable parameters using P32's binary focal loss, no Adam step.
Freeze exact initialization seed and payload names before execution.

For a weight/bias gradient pair choose the row with greatest absolute observed
bias gradient, ties first row; divide that row by its bias gradient. Zero
denominator is an unresolved observation, not a numerical clamp or reason to
replace the target. Never select a row by reconstruction truth. Validate the
identity on synthetic examples before real targets. V2 known-key reconstruction
must solve the documented sketch least-squares problem, not confuse the server's
scaled transpose lift with a pseudoinverse. Freeze the exact solver and verify
it synthetically before drawing targets. No ratio formula is used for batch2.

TabLeak reuses the official P33B FC(input,[100,100,2])/CrossEntropyLoss,
untrained seed333042, known-label, 1500iteration, lr.06, 30-member ensemble,
median/softmax pooling configuration unchanged except batch1/2. This is a
different architecture/representation from P32's full FraudMLP, as in P33B;
do not describe TabLeak results as recovery of the full P32 IEEE feature vector.
Reuse P33B BAF58 and train-selected IEEE60 complete-block adapters unchanged.
PaySim needs an analogous train-only adapter retaining all13 coordinates.
Continuous/frequency coordinates measure prepared representations, not lost
original categories. Official projection/bounds are frozen evaluator operations,
not numerical-failure repairs. No edits to official code.

Batch1 is the easiest record-recovery setting. All instruments measure a single
FedSGD gradient, not multi-step Adam local training, full FedAvg or BN batch means.

## Gates, calibration, fresh targets and fixed statistics

Each dataset/instrument cell must beat BOTH mean/mode and empirical-marginal
records at n8 and subsequently fresh n24, exact one-sided sign p<.05 with ties
removed. Controls/feature accuracy follow P33B (logical feature equality or
train-standard-deviation tolerance; 30 empirical guesses averaged). Gates are
outside the confirmatory family. A failed gate resolves that cell
NOT_ASSESSABLE; no tuning on confirmatory targets or unplanned gate repetition.

Passing cells: n39 fresh targets paired across v1 conservative, key-known
v2(.95/.01), separate distortion-DP matched to each transform, and the chosen
epsilon10 comparator. No utility-matched claim. V1 evaluator knowledge remains
P33B's plain/structure-debiased payload-only knowledge, not silently key-known;
v2 native TabLeak matches in sketch space with the key. Freeze seeds, target-ID
firewall, model hashes, method-specific payloads and solver rules before runs.
Fresh source-disjoint targets must exclude all historical dataset targets and
remain disjoint across stages/instrument cells; do not substitute IDs after
outcomes. Do not load confirmatory records before qualification and calibration.

Distortion calibration uses only n24 development gradients, whole transmitted
non-BN gradient L2; v2 defender-side decoded reference never enters the attacker.
Reuse P33B's C=1.01*max developmental gradient norm, median distortion matching
and <=5% mismatch gate. Record every calibration point and actual deviations;
no outcome-selected rescaling or confirmatory-based tuning.

Primary feature accuracy, lower means poorer recovery. Fixed Holm family:
3 datasets * 3 instrument/batch cells * 2 transforms * 2 comparator kinds *
2 directions = 72 tests. Distortion comparator is transform-specific; epsilon10
is shared within a cell. Missing/gated contrasts retain p1 reservations and
NOT_ASSESSABLE, never shrink the family. Exact one-sided sign tests in both
directions, wins/losses/ties/effective n, raw/Holm p; median per-arm quality and
median paired DNA-minus-DP effects with ranks13/27 of39. Independently recompute
integer binomial tails and Holm using a separate implementation. Poor recovery
or below-control scores are instrument limitations, not universal privacy proof.

## Execution, audit and remaining freeze work

CPU only; intra-op/inter-op threads1; detached portable subprocess supervisor,
at most4 workers, exclusive ownership, per-job atomic results, stdout/stderr,
commands, payload/model/reconstruction/score hashes. Resume only validated
results with identical config/source/input hashes, preserve interrupted attempts.
progress.json and progress.log after every job; checklist updates only after
gates resolve. Fail closed on nonfinite logits/gradients/objectives/scores and
negative BN running_var; record failures, never clamp, replace or exclude seeds.
No analysis over missing required results; drain already submitted jobs on failure.

Before launch: resolve DP direction; freeze exact seeds, target firewall and
payload/solver contracts in an additive execution annex; implement and test;
hash all original/new sources, inputs, protocol and target-ID manifests.
After jobs: reload/scoring audit, independent statistics, report with commands,
all deviations/knowledge/release boundaries, manifest, py_compile,
git diff --check and no-live-worker audit. No report-complete claim before then.

Primary references:
- Phong et al., IEEE TIFS13(5),1333-1345,2018,
  https://ieeexplore.ieee.org/document/8241854 and
  https://eprint.iacr.org/2017/715.pdf (first-layer gradient disclosure).
- Official TabLeak, https://github.com/eth-sri/tableak.
