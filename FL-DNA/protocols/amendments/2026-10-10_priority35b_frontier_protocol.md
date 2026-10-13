# Priority35B — cross-configuration privacy–utility frontier

Date: 2026-10-10 (Asia/Ho_Chi_Minh). Prospective P35B design, written before
tests, target allocation or measurements. This step authorizes protocol/tests
ONLY. Hash-freeze this document and the passing preflight package; subsequent
execution requires separate user authorization and a source/input/target freeze.
Never revise this document after outcomes; use a dated, explicitly post-hoc
addendum. P24–35 results, reports, gates and Holm families remain unchanged.

## 1. Utility axis: Config-C

BAF, IEEE-CIS, PaySim; immutable P32 prepared train/validation/test arrays and
P34A non-IID partition: K=3, fraud evenly distributed; non-fraud natural
category with 55% assigned to its primary client. P34A FraudMLP, local BN affine
and running buffers persist on clients, never uploaded or aggregated. Transform
only uploaded non-BN trainable deltas, not full state. Full participation,
50 rounds, fresh Adam lr=0.001 each client/round, one local epoch, batch1024,
binary focal(alpha=.95,gamma=2); same initialization/order/partition contract.
Seeds exactly321000–321010 (11 paired replicates). A replicate is the mean of
the three clients' test F1, each with its own local BN and threshold chosen only
on validation. Last round50, no best-round or test-threshold selection.

Arms: baseline, v1 conservative (block256,mix.08,keep.88,shrink.45),
v2(compression.95,quantization.01), and client-side DP global/per_tensor at
epsilon={1,3,10,30,100,1000,10000,1000000}. Use the exact P34A/P34B transform
seed derivation and tensor order, not a newly chosen defense realization.
Reuse P34A baseline/v1/v2 for the eleven seeds and P34B **local-DP extension**
epsilon1/3/10 only after config, sources, partition, receipt, result and
checkpoint/prediction hashes verify. Central P34B is NOT an upload comparator.
Reuse is identity-based, not approximate F1 equality. Missing/drifted evidence
stops preparation; any non-identical baseline needs a disclosed pre-run annex
before execution, not a silent substitution. Expected eligible historical
utility jobs99 DNA/baseline +198 localDP;330 new jobs for five added epsilons.

Report all eleven values, mean paired method-minus-baseline Delta-F1 and
two-sided95% Student-t interval df10; also absolute mean F1 and descriptive
ROC-AUC/PR-AUC. No new NI/equivalence claim from mean matching or t-bars.

## 2. DP release and accounting

Exactly P34B local extension: each client clips its non-BN delta BEFORE
transmission, C=.01; replace-one sensitivity S=2C=.02; q=1,50 sequential
releases per client,delta=1e-5. Joint/global clipping bounds the concatenated
vector by C; per_tensor clips each of L uploaded tensors at C/sqrt(L), keeping
the same joint sensitivity2C and isotropic Gaussian SD on every coordinate.
Do not divide noise by K or L or add a second central noise mechanism.
Retain P34B's inward float32 multiplier(1−4*float32 epsilon); float64 Gaussian
noise then cast to original dtype. L excludes local BN and buffers.

RDP epsilon(alpha)=50*alpha/(2*sigma^2). Let d=log(1/delta),
rho=(epsilon/(sqrt(d+epsilon)+sqrt(d)))^2, sigma=sqrt(50/(2*rho)),
alpha*=1+sqrt(d/rho). Per-upload SD=S*sigma. Reproduce old epsilon1/3/10
SDs exactly: .6930431582631877/.24008845279270588/.08031273039270019.
New high-epsilon inversion uses the SAME continuous-order conversion; numerical
validation allowance is sixteen float64 ulps of the target epsilon, not a
different accountant, changed calibration, or scientific comparison tolerance.

Accounting unit: one client's clipped-update transcript conditional on public
history, NOT record-level DP. Trial arms/replicates are separate hypothetical
deployments; releasing them all composes further. Large epsilon is weak formal
accounting, not strong privacy. This is ideal-Gaussian research simulation,
not certified finite-precision production DP. OS-entropy private noise keys,
domain-separated client/round/arm/target, mode0600; never attacker-visible or
derived from public seeds. Truth/checkpoints/local BN/evaluation/audit materials
are private research evidence outside the accounted release.

## 3. Leakage axis: P34C single-gradient setting, NOT Config-C uploads

Use all and only P34C cells passing its original n8 AND n24 gates: BAF and
PaySim ratio_batch1/tableak_batch1/tableak_batch2; IEEE-CIS
ratio_batch1/tableak_batch1. IEEE-CIS tableak_batch2 remains NOT_ASSESSABLE.
No requalification, retuning, b1 Adam/Config-C or P35A estimator substitution.
Ratio uses fixed public untrained P32 FraudMLP seed340042, evalBN/dropout off,
full prepared coordinates(BAF58/IEEE476/PaySim13), focal loss and first-layer
weight/bias ratio. TabLeak uses official P34C native FC[100,100,2], seed333042,
known labels, CrossEntropy,1500iterations,lr.06,30-member ensemble, original
median/softmax pooling, frozen selected adapters(BAF58/IEEE60/PaySim13).
Ratio b1 only; TabLeak b1/b2. Server with seed access uses frozen v2 LSMR
inverse for ratio and sketch-space objective for TabLeak. V1 plain/debiased
candidates selected by observable objective only, plain wins ties; never truth.
All local BN tensors omitted from uploads; ratio's BN is public initialization,
not trained Config-C client BN.

For each of eight eligible cells reserve39 fresh disjoint row sets, paired
across every epsilon/clipping and baseline/v1/v2. Same set across grid points,
not independent draws per point. Across cells, targets are source-disjoint:
312 batches/390 original records. Dataset-namespaced ORIGINAL source IDs,
not prepared row indices. Firewall inventories every P24–35 target/reservation
manifest, verifies authoritative hashes/provenance, excludes reserved as well
as used records, including all P35A diagnostics. Unknown ID schemas, ambiguous
dataset, insufficient fresh rows, duplicates or hash drift fail closed. No
real target allocation in this protocol/tests step. Before authorized execution
freeze deterministic sampling seeds, all target IDs and original-index mapping;
do not replace targets, seeds, failures or enlarge samples after outcomes.

Arms at every eligible cell: unprotected, v1, v2,16 localDP points; mean/mode
row and original P34C empirical-marginal control (30 guesses averaged per
target). Apply each P34B per-upload SD and clipping rule to this single-gradient
query. This is a conditional query/sensitivity transfer, NOT the actual
Config-C Adam update or final-model inversion. No distortion-matched DP arm.
Expected5928 estimator-arm outputs plus controls; no old recovery target reused.

Per-row-set metrics after frozen batch alignment: TabLeak logical feature
accuracy, exact categorical match and continuous tolerance .319 training SD;
standardized coordinate MSE using the frozen P34C train-only mean/std convention
(zero/near-zero std follows that adapter), bounded score=min(MSE,MSE_mean/mode)
on the SAME row set. Higher accuracy means more leakage; larger bounded MSE
means worse recovery. Never treat rows/features/ensemble starts as replicates.
Median per arm,39-unit order interval ranks13/27 (one-based; at least95%
distribution-free coverage for a continuous population median, ties conservative),
mean/mode and empirical-marginal baselines shown with their actual per-target
scores. Exact zero differences are ties, removed only from sign-test effective n.

## 4. Matched utility and fixed inference

For each dataset, transform and clipping separately, require all eight complete
eleven-seed DP results and the same transform seeds. Let threshold=mean F1
transform−.005. Sort by actual calibrated noise SD descending (largest noise
first). Select the first/largest-noise point with mean F1>=threshold. Its
immediate next NOISIER grid neighbor must have mean F1<threshold. If no point
passes or selected point has no noisier neighbor, NOT_ASSESSABLE: no bracket.
No interpolation, extrapolation, grid extension or noise/clip tuning. Equality
passes exactly; no additional tolerance. Record every point and nonmonotonicity.
Selection uses utility only before protected leakage results are inspected;
mean eligibility is not statistical non-inferiority. Existing utility has
already been observed, so new claims are conditional on this predeclared
matching rule and disclosed historical reuse.

Matched DP vs transform:39 complete paired units for EACH qualified estimator,
not only the plotted one. Report median transform-minus-DP differences,
ranks13/27 intervals,wins/losses/ties/effective n and exact one-sided binomial
sign tests both directions for feature accuracy AND bounded MSE. For accuracy
lower is stronger protection; for bounded MSE higher is stronger protection.
One fixed Holm family144 =3datasets*3instrument/batch slots*2transforms*
2clippings*2metrics*2directions. Gated IEEE b2 or unbracketed contrast retains
p=1 and explicit NOT_ASSESSABLE; no family shrink. Alpha .05 after Holm.
Other grid contrasts, plotting estimator selection and control summaries are
descriptive, with no unplanned tests or significance stars. No pooling across
architectures/batch sizes. Independently recompute exact integer tails and Holm
with a separate implementation before final completion. Missing/nonfinite
required output stops analysis, not an NA privacy-success substitution.

## 5. Figure locked from user specification, before measurements

Save PDF,300dpi PNG and exact plotted/source CSV under
artifacts/priority35b/figure/. Three vertically stacked panels BAF/IEEE-CIS/
PaySim, width8.8cm (IEEE single-column), planned height18cm; font>=7pt,
distinct line styles/markers readable in grayscale, identical axis ranges
across panels. No figure is generated in this preparation step.

X reports utility loss =−mean paired Delta-F1 (baseline minus method):0 is
zero-loss reference, larger loss to right. Show95% paired t df10 bars reflected
with the sign. The requested wording “Delta-F1 ... loss to right” has this
explicit sign convention; do not label loss as method-minus-baseline.
If gains/intervals are negative, extend all x axes left of0 rather than hiding,
clamping or cropping them; otherwise0 is the left reference. Shared ranges
include all points/bars and0. Y=median feature accuracy(0–100%, lower leakage),
ranks13/27 bars. Shared y range0–100%. These are marginal intervals on different
units, not joint frontier confidence regions.

For EACH panel select the qualified estimator with highest median accuracy on
its39 new UNPROTECTED uploads, before reading any transform/protected recovery
point. Tie order ratio_batch1,tableak_batch1,tableak_batch2. Persist a hashed
selection receipt with all unprotected medians; use that one estimator for ALL
panel points/controls, never per-arm switching or historical outcome selection.
Caption names selected estimator and selection on unprotected accuracy; this
is a limited instrument maximum, not a universal best adversary. Different
cells may have different target sets/representations and selection bias; all
estimator tables remain visible, no inferential contrast omitted.

Horizontal dashed mean/mode line and separately styled empirical-marginal
line use median control accuracy on the selected estimator's39 row sets.
DP global and per_tensor are separate lines, markers at all eight epsilons,
labels1/3/10/30/100/1e3/1e4/1e6. Connect in epsilon order only, no smoothing
or inferred interpolation. v1 conservative and v2 .95/.01 are distinct labelled
points with BOTH error bars. Deterministic label offsets/leader lines, enlarge
height if needed for readability without changing width/statistics. CSV holds
all raw11utility seed values, all39leakage/control values, summaries/bounds,
epsilon/SD/clip/estimator, reuse hashes and both configuration identifiers.

Mandatory caption: “The x-axis comes from Config-C local-BN Adam training;
the y-axis comes from the Priority34C single-gradient setting. The two axes
come from different configurations.” Do not describe this figure as direct
privacy of deployable Config-C, a formal DNA guarantee, or equivalence to DP.

## 6. Execution boundaries, integrity and completion

This preparation permits source/protocol inspection, synthetic unit tests and
read-only history/hash audits, not training, gradient capture, recovery,
calibration targets, sampling fresh targets, figure generation or launch.
Only new P35B experiments/tests/artifacts/amendment/report; no edits Latex/,
external_defenses/,datasets/ or earlier artifacts/reports. No commit/push.
CPU only, torch intra/inter-op1 per process, optional max4 detached workers
later with serial-identical results. Nonfinite tensors/objectives/metrics or
negative BN variance fail closed, no clamping, dropped failures or seed changes.
Preserve failed attempts/logs; infrastructure resume only after no-live audit,
unchanged freeze and hash-valid completed skips, with explicit authorization.
Progress/checklist after each stage; no busy polling. No full P35B COMPLETE
until all measurements, independent scoring/accounting/stats/firewall/hash audit,
figure visual QA at8.8cm,report/manifest,compile/diff and worker-exit checks.

Sources: immutable P34A local-BN protocol, P34B local-DP extension and P34C
confirmatory execution/qualified gates; [Gaussian RDP](https://arxiv.org/abs/1702.07476),
[official TabLeak benchmark](https://github.com/eth-sri/tableak). Frozen local
reference code is authoritative for reproduction, not an updated upstream copy.
