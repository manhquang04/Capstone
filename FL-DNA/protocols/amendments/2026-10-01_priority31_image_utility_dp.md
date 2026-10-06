# Priority 31 — image utility-matched DP

Written 2026-10-01T21:40:00+07:00, before any Priority 31 run.
Status: FROZEN / AUTHORIZED by supervisor request.

No Latex/ or external_defenses/ edits; no modification of earlier artifacts.
All outputs under artifacts/priority31_image_utility_dp/. Every process uses
torch.set_num_threads(1), and BLAS thread environment variables=1.
Every run, error, interruption/resume and discarded output is disclosed.

## Utility contract and compute

Official inversefed.construct_model('LeNetZhu',seed=42), no architecture change.
CIFAR normalized with inversefed public CIFAR mean/std. CIFAR train is
stratified into 12,000 utility-development training and 3,000 validation images
(split seeds 51001/51002); remaining train images unused. This is explicitly
a development-scale image utility comparator, not a full CIFAR convergence
benchmark. Official 10,000 CIFAR test images evaluated descriptively after
selection; never used for model, sigma or length selection. No augmentation.

FedSGD-style minibatch updates: batch 256, SGD without momentum, one gradient
per optimizer step, CE loss, shuffle anew per epoch. Model initialization=42
for ALL paired branches; training-order/defense seeds=51016..51031. Thus seed
replicates vary sample order and defense randomness, not initial weights.
Train/validation IDs explicitly saved and disjoint. Reused reconstruction
targets belong to CIFAR test and are excluded from utility model selection.

Validation-only length/lr selection: seeds 51000 and 51003, lr grid
[0.03,0.1,0.3], all trained for 100 epochs, accuracy recorded at epochs
[30,60,100]. Choose highest mean validation accuracy; exact tie -> fewer
epochs, smaller lr. No test evaluation during selection. Freeze choice in
selection.json before grid training. Baseline quality gate on the 16 paired
replicates: mean validation accuracy >=0.40 AND >=0.10+0.25. No retry if fail;
mark utility comparator NOT_ASSESSABLE. Report test descriptively.

Clip convention: clip entire trainable GRADIENT vector to C, add independent
Gaussian N(0,(sigma*C)^2) to each gradient coordinate, then SGD step.
Estimate C as p95 gradient norm of every minibatch in one epoch, under seed42
initial model and two unprotected development training-order seeds
[51000,51003], before utility-grid execution. No calibration on BN buffers
(LeNet-Zhu has no BN). Freeze C in clip.json. Report clipping rates per run.
Accounting Gaussian/RDP update-level add/remove sensitivity ratio1, delta1e-5,
one release; no record-level DP or sampling amplification claim. Privacy
noise for reconstruction is applied to the same gradient coordinate space.

Eight-point sigma grid [0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3].
16 paired replicates baseline/v1 conservative/v2 0.95-0.01/each DP point.
Utility transforms use same frozen P30 adapters, including v2 lift at learning
server, so utility describes the existing transform protocol. No tuning.
For each T choose LARGEST sigma whose mean validation paired delta >=
T's OWN mean paired validation delta -0.005. Require next larger point below
threshold. One predeclared extension [1,3,10] if needed. No eligible point
or still unbracketed -> that cell NOT_ASSESSABLE, no reconstruction run.

Compute estimate (arithmetic corrected before execution): 28,200 minibatch
steps for six selection jobs plus at most 176*4,700 initial-grid utility steps,
and 48*4,700 additional steps if the predeclared extension is required.
8 CPU workers, estimated 6–36 hours depending on throughput; this is an
estimate, not a measured runtime. No outcome-based reduction.
Reconstruction 78*~10s/8 workers, minutes. All stages resumable per completed
file. All predeclared selection lengths/grid points executed; no early stop.

## Reconstruction and tests

Reuse EXACT S1 n39 targets and frozen IMAGE_CONFIG (4800 iterations, signed
Adam lr0.1, cosine, TV0.01, one restart, boxed, decay) from run_audit.py.
DP attack sees noisy clipped gradient ONLY. Attack seeds=30600+target_id as
P30 DP image branches; noise seed=510000+101*target_id+frozen defense offset.
Existing E2 transform results from S1c, paired source-ID equality checked
against S1 and S0u. No transforms or attack budgets changed.

Primary exact sign test on input-pixel MSE, larger=less leakage, both
directions, strict zero ties. Report per-arm median PSNR/SSIM and paired
PSNR differences (defense minus DP). Interval endpoints ranks13/27 of39
(one-based), explicitly requested order-statistic interval; not refitted.
Add four new hypotheses to the P30 repaired 111-test Holm family if both
comparators available (115 total); unavailable tests excluded with reason.
Recompute every old adjusted p as well; do not overwrite old CSVs.
Quality effects also exported for ALL image E2/E3 cells already run, including
superseded outputs separately marked as such. No invalid cell enters Holm.

Do not claim RQ1 superiority from unqualified or unbracketed comparators.
This adds the image utility arm; it does not silently replace earlier RQ1
answers. Report statistical direction, practical baseline context, and
non-equivalence of 'no significant DP advantage' and proof of equality.

Deliverables reports/priority31_image_utility_dp_report.md, per-seed utility
CSV, per-target paired CSV, expanded Holm CSV, all-image SSIM CSV, hashes of
all new artifacts and exact commands. Final py_compile and git diff --check.
