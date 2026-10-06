# Priority33c complete execution annex — pre-run freeze

Saved before additional scientific workloads. CPU, one torch thread, four
concurrent workers. Earlier artifacts, raw datasets, Latex/ and external_defenses/
are immutable. Validated outputs are skipped, never selected by outcome.

## Image configurations and utility

Official P30 LeNetZhu seed42, native Inverting Gradients configuration unchanged:
4800 iterations, TV0.01, signed/boxed, Adam lr0.1, cosine matching, one restart,
randn init, no filter, decay enabled, scoring loss. Known labels.
v1 block256 parameters from experiments/run_attack_sweep.py and P23:
conservative mix.08/keep.88/shrink.45; medium .10/.85/.40; stronger .12/.82/.35.
Image transform seed30001 as P30, not encoder diagnostics disclosed to attacker.
v2 ratio.95/eta.01 key30002, actual sketch transmitted, key-holder sketch-space
cosine loss; never a naive lift or raw gradient passed to reconstruct().
v1 compares plain and blockwise debiased matching `(T-m*mean(T))/(1-m)`;
two official full-budget candidates, same initialization, choose solely by
their attacker-observable loss (plain wins exact ties). Report both objectives.
No ground-truth candidate or restart selection.

Utility uses exact P31 split/model/SGD/order seeds51016..51031, 100epochs,
lr0.1, batch256. Reuse immutable 16 baselines, conservative/v2 and single-clip
DP grid outputs with source hashes. Missing checkpoint replay seed51016 passed
validation/test bit-exact; save and verify its SHA before C2.
Train medium/stronger16 each, per-tensor DP16 per sigma. Grid
[.0001,.0003,.001,.003,.01,.03,.1,.3], conditional extension [1,3,10]
ONLY if initial bracketing fails. No other grid/lr/length changes.
Each tensor C_l is p95 initial-gradient norms on P31 clip probe's identical
94minibatches, seeds51000/51003; never from reconstruction targets.
Per-tensor Gaussian std=sigma*C_l independently, clip each tensor to C_l.
This is layerwise Gaussian, NOT P27's isotropic total-sensitivity variant.
Joint add/remove update-level RDP uses whitened sensitivity sqrt(L), delta1e-5,
one release, where L is number of nonzero clipped tensors. Report all C_l,
sigma, accounting and distinguish weak update-level from record-level DP.
Single clip uses existing P31 C=16.82906017303467. Select largest finite sigma
with mean paired validation delta >= transform's own delta-.005 and next larger
finite sigma below threshold. Baseline mean >=.40; all16 jobs finite required.
No bracket or failed numeric gate => NOT_ASSESSABLE, not privacy evidence.
Test accuracy descriptive only, never selection. Reuse single-clip conservative
and v2 calibration only for same untrained native setting, label transfer to
trained/batch4 sensitivity (not recalibration on confirmatory targets).

## Image sampling, distortion, sensitivity

CIFAR10 test split; exclude all earlier CIFAR indices discoverable from artifact
JSON/CSV and explicit P29/P30 sampling functions including development gates.
Persist exclusion inventory, source hashes and overlap checks. Development n24,
C1 n39 shared across paired arms, D image uses same C1 targets, C2 batch4 n24
and C2 trained batch1 n24 disjoint from development/C1 and each other.
Draw once with NumPy generator seed333600; no redraw based on scores.
C1/D model untrained seed42. C2 checkpoint SHA fixed by passed P31 replay.
For distortion-DP medium/stronger: C=1.01*max development raw-gradient norm;
sigma=median transform L2 distortion divided by C*median unit Gaussian norm,
Gaussian seeds333610+i, same development gradients; assert achieved median L2
within5%. Neither utility nor attack tuning uses these confirmatory images.
Attack init seeds333700+target ID, noise333800+101*target ID+arm offset persisted
per job; fixed same target init across arms. Unknown objectives raise.
Full transmitted payload is the only reconstruct() input. Finite gates before
any metric-domain clipping. Batch4 scoring uses Hungarian MSE alignment after
optimization; average per-record PSNR/SSIM, never alignment for restart selection.
Gray/CIFAR-mean references descriptive. No attack-budget reduction/early stop.
Maximum scheduled image jobs: C1 7*39=273, D 4*39=156,
C2 2settings*5arms*24=240; gated utility arms absent explicitly, not silently.

## D PaySim

Use original P24/P25 data/MLP/local-step BN-mean channel, batch4,
one fraud+three nonfraud per target; first BN running_mean vector only for scoring.
v1 conservative and v2 key-holder least-squares, never ground-truth selection.
Fresh source-disjoint development n8/n24 and confirmatory39; targets cannot
overlap any prior source manifest. Qualification raw beats train Prior and
next-target decoy exact sign p<.05 at n8 and n24; otherwise D NOT_ASSESSABLE.
Utility target must be measured on CPU with full-state transform, not historical
MPS numbers or repaired raw-BN variant. Use original P27 model/data/training
(500000 rows,50rounds,3clients,1epoch,focal alpha.95/gamma2), seeds270201..270216,
baseline/conservative/v2 paired; validation F1 only. Guard every transmitted state,
aggregate and evaluation: nonfinite or negative running_var invalidates that
method's entire utility target, no clamping/exclusion/replacement.
Per-tensor DP uses same original P27 p95 tensor clip norms, tensor-specific
sigma*C_l noise, grid[1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,.001,.003], conditional
extension [.01,.03,.1],16pairedseeds. Largest sigma with paired validationF1
delta >= own transform delta-.005 and next larger valid point below. Run this
grid only for a valid utility target. No valid target => utility comparator NA.
Accounting whitened sqrt(L) as above. New training CPU/thread1, no BN repair.

## Statistics and completion gates

P33c fixed20hypotheses: C1 medium/stronger x (unprotected,distortion,single-utility)
x2directions=12; D image/PaySim x conservative/v2 x2directions=8.
PSNR primary for images, SSIM descriptive; standardized featureMSE primary for
BN batch means. Ties excluded, all ties p1. Reserve NA hypotheses at p1 and
label them NA, never a substantive negative result. Holm over20 and combined135
(unchanged115 P31 expanded_holm.csv plus20). Report raw/adjusted p,wins/loss/ties,
median each arm, paired median difference and one-based ranks13/27 of39.
C2 n24 descriptive only: paired medians/range, no ranks13/27 interval.
Independent SciPy binomtest and separate Holm recomputation must agree.
Report includes all P33a/b/c dataset/comparator results and NA reasons, runtimes,
commands, interruptions, source/raw/checkpoint/output SHA256 manifest,
py_compile/git diff --check and no live workloads before COMPLETE.
