# Priority34B — whole-training client-update DP (pre-run amendment)

Written before tests, target creation or experimental runs. P34A COMPLETE and
its final receipt/manifest are prerequisites; all earlier evidence is immutable.

## Accounting and release boundary

Trusted central aggregator; the released transcript comprises only the 50 noisy
global non-BN model broadcasts. Raw client uploads are not protected from that
trusted server: deployment would require secure aggregation for that threat.
Fixed three client slots, all participate (q=1), replace-one adjacency over one
client's complete update-generating dataset/state, conditional on the public
history. P34A partition weights are fixed public metadata, not re-estimated under
adjacency. No privacy amplification. Evaluation metrics, local BN checkpoints,
validation thresholds, audit journals and reproducibility noise seeds are NOT
DP releases. Publishing those artifacts would invalidate privacy protection;
this research estimates the ideal Gaussian mechanism's formal parameters, not
a claim that its fully disclosed research bundle is a private deployment.

Clip only non-BN trainable *deltas*, never absolute model parameters. Freeze
C=0.01, a public, preselected update-space L2 radius, without data-driven tuning.
Global clipping: each entire delta has norm <= C. Per-tensor clipping: L tensors
each have radius C/sqrt(L), hence the concatenated bound is still C. This second
mechanism has the same total sensitivity, not L unaccounted Gaussian releases.
For fixed weights w_i summing to one, replace-one aggregate sensitivity is
S=2 max(w_i) C. One isotropic Gaussian N(0,(sigma*S)^2 I) is added AFTER averaging
the clipped deltas, then the previous global parameters are added.
No extra noise to BN, no averaging or transmission of BN weights/bias/buffers.

Gaussian RDP per round is alpha/(2 sigma^2). T=50 rounds compose adaptively to
T alpha/(2 sigma^2). Delta=1e-5. The certified standard RDP conversion is
epsilon(alpha)=T alpha/(2 sigma^2)+log(1/delta)/(alpha-1).
Let rho=(sqrt(log(1/delta)+epsilon)-sqrt(log(1/delta)))^2 and
sigma=sqrt(50/(2 rho)); alpha*=1+sqrt(log(1/delta)/rho). Targets 1,3,10 are
upper bounds from this conversion, not claims of tight minimum epsilon.
Independently verify with numerical minimization and report sigma, alpha, S,
noise standard deviation, round count, adjacency and accounting unit.
Finite-precision/PRNG implementation is research simulation of ideal Gaussian
DP; noise keys must remain secret in deployment. C is not utility-optimized;
results do not establish the best achievable DP trade-off with only three users.

## Tabular utility (descriptive)

Same prepared matrices, models, partitions, 50 rounds, Adam(.001), focal(.95,2),
one local epoch, batch1024, CPU/thread1 as P34A. Seeds321000–321010: first eleven
predeclared P34A seeds, no selection by metrics. 3 datasets x 3 epsilons x 2
mechanisms x11 =198 new jobs. Reuse33 validated baseline jobs and their bytes;
also show corresponding eleven-seed v1/v2 metrics, and label P34A full21-seed
results separately. Client BN persists locally, final validation-only thresholds
and common-test per-client scores as P34A; seed score is mean of three clients.
Report F1/ROC-AUC/PR-AUC means, paired mean deltas, SD and two-sided95% Student-t
CI(df10). No noninferiority/equivalence tests or replicate increase.

## CIFAR utility and recovery

P31 LeNetZhu seed42, immutable 12000-train/3000-validation split/normalization,
lr0.1, minibatch256, SGD(no momentum). LeNet has no BN. To use the requested
50-round accounting, introduce three fixed balanced IID clients and50FedAvg
rounds with one local epoch (not P31's100 centralized epochs). Partition/order
seeds321000–321010, eleven paired replicates of baseline and both DP mechanisms
at epsilon1/3/10:77new jobs. All three client sizes4000, weights1/3, same C=.01.
Validation accuracy, mean/delta95%CI and every seed reported; not equated to P31
utility numbers or to the tabular Adam setting.

Recovery uses the frozen corrected P33c C1/D single-image, untrained LeNet42,
known-label, native4800iterations, TV.01, one restart, same input metrics and
payload-only key-known v1 conservative/v2 sketch objectives. Freeze39 test IDs
from RNG seed344000, excluding all historical image targets and the entire
P33c target manifest. Same attack seed344100+i per arm; noise344200+i isolated.
Four arms: v1, v2, epsilon10/global, epsilon10/per_tensor (156 jobs). Include
unprotected39 jobs as a reference/gate, total195 recovery jobs.
For DP recover a conditional noisy aggregate with the other two contributions
fixed/known zero; divide by target weight1/3 so payload=clip(g)+N(0,(2sigma C)^2I).
This is a single-round conditional gradient instrument using the same sensitivity
and noise-to-sensitivity ratio calibrated for50 releases; it is NOT inversion
of a final50-round model, not utility matching, not trained-checkpoint matching.
Known labels, untrained checkpoint and other clients' known contributions are
explicit side-information assumptions; don't generalize to unknown-label users.

Unprotected must beat both gray and CIFAR-mean references by PSNR exact sign
test p<.05 at n39; failed gate -> NOT_ASSESSABLE, retain all scores, no tuning.
DP weaker-than-prior recovery is a descriptive instrument limitation, not
proof of superiority by attack failure. Report each arm relative to references.
Primary PSNR, secondary SSIM/MSE. For four DNA-vs-DP pairs x3metrics x2directions,
24 exact one-sided sign tests, ties removed with effective n reported;
Holm over all24 P34B tests. Show raw/Holm p, medians and paired effects with
order-statistic intervals ranks13/27 of39. No existing-family cherry-picking.

## Execution and gates

Source/input/job manifest frozen before detached execution. CPU only,
torch intra/inter-op1, four worker processes maximum. Per-job success arrays,
checkpoint, journal and result hashes validated before resume skip; preserve
attempts/interruption receipts. Negative BN/nonfinite is recorded failure, never
clamped or counted as privacy evidence. Drain already submitted jobs; no new
stage with failed required jobs and no scientific repair without direction.
progress.json/progress.log after each completed/failed job, checklist current.
One supervisor owns exclusive lock; no duplicate pool. Training then recovery,
independent metrics/accounting/statistics, report and final no-worker audit.
Report reports/priority34b_report.md with hashes, all commands/deviations and
utility/recovery threat-boundary disclosures; py_compile/git diff --check.

References: Mironov, RDP (2017), https://arxiv.org/abs/1702.07476;
McMahan et al., user-level DP-FedAvg (ICLR2018), https://arxiv.org/abs/1710.06963.
