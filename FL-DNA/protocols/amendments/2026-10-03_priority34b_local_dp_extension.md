# Priority34B extension — client-side Gaussian update DP

FROZEN DESIGN before tests or scientific execution, 2026-10-03. Authorized by
the user's explicit option2 decision. Separate namespace:
artifacts/priority34b_local_dp; separate report:
reports/priority34b_local_dp_extension_report.md. Preserve the complete central
P34B results, report and manifests byte-identically. Their required label is
**central DP; requires secure aggregation to protect against the server**.
The companion central-scope addendum is the label/interpretation correction;
no central recovery measurement is a comparator for P34C client recovery.

## Threat and accounting unit

Honest-but-curious server observes each client's individual transmitted update.
Each client clips its own non-BN trainable DELTA and adds its own independent
Gaussian BEFORE transmission in every round. Server merely weight-averages the
already noised uploads; it adds no central noise. BN affine parameters and
running buffers stay local and are neither noised, transmitted nor aggregated.

Per-client update-level fixed-slot replace-one adjacency: conditioned on each
public training history, any two possible clipped contributions of that client
have L2 distance at most 2C. Accounting unit is ONE client's contribution
transcript across all50 rounds, not record-level DP and not an epsilon for
publishing all clients/arms/replicates together. Public fixed partition weights,
participation q1, no subsampling amplification. Whole-training delta1e-5.
RDP at order alpha: 50*alpha/(2*sigma^2), noise SD = sigma*(2C) on EACH client
upload. No factor1/3 or max(weight) in local noise scale. Targets epsilon1/3/10
use sigma34.652157913159385/12.004422639635294/4.0156365196350094 from independent
inversion of the standard Gaussian RDP conversion. Releasing a noisy averaged
model afterward is postprocessing of the local release transcript.

Single clip: concatenated non-BN vector norm <= C=.01. Per-tensor clip: L tensors
each <= C/sqrt(L), joint norm <= C, joint replace-one sensitivity2C; one joint
isotropic Gaussian draw represented as per-tensor blocks, not unaccounted L
mechanisms. Keep P34B's disclosed inward float32 clipping rounding unchanged.
Float64 noise/aggregation then model float32, ideal-Gaussian research simulation
not certified finite-precision production DP. OS-entropy private master key
per job, domain-separated by round AND client; never derive privacy noise from
public training seeds, log realized keys or expose them to reconstruction.
Research checkpoints/local BN/validation/test scores and audit bundle are private
evaluation material, not covered DP releases. Publishing them breaks the release
boundary. No private empirical calibration of C or utility optimization.

## Utility execution

Same P34B datasets including CIFAR-10; seeds321000–321010,11paired replicates,
epsilon1/3/10 x global/per_tensor. Tabular198 new jobs: PaySim/IEEE-CIS/BAF with
P34A prepared data, nonIID K3 partition, FraudMLP, local-BN,50 rounds, fresh
Adam.001 each round, focal(.95,2), epoch1/batch1024. Reuse33 P34A baseline
results and corresponding v1/v2 n11 context unchanged, without rerunning.
CIFAR66 new DP jobs: exact P34B LeNet42/12000train3000validation, balanced K3,
50FedAvg rounds, local SGD.1/epoch1/batch256; reuse11 P34B baseline jobs. No BN
exists in that model. Total264 NEW training jobs,44 reused baselines. The CIFAR
extension has no image recovery: prior central recovery remains historical only.

Each tabular client evaluates common validation/test with its own BN and
validation-only F1 threshold. Replicate is mean of three clients. Report all
per-client/endpoints and paired F1/ROC-AUC/PR-AUC delta95% Student-t CIs(df10).
CIFAR validation accuracy paired CIs. Descriptive, no new NI/equivalence tests.
Report local vs central DP explicitly, include prior DNA utility context, never
rename a central result as local or conflate different threat models.

## Runtime, integrity and gate

CPU only, intra/inter-op1, at most4 subprocess workers, detached portable
exclusive supervisor, per-job immutable attempted journals/checkpoints/results,
validated resume skips only complete results with same frozen config/hashes.
Preserve incomplete attempts, record infrastructure interruptions; explicit
--resume only, no automatic scientific replay. Fail closed on negative BN
variance or nonfinite state/logits/gradient/loss/evaluation, record every failure,
drain already submitted jobs and require direction; never clamp/exclude/replace.
progress.json/progress.log after every job, checklist gates current.

Before launch: synthetic accounting/sensitivity/noise-domain/wire-order/BN and
same-runner round-contract tests; py_compile; git diff --check; disk gate;
full sources/prepared-data/reused-result/checkpoint hash freeze. No changes to
frozen sources during runs. After264: independently recompute all endpoint CIs,
reload prediction metrics, every round3 upload noise audit/noBN assertion,
independent RDP minimization, hashes/full manifest; verify checkpoint evaluation,
no-live-worker audit and final report before verified COMPLETE. P34C confirmatory
execution is blocked until this utility extension is independently verified;
utility collapse is descriptive, not permission to change C/sigma.

References: Mironov RDP, https://arxiv.org/abs/1702.07476;
P34B/P34A frozen protocols and runners remain immutable references.
