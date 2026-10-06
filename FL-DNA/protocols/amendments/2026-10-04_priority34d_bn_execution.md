# P34D BAF BN measurement execution annex

Written before new BAF targets are reserved or accessed. Three already validated
checkpoints342000/342001/342002 are all retained. Qualification n8 then fresh
n24 each; both prior and cyclic-decoy std-MSE comparisons must have exact
one-sided sign p<.05. Failure is NOT_ASSESSABLE, not privacy evidence, and cannot
select a replacement checkpoint. Development24 and confirmatory39 are separate
fresh four-record batches/checkpoint. Reserve all1140 source records from P32
test split using fixed seed342321, excluding all audited BAF history and all
P34C reservations. No overlap across checkpoints or stages.

Reuse P33A actual one-step Adam .001/focal .95,gamma2, first BN running-mean
delta and public pre-step checkpoint/BN-state recovery. Capture RNG is
derive_seed(342322, checkpoint_seed, stage, target). Strict finite/BN checks
before/after actual forward and optimizer. Inputs and recovered means finite.
BN affine/statistics are the measured legacy instrument, unlike P34A local-BN
payload exclusion. A transformed vector is not loaded into running_var.

V1 P33A plain recovery; v2 P33A known-key exact sketch least squares; never
scaled-transpose recovery. Separate fresh development24 uses unchanged P33A
distortion rule: C=1.01*max raw BN-vector norm, sigma=median DNA L2 distortion
divided by C times median Gaussian unit-noise L2. V2's original defender-side
decode is used only to define distortion, not as the attack decoder. No tuning
on confirmatory targets; only distortion comparator in Part3.

All DP draws use private OS-random seeds generated before execution freeze,
file0600; never print or pass seeds/noise realizations/truth/raw unprotected
vector to protected decoder. Development and confirmatory draws differ.
Public attack inputs are checkpoint plus received payload/key metadata only.
Save exact transmitted payloads and private truth/capture bundles separately;
private bundles are audit evidence outside any DP release claim.

One BN supervisor (one process/thread1), launched only when image utility pool
has exited; no aggregate CPU oversubscription. Full dependencies, prepared data,
checkpoint receipts/results/state, provenance, reservations and private seed
file frozen before first batch access. Per-target outputs/validated receipts,
hash-based skip only, progress after every job. Numerical/solver/negativeBN
failure preserves attempt and stops this sequential pool requiring direction.
Stage completion is not P34D completion; fixed76/combined211 statistics and
independent reload audits remain required. No change to earlier artifacts.
