# P34D image confirmatory execution annex

Written before new image recovery reservations or runs. Require utility stage
complete and no live utility/BN workers before recovery launch. Four settings
are the three new independently initialized LeNet-Zhu states and the fixed
trained replay state, exactly those sealed for utility calibration. Part1 uses
39 distinct single-image gradients per model; Part2 uses39 distinct four-image
gradients at the trained state. Targets from CIFAR-10 test split, fixed selection
seed342600, source-disjoint across settings and all historical measurements.
Exclude P29/P30 conservative populations, all P33C/P34B reservations plus IDs
in image/CIFAR historical JSON receipts (including nested indices). Document
every contributing path/hash. No utility train/validation split changes.

Per setting39 paired unprotected/v1/v2 targets; add single/per-tensor DP arm
separately for each DNA only when that setting's own finite utility bracket is
MATCHED. NA comparisons reserve p=1 in fixed76; no transferred sigmas or tuning.
At most1092 recovery jobs (4 settings *39*7 arms). Same frozen4800iterations,
1restart, TV.01/signed/boxed Adam.1 cosine and known labels from P33C. Attack
initialization seed342700+target, paired across arms and settings. V1 plain and
structure-debiased candidates select minimum observable objective (plain tie);
v2 known-key sketch-space matching, never scaled-transpose decoding. Received
payload plus public model, labels, known-key metadata only enters attacker.
Truth/raw private gradient/noise realization never enters protected decoder.
Client-side DP adds noise before measurement, same individual-gradient domain;
one-release update distortion/utility comparator, not meaningful record-DP.

Private OS-random noise seeds are generated for each DP arm/target before
execution freeze in file0600 and not printed or passed to attacker. Clip/sigma
are utility calibration outputs for the exact starting state. Normalized inputs
contiguous (value-preserving P33C repair). Strict finite loss/raw/received/
candidate/reconstruction/state checks; do not clamp nonfinite outputs or BN.

Hungarian MSE pairing only AFTER observable candidate selection. Score the
mean per-record PSNR/SSIM/MSE for each four-image batch;39 batches are independent
units, not156 images. Preserve raw reconstructions before metric clipping and
exact receipts separately from private truth arrays. Independent final verifier
must reload and recompute alignment and metrics. Private audit arrays outside
DP release. Full dependencies/calibration/clips/start states/targets/private
seeds frozen before first access. Detached4worker pool, per-attempt artifacts,
validated hash skips only, logs/progress each job; failures drain submitted pool
and require direction. No automatic scientific replay. P34D completion only
after fixed76/combined211 independent stats/manifest/report/no-worker checks.
