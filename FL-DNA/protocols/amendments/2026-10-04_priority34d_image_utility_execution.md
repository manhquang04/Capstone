# P34D image utility execution annex

Written before clip probes or image training. This authorizes only utility
calibration, not recovery target selection or attacks. No earlier source is
edited. The original staged amendment remains authoritative.

Four starting states: LeNet-Zhu initializations 342042, 342043, 342044, and the
fixed P33C replay trained checkpoint (SHA256
6d390aaf2bbb52aab0bc5991a5c682bf1cd88cf3d170957edfe8fbbd834d36c5).
The latter is a warm start; none of the old from-scratch sigmas are transferred.
All four use immutable P31 train/validation split, SGD lr .1, momentum 0,
training batch 256, 100 epochs and paired order seeds 51016 through 51031.
Recovery batch 4 does not change training. CPU, intra/inter torch threads 1.

Before training, each fixed starting state is saved and clip norms recomputed
using both original probe seeds 51000/51003, all 94 initial minibatches.
Single clip is p95 concatenated-gradient norm; per-tensor clips are each p95.
All probes are finite checked. No clipping of BN variances is permitted.
Normalized inputs are contiguous without altering values, gradients or RNG.

For each setting run baseline, v1 conservative (.08/.88/.45, block256,
key30001), P31 v2 (.95/.01, key30002) training adapter, plus single/per-tensor
DP at [.0001,.0003,.001,.003,.01,.03,.1,.3], all16 paired seeds:1216 initial
jobs. Noise seeds follow the original P31 utility convention (orderseed*100000
+ step); this is a utility experiment, not a new privacy release guarantee.
Use original decoded v2 training adapter; later recovery uses sketch-space.

Match each DNA separately: largest finite sigma with mean paired validation
delta >= DNA delta -.005, immediate next higher sigma below threshold,
baseline mean >= .40. Test set cannot select matches. If and only if a passing
sigma is at the top of the observed grid, append preregistered [1,3,10] jobs
for that setting/mechanism (one shared grid for both DNA variants). No smaller
sigmas, alternate clips, new seeds or replicate changes. No eligible sigma
or no bracket after extension is NOT_ASSESSABLE. Numerical failures preserve
attempts, drain already submitted pool and require direction; do not analyze
missing outputs or replace failed replicates.

Preparation freeze seals driver/tests/protocol/transitive project Python
dependencies, immutable CIFAR files, P31 split, checkpoint and prerequisites
BEFORE probe access. Training freeze additionally seals probe outputs and
starting states. Fresh per-job attempt folders, result plus hashed validated
receipt, no skip without hashes/identity checks; infrastructure-only resume
preserves interrupted attempts. No automatic scientific replay. Supervisor
uses at most4 subprocess workers, is detached, refuses duplicate live jobs,
and logs progress after every job. Utility completion is stage-only; P34D
requires later target/firewall/freezes, confirmatory runs and final audit.
