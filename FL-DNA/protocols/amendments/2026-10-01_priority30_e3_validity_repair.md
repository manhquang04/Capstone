# Priority 30 E3 validity repair

Written: 2026-10-01T17:17:00+07:00
Status: FROZEN / AUTHORIZED by supervisor request of this session.

This continues Priority 29/30. Previously emitted verdicts are superseded,
not deleted. All old artifacts remain immutable. No Latex/ or external_defenses/
edits. Every worker uses torch.set_num_threads(1). Results go into
audit/e3_repair_20261001/; paired attacks retain the 39 S1/S2 source IDs.

## Utility calibration, replacing degenerate S4

The former 50-step, lr=0.05 utility calibration attained only the majority
class rate. Its sigma=0.3 selection and dependent S4 comparisons are invalid.
Create a fixed stratified 20% validation split from Adult train, random seed
43001; fit feature mean/std only on the remaining training rows. Test data is
never used for choosing learning rate, sigma or model checkpoints.

Pre-utility clarification, written 2026-10-01T17:26:00+07:00, before any
utility training output (Soteria replay only has begun): select training length
on validation as explicitly requested in the continuation. Full-batch FedSGD
rounds grid [200,400,800], official FullyConnected(input,[100,100,2]),
cross entropy, SGD without momentum. Select (rounds,lr), lr in [0.1,0.3,1.0], by highest
mean validation accuracy on seeds [42000,42001], tie -> fewer rounds then smaller lr. Run every
candidate to completion. Freeze the selected lr before utility-grid training.
PRECODE uses the existing stochastic bottleneck and its official beta=0.001
KL loss; validation/test use 8 stochastic probability-averaged forwards with
separate fixed evaluation seeds. No validation/test gradients.

Sixteen paired grid seeds: 42016..42031. Baseline, v1 conservative, PRECODE,
and DP trained once per seed. Clip C = 95th percentile of initial full-training
gradient norms of these seeds, recorded before noise-grid training.
Baseline must have mean validation AND mean test accuracy >=0.80 and
>= respective majority rate+0.05; otherwise utility arm NOT_ASSESSABLE and
no DP attack is run. Test accuracy is descriptive; its gate can invalidate
the instrument but cannot choose lr/sigma or trigger tuning.

Grid sigma=[0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3]. One fixed
extension [1.0,3.0,10.0] if needed. For each defense select largest sigma
whose mean paired VALIDATION accuracy delta >= defense delta-0.005.
Selection must have a next larger point below threshold; otherwise
NOT_ASSESSABLE (including no eligible point). No additional grid or retry.
Report validation/test means, SDs, deltas, C and one-release RDP epsilon
(add/remove, sensitivity ratio 1, delta=1e-5).

Estimated upper bound: 18 validation candidates + 16*(3+11) up-to-800-round training jobs,
then 78 official Adult attacks (~80s each); 8 single-thread workers,
estimated 1–12 hours depending on training throughput. Per-job resume,
immutable outputs, run journal. No early stopping.

## Distortion repair

Adult Soteria: reproduce the real adapter sensitivity scores from
run_audit.run_adult_cell, prune at 40th percentile, classifier
layers.3.weight. Deterministic per-target initialization seed 42300+target_id;
same model for undefended/defended gradients. C=p95 undefended norm over
the same 39 targets; sigma=median L2 change/(C*sqrt(d)), inherited convention.
Run 39 new comparator attacks; keep old sigma=0 outputs superseded.
Confirm target source IDs match S2 and S0u. No attack settings change.

PRECODE: inspect tensor names/shapes/order and old shared-prefix calculation.
There is no common parameter space for full-vector L2; mark E3(ii)
NOT_APPLICABLE in BOTH domains. Adult uses valid E3(iii) only. Image verdict
is 'E2 passes; DP comparison NOT_ASSESSABLE' if E2 passes the global Holm family.

## Analysis

Reuse unaffected E1/E2/E3 outputs. Recompute exact paired sign tests on input
MSE (image) or 100-feature_accuracy (Adult). Strict zero ties, p0=0.5.
Holm over ONE whole family of all available E1/E2 defended-vs-none/baseline
tests plus BOTH directions of all available E3 comparisons. Invalid and
NOT_APPLICABLE hypotheses excluded with reasons; report family size.
RQ4: E2 must reduce reconstruction significantly, and defense must not lose
significantly to relevant matched DP; missing required DP arm =>
NOT_ASSESSABLE. Unclipped paper noise is descriptive and not the matched-DP
criterion. Existing ATS adaptive exception => NOT_ASSESSABLE.

Every E2/E3 row reports paired median PSNR/SSIM or feature accuracy, medians
of gray/mean-image or Adult mean/mode baselines and pairwise differences.
'both at floor — ranking not practically meaningful' iff both median
reconstruction-quality values are <= the better data-free baseline median.
This floor flag is descriptive and never changes the verdict rule.
Publish new analysis and a new correction report, and a supersession manifest
pointing to the old report/verdict without altering those files.
