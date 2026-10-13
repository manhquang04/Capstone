# Priority35B B3 — prospective execution annex to post-hoc extended frontier

Date: 2026-10-11, Asia/Ho_Chi_Minh. Written before B3 target allocation or
measurement. Original protocol SHA37634510777594f4ac7b65e8face6c305778e9e1c805a0fa3fae5b5032357987
and B2b execution SHA97c088555282c1ec7e2407befbd8c3e8ea6abe58d402dc22a766951d4a0c4376
remain immutable. B2b is post hoc relative to B2 utility outcomes; this annex
prospectively enumerates its leakage arms. No gates, metric, estimator budget,
matching threshold or Holm rule changes. User authorizes eight CPU workers
instead of the original maximum four; each has torch intra/inter-op threads1.
This scheduling-only change must pass serial/spawn equality tests.

## Exact arms, independently measured at every eligible cell

32 distinct arms per dataset (matched arms are references, not extra copies):

1. `unprotected`, `dna_v1_conservative`, `dna_v2_0p95`.
2. Every B2 point: `local_dp_global_epsE` and `local_dp_per_tensor_epsE`,
   E in {1,3,10,30,100,1000,10000,1000000}, C=0.01.
3. `b2b_global_c001_clip_only`, C=0.01, no noise.
4. For each tag in {c01,cmed_half,cmed}: `b2b_global_TAG_clip_only` and
   `b2b_global_TAG_epsE`, E in {10,1000,1000000}; joint clipping only.

Dataset radii from VERIFIED unprotected-baseline telemetry, never DP runs:

| Dataset | c01 | cmed_half | cmed |
| --- | --- | --- | --- |
| BAF | 0.1 | 0.7420758605003357 | 1.4841517210006714 |
| IEEE-CIS | 0.1 | 1.7462964057922363 | 3.4925928115844727 |
| PaySim | 0.1 | 0.5099570453166962 | 1.0199140906333923 |

Exact noise SD is 2*C*sigma(E), with the original continuous-order 50-release
Gaussian RDP accounting, delta1e-5, q1; no K/L division. Global clip uses
concatenated L2 radius C; per-tensor uses C/sqrt(L) for the actual uploaded
non-BN trainable gradient tensors. Preserve original float64 norm/scale,
inward float32 factor(1−4*eps32), float64 Gaussian then original-dtype cast.
No noise draws for clip-only. Independent OS-entropy noise seeds, mode0600,
not present in server receipts. Accounted epsilon is for a conditional client
update transcript, NOT record-level privacy; leakage query transfer is not a
new record-level DP claim.

Matched arms frozen from B2/B2b `arms/matching.csv`, for BOTH transforms:
BAF global=`b2b_global_cmed_half_eps1000000` (C0.7420758605003357,E1e6);
IEEE-CIS global=`b2b_global_cmed_clip_only` (C3.4925928115844727);
PaySim global=`b2b_global_cmed_clip_only` (C1.0199140906333923).
IEEE/PaySim matches have NO DP guarantee. All six per-tensor entries remain
NOT_ASSESSABLE; no tests there. Freeze matching.csv/json/utility tables and
their verified manifest as input evidence; do not recompute selection from
leakage outcomes.

## Targets, estimators, execution and privacy boundaries

Eight original P34C n8+n24 qualified cells only: ratio_batch1, tableak_batch1,
tableak_batch2 in BAF/PaySim; ratio_batch1/tableak_batch1 in IEEE-CIS. Recheck
the historical gates and frozen input hashes. Exactly39 new source-disjoint
row sets/cell, paired across all32 arms and data-free controls:312 sets,
390 original records. P24–35 original-source-ID firewall includes all prior
reservations/diagnostics; reject unknown schemas, duplicates, missing outputs
or insufficient fresh rows. Sample deterministically with seed351042 and
domain `p35b-b3-target`, dataset/cell; freeze IDs and original-index mapping.
Server-side analysis seed uses351400/domain `p35b-b3-reconstruction`, dataset,
cell,index; identical across arms, not private noise seeds. Control seed uses
351500/domain `p35b-b3-control`, dataset,cell,index. Never replace a failure.

Use unchanged P34C public ratio model340042 and native TabLeak333042,
models/losses/adapters, known labels, TabLeak1500iterations/lr.06/30-member
ensemble, observable v1 plain/debiased selection and v2 seed-access solvers.
No privileged Config-C BN or Adam upload inference. Freeze imported scientific
dependencies and source/input manifest before any measurement.

Schedule312 unprotected jobs FIRST. Controls (mean/mode and30 empirical draws)
are computed on these same targets and saved with guesses for independent
audit. Select panel estimator from ALL39 unprotected accuracies per eligible
cell, highest median with original cell-order tie rule. Persist selection
receipt before any protected/transform job is submitted. Then9672 remaining
jobs (31*312); total9984 estimator-arm jobs, no duplicate matched-arm runs.
An eight-process spawn pool is allowed, not eight threads sharing RNG/solver
state. Within each process CPU/thread1; all submitted jobs drain on failure,
then REQUIRES_DIRECTION. No automatic scientific retry. Infrastructure resume
requires human approval, no-live check, unchanged freeze and hash-valid skips.

Timing is based only on successful actual measurements: save job runtimes by
estimator and batch. A preliminary ETA must distinguish fast ratio jobs from
30-member TabLeak, never extrapolate ratio throughput to the whole workload.
Enable the recurring monitor only after several initial jobs have succeeded;
use30-minute notifications for meaningful change/failure/completion, not busy
polling. No independent final verifier until supervisor and all workers exit.

## Metrics and fixed inference (unchanged)

Native TabLeak logical accuracy (.319 trainingSD continuous tolerance, exact
categorical, native Hungarian assignment) and standardized coordinate MSE on
the same aligned, native-continuous-postprocessed reconstruction. Bounded
score=min(MSE,MSE_mean/mode), per paired row set. Save reconstructed/true rows
and controls as private audit bundles outside the accounted release.
All tensors/objectives/metrics finite; negative BN variance fails closed.
No outcome-specific clamp, exclusion, seed change or changed tolerance.

Medians and one-based ranks13/27 intervals over39 COMPLETE row sets.
Transform-minus-matched-arm effects, signs in both directions for accuracy
and boundedMSE. ONE fixed144 Holm family:
3datasets*3estimator slots*2transforms*2clipping modes*2metrics*2directions.
IEEE b2 and every NOT_ASSESSABLE slot reserve p1; no family shrink. Clip-only
comparisons use the same slots, labelled NO_DP_CLIP_ONLY. Independently
recompute every integer-tail sign test and a separate NumPy Holm algorithm.
No other grid-point inferential tests; controls/frontier are descriptive.

## Figure, report and finish

Original single-column8.8cm, >=7pt, shared axes, x=utility loss minus mean
pairedDeltaF1 with95%t(df10) intervals, y=selected estimator medianaccuracy
with ranks13/27, data-free lines; add jointDP lines for each of fourC values
with epsilon labels, original per-tensorC.01 line, allclip-only points and
v1/v2. No interpolation. Full tables for every estimator, not only selected.
Save PDF/PNG/exact-source CSV under artifacts/priority35b/figure/ and visually
verify readability. Caption explicitly: x from Config-C local-BN Adam training,
y from P34C single-gradient setting; two different configurations. This is
not record recovery from deployable Config-C and not formal DNA privacy.

Preserve earlier reports/priority35b_report.md byte-for-byte as a prefix;
append the final B2/B2b/B3 section and exact commands/amendment hashes.
Independent post-exit payload/model/metric/control/firewall/CSV/statistics/
accounting/utility-freeze audit, tests/compile/gitdiffcheck, fullSHA256manifest,
reportprefix assertion and no-live checks required before Priority35B COMPLETE.
Do not modify Latex/, external_defenses/, datasets or earlier artifacts/report
text. No commit. Existing Goals unchanged; progress/checklist records B3 stages.
