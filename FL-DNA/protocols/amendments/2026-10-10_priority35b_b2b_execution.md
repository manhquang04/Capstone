# Priority35B B2b utility extension — 2026-10-10, post hoc

Written before any B2b arm measurement. Post hoc because B2 showed all uploads
clipped at C=.01. Original B2 results, grid, margin .005, matching logic, Holm
family144 and earlier reports remain immutable. Utility only; no B3, leakage,
reconstruction, figure or full-P35B completion is authorized here.

## Baseline-only choice of C (already independently verified)

The prespecified statistic is the median of all 3 clients x 50 rounds x 11
seeds = 1650 unprotected non-BN upload L2 norms per dataset. No DP norm enters
selection. Float32 detached-copy telemetry consumed no RNG and changed no
training operation, device or dtype. All33 replays matched original P34A final
checkpoint tensor bytes,198 probability arrays, metrics and thresholds exactly.

| Dataset | C_med | C_med/2 | Fixed C |
| --- | --- | --- | --- |
| BAF | 1.4841517210006714 | 0.7420758605003357 | 0.1 |
| IEEE-CIS | 3.4925928115844727 | 1.7462964057922363 | 0.1 |
| PaySim | 1.0199140906333923 | 0.5099570453166962 | 0.1 |

Source files and full SHA256 list:
`artifacts/priority35b/b2b/baseline_telemetry/baseline_clip_values.json` and
`BASELINE_TELEMETRY_VERIFIED.json`. Values annex SHA256
e185e292df12f29dec2f0c43a51d896e5d8c5373880a2ed2d9e409411b05be16.
Pre-norm statistic amendment SHA256
4180b481aef73a5e8151d48882abc688992d2e20dbc79e6b3a41e8a0ee7cc824.
The execution freeze hashes every source file listed in the values JSON, the
values/verified receipts, original B2 sources/inputs/results, and this amendment.

## Fixed design and numerical operations

Config-C and P34A training unchanged: K3, full participation, local BN never
uploaded, fresh Adam lr.001, binary focal loss,50 rounds, same partitions,
initialization and seeds321000..321010. Three datasets;11 paired replicates.
Per replicate endpoint is mean of three clients; validation-only thresholds.
P34A baseline/v1/v2 and B2 are hash-validated reuses, not new measurements.

New joint/global arms: clip-only at C=.01,.1,C_med/2,C_med; DP at each of the
three new C values and whole-training epsilon10,1000,1000000. Exactly429 jobs:
3 datasets x11 seeds x(4 clip-only +9 DP). No new per-tensor arm.

Use the original P34B joint clipping implementation with its explicit radius
argument, including inward factor (1-4*float32 epsilon), double norm and scaling
then cast. Clip-only uses that clipping and NO RNG/no Gaussian. DP uses exactly
the original local Gaussian convention: float64 per-coordinate normal noise,
original dtype cast, private OS keys mode0600 and original round/client domain.
For each C, sensitivity S=2C; same q1,50-release RDP accountant,delta1e-5;
SD=2C*sigma_sensitivity(epsilon). Never divide by K, no central second noise.
Use original P34B weighted double-delta aggregation in both clip-only and DP.
Clip-only therefore includes the same clipping inward rounding and upload/
aggregation arithmetic as DP; disclose this numerical contribution rather than
call clipping cost mathematically pure. No record-level DP claim. Clip-only has
NO DP guarantee, epsilon is NOT_APPLICABLE. Trials are separate hypothetical
deployments; joint release would require additional composition.

Save every individual pre-clip and post-clip norm, tensor norms, clipping factor
event (before_norm>C), noise SD, sensitivity, finite/BN/payload checks, client/
round identity, and hashes. There are1650 upload audits per dataset/arm.
Median pre-clip norms and clipped fractions are descriptive telemetry, never
used to change C, epsilons, seeds or thresholds after outcomes.

CPU only, torch intra/inter-op threads1; four spawn workers with identical fixed
keys to serial. Test serial/spawn equality on synthetic data before freeze.
All jobs submitted once; drain pool on failure and preserve every attempt.
Missing/nonfinite tensor, negative BN variance, hash drift or failed output
halts analysis; no exclusion, replacement, tuning or automatic replay. Resume
only explicit approved infrastructure interruption, after no-live audit and
hash-validation of completed jobs. Source/input freeze is immutable.

## Utility statistics, decomposition and union matching

Report all eleven paired seeds, mean method-minus-baseline Delta-F1 and
Delta-ROC-AUC with descriptive two-sided Student-t95% intervals df10; PR-AUC
additionally descriptive. Independent n=11, not clients/rounds/uploads. No
new significance/NI/equivalence or privacy claim and no new Holm tests.
At C=.01, report baseline->clip-only and clip-only->DP eps1,3,10,30,100,1000,
10000,1000000 using saved B2 DP, including paired t intervals for each step.
These contrasts distinguish clipping-wrapper loss from additional noise cost;
they do not establish that either cause is exclusive.

Reapply original largest-noise passing mean-F1 / failing next-noisier neighbor
criterion separately for each dataset, transform and clipping. Per-tensor grid
remains original B2. Global union contains original B2 plus all new DP and
clip-only points; sort by actual SD, not epsilon across different C values.
Threshold remains mean transform F1-.005. No passing point or no strictly
noisier neighbor => NOT_ASSESSABLE. Duplicate SD values are one noise tier:
a tier passes if any point passes; select first passing tier, deterministic
ties by ascending C then arm ID among passing points; every point at immediate
strictly noisier tier must fail. This only resolves added equal-SD controls,
not changes original B2 rule or outcomes. Clip-only is SD0 with epsilon
NOT_APPLICABLE; if selected label MATCHED_CLIP_ONLY_NO_DP, not a matched DP
comparator or privacy evidence. Report SD,C,epsilon,arm and next-noisier tier.

## Verification, timing and stopping

Independent verification ONLY after supervisor and every recorded worker exit.
Reload all new checkpoints, BN, round records, test/validation probabilities,
thresholds and metrics. Independently reconstruct scalar pre/post clip norms
and scaled tensor norms: allow only 8*float32 epsilon relative rounding for the
cast-scaled norm formula, never for the exact C bound or scientific matching.
Recompute all paired summaries from saved CSV using
independent NumPy/t.isf code; independent union bracket check. Rehash original
B2 manifest with its approved supplementary infrastructure log seal; never
overwrite original manifest or log. Preserve new manifests. Runtime logs/locks
sealed only after exit; no repeat of prior verifier-before-exit race.

Run tests, py_compile and git diff --check. Write summary/tables/decomposition/
matching and SHA256 manifest under artifacts/priority35b/b2b/arms/. Only then
mark B2 and B2b utility complete; stop before B3, fullP35B incomplete.

Timing starts with real launch, not preparation. After first successful measured
job, estimate elapsed/(new successful jobs)*(remaining jobs), an initially rough
four-worker throughput estimate. Monitoring cadence: clamp ETA/4 to5..30
minutes rounded up to a5-minute increment; use at least30 minutes when ETA
exceeds120 minutes. Choose once after first successful job, record estimate and
cadence, then schedule short checks (no busy polling). A single event-driven pipe
may wait for the first successful job without file/process polling.
Notify meaningful stage,
failure, major ETA change or verified completion; quiet unchanged.
