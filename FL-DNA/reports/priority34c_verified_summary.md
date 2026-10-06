# Priority34C — verified result summary

Administrative companion to `priority34c_report.md`; it does not replace or
modify the registered evidence. All 1326 confirmatory jobs and the independent
audit passed. Qualification: 8/9 cells; development: 2/16 distortion comparators.

In this easiest, known-label/public-model gradient-recovery setting, the
transforms do **not** show lower recovery than meaningful local DP. DNA recovery
is significantly higher in 15/16 eligible local-epsilon10 comparisons after
the fixed 72-test Holm correction. IEEE-CIS ratio v2 is inconclusive, not
equivalent. The two matched-distortion comparisons have median accuracy 100%
in both arms, zero median paired effects, and no significant difference.

## Local-DP epsilon10 comparisons

Feature accuracy medians (%); effects are median paired DNA minus DP differences
in percentage points, not differences between marginal medians. Full intervals,
both-direction raw p values and Holm72 values are in the immutable main report.

|Dataset / instrument|DNA v1 median|DNA v2 median|DP median|Paired effect v1|Paired effect v2|
|---|---:|---:|---:|---:|---:|
|PaySim / ratio batch1|100.00|100.00|77.78|+22.22|+22.22|
|PaySim / TabLeak batch1|100.00|100.00|22.22|+77.78|+77.78|
|PaySim / TabLeak batch2|100.00|100.00|27.78|+66.67|+72.22|
|IEEE-CIS / ratio batch1|95.00|64.29|60.71|+31.90|+4.52 (not significant)|
|IEEE-CIS / TabLeak batch1|100.00|100.00|35.19|+64.81|+64.81|
|IEEE-CIS / TabLeak batch2|NOT_ASSESSABLE|NOT_ASSESSABLE|—|—|—|
|BAF / ratio batch1|97.30|94.59|37.84|+59.46|+51.35|
|BAF / TabLeak batch1|100.00|100.00|35.14|+64.86|+64.86|
|BAF / TabLeak batch2|100.00|100.00|36.49|+63.51|+63.51|

IEEE-CIS TabLeak batch2 failed the initial qualification gate. Fourteen
distortion comparators failed development matching; they are NOT_ASSESSABLE,
not evidence of privacy. Neither gate was retuned using confirmatory outcomes.

Local DP adds noise before transmission to the same individual gradient;
epsilon10 is per-client, update-level, 50-round RDP at delta1e-5, **not
record-level DP**. Batch1 is the easiest setting. Native TabLeak uses the P33B FC
variant (IEEE selected60 features), whereas ratio uses the full P32 model/evalBN.
This conditional single-query measurement is not an Adam multi-step or
final-model inversion result. Central DP aggregates are excluded. Keys are known
for v2; public population priors and known labels are assumed. Private audit
bundles are not DP releases. All older artifacts, original report and manifest
are preserved, with no training/recovery rerun for administrative close-out.

Main report SHA256: `66199d839a9072f096fb6fee1350b1548ff74e001271bc087d43a4c755e33188`.
Original manifest SHA256: `b11efe6ad629f0c1e3aa9b74dc9337831b5fb139b06e723035cb99852f58d9a5`.
