# Priority35B — post-hoc presentation and descriptive sensitivity note

Date: 2026-10-12. Authorized after measured visual QA identified label overlap.
Freeze this note and the additive adapter before generating the new version.

Only annotation positions and leader-line endpoints may change in the new
figure under artifacts/priority35b/figure/label_layout_v2/. Points, connecting
curves, error bars, markers, colors, fonts, dimensions, axis scales and captions
stay identical. Preserve all original figure files and the original QA record.
Copy the data CSV byte-for-byte, requiring SHA-256
e815f3ba6261bd70fa7e989e812292398a486aa9e30b012e23389cd90bbf1172.
Reserve marker/error-bar bounding boxes during label placement; check the
non-annotation artist fingerprint against the original renderer. Inspect both
the new PDF rasterization and PNG before publication. No measurement reruns.

Append a descriptive estimator-sensitivity section, with no new inferential
tests, Holm changes, gates, selection rules or qualification changes. For each
dataset and each estimator, extract every distinct matched arm plus unprotected,
v1, v2 and C=0.01 epsilon=10 (both joint and per-tensor) from the existing fully
verified CSVs. Independently sort the same 39 per-target accuracies to check the
median and ranks 13/27 against frontier.csv. Show explicit NOT_APPLICABLE for
IEEE-CIS TabLeak b=2, which was not qualified; do not fabricate that cell.

The pre-registered fixed-order tie break selected ratio b=1 from an unprotected
100% three-way tie for BAF and PaySim and a two-way tie for IEEE-CIS (the third
estimator is not qualified there). Explain ratio invariance to positive common
gradient rescaling, the 100% clip-only medians, and sensitivity to additive noise.
Report the observed TabLeak feature-accuracy summaries descriptively: BAF's
matched DP and transforms all have 100% medians at b=1, while b=2 has 93.24%
for matched DP versus 100% for transforms. IEEE-CIS and PaySim matches are
clip-only, not DP, with 100% TabLeak medians. Thus similar/high leakage under
TabLeak is not an equivalence claim; retain all existing paired sign/Holm
results, including any significant differences. Do not claim a three-way tie
on IEEE-CIS or a formal matched DP comparison on clip-only datasets.

Use additive runtime adapters only; do not edit frozen code. Bind the new QA,
presentation adapter/note/freeze, copied CSV hash, sensitivity extraction and
all original assets to the final manifest. Append to the unchanged report byte
prefix. Existing approved numerical tolerances remain unchanged. Preserve all
failed verifier outputs/logs and stop on any new scientific discrepancy.
