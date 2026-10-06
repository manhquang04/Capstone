# Priority34E — pre-execution amendment

Authorized by the user's capstone request after verified P34D. This new
descriptive extension does not replace previous evidence. No Latex/,
external_defenses/, datasets/ or prior sources/artifacts/reports are edited.

## Part 1: fixed total data, more clients

Reuse all three Priority32/P34A prepared datasets (PaySim13, IEEE-CIS476,
BAF58), preprocessing, model and final-round budget: 50 rounds, one local
epoch, Adam lr0.001, focal alpha0.95/gamma2, batch1024, fresh optimizer each
client/round. K=10 and20; baseline, v1 block256/mix0.08/keep0.88/shrink0.45,
v2 ratio0.95/eta0.01;11 paired seeds per dataset/K: 343000..343010.
Total198 jobs. Same seeds across K are contextual paired draws, not more
independent replicates. Use the original generalized partition helper:
fraud shuffled/split evenly; sorted natural category index modulo K selects
primary client,55% nonfraud there and45% divided among other clients.
All records occur exactly once. No rebalancing or category remapping.
Check nonempty clients and singleton final batches before launch; if one
fails, preserve failure and request direction, never silently drop a record.

All BN affine parameters and buffers remain local. Only non-BN parameters
are broadcast/transformed/sample-weighted aggregated. Use P34A guarded
training/evaluation/seed derivation unchanged apart from K and new seed list.
Common validation/test split, threshold tuned per client on validation only;
replicate endpoint is unweighted mean of K client metrics. Paired DNA−baseline
test F1, ROC-AUC, PR-AUC: mean, sample SD, median, IQR and two-sided Student-t
95% CI df10. These are pointwise descriptive CIs, no NI/equivalence/Holm claim.
Independent reload of probability metrics, thresholds, final checkpoints,
round journals, partition hashes and separate fsum paired-CI calculation.

## Part 2: exact protocol-metadata recognition, before outcomes

The original v2 transform metadata contains seed and sampled_indices; literal
full metadata reveals the derived seed directly (no brute force necessary).
The measured counterfactual removes seed but retains sampled_indices, shape,
tensor index, delta and q. This is explicitly a metadata-assisted feasibility
benchmark, not an attack against a metadata-free/hidden-projection sketch.
The server simulation also lifts locally before aggregation; no claim that
this repository has an implemented wire protocol exposing this metadata.

Create exactly one own-data BAF model first-layer update (128x58), after one
Adam/focal step on the first1024 frozen training rows, initialized343100.
Use v2 base seed343101, tensor_index0, ratio0.95, eta0.01,
quantization_seed343102. Attacker function never receives raw update or true
seed; a separate validator scores its ranks after exhaustive search.
Search every candidate base seed in[0,2^20), containing343101, no early stop.
Criterion: regenerate the candidate's ordered sampled index list with the
original v2 RNG/derive routine and require exact equality to observed metadata.
Binary score1 for equality,0 otherwise; no outcome-tuned norm threshold or
ground-truth-distance selection. Rank is competition rank (1+number scoring
strictly better); also report tied-first count, false positives, complete
matching candidate list. Validation succeeds only if the true base seed is
the UNIQUE rank1 candidate. Failure is retained as recognition NOT_VALIDATED,
not fixed/tuned; throughput is still reportable but no validated full-space
identification claim. Decode matching candidates using original lift and
report norm/quantization-grid consistency as diagnostics, not the selection.
Norm/grid alone are not seed identifiers: grid is a property of q shared by
all candidates, and full padded orthogonal lift has seed-invariant L2 norm.

Save public seed-redacted observation separately from private validation
seed/raw update (0600); do not feed private bundle to recognizer.64 immutable
chunks of16384 candidates, exact coverage/receipt hashes and per-chunk time.
Throughput =2^20/sum active search seconds (not setup or resume downtime).
Full32 worst-case seconds=2^32/throughput; uniform-position expected search
half that. Report min/max chunk-rate sensitivity, platform/CPU/versions and
single-process CPU conditions. Do NOT execute a full32 search.32-bit coverage
and uniqueness are NOT proven by a20-bit test; no cryptographic-security claim.

## Execution, progress and completion

CPU only; torch intra/inter-op1 and numerical libraries1 per worker.
Detached single locked supervisor, four spawned utility workers, seed search
sequential before utility so its timing is not contaminated by own training.
Immutable source/input/config freeze before capture/training/search. New
artifacts/priority34e progress.json/progress.log/checklist.json, command and
stdout/stderr, attempt directories, receipts and failures retained. Resume
only skips hash-validated completed jobs/chunks, never replaces scientific
failures; supervisor drains submitted utility work before requiring direction.
OS lock portable through exclusive-create lock plus live PID check. Detached
launcher refuses live duplicate; explicit --resume required after interruption.
Final report reports/priority34e_report.md requires full198 jobs, full20-bit
coverage, independent audit, hashes, tests, py_compile, git diff --check and
no remaining science workers. Parent PROJECT snapshot then additive note.
Old paused P34B Goal is untouched; use persisted checklist/progress for P34E.
