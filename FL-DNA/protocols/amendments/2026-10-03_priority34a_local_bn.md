# Priority34A — local BN utility variant (pre-execution freeze)

Authorized by the user's Priority34A request; new scientific variant, not a
repair or replacement of any earlier evidence. Written before tests or runs.

## Immutable inputs and budget

Reuse byte-identical Priority32 prepared data, audits, preprocessing and
execution_freeze.json: PaySim13, IEEE-CIS476, BAF58 features. No new feature
selection, splits, cap, imputation, scaling or partition rules. Reuse the
Priority32 mild non-IID partition helper with K=3 and seeds321000–321020.
All three selected budgets are50rounds, Adam lr.001, focal alpha.95/gamma2,
one local epoch, batch1024, fresh optimizer per client per round. Final round
only; no post-hoc increase. Baseline/v1 conservative/v2 .95/.01:189jobs.
CPU float32, torch intra/inter-op threads1, four spawned workers, detached
supervisor, validated completed outputs skipped on infrastructure resume.

## Local BN algorithm and evaluation

Discover BN tensors by module type, not substring. Each client starts with
identical initialized BN affine parameters and running buffers; all BN weights,
biases, running_mean/running_var/num_batches_tracked persist locally across
rounds. Broadcast ONLY non-BN trainable parameters; local Adam trains all local
parameters. Upload ONLY non-BN trainable parameter states, equivalent to
updates relative to the broadcast reference. Assert exact allowlist and zero
BN/buffer keys for every client upload, transformed payload and aggregation.
V1 block256/mix.08/keep.88/shrink.45 and v2 compression.95/eta.01 retain P32
seed derivation. Transform each client's allowed delta before sample-weighted
FedAvg (v2 lift before averaging). BN is neither transformed nor averaged.
Round journals record per-layer BN minima and payload key inventories/assertions.

At the final round broadcast final non-BN parameters to each client's local
model. Each client tunes its own threshold on the common validation split
using unchanged P32 tune_threshold, and evaluates once on the common test
split with its own BN. Save each client's threshold, validation/test outputs
and metrics. Arithmetic, unweighted mean of three client F1/AUC/PR metrics
is the primary per-seed endpoint (not metric of averaged predictions).
Clients are not independent replicates: paired inference uses21seed means.

## Gates and statistics

Fail closed on any nonfinite loss/logit/gradient/state/probability/metric or
negative BN running_var at initialization, each optimizer step, each round and
evaluation. Record failure/location without clamp, replacement or exclusion.
Drain submitted jobs safely, then require direction for scientific failures;
do not analyze incomplete pairs. Singleton batches also fail closed as P32.
No new tuning or baseline selection: reuse frozen P32 selected configs.

For each dataset and DNA method: delta=local-BN DNA mean-client metric minus
local-BN baseline mean-client metric, paired21seeds. Mean, sample SD, median,
IQR and two-sided Student-t95%CI(df20). NI requires STRICT lower CI bound>
-.02 for F1 and>-.005 for ROC-AUC, both endpoints required. PR-AUC descriptive.
No Holm/RQ1 sign family: this is the frozen intersection NI rule from P32.
Independent recomputation from individual client probability files and JSON
metrics, using a separate math.fsum/sample-variance implementation.

Compare with P32 trainable-only/raw-BN variant descriptively using identical
seeds, with its own baseline reference and evaluation rule explicitly stated.
P32 transmits transformed BN affine parameters and raw averaged BN buffers;
P34A transmits neither. Local-BN evaluation is different; cross-variant shifts
are not isolated transform effects or a new confirmatory superiority claim.
No BN transmission is an architectural assertion, not proof of privacy or DP.
FedBN reference: Li et al., ICLR2021, https://arxiv.org/abs/2102.07623.

## Deliverables and preservation

New namespace artifacts/priority34a/, report reports/priority34a_report.md.
Protocol/code/input hashes frozen before jobs; per-job result/checkpoint,
probability arrays, round journals, payload inventory, failure registry,
progress.json/progress.log/checklist.json, exact launch receipt/stdout/stderr.
Report all failures/interruptions/deviations. Independent recomputation, full
SHA256 manifest, tests, py_compile, git diff --check and no-live-worker audit
must pass before completion. Update parent PROJECT.md only after completion.
Never edit Latex/, external_defenses/, datasets/, or earlier artifacts/reports.
