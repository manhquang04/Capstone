# Priority 29 protocol amendment: native literature-setting defense audit

Date written: 2026-09-30

Status: FROZEN / AUTHORIZED

Scope: This is a new extension audit using the published/native attack
settings validated in `external_defenses/REFERENCE_ATTACKS_REPORT.md`. It does
not modify any earlier RQ1/RQ2/RQ3 conclusion or any Priority 28 result.

Read-only boundary: everything under `external_defenses/` is read-only for this
priority. Reference code may be imported. Driver logic may be copied into
`experiments/priority29/` with attribution, but no file under
`external_defenses/` may be edited or used as an output location. If a dependency
is missing from `external_defenses/.venv`, the run is reported as blocked.

No `Latex/` edits are allowed.

Threading: every process must keep `torch.set_num_threads(1)`.

Disclosure: every run, including interrupted, resumed, replayed, failed, or
discarded runs, must be listed in the report.

## Native instruments

### IMAGE primary instrument

Domain: CIFAR-10 test split.

Model: official `construct_model("LeNetZhu", seed=42)` from
`external_defenses/invertinggradients`, untrained.

Setting: FedSGD, one gradient, batch 1, known label.

Attacker: official `inversefed.GradientReconstructor` with the paper setting
validated in `REFERENCE_ATTACKS_REPORT.md`:

- cost function: cosine (`sim`);
- signed Adam;
- learning rate `0.1`;
- iterations `4800`;
- restarts `1`;
- total variation `0.01`;
- boxed image projection;
- learning-rate decay enabled;
- score choice: attacker-observable loss.

Scoring: raw-input MSE, PSNR, SSIM against gray `0.5` and CIFAR training mean.

Positive-control gates: fresh CIFAR-10 test targets, source-disjoint within this
priority and disjoint by split from prior project CIFAR train-target artifacts.
Run n=8 and then n=24. The attack must beat Prior, decoy, gray, and CIFAR mean
baselines by exact one-sided sign test, alpha `0.05`.

### TABULAR primary instrument

Domain: Adult, official TabLeak native FedSGD setting.

Model/attack: official TabLeak configuration 46, as in
`external_defenses/reference_tabular/run_reference.py`, copied into
`experiments/priority29/` if needed so outputs remain outside
`external_defenses/`.

Setting:

- Adult dataset from the official checkout;
- batch 8;
- known labels;
- initialized fully connected network `105 -> 100 -> 100 -> 2`;
- cross-entropy;
- cosine discrepancy;
- signed Adam `0.06`;
- `1500` iterations;
- `30` restart/ensemble members;
- uniform initialization;
- categorical softmax;
- continuous sigmoid bounds;
- median + softmax pooling;
- no learning-rate schedule.

Metric: official TabLeak mixed-feature accuracy. Baselines: mean/mode and
empirical marginals.

Positive-control gates: n=8 and n=24 fresh Adult batches. Exact one-sided sign
test that official accuracy beats both data-free baselines.

### PaySim secondary

The deterministic PaySim TabLeak adaptation from `REFERENCE_ATTACKS_REPORT.md`
is secondary/descriptive only and is dropped by compute reduction in this pass
unless image and Adult primary audits complete under the wall-clock ceiling.

## Defenses

Defenses to audit when instruments qualify:

- Soteria;
- PRECODE as a model-level variational bottleneck, not an update pass-through;
- gradient pruning;
- ATS, image only;
- Count-Sketch aggregation;
- DNA Transform v1 conservative;
- DNA Transform v2 0.95 / eta 0.01.

Each defense must pass Priority 28 Step 0 checks: lossless sanity,
defense-not-identity, server-knowledge receipt, decoy correctness, and positive
control.

## Evaluations

For every qualified `(defense, domain)` cell, use fresh confirmatory n=39
targets and exact one-sided sign tests. Holm correction is applied across the
whole Priority 29 family for E2 and E3 p-values.

E1 original-style: run the non-adaptive official attacker and original paper
metric. Question: does the published qualitative protection claim reproduce?

E2 adaptive: run the adaptive attack matching server knowledge. For v2, the
key-holder path must use least-squares through the sketch map, not the naive
lift. Primary question: does the defense significantly reduce reconstruction
quality relative to no defense on the same targets?

E3 DP comparator:

1. paper-documented unclipped Gaussian noise at paper sigma;
2. properly clipped DP at the same L2 distortion as the defense;
3. tabular only: utility-matched DP by Adult test accuracy under FedSGD
   training, bracketed grid and Priority 25b matching rule.

Test defense versus each comparator in both directions.

## Answer rule: RQ4 per defense

`SURVIVES`: E2 shows a significant reconstruction-quality reduction versus no
defense, and the defense is not significantly worse than properly clipped DP at
matched distortion and, for tabular, matched utility.

`DOES_NOT_SURVIVE`: otherwise; the report must name the check that flips or
blocks it.

`NOT_ASSESSABLE`: the native positive control fails or the required model-level
defense/attacker adapter is unavailable without modifying read-only reference
code.

## Compute estimate and reduction

Image reference attacks take roughly 9-16 seconds per target in the validated
single-instance report. Adult TabLeak takes roughly 87 seconds per batch.
The unreduced defense x domain x E1/E2/E3 matrix is expected to exceed four
days once Adult confirmatory and E3 grids are included.

The following reduction is frozen before execution:

1. Drop E3(iii) utility-matched DP unless E2 survives for a tabular defense and
   the remaining projected wall time stays under four days.
2. Drop ATS after image instrument qualification if non-ATS image cells already
   push projected runtime over four days.
3. Drop PaySim secondary for this pass. PaySim is reported as not run by frozen
   compute reduction.

If either IMAGE or TABULAR native positive control fails at n=8, that domain
stops immediately as `NOT_ASSESSABLE` under the pre-registered gate.

## Deliverable

Report path:
`reports/priority29_native_audit_report_20260930.md`

The report must include exact commands, SHA-256 hashes, positive-control
tables, E1/E2/E3 tables when available, Holm correction, RQ4 verdicts,
reconstruction image grids, deviations from original papers, and per-target
CSVs.
