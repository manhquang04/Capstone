# RQ2 Confirmatory Protocol — Paired Multi-Seed Utility

**Protocol ID:** `RQ2-MULTISEED-V1`  
**Version:** `1.2-confirmatory-execution-authorized`  
**Status:** `CONFIRMATORY COMPLETE — SINGLE EXECUTION CONSUMED`  
**Parent plan:** `../../PROJECT.md`  
**Required approval:** research team and supervisor  

## 1. Research question

> How does applying DNA encoding affect the final accuracy of the global model,
> measured by F1-score and AUC-ROC?

The analysis separates bit-exact lossless DNA transport from DNA Transform.
The former should be algorithmically equivalent after decoding, whereas the
latter modifies updates and may affect model utility.

## 2. Estimands and primary hypotheses

For confirmatory seed `s` and method `m`:

```text
Delta_F1(m,s)  = F1(m,s)  - F1(BASELINE,s)
Delta_AUC(m,s) = AUC(m,s) - AUC(BASELINE,s)
```

The sole primary estimand is the mean paired difference for:

1. `DNA_TRANSFORM - BASELINE` in final F1 and AUC-ROC.

`LOSSLESS_DNA - BASELINE` is a separately reported technical/sanity check of
transport invariance, not a member of the primary hypothesis family.

The intended confirmatory framework is non-inferiority for utility. The team
must freeze whether both F1 and AUC must pass or whether F1 is primary and AUC a
key secondary endpoint.

```yaml
analysis_framework: non_inferiority
primary_endpoint_rule: both_f1_and_auc_must_pass
f1_noninferiority_margin: 0.02
auc_noninferiority_margin: 0.005
confidence_level: 0.95
alpha: 0.05
target_power: 0.80
multiple_endpoint_rule: both_endpoints_required
maximum_feasible_seed_count: 30
required_confirmatory_seed_count: 21
```

The method-family multiplicity rule was approved at
`2026-09-13T00:41:34+07:00`, before execution: DNA Transform retains the full
alpha 0.05; lossless DNA is not adjusted jointly with it by Holm or
Bonferroni. See
`amendments/2026-09-13_rq2_multiplicity_and_execution_authorization.md`.

Margins must reflect the largest deployment-relevant utility loss, be approved
before the shared development batch and not be derived from existing
single-seed differences.

## 3. Methods

### 3.1 Primary methods

1. `FL_BASELINE`: raw-update federated learning.
2. `FL_DNA_LOSSLESS`: binary/DNA/AES-GCM transport followed by bit-exact decode.
3. `FL_DNA_TRANSFORM`: exactly the configuration frozen for RQ1.

### 3.2 Secondary/contextual methods

4. `FL_DP_DISTORTION_MATCHED` from RQ1 development calibration.
5. `FL_DP_UTILITY_MATCHED` was `NOT_FOUND` under the frozen rule and is omitted;
   the calibration search is closed permanently as of 2026-09-13.
6. `CENTRALIZED_REFERENCE`, which contextualizes utility but is not a substitute
   for the paired FL baseline.

Do not run the full DNA or DP ablation grid in the confirmatory experiment.

## 4. Dataset and evaluation contract

- Use the frozen fraud dataset checksum and preprocessing implementation.
- Freeze train, validation and untouched test partitions before the development
  batch.
- Training and client partitioning may use train data only.
- Choose classification threshold on validation data by one frozen procedure.
- Evaluate final F1 and AUC-ROC once on the test set for every method/seed.
- Never choose configuration, round, seed or threshold using test performance.
- Freeze whether “final” means last round; best-test-round selection is banned.
- If best-validation-round is reported, define it before execution and retain
  last-round results as the primary outcome unless formally amended pre-freeze.

## 5. Shared development/calibration batch

Use one batch of 5–8 independent development seeds for all three purposes:

1. select `DP_UTILITY_MATCHED` using the frozen candidate grid and distance
   rule;
2. estimate paired variance for F1/AUC sample-size calculations;
3. exercise the complete training, evaluation and artifact pipeline.

Before running the batch, freeze:

- exact development seed list;
- DNA Transform configuration;
- DP candidate grid and utility matching tolerances;
- threshold-selection and final-round rules;
- paired differences and variance estimator;
- retry/failure rules.

The same development results may serve these predefined purposes; no additional
purpose may be invented after seeing them. Development seeds must be disjoint
from confirmatory seeds and may not be pooled into the final estimate.

The frozen paired-seed contract is: one `FL_RUN_SEED` controls dataset
subsampling/split, client partition, model initialization and loader order;
every method for that seed receives the same value. Split and partition
checksums are stored separately for each seed, not as one checksum for the
whole eight-seed batch.

## 6. Utility-matched comparator selection

For each DP-style candidate, compute paired utility degradation from baseline
on every development seed. Compare its aggregate F1/AUC degradation with the
frozen DNA Transform degradation using the predeclared standardized distance
and tolerance.

```yaml
dp_candidate_grid: [0.0001, 0.0005, 0.001, 0.005, 0.01]
f1_matching_tolerance: 0.01
auc_matching_tolerance: 0.0025
distance_function: max_of_normalized_f1_and_auc_absolute_deviation
candidate_tie_breaker: smallest_noise_multiplier_among_equal_absolute_distances
```

Archive all candidates and results, not only the winner. If no candidate is
eligible, record `DP_UTILITY_MATCHED = NOT_FOUND`; do not expand the grid after
confirmatory work starts and do not relabel a nearest candidate as matched.

## 7. Confirmatory sample size

RQ2 uses paired-continuous power analysis, separately for paired F1 and AUC-ROC
differences. It must not use the RQ1 sign-test formula.

1. Estimate paired variance from the shared development batch.
2. Because 5–8 seeds give uncertain variance, use a conservative upper
   confidence bound or publish a sensitivity range and select the largest
   required `n` under the frozen rule.
3. Calculate `n` from the approved non-inferiority margin, alpha, target power,
   paired variance and endpoint/multiplicity rule.
4. Use the maximum `n` required by every endpoint that must pass.
5. Freeze the exact confirmatory seed list before any confirmatory run.

`maximum_feasible_seed_count` is a resource ceiling frozen before the pilot;
`required_confirmatory_seed_count` is the power-analysis result after the pilot.
They must never be treated as the same value.

If required `n` exceeds the compute ceiling, report a precision-limited or
underpowered design and narrow the claim. Do not reduce `n`, increase alpha or
change margins based on observed confirmatory performance.

## 8. Paired execution

For each seed, every method uses the same:

- train/validation/test split;
- client partition and non-IID assignment;
- model initialization;
- client-selection schedule;
- local sample order where technically possible;
- clients, rounds, local epochs, optimizer and evaluation schedule;
- threshold-selection rule.

Defense randomness must use method-specific namespaces derived from the shared
seed and be logged. Run ordering must be randomized or balanced by a frozen rule
to reduce thermal/load bias. No seed may be replaced because its performance is
poor. A demonstrable technical failure may be rerun with identical seed/config;
both the failure and retry must remain in the registry.

## 9. Metrics

### 9.1 Primary

- final F1 at a threshold selected only on validation data;
- final AUC-ROC on the untouched test set.

### 9.2 Secondary

- PR-AUC, precision, recall and confusion matrix;
- recall at a predeclared FPR if feasible;
- validation-selected threshold;
- per-round train/validation metrics and convergence stability;
- final loss and completed-round count.

Accuracy is not a primary metric for the imbalanced fraud task.

### 9.3 Lossless integrity

For `FL_DNA_LOSSLESS`, also report byte equality, dtype/shape restoration and
maximum absolute update error before aggregation. Bit-exactness does not replace
the end-to-end multi-seed utility analysis.

## 10. Statistical analysis

The independent unit is the training seed. For every primary method/endpoint:

- publish all paired per-seed values and differences;
- report mean, SD, median, IQR and a paired confidence interval;
- run the frozen paired non-inferiority test;
- report convergence failures and completed rounds;
- apply the frozen multiple-endpoint/multiple-method rule.

Non-inferiority is established only if the relevant confidence bound lies
within the approved negative margin under the frozen endpoint rule. `p > 0.05`
in a difference test is not evidence of equivalence or non-inferiority.

Secondary DP comparisons are contextual unless explicitly included in the
frozen multiplicity plan. Exploratory analyses must be labeled as such.

### 10.1 Precommitted interpretation for DNA Transform F1

This rule was frozen at `2026-09-13T00:24:52+07:00`, before the confirmatory
seed list was generated and before any RQ2 confirmatory outcome was observed:

> At `n = 21`, if the lower bound of the 95% paired confidence interval for
> `F1(DNA Transform) - F1(Baseline)` is below `-0.02`, conclude officially:
> “DNA Transform không thiết lập được non-inferiority F1 ở margin đã duyệt.”

That outcome is valid, including when negative. It must not trigger a relaxed
margin, a changed endpoint rule, additional seeds, or exclusion of any
completed observation. Approval is recorded in
`amendments/2026-09-13_stage1_supervisor_approvals.md`.

## 11. Missing data and deviations

- Do not discard a completed low-performing run.
- Do not replace a confirmatory seed.
- Mark infrastructure failures separately from algorithmic divergence.
- Rerun only demonstrable infrastructure failures with the same seed and config.
- Freeze how irrecoverable missing pairs affect analysis before execution.
- Report both intention-to-run and valid-paired analysis counts.
- Every protocol deviation requires a timestamped amendment and impact label.

## 12. Required artifacts

Each seed/method run stores:

- protocol/config/code/environment/dataset hashes;
- complete command and stdout/stderr;
- shared seed plus derived partition/model/order/defense seeds;
- split and client-partition provenance;
- model initialization and final checkpoint hashes;
- per-round metrics and threshold-selection trace;
- final predictions or sufficient immutable data to recompute metrics;
- timing, completed rounds and failure status;
- for lossless DNA, update-integrity results.

Aggregate files must be generated from raw artifacts by a versioned script.

## 13. Execution order

1. Resolve scientific thresholds and freeze the resource ceiling, split,
   methods, grids, endpoint rules and 5–8 development seeds.
2. Approve/hash protocol and pre-pilot config; set `PRE-PILOT FROZEN`.
3. Run the shared development/calibration batch exactly once.
4. Select/reject `DP_UTILITY_MATCHED` and estimate conservative paired variance.
5. Compute `required_confirmatory_seed_count` and compare it with the previously
   frozen `maximum_feasible_seed_count`.
6. Freeze the exact confirmatory `n`, seed list and methods in a separate
   confirmatory config; approve/hash it and set `CONFIRMATORY FROZEN`.
7. Run paired confirmatory methods without configuration changes.
8. Verify artifacts, then generate the statistical report.

Steps 3–5 require `PRE-PILOT FROZEN`; steps 7–8 require
`CONFIRMATORY FROZEN`. No later gate may be inferred merely because an earlier
gate was approved.

## 14. Required outputs

```text
protocols/rq2_multiseed_protocol.md
protocols/config/rq2_pre_pilot.yaml
protocols/config/rq2_multiseed.yaml
artifacts/rq2/development_calibration/
artifacts/rq2/run_<seed>_<method>/
results/rq2/rq2_per_seed.csv
results/rq2/rq2_summary.json
results/rq2/rq2_convergence.csv
reports/rq2_report.md
```

## 15. Two-gate freeze checklist

### 15.1 `PRE-PILOT FROZEN`

- [x] Non-inferiority margins and endpoint decision rule approved.
- [x] Alpha, power, multiplicity rule and compute ceiling approved.
- [x] `maximum_feasible_seed_count` approved independently of pilot outcomes.
- [x] Per-seed split contract, threshold selection and final-round rule frozen.
- [x] Eight development seeds and all candidate grids frozen.
- [x] Paired variance estimator and conservative rule frozen.
- [x] Failure, retry and missing-pair rules machine-readable.
- [x] Method configurations and `FL_RUN_SEED` namespace recorded.
- [x] Pre-pilot config and code/protocol/environment hashes recorded.
- [x] Team and supervisor pre-pilot approvals recorded below.

### 15.2 `CONFIRMATORY FROZEN`

- [x] Pilot/calibration artifacts archived; the frozen utility rule returned
  `DP_UTILITY_MATCHED = NOT_FOUND`.
- [x] Conservative paired variance and power-analysis artifacts archived.
- [x] `required_confirmatory_seed_count` compared with the frozen feasible cap.
- [x] Exact confirmatory seed list and method configs frozen.
- [x] Analysis plan complete; DNA Transform is the sole primary method-family
  hypothesis at alpha 0.05 and lossless DNA is a separate sanity check.
- [x] Confirmatory config and all currently relevant hashes recorded.
- [x] Team and supervisor approvals for the five enumerated Stage-1 decisions
  recorded below; confirmatory execution remains held.

## 16. Approval record

| Gate | Role | Name | Decision | Timestamp | Signature/reference |
| --- | --- | --- | --- | --- | --- |
| Pre-pilot | Research lead | Manh Quang | Approved | 2026-09-12 | Chat session approval; decision package v1.0 |
| Pre-pilot | Supervisor | Manh Quang | Approved | 2026-09-12 | Chat session approval; decision package v1.0 |
| Confirmatory freeze inputs | Research lead | Manh Quang | Approved | 2026-09-13 | Chat approval; amendment `2026-09-13_stage1_supervisor_approvals.md` |
| Confirmatory execution | Supervisor | Manh Quang | Approved | 2026-09-13 | Single execution; multiplicity amendment timestamped before outcomes |

`DRAFT` authorizes writing only. `PRE-PILOT FROZEN` authorizes the shared
development/calibration batch. The confirmatory seed/config snapshot is
immutable and the single confirmatory execution is authorized.
