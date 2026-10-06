# Priority 32b: BN variance reconciliation

Status: COMPLETED. Interim sections are retained chronologically; final findings below govern. P33 was deferred throughout this reconciliation.

Frozen diagnosis amendment:
`protocols/amendments/2026-10-02_priority32b_bn_variance_reconciliation.md`.
All original artifacts, reports and source runners remain unchanged.

## Static pipeline comparison

| Factor | Registered PaySim RQ2 | Priority 32 original full-state | Evidence |
|---|---|---|---|
| Raw data | Same PaySim CSV, isFraud | Same | data/load_creditcard.py; P32 RAW mapping |
| Cap placement | Stratified 500k BEFORE split | Split full dataset, stratified cap WITHIN partitions | _read_dataset vs capped_parts |
| Cap RNG | Per-run FL_RUN_SEED | Fixed 320032 (+ partition index) | Same functions |
| Split RNG | Per-run FL_RUN_SEED | Fixed 320032, reused by all training seeds | loader vs prepare_dataset |
| Split sizes | Sequential .35 then .20/.35, rounding may produce 74999/100001 | Selected counts 325000/75000/100000 | sklearn splitting vs proportional cap |
| Imputation scope | Numeric medians before split | Train-only medians | _build_features vs preprocess |
| Scaling | Train-fitted RobustScaler | Train-fitted RobustScaler | _fit_transform_features vs preprocess |
| Categorical encoding | Train-fitted OneHotEncoder, ignore unknown | Train-fitted manual one-hot, unknown all-zero | Same five type categories |
| Features | Same 8 numeric +5 type columns, d=13 | Same | stored metrics/preprocessing.json |
| Client partition rule | Same mild non-IID helper: fraud split evenly, nonfraud type55% primary | Same helper | _mild_non_iid_client_indices |
| Partition RNG | Run seed, seed-dependent data | Run seed, fixed data | Same helper inputs differ |
| Model | FraudMLP 128/64/32, three BN, dropout .3/.2/.1 | Same class | models/fraud_mlp.py |
| Device | Native automatic MPS if available, otherwise CPU | Explicit CPU | fraud_fl_common._get_device; P32 job_config |
| Optimizer/loss | Fresh Adam each client-round, lr.001, focal alpha.95 gamma2 | Same | train_local_model vs train_job |
| Training numerical guard | No nonfinite-loss guard in train_local_model | Raises on nonfinite local loss | fraud_fl_common.train_local_model vs P32 train_job |
| v1/v2 hyperparameters | v1 block256/mix.08/keep.88/shrink.45; v2 ratio.95/eta.01 | Same | stored configs and frozen P32 configuration |
| Dtypes | Numeric scaling arithmetic float64; model inputs/state float32; labels float32 | Same model/scaler arithmetic; cached labels float32 | scaler paths and cache construction |
| Budget | 50 rounds, K3, epoch1, batch1024 | Same | frozen configs and P32 |
| Train loader | Seed+client private generators, shuffle, four persistent workers by default | Same generators/shuffle, zero workers | _create_loader vs train_job |
| Transform scope/order | All floating state deltas including BN; transform each client, then weighted FedAvg | Same | dna_transform_state / dna_transform_v2_state |
| v1 root seed label | rq2-dna-transform | Same | registered runner vs train_job |
| v2 root seed label | dna-transform-v2 | rq2-dna-transform-v2 | registered v2 run constant vs train_job |
| Client-round/quantization seeds | Same derivation downstream of root | Same downstream; v2 root differs | derive_seed call sites |
| Global-state snapshot | Nonmutated live state_dict view | deepcopy | Equivalent while clients are trained on copies |
| Floating aggregation | Same sample-count weighted fed_avg; integer buffers copied | Same | fraud_fl_common.fed_avg |
| Evaluation cadence | Validation and test every round | Only after round50 | main vs train_job |
| Evaluation mode/batch | model.eval(), loader batch1024 | model.eval(), array blocks4096 | evaluate_model vs probabilities |
| Threshold | Same tune_threshold, validation F1 only | Same | imported common function |
| Score errors | AUC/PR ValueError caught as null; may still write SUCCESS | Explicit metrics exception prevents completed JSON | evaluate_model vs metrics/valid_result |
| Seed family | Frozen v1 n21 / v2 n52 31-bit hashed seeds | 321000–321020; now replayed on registered seeds | frozen manifests |
| Metric names | f1_score/optimal_threshold, full confusion table, all rounds | f1/threshold, final-round scores only | metrics schema |

Data-bundle swaps are coupled changes. Their decomposition and no-op handling
are fixed in the amendment; no single cause is inferred from this table.

## Original stored-output audit (read-only)

All 21 v1 and 52 v2 transform files exist. None has null AUC in any stored
round or in its final round. This does not prove all intermediate running
variances were nonnegative, since those states were not recorded originally.
The claim of a completed process is weaker than finite model-output evidence;
this diagnosis explicitly records both logits and probabilities.

Read-only raw numeric audit found 6,362,620 rows and zero missing values in
all six raw numeric columns. The two balance differences inherit no missing
values; median-imputation scope is therefore an algebraic no-op for this data.
Source/data/prepared-cache/stored-metric SHA-256 evidence (99 files) is in
`artifacts/priority32b_bn_reconciliation/source_provenance.json`.

## Frozen replays

v1: 1984441809,653660776,650446433. v2: 94400035,1530452029,1646028911.
First three seeds in frozen manifest order, not selected by outcomes.
39 fixed jobs; up to12 predeclared data-decomposition jobs if triggered.
Torch threads1, at most4 concurrent processes. Hooks do not add forwards,
loader iterations or RNG draws; they copy/read actual evaluation tensors.

## Commands and disclosures

```sh
.venv-phase1/bin/python -B -u experiments/priority32b_bn_reconcile.py --supervise
git diff --check
```

No replays launched before the frozen amendment. Per-job outputs, exceptions,
round BN minima, probability arrays, final state, exact subprocess commands,
progress and source/config hashes are saved in
`artifacts/priority32b_bn_reconciliation/`. Final conclusions, all run outcomes,
SHA-256 manifest and final checks will be added after the diagnostic queue.

Additional read-only audit command:
`.venv-phase1/bin/python -B experiments/audit_priority32b_provenance.py`.
Detached supervisor PID2263 started 2026-10-02T14:04:11Z. Four registered jobs
are active; no completed job yet at the first observation. The registered v1
seed1984441809 replay has negative aggregate `network.1.running_var` at round22
(minimum -0.2127340138), while recorded round24 test logits/probabilities remain
finite. These are interim observed tensors, not a reproduction verdict or proof
of undistorted outputs. Final metric matching and paired P32 diagnostics remain
pending. Earlier artifacts and original sources have not been modified.

## Interim completed replay evidence (not final reconciliation)

At 2026-10-02T14:11:44Z, six diagnostic jobs are complete. An independent
analysis recomputed every stored/replay core-metric difference from the original
JSON files, agreeing with the instrumentation runner. Registered replay metrics
are bit-identical across all 50 rounds in the four completed registered jobs:

| Pipeline | Method | Seed | First negative aggregate BN round | Final test probabilities finite | Stored metrics max absolute error |
|---|---|---:|---:|---|---:|
| Registered native MPS | v1 | 1984441809 | 11 | Yes | 0 |
| Registered native MPS | v1 | 650446433 | 5 | Yes | 0 |
| Registered native MPS | v1 | 653660776 | 7 | Yes | 0 |
| Registered native MPS | v2 | 94400035 | 2 | Yes | 0 |
| P32 CPU | v1 | 1984441809 | 2 | No | Not a registered-metric replay |
| P32 CPU | v1 | 653660776 | 5 | No | Not a registered-metric replay |

Thus "successful original job" does not establish nonnegative BN variance.
These observations demonstrate numerically reproducible original metrics despite
invalid BN variance states in the selected jobs; they do not yet isolate which
factor permits finite output or quantify output distortion. Remaining registered,
P32 and one-factor jobs are required before the conclusion. Diagnostic P32 jobs
save nonfinite output evidence as completed *diagnostics*, not successful RQ2
utility runs. No scores are imputed or NaNs converted into utility metrics.
Independent analysis command:
`.venv-phase1/bin/python -B experiments/analyze_priority32b_bn_reconciliation.py`.
The timestamped analysis snapshot is retained under the new artifacts namespace.

## Completed registered/P32 replay stage and backend diagnosis

All six registered replay core metrics reproduce bit-for-bit for all50 rounds;
all600 recorded validation/test evaluations have finite logits/probabilities.
All six have at least one negative aggregate BN variance round. Final checkpoint
BN variance is still negative in five; v1 seed650446433 has nonnegative final
variance despite earlier negative rounds. The remaining registered rows are:

| Pipeline | Method | Seed | First negative BN round | Final test finite | Stored metric max error |
|---|---|---:|---:|---|---:|
| Registered MPS | v2 | 1530452029 | 1 | Yes | 0 |
| Registered MPS | v2 | 1646028911 | 1 | Yes | 0 |
| P32 CPU | v1 | 650446433 | 4 | Yes | Not applicable |
| P32 CPU | v2 | 94400035 | 1 | No | Not applicable |
| P32 CPU | v2 | 1530452029 | 1 | No | Not applicable |
| P32 CPU | v2 | 1646028911 | 1 | No | Not applicable |

All six P32 diagnostics reached50 rounds without modifying/clamping any BN
state. Five have nonfinite final probabilities; the one finite final job still
has earlier negative variance. P32 original missing-output behavior and these
diagnostic retained NaN arrays are distinguished deliberately.

Additional diagnosis amendments were written before their inference-only runs:
`2026-10-02_priority32b_backend_probe_diagnosis.md` and
`2026-10-02_priority32b_activation_probe_diagnosis.md`. They do not change the
39/51 training-job design. Both evaluate ALL six unchanged final checkpoints
on identical first1024 P32 validation inputs, restore constructor CPU RNG,
never correct any tensor, and save results in separate new namespaces.

| Registered final checkpoint | CPU finite logits /1024 | MPS finite logits /1024 | Final BN state |
|---|---:|---:|---|
| v1/1984441809 | 0 | 1024 | Negative variance |
| v1/653660776 | 0 | 1024 | Negative variance |
| v1/650446433 | 1024 | 1024 | Nonnegative variance |
| v2/94400035 | 0 | 1024 | Negative variance |
| v2/1530452029 | 0 | 1024 | Negative variance |
| v2/1646028911 | 0 | 1024 | Negative variance |

The nonnegative final checkpoint's maximum finite CPU/MPS logit difference is
2.86102294921875e-6. For the other five there is no finite CPU/MPS pair from
which a meaningful numerical output difference can be computed.

The isolated eval-mode BN operator with running_var[-.25,0,.25] produces NaN
in the negative-variance coordinate on BOTH backends, agreeing with the
defining normalization denominator sqrt(var+epsilon). Fixed isolated ReLU
on [NaN,-1,0,1] gives [NaN,0,0,1] on CPU and [0,0,0,1] on this MPS environment.
Read-only layer hooks in the actual checkpoints establish the same path:
v1/1984441809 first BN emits1024 NaNs on each device; the following MPS ReLU
has0 NaNs while CPU ReLU retains1024, followed by full NaN propagation through
the CPU linear layer. Parameter/buffer equality and CPU RNG equality pass.
Thus the finite outputs are not evidence of valid BN: this backend's activation
semantics suppress invalid normalized channels. This is directly measured in
the local environment, not a general claim about all MPS/PyTorch versions.

The registered numbers are genuine reproduced backend-specific endpoints,
not fabricated measurements. They cannot be certified as ordinary valid-BN,
backend-independent full-state utility evidence: at least five selected FINAL
models and all six selected INTERMEDIATE histories are affected. The one
nonnegative final checkpoint is not declared final-invalid solely because its
earlier evaluations were affected. Neither these six replays nor stored null
audits establish the fraction affected across all21v1/52v2 seeds. A numerical
utility bias relative to a valid-BN counterfactual is not identified by these
probes; no such training experiment is substituted for the registered runs.
Final one-factor causal reconciliation remains pending.

Commands:
`.venv-phase1/bin/python -B experiments/probe_priority32b_bn_backend.py`;
`.venv-phase1/bin/python -B experiments/probe_priority32b_activation_path.py`.

Supplemental read-only transform dependency/environment fingerprint command:
`.venv-phase1/bin/python -B experiments/audit_priority32b_transform_dependencies.py`.
Hashes of the actual DNA transform modules and binary/DNA mapping dependencies
are retained in transform_dependency_provenance.json; these immutable modules
were not changed. The additional fingerprint was collected during the queue,
not falsely described as a pre-launch measurement.

## Completed one-factor stages (remaining swaps still in progress)

At snapshot20261002T142608872780Z, all six registered-data swaps and all six
native-device swaps are complete. Neither changes whether negative aggregate
variance occurs at least once: all12 have negative rounds. Therefore the frozen
conditional data-decomposition trigger has not fired for these seeds; cap/split
components are not independently assigned causal effects. Registered-data
swaps on CPU have finite final outputs for3/3v1 and0/3v2, compared with1/3v1
and0/3v2 for the P32 full-state baseline. Data changes final-state persistence,
but does not remove the underlying variance-domain violation. Device-only MPS
swaps have finite final outputs for6/6 despite negative variance histories.
The same-checkpoint probes above isolate inference semantics more cleanly than
device-only TRAINING swaps, which also alter numerical training trajectories.

All six evaluation-cadence swap final checkpoints are byte-identical (SHA-256)
to their corresponding P32 full-state checkpoints. Evaluation-every-round vs
final-only does not cause a training-trajectory difference in these fixed
seed cases. Read-only check command:
`shasum -a 256 artifacts/priority32b_bn_reconciliation/jobs/{p32_full,evaluation_cadence}/{v1,v2}/seed_*/final_state.pt`.

## Final diagnostic queue and one-factor reconciliation

All 39 frozen diagnostic jobs finished. No early stopping or scientific tuning was used.

| Factor | Method | Seed | First negative BN round | Negative rounds | Final validation finite | Final test finite | Runtime seconds |
|---|---|---:|---:|---:|---|---|---:|
| evaluation_cadence | v1 | 1984441809 | 2 | 13 | False | False | 145.737 |
| evaluation_cadence | v1 | 650446433 | 4 | 22 | True | True | 146.353 |
| evaluation_cadence | v1 | 653660776 | 5 | 18 | False | False | 166.727 |
| evaluation_cadence | v2 | 1530452029 | 1 | 45 | False | False | 170.626 |
| evaluation_cadence | v2 | 1646028911 | 1 | 44 | False | False | 145.406 |
| evaluation_cadence | v2 | 94400035 | 1 | 48 | False | False | 141.077 |
| loader_workers | v1 | 1984441809 | 7 | 13 | False | False | 91.765 |
| loader_workers | v1 | 650446433 | 3 | 25 | False | False | 95.236 |
| loader_workers | v1 | 653660776 | 4 | 20 | False | False | 92.669 |
| loader_workers | v2 | 1530452029 | 1 | 47 | False | False | 88.794 |
| loader_workers | v2 | 1646028911 | 1 | 44 | False | False | 85.891 |
| loader_workers | v2 | 94400035 | 1 | 47 | False | False | 93.460 |
| native_device | v1 | 1984441809 | 5 | 19 | True | True | 260.763 |
| native_device | v1 | 650446433 | 10 | 18 | True | True | 272.204 |
| native_device | v1 | 653660776 | 9 | 15 | True | True | 260.544 |
| native_device | v2 | 1530452029 | 1 | 42 | True | True | 248.355 |
| native_device | v2 | 1646028911 | 1 | 48 | True | True | 249.390 |
| native_device | v2 | 94400035 | 1 | 47 | True | True | 253.568 |
| p32_full | v1 | 1984441809 | 2 | 13 | False | False | 108.138 |
| p32_full | v1 | 650446433 | 4 | 22 | True | True | 124.129 |
| p32_full | v1 | 653660776 | 5 | 18 | False | False | 107.343 |
| p32_full | v2 | 1530452029 | 1 | 45 | False | False | 106.755 |
| p32_full | v2 | 1646028911 | 1 | 44 | False | False | 107.899 |
| p32_full | v2 | 94400035 | 1 | 48 | False | False | 106.358 |
| registered | v1 | 1984441809 | 11 | 19 | True | True | 331.715 |
| registered | v1 | 650446433 | 5 | 27 | True | True | 329.760 |
| registered | v1 | 653660776 | 7 | 11 | True | True | 336.878 |
| registered | v2 | 1530452029 | 1 | 47 | True | True | 283.085 |
| registered | v2 | 1646028911 | 1 | 46 | True | True | 276.563 |
| registered | v2 | 94400035 | 2 | 46 | True | True | 334.445 |
| registered_data | v1 | 1984441809 | 2 | 21 | True | True | 129.736 |
| registered_data | v1 | 650446433 | 3 | 23 | True | True | 123.140 |
| registered_data | v1 | 653660776 | 9 | 15 | True | True | 131.648 |
| registered_data | v2 | 1530452029 | 1 | 49 | False | False | 149.407 |
| registered_data | v2 | 1646028911 | 1 | 37 | False | False | 131.993 |
| registered_data | v2 | 94400035 | 1 | 46 | False | False | 115.849 |
| transform_seed_label | v2 | 1530452029 | 1 | 46 | False | False | 120.836 |
| transform_seed_label | v2 | 1646028911 | 1 | 43 | False | False | 99.452 |
| transform_seed_label | v2 | 94400035 | 1 | 49 | False | False | 104.888 |

One-factor contrasts are conditional on the six specified seed/method pairs, not a population causal estimate.

| Factor | Completed jobs | Negative variance ever | Nonfinite final test | Changed negative-presence vs P32 |
|---|---:|---:|---:|---:|
| evaluation_cadence | 6 | 6 | 5 | 0 |
| loader_workers | 6 | 6 | 6 | 0 |
| native_device | 6 | 6 | 0 | 0 |
| p32_full | 6 | 6 | 5 | Not a swap |
| registered | 6 | 6 | 0 | Not a swap |
| registered_data | 6 | 6 | 3 | 0 |
| transform_seed_label | 3 | 3 | 3 | 0 |

The source of invalid variance is the unconstrained full-floating-state transform of BN variance deltas before FedAvg; neither transform enforces a nonnegative reconstructed running_var. The registered pipeline ALSO has this failure, so a difference in dataset preprocessing is not required to explain its presence. Data/seed/trajectory may change its onset or final persistence. Backend inference probes isolate the finite-output discrepancy independently of those coupled training factors.

Plain conclusion: the selected stored metrics are exactly reproducible, but successful jobs and finite endpoints masked invalid BN channels. The five final-negative selected registered models are not valid ordinary-BN utility evidence; MPS ReLU suppresses NaNs into zeros. All six selected intermediate evaluation histories were affected. The one nonnegative final checkpoint has valid-domain final inference in the cross-backend probe. This cannot certify or reject every unreplayed final endpoint across 21 v1 and 52 v2 runs. The original aggregate RQ2 claims must carry this numerical-validity caveat; the observed failure is not repaired by reinterpreting the separate trainable-only/raw-BN P32 variant as the registered experiment. No valid-BN counterfactual utility effect size is identified here.

Every job command and result SHA is in runs.jsonl; every per-layer, per-round client/aggregate variance is in bn_rounds.jsonl. Full evaluation probability arrays and final checkpoints are retained. There were no diagnostic training interruptions or scientific reruns if all frozen round-count gates above pass. Additional inference-only probes were preregistered after observing finite-on-negative states, disclosed above, and did not modify frozen training or earlier artifacts.

Evaluation cadence is excluded as a trajectory-changing cause on these six seeds: all six final_state.pt SHA-256 values are byte-identical to their P32 full-state counterparts. This is stronger than merely equal endpoint metrics.

Finalization command: `.venv-phase1/bin/python -B experiments/finalize_priority32b_bn_reconciliation.py`.
