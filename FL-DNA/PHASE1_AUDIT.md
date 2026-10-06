# FL-DNA Phase 1 Audit

## Scope

This audit checks the implementation used by the PaySim-style fraud experiments. It does not treat existing result files as proof of a new run. Results are separated into implementation observations, existing artifacts, and checks that remain blocked by the local environment.

## Implementation observations

| Area | Evidence | Finding | Impact |
|---|---|---|---|
| Data schema | `data/load_creditcard.py:_read_dataset` | The loader expects `step`, `type`, six balance/amount fields, and `isFraud`. `nameOrig`, `nameDest`, and `isFlaggedFraud` are not loaded. | Consistent with the intended PaySim-style feature set. |
| Features | `data/load_creditcard.py:_build_features` | `balance_diff_orig = oldbalanceOrg - newbalanceOrig` and `balance_diff_dest = newbalanceDest - oldbalanceDest` are computed after loading. | These are post-transaction balance features; the task should be described as post-transaction fraud classification. |
| Preprocessing | `data/load_creditcard.py:_fit_transform_features` | `RobustScaler` and `OneHotEncoder` are fit on the pooled training split and reused by all clients. | This is a centralized preprocessing assumption, not a fully decentralized preprocessing protocol. |
| Splits | `data/load_creditcard.py:load_creditcard_data` | Train/validation/test are stratified with the same seed; the sampled subset is stratified when `MAX_ROWS` is used. | The split is reproducible and label ratios are preserved, subject to the seed and source file. |
| Client partition | `data/load_creditcard.py:_mild_non_iid_client_indices` | Positive examples are distributed across clients, while negative examples are skewed by transaction type. | Mild type-based non-IID is implemented; client counts and fraud rates are reported in result metadata. |
| Loss | `experiments/fraud_fl_common.py:BinaryFocalLoss` | `alpha_t = alpha*y + (1-alpha)*(1-y)` is class-dependent. | The implementation is valid focal loss, but the manuscript must describe alpha as a positive-class weighting parameter rather than a common multiplier. |
| Optimizer | `experiments/fraud_fl_common.py:train_local_model` | Adam is recreated for each client on each round. | Optimizer state is not carried across rounds. This is a deliberate implementation detail that should be reported. |
| Aggregation | `experiments/fraud_fl_common.py:fed_avg` | Floating tensors in the model state, including BatchNorm buffers, are sample-count averaged; non-floating tensors use the first client value. | FedAvg behavior is explicit, but BatchNorm buffer handling should be treated as part of the protocol. |
| Lossless DNA | `dna_encoder/encoder.py`, `dna_encoder/binary_mapper.py` | Float32 bytes are mapped to binary and DNA symbols, encrypted, then restored and reshaped. | The transport path is intended to preserve IEEE-754 float32 bits. |
| DNA Transform | `dna_encoder/transform_defense.py:transform_update_array` | Each block is DNA-derived for deterministic seeding, then permuted, selectively attenuated, and mixed with the original block. | This is a lossy update transformation; it is not equivalent to lossless encoding and has no formal privacy guarantee. |
| Secure aggregation | `privacy/secure_agg.py:secure_aggregate_states` | Pairwise random masks cancel in the sum. | This is a numerical simulation of the Secure Aggregation threat model, not production cryptography. |
| Gradient inversion | `attacks/gradient_inversion.py` | The attack optimizes a dummy input against per-parameter gradients with a known label. | The current attack is sample-level, known-label, gradient-matching analysis; it is not an update-level attack over a complete FedAvg delta. |
| MPS boundary | `attacks/attack_runner.py` | Warm-up can use MPS, while NumPy conversion and inversion tensors are CPU-oriented. | Fixed by moving the model to CPU at the attack boundary and using `.cpu().numpy()` for defense diagnostics. |

## Added checks

`tests/test_phase1_invariants.py` covers:

- float32 bit preservation through the DNA encoder;
- identity behavior when transform `mix_ratio=0`;
- direct FedAvg arithmetic versus the SecureAgg simulation;
- identity reconstruction metric behavior, including SSIM and sign match.

## Existing artifact checks

The repository contains prior fraud results and inversion sweep artifacts, including 50-round utility files and attack summaries. These files were not overwritten by this audit. They are evidence of previous runs only; they are not regenerated in this environment.

## Phase 1 smoke and attack pilot

The smoke run used the same seed and split for all three methods: `MAX_ROWS=5000`, `FL_NUM_CLIENTS=3`, `NUM_ROUNDS=1`, `LOCAL_EPOCHS=1`, and `DATALOADER_NUM_WORKERS=0`. It completed CSV loading, preprocessing, local training, update protection, FedAvg, thresholded test evaluation, and JSON output.

| Method | Train samples/client | Fraud rate/client | Loss | F1 | ROC-AUC | PR-AUC | Protection time |
|---|---:|---|---:|---:|---:|---:|---:|
| FL Baseline | 1323 / 1187 / 740 | 0.001512 / 0.000842 / 0.001351 | 0.007760 | 0.000000 | 0.202000 | 0.001252 | -- |
| FL + DNA lossless | 1323 / 1187 / 740 | 0.001512 / 0.000842 / 0.001351 | 0.007760 | 0.000000 | 0.202000 | 0.001252 | 485.979 ms |
| FL + DNA Transform | 1323 / 1187 / 740 | 0.001512 / 0.000842 / 0.001351 | 0.007760 | 0.000000 | 0.194000 | 0.001239 | 245.921 ms |

These utility values are smoke diagnostics, not official results. One round and the small validation split are insufficient for a meaningful fraud-detection estimate. The lossless DNA row is numerically identical to baseline in this run, as expected.

The attack pilot used one validation sample, one warm-up round, five Adam iterations, and a separate output directory. Total runtime was 6.34 s. The raw-visible rows were:

| Method | MSE | PSNR | SSIM | Cosine | Observation |
|---|---:|---:|---:|---:|---|
| FL Baseline | 59.229404 | 12.441 | 0.043 | 0.312 | Raw per-sample gradient |
| FL + DNA lossless | 59.229404 | 12.441 | 0.043 | 0.312 | Lossless representation |
| FL + DP | 59.808018 | 12.400 | 0.036 | 0.245 | Clipping/noise |
| FL + DNA Transform | 59.232361 | 12.440 | 0.043 | 0.313 | Transformed gradient |

SecureAgg rows were marked `not_directly_applicable` under the true server-side threat model. `PreAggregationLeakage` rows are analysis-only upper bounds if an individual update leaks before aggregation. This one-sample pilot confirms that the diagnostic runs, but it does not establish privacy superiority and does not evaluate a complete client FedAvg update.

Smoke and pilot outputs are under `artifacts/phase1_smoke/`. The attack JSON records total and per-attack runtime; Transform metrics record protection time.

## Environment and reproducibility

The original repository virtualenv remains broken because its Python symlink points to `/Users/manhquang/.local/bin/python3.11`, which is absent. A separate `.venv-phase1` was created with Python 3.9.6 arm64 and compatible packages: `numpy==1.26.4`, `pandas==2.0.3`, `scikit-learn==1.3.2`, `torch==2.2.2`, `matplotlib==3.7.5`, `pycryptodome==3.20.0`, `pytest==8.3.5`, and `psutil==6.1.1`. PyTorch reported `mps_built=True` and `mps_available=True`.

The project manifest pins newer versions requiring a newer Python baseline, so the compatibility environment is separate and the manifest was not changed. Recreate it with:

```bash
cd FL-DNA
python3 -m venv .venv-phase1
.venv-phase1/bin/python -m pip install 'numpy==1.26.4' 'pandas==2.0.3' 'scikit-learn==1.3.2' 'torch==2.2.2' 'matplotlib==3.7.5' 'pycryptodome==3.20.0' 'pytest==8.3.5' 'psutil==6.1.1'
.venv-phase1/bin/python -m pytest -q tests/test_phase1_invariants.py
QUICK=1 MAX_ROWS=5000 NUM_ROUNDS=1 DATALOADER_NUM_WORKERS=0 BASELINE_OUTPUT_PATH=artifacts/phase1_smoke/baseline_metrics.json .venv-phase1/bin/python -m experiments.run_fraud_fl_baseline
QUICK=1 MAX_ROWS=5000 NUM_ROUNDS=1 DATALOADER_NUM_WORKERS=0 DNA_OUTPUT_PATH=artifacts/phase1_smoke/dna_metrics.json .venv-phase1/bin/python -m experiments.run_fraud_fl_dna
QUICK=1 MAX_ROWS=5000 NUM_ROUNDS=1 DATALOADER_NUM_WORKERS=0 DNA_TRANSFORM_OUTPUT_PATH=artifacts/phase1_smoke/transform_metrics.json .venv-phase1/bin/python -m experiments.run_fraud_fl_dna_transform
MAX_ROWS=5000 ATTACK_NUM_SAMPLES=1 ATTACK_WARMUP_ROUNDS=1 ATTACK_ITERATIONS=5 DATALOADER_NUM_WORKERS=0 ATTACK_OUTPUT_DIR=artifacts/phase1_smoke/attack .venv-phase1/bin/python -m attacks.attack_runner
```

## Phase 1 status

- **Implementation validation: PASS.** The environment, invariant tests, FL smoke path, DNA paths, artifact reload checks, objective decomposition, and metric edge cases execute successfully.
- **Diagnostic effectiveness: INCONCLUSIVE.** In the bounded pilot, gradient matching reduced its own objective and the fraud/non-fraud cases were exercised, but the two targets are not enough to establish a general reconstruction advantage over controls. The baseline and zero-gradient MSEs are mixed across settings.
- **Phase 1 closure: COMPLETE_WITH_LIMITATIONS.** The diagnostic package is sufficiently checked for the next development step, but it is not a complete FL privacy evaluation: the attack is known-label/sample-level, SecureAgg is simulated, the Transform attacker is not differentiable through the discrete transform, and no formal DP accountant is present.

## Draft next-round protocol

Use the same immutable dataset snapshot and recorded split, model state, preprocessing convention, and client partition across defenses. Capture a complete client model delta after one local epoch, define attacker knowledge before running the attack, and evaluate at least 10 matched target samples with fixed iterations/restarts. Report feature-space MSE/MAE and categorical accuracy separately from pseudo-image PSNR/SSIM. Reconstruction-success thresholds remain **TBD** until calibrated separately; do not select them using test records or reconstruction error. Record round, client, seed, model mode, labels-known status, batch size, optimizer-state assumptions, runtime, and all loss curves.

## Five attack-validation checks (updated)

The bounded runner is `experiments/run_phase1_attack_validation.py`. It writes to `artifacts/phase1_validation/` and does not modify official results.

1. **Baseline/pre-aggregation equivalence:** the same CPU model state, observed per-sample gradients, known label, dummy seed, and 10-iteration budget were used twice. Both trials used the real validation feature vector as the reconstruction target. Saved reconstructions and best losses were identical; the check passed. This verifies reproducibility of the current call path, but does not execute two distinct runner implementations.
2. **Artifact and best-candidate check:** original vectors, initial dummy vectors, and best reconstructions are saved as `.pt` files. The objective is checked against `best_attack_loss` before writing and again after loading the saved artifact. The closure pilot passed 102 in-memory checks and 102 reload checks. Twelve prior controls intentionally skip optimization checks.
3. **Budget, multi-start, and regularization pilot:** one fraud and one non-fraud validation target were evaluated at 0, 20, 100, and 300 Adam iterations, with three restarts and both `l2_weight=1e-4` and `l2_weight=0`. This produced 48 baseline and 48 zero-gradient trials. The seed depends on target and restart, not budget, so initialization is paired across budgets.
4. **No-update controls:** zero-gradient controls use the same model, label, initialization, budget, restart, and regularization setting as the corresponding baseline trials. Twelve unoptimized dummy priors are also recorded. The zero-gradient case means “fit a dummy input to an all-zero gradient”; it is not a universal no-update attacker.
5. **Paired DNA comparison:** lossless DNA executes an actual float32 encode/decode round trip before attack matching. Lossless and Transform trials use the same target and initialization seed for each target. Transform parameters are disclosed in metadata, but the optimizer still matches raw gradients to the transformed signal; this is not yet a differentiable, transform-aware attacker.

The closure report is `artifacts/phase1_validation/run_phase1_closure_v2/validation_report.json`. It records validation indices `[0, 576]`, labels `[0, 1]`, budgets `[0, 20, 100, 300]`, three restarts, two L2 settings, and 42.95 s total runtime. It contains 111 top-level records and 114 flattened attack calls: 48 baseline, 48 zero-gradient, 12 prior, 2 equivalence children, and 4 paired DNA children. All 102 optimized calls passed both candidate checks. Metric version `joint_range_pseudo_image_v2` uses a joint range, so constant-vector errors are not reported as perfect pseudo-image matches.
