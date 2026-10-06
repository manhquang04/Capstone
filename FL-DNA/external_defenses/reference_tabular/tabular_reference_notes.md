# TabLeak tabular reference branch

## Scope and outcome

This is a working **undefended FedSGD gradient inversion instrument**, not a reproduction of full project training. Three complete runs use official experiment 46 with batch eight, known labels, an initialized network, 30 ensemble members, 1500 Adam iterations each, learning rate 0.06, cosine gradient matching, gradient-sign updates, uniform initialization, categorical softmax, continuous sigmoid bounds, and median+softmax pooling. `perfect_pooling=False`; the attacker receives an empty shape-equivalent tensor instead of actual target feature values. The 30 restart candidates are selected/aligned by gradient objective, never by true reconstruction accuracy.

The official source is untouched. `run_reference.py` reads configuration 46 from `tableak/run_inversion_attacks.py` using Python AST; it bypasses only the Linux taskset/multiprocessing launcher to enforce one CPU process on macOS. Intra-op=2, inter-op=1, all BLAS/OpenMP environment limits=2; `DATALOADER_NUM_WORKERS=0` is set **before** importing the read-only project loader. No packages were installed. Existing `external_defenses/.venv` has PyTorch 2.5.1, NumPy 2.4.6, pandas 3.0.6, sklearn 1.9.1 (preflight measured).

## Published native target and measured results

Primary source read through literature: Vero, Balunović, Dimitrov, Vechev, *TabLeak: Tabular Data Leakage in Federated Learning*, ICML 2023, [arXiv:2210.01785](https://arxiv.org/abs/2210.01785), Table 1 (p.6) and Appendix A/B (p.12–13). **True-label Adult batch-eight TabLeak: 95.2 ± 8.8%**, over 50 random batches. Random baseline there is 53.9 ± 4.4%. This published mean is a reference, not a single-batch pass criterion. We run one batch under this published setting, not 50-batch statistics and not precisely their entire seed trajectory. No significance test or confidence interval is justified for one batch. Adult's genuine raw train/test files were already present in the clone; no download was needed.

Official metric: clamp continuous reconstructions to admissible bounds; decode Adult with official integer rounding and categorical argmax; Hungarian-align rows on the combined feature error; a categorical feature is correct if exact match, a numerical feature if within inclusive ±0.319 SD; accuracy = 100*(1-mean error). Adult global SD is computed from concatenated train/test as in the official code. The denominator is 14 mixed features, not 105 one-hot dimensions.

| Scenario | Combined accuracy % | Category accuracy % | Continuous accuracy % | Inversion runtime s | Positive labels in attacked batch |
|---|---:|---:|---:|---:|---:|
| adult_native | 95.535714 | 98.437500 | 91.666667 | 87.277 | 5 |
| paysim_official_fc | 25.000000 | 62.500000 | 20.312500 | 68.748 | 0 |
| paysim_project_eval | 34.722222 | 62.500000 | 31.250000 | 91.143 | 0 |


All displayed measurements come from `run_reference.py` end-to-end, frozen `arrays.npz`/`result.json`, and fresh-process `verify_reference.py`. Official scores and independent cost-matrix/Hungarian scores agree. Times cover inversion, not dataset preparation.

## Real PaySim protocol and limitations

Read-only input: `datasets/creditcard.csv` (the real PaySim-schema file, not the similarly named European credit-card dataset). `data/load_creditcard.py:65–166` loads a seed-42 stratified 100,000-row sample from the full CSV, splits 65/15/20%, fits RobustScaler/one-hot on train only, and partitions three clients. We sample eight random records with replacement from client 0. Numeric columns are step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, balance_diff_orig, balance_diff_dest. Type is one mixed categorical feature represented by five one-hot dimensions: 9 mixed features, 13 encoded dimensions. IDs nameOrig/nameDest and isFlaggedFraud are excluded by the project loader.

`PaySimAdapter` preserves project robust scaling (center/scale treated as the official adapter's affine parameters), derives sigmoid bounds and tolerance SD from pooled training data only, and does not re-standardize the network input. It uses the unchanged official gradient inversion and official Hungarian metric, with **raw-money continuous decode, no integer rounding**. This adaptation is explicit: Adult's default integer decode would corrupt scaled or monetary data. Numeric tolerance is 0.319 raw training SD, **not 0.319 RobustScaler IQR**. Pooled training feature bounds/statistics/categories are assumed server-known auxiliary information.

PaySim official-FC is a dataset-transfer control using the published FC(100,100,2)/cross-entropy, not the project's classifier. PaySim project-eval uses the real `FraudMLP` class unchanged but deterministic `.eval()` mode and weighted BCE. The project default is focal loss; this branch does not claim to test that default. Evaluation mode freezes BN running statistics and disables dropout; it is an instrument sanity check, **not train-mode/FedAvg reconstruction**. Batch-zero initialization is not a trained checkpoint. One batch cannot establish general reconstruction quality across time steps, transaction types, fraud cases, clients, model states or defenses. Heavy-tailed monetary SD tolerance and eight of nine continuous/derived fields can make coarse accuracy look favorable while raw monetary error or constraint violation remains large. Derived balance differences are optimized independently by official TabLeak; no accounting-identity prior is added.

## Concrete differences: official vs project

| Dimension | Published/official branch | Actual project | Concrete source locations |
|---|---|---|---|
| Input schema | Adult: 14 mixed / 105 encoded, 8 categorical + 6 numerical | PaySim: 8 numerical + type, 13 encoded, two derived differences, identifiers excluded | `tableak/datasets/adult.py:16–51,75–88`; `data/load_creditcard.py:24–43,263–303` |
| Scaling | mean/SD of joint train+test, including one-hot columns | train-only RobustScaler median/IQR for numeric; one-hot not standardized | `tableak/datasets/base_dataset.py:106–139`; `data/load_creditcard.py:279–310` |
| Architecture | FC 105→100→100→2, ReLU, no BN/dropout | FC 13→128→64→32→1, BN after hidden linears, dropout 0.3/0.2/0.1 | `tableak/models/fully_connected.py:7–49`; `models/fraud_mlp.py:14–36` |
| Classification loss | two-logit cross entropy, known true integer labels | one-logit default focal (alpha=.95,gamma=2), optional weighted BCE | `tableak/single_inversion_fedsgd.py:44–56`; `experiments/fraud_fl_common.py:36–38,51–69,110–116` |
| Training/observation | initialized weights, one raw FedSGD minibatch gradient | train mode, Adam lr=.001, local epochs, batch1024; not just one raw gradient | `tableak/run_inversion_attacks.py:377–383`; `experiments/fraud_fl_common.py:33–35,82–107` |
| Randomness/state | deterministic ReLU forward | fresh dropout masks; BN batch-dependent and running-state updates | `models/fraud_mlp.py:17–30`; `experiments/fraud_fl_common.py:90–103` |
| Inversion | 30 x 1500 sign-Adam/cosine, structured softmax/sigmoid, median ensemble | not substituted by project proxy; official code imported unchanged | `tableak/run_inversion_attacks.py:243–264`; `tableak/attacks/gradient_inversion_attack.py:65–143,666–836` |
| Metric | mixed-feature correctness, tolerance .319 SD, Hungarian matching | project's separate metrics flatten per-record normalized errors, category argmax and balance consistency | `tableak/utils/matching.py:29–50`; `tableak/utils/eval_metrics.py:68–105`; `attacks/tabular_metrics.py:10–80` |

Paths beginning `tableak/` above are relative to `external_defenses/`; project paths are root-relative. These differences mean that a successful native run must not be reported as proof that a project train-mode FedAvg attack is valid.

## Same-input gradient replay diagnostic


On the actual project model and actual attacked PaySim inputs, five train-mode gradient replays with unchanged inputs/labels had cosine similarities to the first gradient: `[0.37854623794555664, 0.24451710283756256, 0.39104539155960083, 0.37306272983551025, 0.3161936402320862]`. Evaluation mode same-input cosine: `1.0`; maximum absolute gradient difference: `0.0`. This measured mismatch is due to the stochastic training-forward setting, not imperfect input reconstruction. Exact dropout masks and BN semantics/state must be modeled or controlled before applying a deterministic gradient objective to the actual training gradient.


## Compatibility, blocked cases and preserved attempts

1. Job `c8854425-0b5`, exit 1: pre-run JSON serialization failed with `TypeError: Object of type float32 is not JSON serializable`. Preserved `failed_preflight.log`, `runs/default_batch8_seed42/` (metadata and inputs). Fixed only the driver by converting scalar metadata to Python float.
2. Job `3537d002-c76`, exit 1: official `utils/differentiable_bounds.py:30` final leaf in-place update failed with `RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation.` This is the README's documented modern-PyTorch incompatibility. Preserved `failed_torch_leaf.log`, `runs/default_batch8_seed42_v2/`. No upstream source modification or package downgrade. The driver locally wraps `attacks.gradient_inversion_attack.continuous_sigmoid_bound` by cloning its input tensor before calling the identical official transform; values and autograd preserved. This is an explicit PyTorch 2.x API compatibility adaptation, not a replacement inversion algorithm.
3. Completed measured job `7d8cdb2e-8df` uses `runs/default_batch8_seed42_v3/`; older directories are retained. Full official default ensemble/iterations were run, not just inspected or shortened.
4. Full 50-batch Monte Carlo reproduction, project train-mode/stochastic FedAvg, project default focal criterion and full-data PaySim generalization are **not evaluated**. These are scope limitations, not synthetic/placeholder results.

## Verification and commands

Run from FL-DNA root:
```sh
env PYTHONDONTWRITEBYTECODE=1 DATALOADER_NUM_WORKERS=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 external_defenses/.venv/bin/python -u external_defenses/reference_tabular/run_reference.py --run-name default_batch8_seed42_v3 > external_defenses/reference_tabular/run.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 external_defenses/.venv/bin/python external_defenses/reference_tabular/verify_reference.py
PYTHONDONTWRITEBYTECODE=1 external_defenses/.venv/bin/python external_defenses/reference_tabular/write_notes.py
```

For a new inversion run, choose a new `--run-name` (the script rejects existing output directories). Verification is fresh-process and checks final arrays for finite values, batch shapes, 30 ensemble members, objective count, config, officially recomputed all/category/continuous and per-feature accuracies, independently recomputed assignment accuracy, upstream commit, unchanged upstream source and relevant project source SHA-256 hashes. `verification.json` records raw PaySim numeric errors and derived-balance consistency errors so the tolerance score is not mistaken for exact transaction reconstruction.


Official commit: `6b8c1c82ffe85bc6fdd96938ac42b6c9b5685300`. Project commit at verification: `7df4f3add9ed17d6d5661b08d8aee8eea0c0142d`. Official tracked diff: empty.


## PaySim reparameterization diagnostic and retained best instrument

Default Adult transfer was poor on real PaySim: 25% official FC and 34.7222% project eval, with very large raw monetary errors. A supplementary `run_paysim_ablation.py` reloads the **same saved true batch, model checkpoint and criterion**, keeps official 30×1500 iterations, softmax and median pooling, and disables only the official sigmoid option. Restart RNG is explicitly reset to 42; the initial random stream is not paired to the original run's post-model-initialization/post-replay RNG state. Thus this is evidence for a useful setting and **consistent with** adverse wide-bound sigmoid conditioning, not a perfectly randomized one-variable causal experiment. Native/default artifacts remain unchanged; no published-default success is claimed for this variant.

| Same real PaySim batch/checkpoint, sigmoid disabled | Combined % | Category % | Continuous % | Runtime s |
|---|---:|---:|---:|---:|
| paysim_official_fc | 83.333333 | 87.500000 | 82.812500 | 51.474 |
| paysim_project_eval | 76.388889 | 87.500000 | 75.000000 | 82.418 |


Both attacked PaySim batches contain **zero fraud-positive examples**. The result establishes recovery on these benign transactions only, not fraud-case leakage. The no-sigmoid variant is the better measured undefended PaySim instrument on this same real batch; preserved beside, not over, the default outputs. The metadata/source checkpoint remains under `runs/default_batch8_seed42_v3/<scenario>/`, and variant arrays/config/results are under `runs/paysim_no_sigmoid/<scenario>/`. No monetary rounding or accounting-identity enforcement is introduced.

Supplementary job `045d7775-d16` completed, exit 0. Fresh-process verification recomputed both variants using official and independent scores, exit 0:
```sh
env PYTHONDONTWRITEBYTECODE=1 DATALOADER_NUM_WORKERS=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 external_defenses/.venv/bin/python -u external_defenses/reference_tabular/run_paysim_ablation.py > external_defenses/reference_tabular/ablation.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 external_defenses/.venv/bin/python external_defenses/reference_tabular/verify_reference.py --run-name paysim_no_sigmoid --scenarios paysim_official_fc paysim_project_eval --output-name verification_ablation.json
```

## Raw PaySim errors (official Hungarian-aligned, no unit mixing)

These errors distinguish tolerance-based feature recovery from exact transaction recovery. Raw amount and balances are in the CSV's simulated monetary units; step is a simulation time-step index. They are computed by `verify_reference.py` from final saved arrays. Do not combine step and money into a single physical-unit MAE.

| Scenario | Setting | Amount MAE (raw) | Origin derived consistency MAE (raw) | Destination derived consistency MAE (raw) |
|---|---|---:|---:|---:|
| paysim_official_fc | default_sigmoid | 19026305.719850 | 5801778.652500 | 22475893.048375 |
| paysim_project_eval | default_sigmoid | 18198768.516100 | 13521457.839002 | 25151117.858238 |
| paysim_official_fc | no_sigmoid | 496202.447920 | 25621.285900 | 209699.152786 |
| paysim_project_eval | no_sigmoid | 235700.974931 | 76645.711688 | 581166.970437 |

Verification and note generation commands above all exit 0 on the final scripts/files. No lead report, upstream source, project data/model, or environment package was edited.
