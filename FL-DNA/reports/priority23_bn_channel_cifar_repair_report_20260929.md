# Priority 23 — BN channel, repaired CIFAR-10 attack, DP recalibration, and RQ2 facts

Date: 2026-09-29  
Scope: supplementary evidence only. No `Latex/` files or pre-existing artifacts/reports were modified.

## Commands run

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority23_bn_rq2_analysis.py \
  --output-dir artifacts/priority23_bn_channel_rq2_20260929_rq2maxrows

PYTHONPATH=. .venv-phase1/bin/python experiments/priority23_repaired_cifar_attack.py \
  --output artifacts/priority23_cifar_repaired_20260928 \
  --groups 39 --restarts 4 --resume

.venv-phase1/bin/python -m py_compile \
  experiments/priority23_bn_rq2_analysis.py \
  experiments/priority23_repaired_cifar_attack.py

git diff --check
```

Note: an earlier Part-B run was interrupted after writing only `execution_manifest.json` and before any summary artifact existed. I then added per-group checkpoint/resume output to the runner and resumed. This was a technical robustness change; it did not change the frozen target bundle, branches, budgets, objective, seeds, restarts, or metrics.

## Part A — BN-statistics leakage channel

Target bundle:

`artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_targets_20260927/paysim_priority16_v1_medium_confirmatory_targets.pt`

The recovery uses the first BatchNorm running-mean update:

`batch_mean_z = running_mean_before + delta_running_mean / momentum`, then `W1 * mean_x = batch_mean_z - b1`.

### BN mean recovery vs population-mean baseline

Raw MSE is on the robust-scaled feature vector. Standardized MSE divides each feature error by its empirical RQ2 training-set feature standard deviation (`max_rows=500000`, seed 42).

| Defense | Raw wins vs population | Raw p-value | Median raw recovered MSE | Median raw population MSE | Std wins vs population | Std p-value | Median std recovered MSE | Median std population MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 39/39 | 1.818989e-12 | 4.891e-13 | 14.7925 | 39/39 | 1.818989e-12 | 8.656e-13 | 0.1845 |
| v1 conservative | 39/39 | 1.818989e-12 | 0.1245 | 14.7925 | 27/39 | 0.01185 | 0.0909 | 0.1845 |
| v1 medium | 39/39 | 1.818989e-12 | 0.1940 | 14.7925 | 21/39 | 0.37463 | 0.1441 | 0.1845 |
| v1 stronger | 39/39 | 1.818989e-12 | 0.2785 | 14.7925 | 18/39 | 0.73880 | 0.2020 | 0.1845 |
| v2 0.95/0.01 decoded | 39/39 | 1.818989e-12 | 0.0520 | 14.7925 | 2/39 | ~1.0 | 0.6307 | 0.1845 |
| DP 0.000315 | 39/39 | 1.818989e-12 | 0.0396 | 14.7925 | 19/39 | 0.62537 | 0.3410 | 0.1845 |
| DP 0.0004 | 39/39 | 1.818989e-12 | 0.0602 | 14.7925 | 13/39 | 0.98815 | 0.4211 | 0.1845 |
| DP 0.00105 | 39/39 | 1.818989e-12 | 0.4072 | 14.7925 | 0/39 | 1.0 | 3.5891 | 0.1845 |
| DP recalibrated v1 conservative | 39/39 | 1.818989e-12 | 3.555e-07 | 14.7925 | 39/39 | 1.818989e-12 | 6.050e-06 | 0.1845 |
| DP recalibrated v1 medium | 39/39 | 1.818989e-12 | 5.546e-07 | 14.7925 | 39/39 | 1.818989e-12 | 6.678e-06 | 0.1845 |
| DP recalibrated v1 stronger | 39/39 | 1.818989e-12 | 7.332e-07 | 14.7925 | 39/39 | 1.818989e-12 | 1.589e-05 | 0.1845 |
| DP recalibrated v2 0.95/0.01 | 39/39 | 1.818989e-12 | 1.631e-06 | 14.7925 | 39/39 | 1.818989e-12 | 2.321e-05 | 0.1845 |

`running_var` check: the transmitted `running_var` gives first-layer activation variance statistics. It does not uniquely identify input-feature second moments without additional covariance assumptions.

Realistic RQ2-style full-client descriptive check (`max_rows=500000`, seed 98089969, client 0, one local epoch, batch 1024): the recovered vector is an EMA effective feature mean, not the unweighted full-client mean. It used 132,169 client samples / 130 batches. MSE vs full-client mean: `1101.18`; vs last batch mean: `1119.20`; vs last-5-batch mean: `1089.07`.

Verdict: the implemented BN channel leaks target batch mean information under raw robust-scaled scoring for none, v1, v2, and DP. Under per-feature-standardized scoring, the strongest evidence remains for none, v1-conservative, and recalibrated/very-weak DP; v1-medium/stronger, decoded v2, and the old DP multipliers are much less useful because the recovered errors concentrate in high-variance features.

## Part B — repaired CIFAR-10 attack

Target bundle:

`artifacts/priority7_image_domain/confirmatory_v2_cosine_targets_20260917/cifar10_priority7_targets.pt`

Branches: `none`, `v2_ratio0p95_eta0p01`, `v1_medium`.  
Budgets: original 25-iteration cosine-only; strong 250-iteration cosine + TV.  
Primary test: own-signal reconstruction vs decoy-signal reconstruction, exact one-sided sign test. Secondary controls: gray `0.5`, CIFAR training-set mean, and prior.

| Budget / branch | n | Own MSE mean | Own PSNR mean | Own SSIM mean | Wins vs decoy | p vs decoy | Wins vs gray | p vs gray | Wins vs mean | p vs mean | Wins vs prior | p vs prior | PSNR ratio vs none |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original_25 / none | 39 | 0.09208 | 10.5912 | 0.0513 | 39/39 | 1.818989e-12 | 0/39 | 1.0 | 0/39 | 1.0 | 39/39 | 1.818989e-12 | N/A |
| original_25 / v1 medium | 39 | 0.09206 | 10.5843 | 0.0543 | 39/39 | 1.818989e-12 | 0/39 | 1.0 | 0/39 | 1.0 | 39/39 | 1.818989e-12 | 0.9993 |
| original_25 / v2 0.95/0.01 | 39 | 0.09184 | 10.6020 | 0.0541 | 39/39 | 1.818989e-12 | 0/39 | 1.0 | 0/39 | 1.0 | 39/39 | 1.818989e-12 | 1.0010 |
| strong_250+TV / none | 39 | 0.08122 | 11.3399 | 0.1355 | 39/39 | 1.818989e-12 | 9/39 | 0.99985 | 5/39 | ~1.0 | 36/39 | 1.804437e-08 | N/A |
| strong_250+TV / v1 medium | 39 | 0.08167 | 11.2879 | 0.1289 | 39/39 | 1.818989e-12 | 8/39 | 0.99996 | 4/39 | ~1.0 | 37/39 | 1.420631e-09 | 0.9954 |
| strong_250+TV / v2 0.95/0.01 | 39 | 0.08097 | 11.3425 | 0.1338 | 39/39 | 1.818989e-12 | 8/39 | 0.99996 | 5/39 | ~1.0 | 37/39 | 1.420631e-09 | 1.0002 |

Example grids:

- `artifacts/priority23_cifar_repaired_20260928/examples/original_25_cosine__none.png`
- `artifacts/priority23_cifar_repaired_20260928/examples/original_25_cosine__v1_medium.png`
- `artifacts/priority23_cifar_repaired_20260928/examples/original_25_cosine__v2_ratio0p95_eta0p01.png`
- `artifacts/priority23_cifar_repaired_20260928/examples/strong_250_cosine_tv__none.png`
- `artifacts/priority23_cifar_repaired_20260928/examples/strong_250_cosine_tv__v1_medium.png`
- `artifacts/priority23_cifar_repaired_20260928/examples/strong_250_cosine_tv__v2_ratio0p95_eta0p01.png`

Verdict: the repaired CIFAR attack consistently extracts target-specific signal: own-signal beats decoy 39/39 for none, v1, and v2 under both budgets. However, even the stronger budget usually does not beat simple gray/mean-image controls, so the reconstructions remain weak in absolute image quality. V2 is not meaningfully weaker than none in this run by PSNR ratio; the branch PSNRs are almost identical.

## Part C — DP recalibration on trainable-only vectors

Clip norm fixed at `100`; delta `1e-5`; update-level add/remove RDP accounting, one release and 50 releases. The multipliers below match median transform L2 distortion on the trainable-parameter vector of Priority-16 captures.

| Transform | New multiplier | Old multiplier | New/old ratio | Single-release epsilon | 50-release epsilon |
|---|---:|---:|---:|---:|---:|
| v1 conservative | 9.8487e-07 | 0.00025 | 0.00394 | 5.1549e11 | 2.5775e13 |
| v1 medium | 1.2292e-06 | 0.000315 | 0.00390 | 3.3094e11 | 1.6547e13 |
| v1 stronger | 1.4698e-06 | 0.0004 | 0.00367 | 2.3145e11 | 1.1573e13 |
| v2 0.95/0.01 | 1.9848e-06 | 0.00105 | 0.00189 | 1.2693e11 | 6.3463e12 |

Verdict: on the trainable-only vector, the old DP multipliers used in Priority 14/16/20 were hundreds of times larger than the multiplier needed to match transform L2 distortion. The trainable-vector L2-matched DP multipliers would have even larger epsilon values, i.e. even weaker formal DP protection.

## Part D — RQ2 facts and PR-AUC

Configuration evidence:

- RQ2 v1 confirmatory used **DNA Transform v1 conservative**, not medium/stronger. Evidence: `protocols/config/rq2_confirmatory.yaml` freezes `DNA-TRANSFORM-CONSERVATIVE-V1` with `mix=0.08`, `keep=0.88`, `shrink=0.45`; metrics also record the same config.
- RQ2 v2 confirmatory used `compression_ratio=0.95`, `quantization_eta=0.01`.
- I found no confirmatory RQ2 run for v1-medium or v1-stronger.
- BN buffers were transformed/noised in RQ2 paths: v1 and v2 transform runners iterate over `state_dict.items()` and transform each floating tensor; DP clips/noises each floating `local_state` item.

| RQ2 run | n | Baseline F1 mean | ΔF1 mean [95% CI] | Baseline ROC-AUC mean | ΔROC-AUC mean [95% CI] | Baseline PR-AUC mean | ΔPR-AUC mean [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 conservative vs baseline | 21 | 0.7409 | +0.00143 [-0.00813, +0.01098] | 0.99284 | -0.000391 [-0.001104, +0.000323] | 0.76419 | -0.00262 [-0.00703, +0.00179] |
| v2 0.95/0.01 vs baseline | 52 | 0.7322 | -0.01171 [-0.01951, -0.00392] | 0.99299 | -0.001981 [-0.004043, +0.000080] | 0.75603 | -0.00735 [-0.01422, -0.00049] |

Verdict: PR-AUC is directionally worse for both RQ2 transform runs, with a statistically clear negative CI for v2. The existing frozen RQ2 endpoint rule did not include PR-AUC, but PR-AUC should be disclosed as an additional fraud-relevant metric.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `protocols/amendments/2026-09-28_priority23_bn_channel_and_cifar_repair.md` | `be47dc976a05cb1ae5cd0b48a9a001a63ac64c1f1381ac52aa47d9f7e7af1d74` |
| `experiments/priority23_bn_rq2_analysis.py` | `2bfbf865b9f024e626c6cede981816a73416bd10b0d58fb973cf63424cda3c1d` |
| `experiments/priority23_repaired_cifar_attack.py` | `49bdb056f81d4880f23cd43e040610a8b1639a73fe998b96f989628a3fac4343` |
| `artifacts/priority23_bn_channel_rq2_20260929_rq2maxrows/priority23_part_a_c_d_summary.json` | `e94a77a556c481568dc8a916907836eaf4a388c3dd37929856ba9c6ed694e849` |
| `artifacts/priority23_bn_channel_rq2_20260929_rq2maxrows/part_a_bn_mean_recovery_per_target.csv` | `1a3c90ee07f64aae0a94c614c287da60a34f1707faf797d609d40601887c096b` |
| `artifacts/priority23_bn_channel_rq2_20260929_rq2maxrows/part_a_running_var_descriptive_per_target.csv` | `00768b0c6a21eb9875af70583ddf51fe6b1c6da619fa890939da6010a540a435` |
| `artifacts/priority23_bn_channel_rq2_20260929_rq2maxrows/part_c_trainable_l2_calibration_per_target.csv` | `6a0c1cbb805e09b4a0a8d1aed036edf05dc96527f93865b92b6aab174fc4d62e` |
| `artifacts/priority23_cifar_repaired_20260928/priority23_cifar_repaired_summary.json` | `e61d909edc0ef6ae4eedfc7df5b1d31f2c0366a19a5fb1388bd314683e30d583` |
| `artifacts/priority23_cifar_repaired_20260928/priority23_cifar_repaired_per_group.csv` | `487c978a3a436fc81072382c998be8e9abbd8bfa957adf641656b14719346b1a` |
| `artifacts/priority23_cifar_repaired_20260928/manifest.json` | `ffd49a5b6abc1f0a21597064973a8777a5d3bd7c17ddee1fecb67b80be8762aa` |

## Final plain answers

1. **Does the transmitted BN channel leak the target batch mean under none, v1, v2, and DP?** Yes in raw robust-scaled feature MSE for every evaluated defense. In standardized feature MSE, leakage evidence is strong for none, v1-conservative, and recalibrated DP, but not for all old DP/v1/v2 settings.
2. **Does the repaired CIFAR attack beat decoy, gray, and mean-image controls, and how much weaker is it on v2 than on none?** It beats decoy 39/39 for all branches/budgets, but it does not beat gray or mean-image controls. V2 is not weaker than none by PSNR in this repaired run; PSNR ratios are approximately 1.00.
3. **What are the correct L2-matched DP multipliers on the trainable vector?** Approximately `9.85e-7` (v1 conservative), `1.23e-6` (v1 medium), `1.47e-6` (v1 stronger), and `1.98e-6` (v2 0.95/0.01), all far below the old multipliers.
4. **What does PR-AUC show?** v1-conservative has a small negative mean PR-AUC delta with CI crossing zero; v2 has a negative PR-AUC delta with 95% CI entirely below zero.
