# Priority 21 — head-to-head metric validity audit and positive control

Date: 2026-09-28

Scope: no edits under `Latex/`; existing artifacts and reports were not modified.  Steps 1--3 are re-analysis only.  Step 4 is the pre-registered positive-control run authorized by `protocols/amendments/2026-09-28_priority21_positive_control_protocol.md`.

## Commands actually run

```bash
.venv-phase1/bin/python -m py_compile experiments/priority21_metric_validity.py
.venv-phase1/bin/python experiments/priority21_metric_validity.py reanalyze --output-dir artifacts/priority21_metric_validity/reanalysis_20260928
.venv-phase1/bin/python experiments/priority21_metric_validity.py positive-control --output-dir artifacts/priority21_metric_validity/positive_control_20260928
shasum -a 256 experiments/priority21_metric_validity.py protocols/amendments/2026-09-28_priority21_positive_control_protocol.md artifacts/priority21_metric_validity/reanalysis_20260928/priority21_reanalysis_summary.json artifacts/priority21_metric_validity/positive_control_20260928/priority21_positive_control_summary.json artifacts/priority21_metric_validity/positive_control_20260928/priority21_positive_control_per_target.csv
```

## Step 1 — code facts

### Metric formula and tensors

Priority 14 computes `_mse(candidate, signal)` by concatenating `(candidate[key] - signal[key])^2` over floating keys of the defended-update signal, not input features (`experiments/priority14_fresh_dna_vs_dp_probe.py:85-88`).  The optimizer uses `update_objective(candidate, signal, ..., mode="balanced_tensor")` and the saved metric is `clean_vector_mse` (`experiments/priority14_fresh_dna_vs_dp_probe.py:156-180, 195`).  Priority 16 uses the same pattern (`experiments/priority16_v1_medium_dna_vs_dp_probe.py:80-83, 151-175, 190`).  Priority 20 also uses the same update-space residual and saves it as `clean_vector_mse` (`experiments/priority20_multi_checkpoint_head_to_head.py:245-269, 285`).

Therefore the published head-to-head statistic `D = dna_mse - dp_mse` is a defended-update/parameter-space fit residual.  Input-space metrics are saved separately as `feature_metrics` by `score(...)` but were not used for `D`.

### DP branch noise formation

The DP observed signal is built by `simple_defense_plan("clipping_noise_mc", ...)` and `_apply_observed_defense(...)` (`experiments/priority14_fresh_dna_vs_dp_probe.py:121-134`; Priority 16 analogous at `experiments/priority16_v1_medium_dna_vs_dp_probe.py:116-129`; Priority 20 at `experiments/priority20_multi_checkpoint_head_to_head.py:196-209`).  The underlying helper draws a single observed noise tensor with seed `derive_seed(defense_seed, "clipping-noise", group_id)` (`experiments/run_phase4_simple_defense_attack.py:180-196, 285-295`).  The observed defended update is `clip_factor * value + plan["noise"][key]` (`experiments/run_phase4_simple_defense_attack.py:199-214`).

The candidate DP update is not compared with that same noise draw.  It uses `clip_factor * value + plan["mc_noise_mean"][key]` (`experiments/run_phase4_simple_defense_attack.py:217-239`).  The MC mean is formed from `mc_noise_samples` samples with a different seed family, `derive_seed(defense_seed, "clipping-noise-mc", group_id)` (`experiments/run_phase4_simple_defense_attack.py:306-314`).  In these cells `mc_noise_samples=100` (Priority 14/16 DP config lines `experiments/priority14_fresh_dna_vs_dp_probe.py:123-130`, `experiments/priority16_v1_medium_dna_vs_dp_probe.py:118-125`).

### DNA branch attacker knowledge and level

For v1, the observed transform uses the true v1 transform on the observed update (`experiments/run_phase4_dna_level1_forward_attack.py:91-99`), and the true v1 per-block seed is derived from the observed block contents (`dna_encoder/transform_defense.py:57-66`).  The attacker candidate does not receive that exact realization; it uses `_surrogate_plan_from_state(...)` (`experiments/priority14_fresh_dna_vs_dp_probe.py:91-96`, `experiments/priority16_v1_medium_dna_vs_dp_probe.py:86-91`, Priority 20 at `experiments/priority20_multi_checkpoint_head_to_head.py:185-193`).  `_surrogate_plan_from_state` derives surrogate matrices from `(config.seed, group_id, realization_id, tensor_index, block_index)` rather than the true data-derived block seed (`experiments/run_phase4_dna_level1_forward_attack.py:57-74`).  I classify v1 head-to-head as Level 1 under the paper's terms.

For v2, the observed branch stores the true sketch payload including tensor index and quantization delta (`experiments/run_phase4_dna_v2_sketch_space_attack.py:107-127`), and the candidate branch is given that plan plus `v2_base_seed=int(config.seed)` (`experiments/priority14_fresh_dna_vs_dp_probe.py:97-117`; candidate sketch uses `base_seed`, `tensor_index`, `quantization_delta` at `experiments/run_phase4_dna_v2_sketch_space_attack.py:130-147`).  I classify the v2 head-to-head as Level 2 / seed-known relative to the paper's Level 1 wording.

### Restart selection and target setting

The saved branch artifact records both `best_objective` and `clean_vector_mse`; the best latent is selected during optimization only by the attacker-observable update objective (`experiments/priority14_fresh_dna_vs_dp_probe.py:156-180`; Priority 16 `experiments/priority16_v1_medium_dna_vs_dp_probe.py:151-175`; Priority 20 `experiments/priority20_multi_checkpoint_head_to_head.py:245-269`).  The run-level summary then selects the restart with minimum saved update residual per branch, not ground-truth input error (`priority14` logic is reflected by the selected `dna_mse`/`dp_mse` fields in the saved JSON; the artifacts used here are selected the same way).

Targets are 4-record bounded local updates: Priority 14/16 call `capture_paysim(..., 4)` (`experiments/priority14_fresh_dna_vs_dp_probe.py:213-216`, `experiments/priority16_v1_medium_dna_vs_dp_probe.py:208-211`), and Priority 20 calls `_capture_with_checkpoint(..., 4, checkpoint)` (`experiments/priority20_multi_checkpoint_head_to_head.py:300-306`).  The optimizer loop is 601 objective evaluations / 600 Adam updates (`experiments/priority14_fresh_dna_vs_dp_probe.py:156-173`; Priority 16 `experiments/priority16_v1_medium_dna_vs_dp_probe.py:151-168`; Priority 20 `experiments/priority20_multi_checkpoint_head_to_head.py:245-262`).

## Step 2 and Step 3 — re-analysis results

Columns: `D_update` is the originally reported update-space mean `dna_mse - dp_mse`.  `floor_dp` and `floor_dna` are perfect-reconstruction residuals.  `explained` is `(floor_dna - floor_dp) / D_update`.  The floor-corrected sign test column reports DP wins / DNA wins / ties after using `D' = (dna_mse-floor_dna) - (dp_mse-floor_dp)`, with DP wins represented by `-D' > 0`.  Input sign tests use `D_in = dna_feature_mse - dp_feature_mse`, so DP wins when `D_in > 0`.

| Cell | n | mean D_update | mean floor_dp | mean floor_dna | mean explained | floor-corrected DP/DNA/tie | input strict DP/DNA/tie | input tie-banded DP/DNA/tie |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| priority14_v1_stronger_vs_dp_0p0004 | 39 | -0.001617 | 0.001617 | 2.017e-08 | 1 | 8/31/0, p=1 | 18/21/0, p=0.7388 | 18/21/0, p=0.7388 |
| priority14_v2_ratio0p95_vs_dp_0p00105 | 39 | -0.01114 | 0.01114 | 2.005e-11 | 1 | 7/32/0, p=1 | 15/24/0, p=0.9459 | 15/24/0, p=0.9459 |
| priority16_v1_medium_vs_dp_0p000315 | 39 | -0.001003 | 0.001003 | 1.418e-08 | 1 | 13/26/0, p=0.9881 | 14/25/0, p=0.9734 | 14/25/0, p=0.9734 |

Priority 20 multi-checkpoint rows are included because the user asked for every head-to-head harness used in the paper:

| Cell | n | mean D_update | mean floor_dp | mean floor_dna | mean explained | floor-corrected DP/DNA/tie | input strict DP/DNA/tie | input tie-banded DP/DNA/tie |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| priority20_ckpt0_v1_medium_vs_dp_0p000315 | 8 | -0.001012 | 0.001013 | 1.4e-08 | 1.001 | 1/7/0, p=0.9961 | 3/5/0, p=0.8555 | 3/5/0, p=0.8555 |
| priority20_ckpt1_v1_medium_vs_dp_0p000315 | 8 | -0.001012 | 0.001013 | 1.407e-08 | 1 | 1/7/0, p=0.9961 | 2/6/0, p=0.9648 | 2/6/0, p=0.9648 |
| priority20_ckpt2_v1_medium_vs_dp_0p000315 | 8 | -0.001012 | 0.001013 | 1.439e-08 | 1.001 | 1/7/0, p=0.9961 | 1/7/0, p=0.9961 | 1/7/0, p=0.9961 |
| priority20_ckpt3_v1_medium_vs_dp_0p000315 | 8 | -0.001013 | 0.001013 | 1.442e-08 | 1 | 3/5/0, p=0.8555 | 3/5/0, p=0.8555 | 3/5/0, p=0.8555 |
| priority20_ckpt4_v1_medium_vs_dp_0p000315 | 8 | -0.001012 | 0.001013 | 1.433e-08 | 1 | 2/6/0, p=0.9648 | 2/6/0, p=0.9648 | 2/6/0, p=0.9648 |

Perfect-reconstruction sanity check: the script explicitly re-derived the same DNA/DP plans from saved target/group seeds, passed the true trainable update through each candidate-defense function, and computed `floor_dna`/`floor_dp` from those tensors.  This is the direct sanity check requested; the DP floor equals the candidate residual of a perfect `u_hat = u` under the harness's MC-mean candidate rule.

Zero-update input-space controls for Step 3 were not present in the saved head-to-head artifacts.  I did not synthesize them in Steps 1--3 because those steps were constrained to no new attacker runs.  The new Step 4 positive control supplies a fresh zero-update control under a pre-registered amendment.

## Step 4 — positive control

Amendment: `protocols/amendments/2026-09-28_priority21_positive_control_protocol.md`.

Target: `artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_targets_20260927/paysim_priority16_v1_medium_confirmatory_targets.pt` (`sha256=645fe7ff62403e0bce81a3fc6cbed4d857306d36719cc2e3bad740acb73297fe`).

Result, input-space mean MSE:

| Branch/control | mean | median | min | max | sd |
|---|---:|---:|---:|---:|---:|
| none | 3.399e+04 | 2.773e+04 | 4610 | 9.025e+04 | 2.329e+04 |
| Prior | 3.376e+04 | 2.762e+04 | 4480 | 8.961e+04 | 2.307e+04 |
| Zero-update | 3.428e+04 | 2.821e+04 | 3926 | 8.921e+04 | 2.147e+04 |

Gate results with the frozen v1 tie band `0.0390625`:

| Gate | wins | losses | ties | non-tied | exact one-sided p |
|---|---:|---:|---:|---:|---:|
| none beats Prior | 19 | 20 | 0 | 39 | 0.6254 |
| none beats Zero-update | 21 | 18 | 0 | 39 | 0.3746 |

Pre-registered interpretation applies: `none` does not pass against Prior at n=39.

## Plain verdict

1. Is the published 39/39 DP-win result explained by the DP noise floor?  Yes, for the three main confirmatory head-to-head cells.  The mean `explained` fraction is approximately 1.0003 for v1-stronger, 1.0002 for v2, and 1.0003 for v1-medium.  After floor correction, the DP-win sign test no longer supports DP: v1-stronger has 8 DP wins / 31 DNA wins, v2 has 7 / 32, and v1-medium has 13 / 26.

2. Does DP still beat the transforms when scored in input space?  No.  Using the same attacker-observable restart selection, input-space strict/tie-banded counts are v1-stronger 18 DP wins / 21 DNA wins, v2 15 / 24, and v1-medium 14 / 25.  None is significant in the DP-winning direction.

3. Which cells were Level 1 vs Level 2?  v1 head-to-head cells (Priority 14 v1-stronger, Priority 16 v1-medium, Priority 20 v1-medium) use a surrogate v1 realization and are Level 1 in the relevant sense.  The v2 head-to-head cell passes the true v2 sketch payload plus base seed into the candidate sketch and is Level 2 / seed-known.

4. Can the attacker break an undefended update on this tabular setting?  No under the pre-registered Priority 21 positive-control gate.  `none` vs Prior is 19/39 wins, p=0.625; `none` vs Zero-update is 21/39 wins, p=0.375.  Therefore this attacker is not a valid instrument for interpreting tabular DNA-vs-DP privacy comparisons in this bounded 4-record setting.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `experiments/priority21_metric_validity.py` | `6e2235bc3aee8cfa60fe2bc59decfd721247258994672d51eca05434d9b3885c` |
| `protocols/amendments/2026-09-28_priority21_positive_control_protocol.md` | `11db1b6b965cf5e46e4aa41f54120b12d6e925252cd8fd8ca994b35a3b0d7846` |
| `artifacts/priority21_metric_validity/reanalysis_20260928/priority21_reanalysis_summary.json` | `cf25a569f2051c5b72362f600f58d276cdf1b97193cdaba4ae0edb2e020752d2` |
| `artifacts/priority21_metric_validity/positive_control_20260928/priority21_positive_control_summary.json` | `81896f8b3b782aec920fe04f395c06e96d8df7462c6b5f7facf9b4cba0e730af` |
| `artifacts/priority21_metric_validity/positive_control_20260928/priority21_positive_control_per_target.csv` | `0343f4e57301c7dc3f1c184461e1260ae6934aaa052a0bae81d073242e3a1e38` |
| `artifacts/priority21_metric_validity/reanalysis_20260928/priority14_v1_stronger_vs_dp_0p0004_priority21_reanalysis.csv` | `84898b0f11669a764c5b3371f9e1bc805c88f4e06f6135b1bfff78c180f4e60c` |
| `artifacts/priority21_metric_validity/reanalysis_20260928/priority14_v2_ratio0p95_vs_dp_0p00105_priority21_reanalysis.csv` | `5592b9d04d5068902bc8afc8f71b90987ec0625b384402a9b2c4e88eecb5a1cc` |
| `artifacts/priority21_metric_validity/reanalysis_20260928/priority16_v1_medium_vs_dp_0p000315_priority21_reanalysis.csv` | `8589556c93448b954f878532e7305f379a4ef72a50ef83656ebd94cad619b9d4` |
| `artifacts/priority21_metric_validity/reanalysis_20260928/priority20_ckpt0_v1_medium_vs_dp_0p000315_priority21_reanalysis.csv` | `692ebdcd9cec9c35ff39a665c71edcad34590526611405dc70f925baa446f202` |
| `artifacts/priority21_metric_validity/reanalysis_20260928/priority20_ckpt1_v1_medium_vs_dp_0p000315_priority21_reanalysis.csv` | `fa18ec03bb3b1b3322b7c5ebd2b964b4a8405ba33ebfcfbefbbb91cbff6dca58` |
| `artifacts/priority21_metric_validity/reanalysis_20260928/priority20_ckpt2_v1_medium_vs_dp_0p000315_priority21_reanalysis.csv` | `d613ab5e5fd8afbaadcc04254be073bd5ba8f1bca445947cb8dea759999e728b` |
| `artifacts/priority21_metric_validity/reanalysis_20260928/priority20_ckpt3_v1_medium_vs_dp_0p000315_priority21_reanalysis.csv` | `2cbb5f806f0e63bfd1cd2f6c96e4128563ddef01b7df0a4b9761a5b36deee311` |
| `artifacts/priority21_metric_validity/reanalysis_20260928/priority20_ckpt4_v1_medium_vs_dp_0p000315_priority21_reanalysis.csv` | `1eb40709a128077ae96990ce91761bd4e4a02e61045f4aea6771dbfe0352fdb2` |
