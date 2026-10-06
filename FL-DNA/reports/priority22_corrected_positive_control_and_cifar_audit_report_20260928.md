# Priority 22 — corrected positive control and CIFAR/tabular gate audit

Date: 2026-09-28

Scope: no `Latex/` edits; no existing artifacts or reports were modified. Priority 21's invalid positive-control output is left untouched and superseded by the corrected run here.

## Commands run

```bash
.venv-phase1/bin/python -m py_compile experiments/priority22_corrected_positive_control_and_audit.py
.venv-phase1/bin/python experiments/priority22_corrected_positive_control_and_audit.py audit --output-dir artifacts/priority22_corrected_positive_control_and_audit/audit_20260928
.venv-phase1/bin/python experiments/priority22_corrected_positive_control_and_audit.py positive-control --output-dir artifacts/priority22_corrected_positive_control_and_audit/corrected_positive_control_20260928_rerun1
shasum -a 256 experiments/priority22_corrected_positive_control_and_audit.py protocols/amendments/2026-09-28_priority22_corrected_positive_control.md artifacts/priority22_corrected_positive_control_and_audit/audit_20260928/priority22_audit_summary.json artifacts/priority22_corrected_positive_control_and_audit/corrected_positive_control_20260928_rerun1/equivalence_check.json artifacts/priority22_corrected_positive_control_and_audit/corrected_positive_control_20260928_rerun1/priority22_corrected_positive_control_summary.json artifacts/priority22_corrected_positive_control_and_audit/corrected_positive_control_20260928_rerun1/priority22_corrected_positive_control_per_target.csv
```

Note: an earlier attempt wrote a partial folder `artifacts/priority22_corrected_positive_control_and_audit/corrected_positive_control_20260928/` and stopped before scientific output because the equivalence folder did not exist before `score(...)` wrote CSV. It is not used in this report. The successful run is `corrected_positive_control_20260928_rerun1`.

## Step A — corrected positive control

Amendment: `protocols/amendments/2026-09-28_priority22_corrected_positive_control.md`.

Target: `artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_targets_20260927/paysim_priority16_v1_medium_confirmatory_targets.pt`  
Target SHA-256: `645fe7ff62403e0bce81a3fc6cbed4d857306d36719cc2e3bad740acb73297fe`

The implementation first reran two Priority 16 DNA branch selected restarts through the same generic optimizer used for the positive control. Both reproduced saved `clean_vector_mse` bit-for-bit:

| Group | Restart | Saved clean_vector_mse | Reproduced clean_vector_mse | Bit-exact |
|---:|---:|---:|---:|---|
| 0 | 3 | 1.2007617985628e-06 | 1.2007617985628e-06 | True |
| 1 | 1 | 1.266959452319495e-06 | 1.266959452319495e-06 | True |

Corrected positive-control result:

| Quantity | mean | median | min | max | sd |
|---|---:|---:|---:|---:|---:|
| none input MSE | 439.398 | 171.572 | 18.6832 | 5471.32 | 939.671 |
| Prior input MSE | 426.283 | 79.355 | 11.5647 | 6122.02 | 1051.88 |
| Zero-update input MSE | 459.442 | 76.0227 | 11.388 | 7160.42 | 1188.71 |

Gate uses tie band `0.0390625`, exact one-sided sign test:

| Gate | wins | losses | ties | non-tied | p |
|---|---:|---:|---:|---:|---:|
| none beats Prior | 15 | 23 | 1 | 38 | 0.928347 |
| none beats Zero-update | 17 | 22 | 0 | 39 | 0.831608 |

Pre-registered interpretation: `none` does **not** beat Prior. Therefore, this attacker is not a valid instrument in the bounded tabular Priority 16 setting; no tabular DNA-vs-DP privacy conclusion can be drawn from this attacker.

## Step B — CIFAR-10 Transform v2 confirmatory break audit

Artifact audited: `artifacts/priority7_image_domain/confirmatory_v2_cosine_gate_20260917/priority7_image_gate_report.json` (`sha256=fd7befa0647ad0589e11d0b38c0214db02b1983b321d595504bd6366eacd634e`).

| Question | Finding |
|---|---|
| Metric space | input-image space image_mse; _score computes MSE/PSNR/SSIM and _evaluate gates on baseline_mse-prior_mse and baseline_mse-zero_mse |
| Formula/code | `experiments/run_priority7_image_domain_gate.py:370-379; experiments/run_priority7_image_domain_gate.py:387-430` |
| Attacker objective/restart selection | GEN_COSINE_TV minimizes global cosine in defended-update sketch space; selected restart is minimum objective Code: `experiments/run_priority7_image_domain_gate.py:210-218; experiments/run_priority7_image_domain_gate.py:323-346; experiments/run_priority7_image_domain_gate.py:391-394` |
| Attacker knowledge | Level 2 / seed-known for v2: candidate uses true v2 base seed plus observed sketch payload containing tensor_index and quantization_delta; sampled/sign pattern derives from fixed seed. Code: `experiments/run_priority7_image_domain_gate.py:135-153; experiments/run_phase4_dna_v2_sketch_space_attack.py:107-147` |
| Controls | Prior is sigmoid(random initial_logits) using the same init family as the attack restarts. Zero-update starts from the same initial logits per restart and optimizes against zero defended-signal. _evaluate averages prior MSE over baseline restarts, rather than using only the selected baseline restart's prior. Code: `experiments/run_priority7_image_domain_gate.py:316-325; experiments/run_priority7_image_domain_gate.py:391-396` |
| Mechanical floor/asymmetry | No DP-style branch noise floor exists. Baseline and zero controls use the same candidate defended transform; the main asymmetry is that v2 is evaluated seed-known. |

CIFAR gate result from the frozen artifact:

| Control | wins/non-ties | mean difference | median difference | p | pass |
|---|---:|---:|---:|---:|---|
| Prior | 39/39 | -0.0167283 | -0.0174395 | 1.81899e-12 | True |
| Zero-update | 39/39 | -0.0163785 | -0.016955 | 1.81899e-12 | True |

Selected input-image metrics:

| Metric | mean | median | min | max | sd |
|---|---:|---:|---:|---:|---:|
| baseline image MSE | 0.0918891 | 0.0894096 | 0.0670641 | 0.137851 | 0.0170335 |
| Prior image MSE | 0.108617 | 0.106492 | 0.086293 | 0.156387 | 0.0156071 |
| Zero-update image MSE | 0.108268 | 0.106042 | 0.0857402 | 0.154911 | 0.0155008 |
| baseline PSNR | 10.6099 | 10.5719 | 8.90848 | 11.9218 | 0.770396 |

Conclusion for Step B: the CIFAR-10 v2 break is an input-image-space break against both controls, but it is **not Level 1** under the paper's definitions; it is Level 2 / seed-known for v2 because the candidate uses true v2 base seed and observed sketch payload.

## Step C — tabular attacker-vs-control gate audit

| Gate family | Metric space | Prior init | Attacker init/lr | Same init as Prior? | Flag |
|---|---|---|---|---|---|
| Priority 6/9 PaySim GEN_IDLG_STYLE/GEN_COSINE_TV attacker-vs-control gates | input-space PaySim fraud-class mean MSE (`classes['1']['mean_mse']`) | standard hard-diff latent init | same initial tensor as Prior; Adam lr=0.1; 601 evaluations | True | valid same-init gate design, but historical results may still be affected by other previously documented contamination unless rerun under corrected-vector pipeline |
| Priority 6 IEEE-CIS GEN_COSINE_TV attacker-vs-control gates | input-space IEEE feature mean MSE | IEEE numeric/categorical/label latent init from `init_ieee` | same numeric/categorical/label initial state as Prior; Adam lr=0.05; 301 evaluations | True | valid same-init gate design, but historical result was later questioned by corrected-vector development gate; interpret with its own reports |
| Legacy Phase-4/RQ1 variant confirmatory gates | PaySim fraud-class MSE / reconstruction metrics from selected artifacts | standard hard-diff init in command wrappers; branch scripts compute Prior from the same initial tensor | same initial tensor as Prior; branch-specific Adam lr/iterations from config | True | same-init Prior design; many legacy gates are contaminated by the separate BN-buffer issue documented in Priority 11 |
| Priority 14/16/20 DP-vs-transform head-to-head | not attacker-vs-Prior/Zero gate; original D was update-space defended residual | not applicable for published D; P21 re-analysis compared saved branch reconstructions against their own initial tensors | standard init; Adam lr=0.1; 601 evaluations | None | not a valid Prior/Zero gate; P21 showed update-space D dominated by DP floor and input-space comparison does not support DP wins |
| Priority 21 positive control (invalidated) | input-space mean MSE | plausible init, not matching P16 | plausible init + Adam lr=0.08 | True | invalid because it does not match P16 init/lr; superseded by Priority 22 corrected positive control |

## Plain verdicts

1. Does the attacker work on undefended tabular updates?  **No** for the corrected Priority 16 positive control. `none` beats Prior only 15/38 non-tied targets, p=0.928347; it also fails Zero-update with 17/39, p=0.831608.

2. Is the CIFAR-10 v2 break a Level-1, input-space break?  It is an input-space break, but **not Level 1**. It is Level 2 / seed-known under the v2 threat-model facts.

3. Which tabular gates are valid?  The gate designs that compare attacker reconstructions against Prior/Zero in Priority 6/9 and legacy Phase-4/RQ1 use the same initialization for the Prior and the attack branch. However, many historical tabular results remain limited by previously documented contamination or corrected-vector failures. The Priority 14/16/20 DP-vs-transform head-to-head rows are not Prior/Zero gates and were already invalidated as privacy evidence by Priority 21's floor/input-space re-analysis.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `experiments/priority22_corrected_positive_control_and_audit.py` | `d04e7fc3a0c79e726d14ef7517b1653f6503c3ad8ee982cd2feb7867d4b9b84e` |
| `protocols/amendments/2026-09-28_priority22_corrected_positive_control.md` | `679fe326f2cab670618f9b2fb1f9b42172ce1b5ba74f0aa0a0dda2af603af6a3` |
| `artifacts/priority22_corrected_positive_control_and_audit/audit_20260928/priority22_audit_summary.json` | `df79a156f03b2e7e41c16959a824c61155a0ba21bbe5a4e1ed1cc7d953372bdd` |
| `artifacts/priority22_corrected_positive_control_and_audit/corrected_positive_control_20260928_rerun1/equivalence_check.json` | `b65617d198915475088ed4db49aa2ccca4f02e29cbbb2d9505b328f318b607c5` |
| `artifacts/priority22_corrected_positive_control_and_audit/corrected_positive_control_20260928_rerun1/priority22_corrected_positive_control_summary.json` | `2b9a244ba8917e0995dc450f11f8c932c6602bdc2dcf9e324e24219f406e9418` |
| `artifacts/priority22_corrected_positive_control_and_audit/corrected_positive_control_20260928_rerun1/priority22_corrected_positive_control_per_target.csv` | `d3c6476b5a4b0017fcb6ab052fa7172af516b39f8ede523dac66d3aa902b5a6b` |
