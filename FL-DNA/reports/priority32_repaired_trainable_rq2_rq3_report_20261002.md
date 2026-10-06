# Priority32: multi-dataset RQ2 and RQ3

Status: COMPLETE. Training and cost supervisors have exited; live-PID checks passed.

This is the separately authorized trainable-only transform / raw BN utility variant. Earlier results remain unchanged. No reconstruction attacks, Latex edits or external_defenses edits.

## Data audit and frozen preprocessing

| Dataset | Raw rows | Fraud rate | Features | Train/validation/test selected |
| --- | --- | --- | --- | --- |
| paysim | 6362620 | 0.00129082045 | 13 | 325000/75000/100000 |
| ieee_cis | 590540 | 0.0349900091 | 476 | 325000/75000/100000 |
| baf | 1000000 | 0.011029 | 58 | 337833/59662/102505 |

Train-only imputation/scaling/encoding; cap500k inside partitions; source-overlap0. IEEE temporal split with identity left join; BAF months0–4/5/6–7 and exact -1 indicators. PaySim retains the unchanged loader feature/partition helpers, but cap is after splitting and medians are train-only. Full preprocessing, category mapping, missing-column lists and source hashes are in prepared/<dataset>/audit.json and preprocessing.json.

## Validation quality gates

| Dataset | Chosen training | Val F1 | All-fraud F1 | Val AUC | Gate |
| --- | --- | --- | --- | --- | --- |
| paysim | {'alpha': 0.95, 'lr': 0.001, 'rounds': 50} | 0.7457627118644068 | 0.002583325626216832 | 0.9953917031663356 | PASS |
| ieee_cis | {'alpha': 0.95, 'lr': 0.001, 'rounds': 50} | 0.40025948751216345 | 0.07546734057995523 | 0.8525038506570792 | PASS |
| baf | {'alpha': 0.95, 'lr': 0.001, 'rounds': 50} | 0.23506988564167725 | 0.0233898755793195 | 0.8804120881813815 | PASS |

Original validation attempts retained under artifacts/priority32_multidataset/quality/ and reused without rerunning. Selection used validation only; no test evaluation in those jobs. execution_freeze.json predates confirmatory jobs.

## RQ2 paired non-inferiority

21 seeds/dataset, final-round checkpoint; same seeds across methods, validation-F1 threshold. Paired mean Student-t two-sided95%CI; strict lower bound > -.02 F1 / -.005 ROC-AUC. PR-AUC descriptive.

| Dataset | Method | Endpoint | Baseline mean | Mean delta | SD | 95% CI | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paysim | dna_v1_conservative | f1 | 0.786175079 | -0.00123853687 | 0.0141876245 | [-0.0076966661, 0.00521959236] | PASS |
| paysim | dna_v1_conservative | auc_roc | 0.997662213 | 2.77896323e-05 | 0.000248356771 | [-8.52610146e-05, 0.000140840279] | PASS |
| paysim | dna_v1_conservative | pr_auc | 0.80726339 | 0.0015280865 | 0.00903726743 | [-0.00258562831, 0.00564180132] | DESCRIPTIVE |
| paysim | dna_v2_0p95 | f1 | 0.786175079 | -0.00179521107 | 0.0156654461 | [-0.00892603663, 0.00533561449] | PASS |
| paysim | dna_v2_0p95 | auc_roc | 0.997662213 | 3.7846903e-05 | 0.000255998715 | [-7.86823148e-05, 0.000154376121] | PASS |
| paysim | dna_v2_0p95 | pr_auc | 0.80726339 | 0.00308207531 | 0.0116745347 | [-0.00223210919, 0.0083962598] | DESCRIPTIVE |
| ieee_cis | dna_v1_conservative | f1 | 0.352168032 | -0.00672025543 | 0.0230595943 | [-0.0172168569, 0.00377634605] | PASS |
| ieee_cis | dna_v1_conservative | auc_roc | 0.838572269 | -0.00132730978 | 0.00247063072 | [-0.0024519274, -0.000202692161] | PASS |
| ieee_cis | dna_v1_conservative | pr_auc | 0.326702814 | -0.00912300054 | 0.0168187624 | [-0.0167788094, -0.00146719168] | DESCRIPTIVE |
| ieee_cis | dna_v2_0p95 | f1 | 0.352168032 | -0.0127650766 | 0.0172303065 | [-0.0206082181, -0.00492193513] | FAIL |
| ieee_cis | dna_v2_0p95 | auc_roc | 0.838572269 | -0.000557864701 | 0.00173717078 | [-0.00134861536, 0.000232885958] | PASS |
| ieee_cis | dna_v2_0p95 | pr_auc | 0.326702814 | -0.0078157807 | 0.010421434 | [-0.0125595606, -0.00307200082] | DESCRIPTIVE |
| baf | dna_v1_conservative | f1 | 0.225726006 | 0.000418350858 | 0.00318017249 | [-0.0010292463, 0.00186594802] | PASS |
| baf | dna_v1_conservative | auc_roc | 0.872388336 | 4.36483457e-05 | 0.00200210149 | [-0.000867697326, 0.000954994017] | PASS |
| baf | dna_v1_conservative | pr_auc | 0.160628842 | -0.000387011307 | 0.00120674147 | [-0.000936313438, 0.000162290823] | DESCRIPTIVE |
| baf | dna_v2_0p95 | f1 | 0.225726006 | -0.000373761332 | 0.00573891588 | [-0.00298608452, 0.00223856186] | PASS |
| baf | dna_v2_0p95 | auc_roc | 0.872388336 | -0.000323448364 | 0.00187202331 | [-0.00117558316, 0.000528686432] | PASS |
| baf | dna_v2_0p95 | pr_auc | 0.160628842 | -0.000547672485 | 0.00154322007 | [-0.00125013784, 0.00015479287] | DESCRIPTIVE |


Overall: {"baf/dna_v1_conservative": "PASS", "baf/dna_v2_0p95": "PASS", "ieee_cis/dna_v1_conservative": "PASS", "ieee_cis/dna_v2_0p95": "FAIL", "paysim/dna_v1_conservative": "PASS", "paysim/dna_v2_0p95": "PASS"}

Independent NumPy/SciPy recomputation from per-job JSON: {"pass": true, "contrasts_endpoints": 18, "max_abs_error": 2.220446049250313e-16, "method": "independent math.fsum/sample variance + scipy.stats.sem/t.interval from original per-job JSON"}

Paired differences: artifacts/priority32_multidataset/repair_trainable_raw_bn_20261002/rq2_analysis/paired_differences.csv.

## RQ3 cost and DP comparison

Same optimized frozen Bundle B harness/config:10 warmups,50 measurements,orderseed271828,bootstrap10000; new input dimensions only. Synthetic trainable parameter shapes (not full-state real RQ2 traffic). Each IEEE/BAF has27 RAW/v1/v2 cells plus9 DP descriptive cells; PaySim has9 RAW+9 DP references. All run in isolated serial subprocesses after training.

| Dataset | Method | Bundle B | Failed criteria |
| --- | --- | --- | --- |
| paysim | DP_CLIP_NOISE | NOT_ACCEPTABLE | peak_client_memory_overhead_fraction |
| ieee_cis | DNA_TRANSFORM_TRANSPORT | NOT_ACCEPTABLE | client_encode_serialize_p95_ms, end_to_end_round_overhead_p95_fraction, peak_client_memory_overhead_fraction |
| ieee_cis | DNA_TRANSFORM_V2_TRANSPORT | NOT_ACCEPTABLE | end_to_end_round_overhead_p95_fraction, peak_client_memory_overhead_fraction, peak_server_memory_overhead_fraction, server_decode_aggregate_p95_ms |
| ieee_cis | DP_CLIP_NOISE | NOT_ACCEPTABLE | peak_client_memory_overhead_fraction |
| baf | DNA_TRANSFORM_TRANSPORT | NOT_ACCEPTABLE | end_to_end_round_overhead_p95_fraction, peak_client_memory_overhead_fraction |
| baf | DNA_TRANSFORM_V2_TRANSPORT | NOT_ACCEPTABLE | end_to_end_round_overhead_p95_fraction, peak_client_memory_overhead_fraction, peak_server_memory_overhead_fraction |
| baf | DP_CLIP_NOISE | NOT_ACCEPTABLE | peak_client_memory_overhead_fraction |


| Dataset | Method | Worst-cell p95 encode ms | Worst-cell p95 server ms | Max payload bytes(total clients) |
| --- | --- | --- | --- | --- |
| paysim | DP_CLIP_NOISE | 0.9031186 | 2.9759102 | 514580 |
| paysim | RAW_FLOAT32 | 0.30281045 | 2.987191 | 514580 |
| ieee_cis | DNA_TRANSFORM_TRANSPORT | 458.5441 | 3.5452215 | 2885170 |
| ieee_cis | DNA_TRANSFORM_V2_TRANSPORT | 51.738677 | 459.953948 | 3894384 |
| ieee_cis | DP_CLIP_NOISE | 1.62917715 | 3.6859497 | 2885170 |
| ieee_cis | RAW_FLOAT32 | 0.43585785 | 3.6882705 | 2885170 |
| baf | DNA_TRANSFORM_TRANSPORT | 119.643673 | 2.951933 | 744990 |
| baf | DNA_TRANSFORM_V2_TRANSPORT | 18.8331167 | 156.66779 | 988956 |
| baf | DP_CLIP_NOISE | 0.9168606 | 2.95767105 | 744990 |
| baf | RAW_FLOAT32 | 0.29874535 | 3.1326769 | 744990 |

DP cost-only configuration: whole synthetic vector clip100,sigma.001; clipping/noise timed, RAW serialization. It is not a new calibrated DP utility/privacy claim. Per-cell overhead/memory/payload and CI acceptance criteria are in rq3/<dataset>/analysis/acceptance_criteria.csv.

## Exact commands and disclosures

```text
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -u experiments/priority32_repaired_trainable.py --supervise
```
All exact subprocess commands, start/skip/failure/resume events and output hashes: runs.jsonl. No unfinished or failed run is silently excluded. Stage progress/checklist in progress.json, progress.log, checklist.json. All round journals retained.

## Provenance and checks

Amendment: protocols/amendments/2026-10-02_priority32_bn_domain_repair_authorized.md
Raw/code/output SHA-256: artifacts/priority32_multidataset/repair_trainable_raw_bn_20261002/sha256_manifest.csv. Manifest SHA and report SHA: final_hashes.json. Manifest excludes itself and final_hashes.json to avoid impossible self-reference.
Compile, unit tests, git diff --check and independent statistics: final_checks.json. All checks passed. torch.set_num_threads(1); no old data/artifacts/reports changed.

## Authorized repair scope and full run disclosure

The original full-state experiment remains unchanged: 189 submitted attempts, 88 finite outputs and 101 missing outputs. One authorized diagnostic replay of BAF/v1/321001 reproduced NaN evaluation and found negative BN running variance from round 2; not all missing original jobs have independently identified causes. The diagnostic report is reports/priority32_numerical_failure_diagnosis_20261002.md.

The human-approved repair transforms trainable parameters only, restoring all nontrainable local BN buffers before ordinary sample-weighted FedAvg. BN buffers remain transmitted RAW. This is NOT FedBN and does NOT protect the BN leakage channel. Results do not replace earlier full-state evidence or the registered PaySim result. All 126 transform jobs, including 25 previously finite jobs, were rerun in the new repair namespace. 63 baseline files were reused as byte-identical copies; every frozen source/copy checksum was verified. No clamping, imputation, seed exclusions or scientific tuning.

Training completed 126/126 with zero failures in 4272.953714 seconds (71.215895 minutes). No repaired training interruption or resume. The postprocess first attempt stopped before statistics/cost because valid_result received a string instead of a Path. The path conversion alone was fixed and the driver resumed; old stderr and the interruption entry remain retained. No training output was recomputed.

Cost stage: all 90 cells completed in 8485.515731 seconds (141.425262 minutes), no failed subprocess or resume.

Postprocess exact launch: .venv-phase1/bin/python -B -u experiments/priority32_repaired_postprocess.py. Finalization: .venv-phase1/bin/python -B experiments/finalize_priority32_repaired.py. Long drivers used nohup, stdin DEVNULL, separate append-only stdout/stderr and detached sessions. All original/repair unit tests passed. Prepared audit/preprocessing source files remain under the original prepared/<dataset>/ namespace; only audit metadata was copied into the repair namespace for cost shape routing.
