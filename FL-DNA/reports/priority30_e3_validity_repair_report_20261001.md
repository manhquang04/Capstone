# Priority 30 — E3 validity repair (2026-10-01)

Status: corrected analysis. The final verdict table in `reports/priority30_native_audit_report_20260930.md` is SUPERSEDED. That report and all old artifacts remain unchanged.

## Protocol and execution

Protocol: `protocols/amendments/2026-10-01_priority30_e3_validity_repair.md`. Utility length/lr selection was clarified before any utility output. No new source targets were drawn; attacks reuse the 39 S1/S2 targets. Every new result asserts source IDs equal the paired S0u artifact. No Latex/ or external_defenses/ edits; one torch thread per process.

Commands executed:

```sh
external_defenses/.venv/bin/python -B experiments/priority30_native_defenses/repair_e3_validity.py --stage all --workers 8
kill -INT 63696
# Initial parent received SIGINT before utility execution after the continuation required validation-selected training length.
# Its queued Soteria jobs were allowed to finish; completed files are preserved and never recomputed.
external_defenses/.venv/bin/python -B experiments/priority30_native_defenses/repair_e3_validity.py --stage utility --workers 8
external_defenses/.venv/bin/python -B experiments/priority30_native_defenses/analyze_e3_validity_repair.py
external_defenses/.venv/bin/python -B experiments/priority30_native_defenses/finalize_e3_validity_repair.py
```

Execution journal: `artifacts/priority30_native_defenses/audit/e3_repair_20261001/journal.log`. An attempted pytest invocation failed because the read-only reference venv has no pytest; the synthetic gate/bracketing tests were invoked directly and passed. No package was installed. See final checks manifest for compile/diff checks.

## PRECODE distortion is not defined in a common parameter space

Old calibration (`run_e3_dp_comparators.py:310–312`) flattened both gradients, used `shared=min(b.numel(),d.numel())`, and took `norm(d[:shared]-b[:shared])`. LeNet-Zhu has 15,826 parameters; PRECODE has 606,930. The first six tensors (body convolutions, 8,136 coordinates) match names/shapes. After that, base `fc.0.weight` (7,680) and bias (10) were compared to the first 7,690 coordinates of PRECODE `bottleneck.encoder.weight` (393,216). Thus the reported median 30.9532146454 is a mismatched-prefix norm, not defense L2 distortion. The gradients through shared convolutions also differ due to a different stochastic forward path; they do not define the full-vector comparator. Adult old calibration used a pass-through (`run_e3_dp_comparators.py:279–284`), yielding zero. PRECODE E3(ii) is NOT_APPLICABLE in BOTH domains. Adult is assessed using repaired E3(iii) only; image has no assessable matched-DP arm.

## Repaired Adult Soteria distortion calibration

| C | sigma | median L2 distortion | dimension | epsilon (one release, delta=1e-5) |
|---:|---:|---:|---:|---:|
| 1.71000306606 | 0.00181114791497 | 0.44775941968 | 20902 | 155076.53087 |

Sensitivity mask uses the real Adult Soteria algorithm at 40th percentile on `layers.3.weight` (`run_audit.py:855–871`), not a pass-through. Seed 42300+target_id; base/defended gradient computed on the same model. Each of the 39 distortions is positive. Sigma matches expected Gaussian L2 scale using the inherited `median_distortion/(C*sqrt(d))` formula. C is the p95 norm; clipping in the top 5% can add distortion, so this is an expected-noise-scale match, not an exact match per target. Accounting: update-level add/remove, sensitivity ratio 1, Gaussian RDP, no subsampling.

## Adult utility repair

Re-reading the old CSV: v1 is exactly 0.7543160915 for all 16 seeds; baseline spans 0.7543160915–0.7544488907; DP sigma<=0.03 spans 0.7543160915–0.7545152903. Thus the majority-class degeneracy is real, but the claim that every baseline/DP value is bit-identical is not exact. The old S4 selections/comparisons are superseded in full.

```json
{
  "status": "CALIBRATED",
  "clip_norm": 0.6888320595026016,
  "lr": 0.3,
  "rounds": 400,
  "grid": [
    0.0001,
    0.0003,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3
  ],
  "seeds": [
    42016,
    42017,
    42018,
    42019,
    42020,
    42021,
    42022,
    42023,
    42024,
    42025,
    42026,
    42027,
    42028,
    42029,
    42030,
    42031
  ],
  "baseline_gate": {
    "means": {
      "validation_accuracy": 0.8549540005624294,
      "test_accuracy": 0.8471032567322254
    },
    "majority_validation": 0.7510359883308411,
    "majority_test": 0.7543160915374756,
    "passed": true
  },
  "selected": {
    "precode": {
      "sigma": 0.01,
      "bracketed": true,
      "threshold": -0.005963453203439713,
      "mean_validation_delta": -0.0018233060836791992,
      "epsilon_delta_1e_minus_5_one_release": 5480.258509299404,
      "epsilon_50_releases": 253393.07076067306,
      "epsilon_training_rounds": 2009597.0522770707
    },
    "dna_v1_conservative": {
      "sigma": 0.01,
      "bracketed": true,
      "threshold": -0.00495855987071991,
      "mean_validation_delta": -0.0018233060836791992,
      "epsilon_delta_1e_minus_5_one_release": 5480.258509299404,
      "epsilon_50_releases": 253393.07076067306,
      "epsilon_training_rounds": 2009597.0522770707
    }
  }
}
```

Validation is stratified 20% of Adult train (seed 43001). Scaling is fit on the remaining train only. LR and rounds are selected on two validation-only seeds; DP selection uses 16 paired validation replicates. Test accuracy is descriptive and used only for the pre-frozen baseline quality gate. PRECODE includes official KL loss and 8-pass probability averaging for evaluation.

| Branch | sigma | validation mean (SD) | paired validation delta | test mean (SD) | paired test delta |
|---|---:|---:|---:|---:|---:|
| baseline | — | 0.854954 (0.001054) | +0.000000 | 0.847103 (0.000815) | +0.000000 |
| dp | 0.0001 | 0.854881 (0.001132) | -0.000073 | 0.847203 (0.000851) | +0.000100 |
| dp | 0.0003 | 0.854830 (0.001057) | -0.000124 | 0.847053 (0.000986) | -0.000050 |
| dp | 0.001 | 0.855047 (0.001147) | +0.000093 | 0.847000 (0.001011) | -0.000104 |
| dp | 0.003 | 0.855058 (0.001369) | +0.000104 | 0.846800 (0.000913) | -0.000303 |
| dp | 0.01 | 0.853131 (0.003479) | -0.001823 | 0.845887 (0.003356) | -0.001216 |
| dp | 0.03 | 0.843838 (0.003669) | -0.011116 | 0.838538 (0.003068) | -0.008566 |
| dp | 0.1 | 0.763809 (0.046988) | -0.091145 | 0.759367 (0.049537) | -0.087737 |
| dp | 0.3 | 0.753325 (0.024310) | -0.101629 | 0.749635 (0.025887) | -0.097468 |
| precode | — | 0.853991 (0.001502) | -0.000963 | 0.847817 (0.001125) | +0.000714 |
| v1 | — | 0.854995 (0.001356) | +0.000041 | 0.847327 (0.000950) | +0.000224 |

## Every E2/E3 test: effects, data-free references and whole-family Holm

Holm uses ONE family of 111 finite hypotheses: all valid E1/E2 vs undefended/data-free tests and both directions of all valid E3 tests. Invalid PRECODE distortion and old S4 tests are excluded. This is more conservative than the old separate-family output. Strict exact-zero ties; one-sided Binomial(non-tied, 0.5). E1 results remain in the family and full CSV.

Quality: image PSNR dB (SSIM in parentheses), Adult feature accuracy %. Reference columns are gray/CIFAR mean for image and mean-mode/empirical marginal for Adult. Higher quality means more leakage. The reference flag compares both medians to the better baseline median, descriptively only. No practical-equivalence claim or verdict adjustment follows from this flag.

| Domain | Defense | Evaluation/comparator | wins/losses/ties | defense quality (SSIM) | comparator quality (SSIM) | paired median quality delta | refs 1/2 | raw p defense / comparator | Holm p defense / comparator | reference flag |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| adult | count_sketch | E2 / empirical_marginal | 1/38/0 | 88.3929 | 54.4643 | +33.9286 | 52.6786/54.4643 | 1 | 1 |  |
| adult | count_sketch | E2 / mean_mode | 0/39/0 | 88.3929 | 52.6786 | +37.5000 | 52.6786/54.4643 | 1 | 1 |  |
| adult | count_sketch | E2 / undefended | 37/0/2 | 88.3929 | 99.1071 | -9.8214 | 52.6786/54.4643 | 7.2759576e-12 | 5.8207661e-10 |  |
| adult | count_sketch | E3 / distortion_matched_clipped | 1/38/0 | 88.3929 | 64.2857 | +22.3214 | 52.6786/54.4643 | 1 / 7.2759576e-11 | 1 / 5.7480065e-09 |  |
| adult | dna_v1_conservative | E2 / empirical_marginal | 0/39/0 | 95.5357 | 54.4643 | +41.9643 | 52.6786/54.4643 | 1 | 1 |  |
| adult | dna_v1_conservative | E2 / mean_mode | 0/39/0 | 95.5357 | 52.6786 | +42.8571 | 52.6786/54.4643 | 1 | 1 |  |
| adult | dna_v1_conservative | E2 / undefended | 29/6/4 | 95.5357 | 99.1071 | -1.7857 | 52.6786/54.4643 | 5.8420934e-05 | 0.0039726235 |  |
| adult | dna_v1_conservative | E3 / distortion_matched_clipped | 26/7/6 | 95.5357 | 99.1071 | -1.7857 | 52.6786/54.4643 | 0.00065936358 / 0.99983797 | 0.042858633 / 1 |  |
| adult | dna_v1_conservative | E3 / utility_matched_clipped | 0/39/0 | 95.5357 | 75.0000 | +19.6429 | 52.6786/54.4643 | 1 / 1.8189894e-12 | 1 / 2.0190782e-10 |  |
| adult | dna_v2_0p95 | E2 / empirical_marginal | 0/39/0 | 100.0000 | 54.4643 | +44.6429 | 52.6786/54.4643 | 1 | 1 |  |
| adult | dna_v2_0p95 | E2 / mean_mode | 0/39/0 | 100.0000 | 52.6786 | +46.4286 | 52.6786/54.4643 | 1 | 1 |  |
| adult | dna_v2_0p95 | E2 / undefended | 7/11/21 | 100.0000 | 99.1071 | +0.0000 | 52.6786/54.4643 | 0.88105774 | 1 |  |
| adult | dna_v2_0p95 | E3 / distortion_matched_clipped | 8/21/10 | 100.0000 | 99.1071 | +0.8929 | 52.6786/54.4643 | 0.99593497 / 0.012059772 | 1 / 0.73564612 |  |
| adult | gradient_pruning | E2 / empirical_marginal | 0/39/0 | 98.2143 | 54.4643 | +41.0714 | 52.6786/54.4643 | 1 | 1 |  |
| adult | gradient_pruning | E2 / mean_mode | 0/39/0 | 98.2143 | 52.6786 | +43.7500 | 52.6786/54.4643 | 1 | 1 |  |
| adult | gradient_pruning | E2 / undefended | 17/12/10 | 98.2143 | 99.1071 | +0.0000 | 52.6786/54.4643 | 0.22912916 | 1 |  |
| adult | gradient_pruning | E3 / distortion_matched_clipped | 4/35/0 | 98.2143 | 91.0714 | +5.3571 | 52.6786/54.4643 | 0.99999998 / 1.6765807e-07 | 1 / 1.2406697e-05 |  |
| adult | gradient_pruning | E3 / paper_unclipped | 0/39/0 | 98.2143 | 45.5357 | +49.1071 | 52.6786/54.4643 | 1 / 1.8189894e-12 | 1 / 2.0190782e-10 |  |
| adult | precode | E2 / empirical_marginal | 39/0/0 | 23.2143 | 54.4643 | -33.0357 | 52.6786/54.4643 | 1.8189894e-12 | 2.0190782e-10 | both at reference level |
| adult | precode | E2 / mean_mode | 39/0/0 | 23.2143 | 52.6786 | -30.3571 | 52.6786/54.4643 | 1.8189894e-12 | 2.0190782e-10 | both at reference level |
| adult | precode | E2 / undefended | 39/0/0 | 23.2143 | 99.1071 | -74.1071 | 52.6786/54.4643 | 1.8189894e-12 | 2.0190782e-10 |  |
| adult | precode | E3 / paper_unclipped | 39/0/0 | 23.2143 | 75.0000 | -50.8929 | 52.6786/54.4643 | 1.8189894e-12 / 1 | 2.0190782e-10 / 1 |  |
| adult | precode | E3 / utility_matched_clipped | 39/0/0 | 23.2143 | 76.7857 | -51.7857 | 52.6786/54.4643 | 1.8189894e-12 / 1 | 2.0190782e-10 / 1 |  |
| adult | soteria | E2 / empirical_marginal | 1/38/0 | 100.0000 | 54.4643 | +44.6429 | 52.6786/54.4643 | 1 | 1 |  |
| adult | soteria | E2 / mean_mode | 0/39/0 | 100.0000 | 52.6786 | +45.5357 | 52.6786/54.4643 | 1 | 1 |  |
| adult | soteria | E2 / undefended | 6/15/18 | 100.0000 | 99.1071 | +0.0000 | 52.6786/54.4643 | 0.98669815 | 1 |  |
| adult | soteria | E3 / distortion_matched_clipped | 2/37/0 | 100.0000 | 90.1786 | +7.1429 | 52.6786/54.4643 | 1 / 1.4206307e-09 | 1 / 1.0796794e-07 |  |
| adult | soteria | E3 / paper_unclipped | 0/39/0 | 100.0000 | 45.5357 | +51.7857 | 52.6786/54.4643 | 1 / 1.8189894e-12 | 1 / 2.0190782e-10 |  |
| image | count_sketch | E2 / cifar_mean | 0/39/0 | 21.1584 (0.7625) | 12.2992 (0.1015) | +8.5156 | 12.0222/12.2992 | 1 | 1 |  |
| image | count_sketch | E2 / gray | 0/39/0 | 21.1584 (0.7625) | 12.0222 (0.1037) | +8.5598 | 12.0222/12.2992 | 1 | 1 |  |
| image | count_sketch | E2 / undefended | 39/0/0 | 21.1584 (0.7625) | 23.4330 (0.8496) | -2.3768 | 12.0222/12.2992 | 1.8189894e-12 | 2.0190782e-10 |  |
| image | count_sketch | E3 / distortion_matched_clipped | 0/39/0 | 21.1584 (0.7625) | 7.9640 (0.0387) | +12.8176 | 12.0222/12.2992 | 1 / 1.8189894e-12 | 1 / 2.0190782e-10 |  |
| image | dna_v1_conservative | E2 / cifar_mean | 0/39/0 | 22.6500 (0.8272) | 12.2992 (0.1015) | +10.1890 | 12.0222/12.2992 | 1 | 1 |  |
| image | dna_v1_conservative | E2 / gray | 0/39/0 | 22.6500 (0.8272) | 12.0222 (0.1037) | +10.2084 | 12.0222/12.2992 | 1 | 1 |  |
| image | dna_v1_conservative | E2 / undefended | 35/4/0 | 22.6500 (0.8272) | 23.4330 (0.8496) | -0.7617 | 12.0222/12.2992 | 1.6765807e-07 | 1.2406697e-05 |  |
| image | dna_v1_conservative | E3 / distortion_matched_clipped | 33/6/0 | 22.6500 (0.8272) | 23.5465 (0.8491) | -0.6724 | 12.0222/12.2992 | 7.1496306e-06 / 0.99999879 | 0.00050762377 / 1 |  |
| image | dna_v2_0p95 | E2 / cifar_mean | 0/39/0 | 23.4945 (0.8523) | 12.2992 (0.1015) | +10.9682 | 12.0222/12.2992 | 1 | 1 |  |
| image | dna_v2_0p95 | E2 / gray | 0/39/0 | 23.4945 (0.8523) | 12.0222 (0.1037) | +11.0884 | 12.0222/12.2992 | 1 | 1 |  |
| image | dna_v2_0p95 | E2 / undefended | 27/12/0 | 23.4945 (0.8523) | 23.4330 (0.8496) | -0.0437 | 12.0222/12.2992 | 0.011851351 | 0.73478378 |  |
| image | dna_v2_0p95 | E3 / distortion_matched_clipped | 0/39/0 | 23.4945 (0.8523) | 22.0672 (0.8024) | +1.3913 | 12.0222/12.2992 | 1 / 1.8189894e-12 | 1 / 2.0190782e-10 |  |
| image | gradient_pruning | E2 / cifar_mean | 0/39/0 | 20.6596 (0.7421) | 12.2992 (0.1015) | +8.1388 | 12.0222/12.2992 | 1 | 1 |  |
| image | gradient_pruning | E2 / gray | 0/39/0 | 20.6596 (0.7421) | 12.0222 (0.1037) | +8.1168 | 12.0222/12.2992 | 1 | 1 |  |
| image | gradient_pruning | E2 / undefended | 38/1/0 | 20.6596 (0.7421) | 23.4330 (0.8496) | -2.6229 | 12.0222/12.2992 | 7.2759576e-11 | 5.7480065e-09 |  |
| image | gradient_pruning | E3 / distortion_matched_clipped | 31/8/0 | 20.6596 (0.7421) | 22.1811 (0.7984) | -1.3416 | 12.0222/12.2992 | 0.00014703844 / 0.99996487 | 0.0098515753 / 1 |  |
| image | gradient_pruning | E3 / paper_unclipped | 0/39/0 | 20.6596 (0.7421) | 18.2722 (0.5758) | +2.7854 | 12.0222/12.2992 | 1 / 1.8189894e-12 | 1 / 2.0190782e-10 |  |
| image | precode | E2 / cifar_mean | 38/1/0 | 9.3893 (0.0932) | 12.2992 (0.1015) | -2.9712 | 12.0222/12.2992 | 7.2759576e-11 | 5.7480065e-09 | both at reference level |
| image | precode | E2 / gray | 39/0/0 | 9.3893 (0.0932) | 12.0222 (0.1037) | -2.8710 | 12.0222/12.2992 | 1.8189894e-12 | 2.0190782e-10 | both at reference level |
| image | precode | E2 / undefended | 39/0/0 | 9.3893 (0.0932) | 23.4330 (0.8496) | -14.2490 | 12.0222/12.2992 | 1.8189894e-12 | 2.0190782e-10 |  |
| image | precode | E3 / paper_unclipped | 33/6/0 | 9.3893 (0.0932) | 23.4876 (0.8498) | -14.0505 | 12.0222/12.2992 | 7.1496306e-06 / 0.99999879 | 0.00050762377 / 1 |  |
| image | soteria | E2 / cifar_mean | 0/39/0 | 24.2869 (0.9006) | 12.2992 (0.1015) | +12.1185 | 12.0222/12.2992 | 1 | 1 |  |
| image | soteria | E2 / gray | 0/39/0 | 24.2869 (0.9006) | 12.0222 (0.1037) | +12.3892 | 12.0222/12.2992 | 1 | 1 |  |
| image | soteria | E2 / undefended | 1/38/0 | 24.2869 (0.9006) | 23.4330 (0.8496) | +1.6426 | 12.0222/12.2992 | 1 | 1 |  |
| image | soteria | E3 / distortion_matched_clipped | 0/39/0 | 24.2869 (0.9006) | 14.1486 (0.3319) | +11.2274 | 12.0222/12.2992 | 1 / 1.8189894e-12 | 1 / 2.0190782e-10 |  |
| image | soteria | E3 / paper_unclipped | 0/39/0 | 24.2869 (0.9006) | 18.2619 (0.5880) | +7.5041 | 12.0222/12.2992 | 1 / 1.8189894e-12 | 1 / 2.0190782e-10 |  |

## RQ4 re-issued verdicts

This table supersedes the old verdict table. Failure of E2 takes precedence; missing required matched-DP evidence is NOT_ASSESSABLE. Paper-unclipped noise is descriptive, not the matched-DP criterion. Failure to reject a DP advantage is not proof of equivalence.

| Domain | Defense | Old (superseded) | New | Changed | Reason |
|---|---|---|---|---|---|
| image | precode | SURVIVES | NOT_ASSESSABLE | True | E2 passes; DP comparison NOT_ASSESSABLE (different parameter spaces) |
| image | soteria | DOES NOT SURVIVE | DOES NOT SURVIVE | False | E2 not significant under whole-family Holm |
| image | gradient_pruning | SURVIVES | SURVIVES | False | E2 passes; no significant loss to required matched DP |
| image | ats | NOT_ASSESSABLE | NOT_ASSESSABLE | False | ATS has no working adaptive attacker |
| image | count_sketch | DOES NOT SURVIVE | DOES NOT SURVIVE | False | matched DP significantly better: distortion_matched_clipped |
| image | dna_v1_conservative | SURVIVES | SURVIVES | False | E2 passes; no significant loss to required matched DP |
| image | dna_v2_0p95 | DOES NOT SURVIVE | DOES NOT SURVIVE | False | E2 not significant under whole-family Holm |
| adult | precode | SURVIVES | SURVIVES | False | E2 passes; no significant loss to required matched DP |
| adult | soteria | DOES NOT SURVIVE | DOES NOT SURVIVE | False | E2 not significant under whole-family Holm |
| adult | gradient_pruning | DOES NOT SURVIVE | DOES NOT SURVIVE | False | E2 not significant under whole-family Holm |
| adult | count_sketch | DOES NOT SURVIVE | DOES NOT SURVIVE | False | matched DP significantly better: distortion_matched_clipped |
| adult | dna_v1_conservative | DOES NOT SURVIVE | DOES NOT SURVIVE | False | matched DP significantly better: utility_matched_clipped |
| adult | dna_v2_0p95 | DOES NOT SURVIVE | DOES NOT SURVIVE | False | E2 not significant under whole-family Holm |

Changed verdicts:

- image / precode: SURVIVES → NOT_ASSESSABLE; E2 passes; DP comparison NOT_ASSESSABLE (different parameter spaces).

## Artifacts and hashes

All new per-target results, arrays, split IDs, per-seed grid results, analysis CSVs and their SHA-256 hashes are listed in `artifacts/priority30_native_defenses/audit/e3_repair_20261001/sha256_manifest.csv`. Full tests/effects/references: `S6/tests_with_effects_and_references.csv`; per-target comparisons: `S6/per_target.csv`; exclusions: `S6/unavailable.json`; supersession map: `S6/supersession.json`.

Final verification is recorded in `checks.json`: py_compile, git diff --check, 3 direct unit/statistical tests, unchanged old summary hashes, and an independent SciPy/NumPy cross-check of all 111 exact-binomial/Holm hypotheses. A final process check is performed after all workload exits.

Supplementary execution and reference-level facts:

- Amendment filesystem creation: 17:17:35 on 2026-10-01; utility-only
  clarification mtime: 17:26:17. First Soteria output: 17:24:12;
  first utility output: 17:32:56. Both affected scopes were specified before
  their outputs. Exact epochs are recorded in `checks.json`.
- New S4 grid has 16 baseline, 16 PRECODE, 16 v1 and 128 DP training results;
  the 18 validation-only learning-rate/length fits are separate. No grid
  extension was needed. All 39 Soteria and all 78 utility comparator attacks
  completed. The utility attack stage took 1227.5 seconds; its final journal
  entry is 18:06:48 +07.
- No **E3 defense–DP pair** has both median reconstruction qualities at/below
  the stronger data-free reference in the corrected data. PRECODE alone is
  below the Adult reference, while its DP comparator remains above it.
  Flags in E2 comparisons against a data-free guess refer to that named
  comparator, not to a DP arm. The medians of the references are
  image gray/CIFAR mean = 12.022220/12.299180 dB and Adult mean-mode/empirical
  marginal = 52.678571/54.464286% on these 39 targets.
- Utility-DP epsilon is 5480.258509 for one release, 253393.070761 for
  50 releases, and 2009597.052277 for the 400 training-round composition
  (same update-level add/remove convention). It remains weak formal
  protection; these are not record-level DP claims.
- Re-issued verdict totals: **3 SURVIVES, 8 DOES NOT SURVIVE,
  2 NOT_ASSESSABLE**. Only image PRECODE changed from the old table.
