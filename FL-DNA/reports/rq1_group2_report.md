# RQ1 Group 2 — Transform variants, Level 2 and utility calibration

**Execution date:** 2026-09-13  
**Status:** `COMPLETE`  
**Primary decisions:** medium and stronger do not establish a DNA advantage over their separately distortion-matched clipping/noise comparators.

## 1. Đã làm gì

1. Calibrate `DP_DISTORTION_MATCHED` independently on development data for
   DNA medium and stronger. Medium required the pre-result interpolation grid
   amendment; stronger matched on the original frozen extension grid.
2. Recomputed exact-binomial planning independently for both families:
   37 non-tied observations, rejection at 24 wins, actual alpha 0.0494359,
   power 0.807096, initial draw 44 after planned ties/dropout.
3. Generated exactly one 44-group target for each variant. Both sets are
   source-disjoint from every historical target checked and from one another.
4. Executed raw, DNA Level-1 and clipping/noise-MC once on each frozen target:
   132/132 successful jobs per variant, at most 9 processes and one Torch
   thread per process.
5. Ran realization-known Level 2 direct inversion on the already frozen
   conservative target (`n=39`).
6. Ran the pre-amended, development-only utility grid extension on the same
   eight development seeds. A strict utility match was found at multiplier
   `0.00001`; the search is now permanently closed. No privacy result was used
   to select it, and it was not retrofitted into either variant's primary run.

## 2. File/config thay đổi

- `protocols/amendments/2026-09-13_rq1_variant_and_utility_extension.md`
- `protocols/amendments/2026-09-13_rq1_medium_distortion_grid_extension.md`
- `protocols/config/rq1_medium_pretarget_freeze.json`
- `protocols/config/rq1_stronger_pretarget_freeze.json`
- `protocols/config/rq1_medium_confirmatory.json`
- `protocols/config/rq1_stronger_confirmatory.json`
- `experiments/calibrate_rq1_transform_variants.py`
- `experiments/run_rq2_utility_extension.py`
- `experiments/analyze_rq2_utility_extension.py`
- `experiments/run_rq1_variant_confirmatory.py`
- `experiments/analyze_rq1_variant_confirmatory.py`
- `experiments/run_rq1_conservative_level2.py`
- `protocols/rq1_group2_execution_manifest_2026-09-13.json`

## 3. Lệnh thực sự đã chạy

```bash
.venv-phase1/bin/python experiments/calibrate_rq1_transform_variants.py ...
.venv-phase1/bin/python experiments/run_rq2_utility_extension.py ...
.venv-phase1/bin/python experiments/analyze_rq2_utility_extension.py ...
.venv-phase1/bin/python experiments/rq1_power_analysis.py ...
PYTHONPATH=. .venv-phase1/bin/python experiments/generate_rq1_confirmatory_targets.py ...

.venv-phase1/bin/python experiments/run_rq1_variant_confirmatory.py \
  --config protocols/config/rq1_medium_confirmatory.json \
  --output-dir artifacts/rq1/group2_medium_confirmatory_run_20260913 \
  --workers 9
.venv-phase1/bin/python experiments/analyze_rq1_variant_confirmatory.py \
  --config protocols/config/rq1_medium_confirmatory.json \
  --run-dir artifacts/rq1/group2_medium_confirmatory_run_20260913 \
  --output-dir results/rq1/group2_medium_confirmatory_20260913

.venv-phase1/bin/python experiments/run_rq1_variant_confirmatory.py \
  --config protocols/config/rq1_stronger_confirmatory.json \
  --output-dir artifacts/rq1/group2_stronger_confirmatory_run_20260913 \
  --workers 9
.venv-phase1/bin/python experiments/analyze_rq1_variant_confirmatory.py \
  --config protocols/config/rq1_stronger_confirmatory.json \
  --run-dir artifacts/rq1/group2_stronger_confirmatory_run_20260913 \
  --output-dir results/rq1/group2_stronger_confirmatory_20260913

.venv-phase1/bin/python experiments/run_rq1_conservative_level2.py \
  --output-dir artifacts/rq1/group2_conservative_level2_20260913
```

A later operator command for stronger encountered `FileExistsError` before
opening the target because the completed official output directory already
existed. It performed no attack and is not a rerun; the existing 132-job
execution is the sole scientific run.

## 4. Kết quả kèm uncertainty

### Distortion calibration

| Variant | DNA median relative-L2 | DP multiplier | DP median relative-L2 | mismatch | 5% gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Medium | 0.098758 | 0.000315 | 0.098386 | 0.376% | PASS |
| Stronger | 0.120965 | 0.000400 | 0.124935 | 3.282% | PASS |

### Confirmatory Level-1 comparison

The primary win means `feature-MSE(DNA) > feature-MSE(DP)` outside the frozen
tie threshold. The sign test is primary; mean differences are descriptive and
can point in the opposite direction when a minority of groups has large
outliers.

| Variant | Raw/DNA/DP gates | DNA wins / non-tied | win probability (exact 95% CI) | one-sided p | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Medium | PASS / PASS / PASS | 26/44 | 0.591 [0.432, 0.737] | 0.1456 | No significant DNA advantage |
| Stronger | PASS / PASS / PASS | 25/44 | 0.568 [0.410, 0.717] | 0.2257 | No significant DNA advantage |

There were zero ties and no dropouts. Descriptive mean DNA-minus-DP feature
MSE was `-461.08` (95% t-CI `[-862.91, -59.26]`) for medium and `-517.53`
(`[-953.12, -81.93]`) for stronger. This indicates sensitivity to magnitude/
outliers and does not override the frozen sign-test decision.

Secondary image-style metrics likewise do not establish a consistent ranking:

- medium DNA-minus-DP PSNR CI `[0.061, 2.203]`, SSIM CI `[0.031, 0.226]`;
- stronger PSNR CI `[-0.581, 1.827]`, SSIM CI `[-0.003, 0.192]`.

### Conservative Level 2

All 2,496 transform blocks were full rank. Median condition number was
`1.19048`; mean relative recovery error was `3.41e-08` and maximum was
`7.88e-08`. The recovered-signal attacker passed both controls: 32/39 wins
against Prior (`p=3.51e-05`) and 34/39 against Zero-update (`p=1.21e-06`).
Thus, when the realization seed is known, the linear transform is effectively
invertible at numerical precision; this attacker-favorable sensitivity does
not replace the Level-1 result.

### Utility-matched extension

Multiplier `0.00001` met the strict development tolerances. Against baseline,
mean DP deltas were F1 `-0.012807` and AUC `-0.0007656`; the corresponding DNA
Transform deltas were `-0.012430` and `-0.0003867`. Absolute deviations were
F1 `0.0003774` and AUC `0.0003788`. This is a calibration result only.

## 5. Gate đạt/chưa đạt

| Gate | Status | Reason |
| --- | --- | --- |
| Variant distortion matching | PASS | Both mismatches below frozen 5% tolerance |
| Source disjointness | PASS | Medium/stronger maximum overlap 0 |
| Execution completeness | PASS | 264/264 jobs successful; no scientific rerun |
| Branch validity | PASS | Raw, DNA and DP each beat Prior and Zero controls for both variants |
| Medium primary hypothesis | NOT REJECTED | 26 wins is below rejection boundary; p=0.1456 |
| Stronger primary hypothesis | NOT REJECTED | 25 wins is below rejection boundary; p=0.2257 |
| Level-2 technical gate | PASS | Both controls passed; transform recovered numerically |
| Utility search | CLOSED — MATCH FOUND | Strict match at 0.00001; no reopening permitted |

## 6. Artifact/run IDs

- `artifacts/rq1/group2_medium_confirmatory_run_20260913/`
- `results/rq1/group2_medium_confirmatory_20260913/`
- `artifacts/rq1/group2_stronger_confirmatory_run_20260913/`
- `results/rq1/group2_stronger_confirmatory_20260913/`
- `artifacts/rq1/group2_conservative_level2_20260913/`
- `artifacts/rq2/utility_extension_20260913/`
- Execution manifest: `protocols/rq1_group2_execution_manifest_2026-09-13.json`

## 7. Amendment

Both amendments were timestamped before observing the associated privacy
outcomes. The medium grid amendment was triggered by a 5.120% development
mismatch and added only interpolation points. No parameter was changed after
either confirmatory target was generated.

## 8. Bước tiếp theo được phép

Group 2 now has its separate report. Group 3 may create independently frozen,
source-disjoint conservative replication sets under separate pre-run
protocol/amendment records. Group 4 remains blocked until Group 3 is reported.

## 9. Post-report wording note from Priority 3 DP accounting

Priority 3 DP accounting, completed on 2026-09-16, found that the
distortion-matched clipping/noise comparators used in this report have finite
but extremely large epsilon under the approved update-level Gaussian/RDP
accountant.  Therefore these comparisons should be described as DNA Transform
versus `DP-style full-client-update clipping/noise with weak formal accounting`
at comparable update distortion.  They should not be described as comparisons
against strong record-level Differential Privacy.  This note does not alter any
historical result or p-value in this report.
