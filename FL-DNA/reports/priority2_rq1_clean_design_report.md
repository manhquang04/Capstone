# Priority 2 — RQ1 clean confirmatory redesign and development Pareto/proxy sweep

**Date:** 2026-09-16  
**Supervisor:** Manh Quang  
**Status:** VIỆC 1 COMPLETE; VIỆC 2 DEVELOPMENT-PROXY COMPLETE WITH LIMITATION  
**Confirmatory execution:** NOT STARTED

## 1. Đã làm gì

### Việc 1 — Clean RQ1 design

Created a pre-target amendment for a cleaner RQ1 follow-up:

```text
protocols/amendments/2026-09-16_priority2_rq1_clean_confirmatory_design.md
```

The amendment locks the smaller practically meaningful win probability
requested by the supervisor:

```text
p0 = 0.50
p1 = 0.65
alpha = 0.05
target power = 0.80
planned tie rate = 0.10
planned dropout rate = 0.05
maximum_feasible_target_count = 150
```

Exact-binomial power analysis gives:

```text
effective non-tied n = 69
rejection threshold = 42 wins
actual alpha = 0.04559324528350576
actual power = 0.8020563639664955
initial draw count = 81
```

The design fits within the locked `maximum_feasible_target_count=150`; no
target-count ceiling amendment is needed for `p1=0.65`.

The amendment also records the `p1=0.60` sensitivity note:

```text
p1 = 0.60 -> 158 effective non-tied / 185 initial draw
```

That exceeds the current ceiling and remains unauthorized unless the supervisor
approves a separate target-count amendment.

### Việc 2 — Development Pareto/proxy sweep

Created and ran:

```text
experiments/priority2_pareto_development_proxy.py
```

The script uses development data only:

```text
artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/development_gate_targets.pt
```

It does not create confirmatory targets and does not use old post-hoc target
outcomes for selection.

The sweep includes:

- 6 DNA Transform points from very light to extra strong;
- 11 DP-style clipping/noise multipliers from `0.00001` to `0.01`;
- one non-selection legacy DP attack-measured point at target relative-L2
  `0.1`, retained only as a historical development measurement.

For each DP noise-multiplier point, the script computes epsilon using the
Priority 3 Gaussian/RDP accountant at `delta=1e-5`, add/remove neighboring
updates and a single release.

## 2. File/config thay đổi

New files:

- `protocols/amendments/2026-09-16_priority2_rq1_clean_confirmatory_design.md`
- `experiments/priority2_pareto_development_proxy.py`
- `reports/priority2_rq1_clean_design_report.md`
- `artifacts/rq1/priority2_power_p1_0p65_20260916/rq1_power_analysis.json`
- `artifacts/rq1/priority2_power_p1_0p65_20260916/rq1_power_sensitivity.csv`
- `artifacts/rq1/priority2_pareto_development_20260916/pareto_report.json`
- `artifacts/rq1/priority2_pareto_development_20260916/pareto_points.csv`
- `artifacts/rq1/priority2_pareto_development_20260916/dna_relative_l2_detail.csv`
- `artifacts/rq1/priority2_pareto_development_20260916/dp_relative_l2_detail.csv`
- `artifacts/rq1/priority2_pareto_development_20260916/pareto_proxy.png`

Updated:

- `../PROJECT.md`

No confirmatory config was created and no confirmatory target set was drawn.

## 3. Lệnh thực sự đã chạy

```bash
.venv-phase1/bin/python experiments/rq1_power_analysis.py \
  --p0 0.50 \
  --p1 0.65 \
  --alpha 0.05 \
  --power 0.80 \
  --tie-rate 0.10 \
  --dropout-rate 0.05 \
  --maximum-n 150 \
  --output-dir artifacts/rq1/priority2_power_p1_0p65_20260916

.venv-phase1/bin/python experiments/rq1_power_analysis.py \
  --p0 0.50 \
  --p1 0.60 \
  --alpha 0.05 \
  --power 0.80 \
  --tie-rate 0.10 \
  --dropout-rate 0.05 \
  --maximum-n 300

python3 -m py_compile experiments/priority2_pareto_development_proxy.py

PYTHONPATH=. .venv-phase1/bin/python \
  experiments/priority2_pareto_development_proxy.py \
  --output-dir artifacts/rq1/priority2_pareto_development_20260916
```

## 4. Kết quả Pareto/proxy development sweep

The development sweep currently supports a distortion/utility proxy view, not
a complete reconstruction-resistance Pareto front for every new point.

| Point | Family | median relative-L2 | F1 delta source | F1 delta | epsilon, delta=1e-5 |
| --- | --- | ---: | --- | ---: | ---: |
| `dna_very_light` | DNA | 0.02885 | missing |  |  |
| `dna_current` | DNA | 0.05068 | historical single-seed | -0.02301 |  |
| `dna_conservative` | DNA | 0.08110 | historical/RQ2 context | -0.02003 single-seed; RQ2 confirmatory non-inferior |  |
| `dna_medium` | DNA | 0.09586 | historical single-seed | -0.04341 |  |
| `dna_stronger` | DNA | 0.11755 | historical single-seed | -0.03217 |  |
| `dna_extra_strong` | DNA | 0.15968 | missing |  |  |
| `dp_0.0001` | DP-style | 0.03126 | historical single-seed | -0.03658 | 5.0048e7 |
| `dp_0.0002` | DP-style | 0.06252 | missing |  | 1.2524e7 |
| `dp_0.00025` | DP-style | 0.07815 | RQ2 confirmatory contextual | -0.06439 | 8.0192e6 |
| `dp_0.0003` | DP-style | 0.09378 | missing |  | 5.5716e6 |
| `dp_0.0004` | DP-style | 0.12504 | missing |  | 3.1370e6 |
| `dp_0.0005` | DP-style | 0.15630 | missing |  | 2.0096e6 |
| `dp_0.001` | DP-style | 0.31261 | historical single-seed | -0.45333 | 5.0480e5 |
| `dp_0.005` | DP-style | 1.11357 | historical single-seed | -0.72057 | 2.0976e4 |
| `dp_0.01` | DP-style | 2.22715 | missing |  | 5.4803e3 |

The exact CSV contains all points, including `dp_0.00001`, `dp_0.00005` and the
non-selection legacy attack-measured point:

```text
artifacts/rq1/priority2_pareto_development_20260916/pareto_points.csv
```

Development reconstruction measurement availability:

- `dna_conservative` has a same-development-pool attack measurement from the
  old development gate artifact.  Mean feature-MSE delta vs raw is
  `+8015.33`.
- The legacy DP target-relative-L2 `0.1` point has a same-development-pool
  attack measurement.  Mean feature-MSE delta vs raw is `+8619.23`, but this
  point is not a noise-multiplier candidate and is not selection-eligible.
- New DNA points and DP noise-multiplier grid points do not yet have direct
  reconstruction attack measurements under this Priority 2 sweep.

Therefore, the current plot is a Pareto/proxy plot, not a final
reconstruction-resistance Pareto front:

```text
artifacts/rq1/priority2_pareto_development_20260916/pareto_proxy.png
```

## 5. Kết quả và uncertainty

For Việc 1, exact-binomial uncertainty is determined by the exact tail
calculation:

- actual alpha: `0.04559324528350576`;
- actual power at `p1=0.65`: `0.8020563639664955`.

For Việc 2, no confirmatory uncertainty is claimed.  The relative-L2 estimates
are development-only medians over:

- 8 development groups for DNA points;
- 8 groups x 5 calibration seeds = 40 observations for each DP noise multiplier.

No confirmatory target outcome was generated.

## 6. Gate đạt/chưa đạt và lý do

| Gate | Status | Reason |
| --- | --- | --- |
| Việc 1 p1=0.65 power/design gate | PASS | 69 non-tied / 81 draw fits within target ceiling 150 |
| Việc 1 p1=0.60 under current ceiling | NOT AUTHORIZED | 185 draw exceeds target ceiling 150 |
| Việc 2 development-only scope gate | PASS | Used `development_gate_targets.pt`; no confirmatory target created |
| Việc 2 complete reconstruction Pareto gate | PARTIAL / NOT YET SUFFICIENT | Most new grid points lack direct attack measurements |
| DP wording gate | PASS | Reports must use weak-accounting DP-style wording, not strong record-level DP |

## 7. Artifact/run ID

Power analysis:

```text
artifacts/rq1/priority2_power_p1_0p65_20260916
```

Pareto/proxy sweep:

```text
artifacts/rq1/priority2_pareto_development_20260916
```

## 8. Amendment

```text
protocols/amendments/2026-09-16_priority2_rq1_clean_confirmatory_design.md
```

## 9. Bước tiếp theo được protocol cho phép

Before Việc 3 confirmatory execution, supervisor should review the
development/proxy Pareto report and choose one of two paths:

1. approve a representative point using distortion/utility proxy evidence
   only, most naturally `dna_conservative` against DP-style
   `noise_multiplier=0.00025`; or
2. require a separate development attack sweep for the new DNA/DP points so the
   Pareto front has measured reconstruction resistance on the Y axis before
   selecting one or two confirmatory points.

Only after that selection is frozen should the team create the new
source-disjoint confirmatory target set of 81 draws for the clean RQ1 run.

The RQ1 conclusion wording must include the Priority 3 limitation: this is a
comparison against weak-accounting DP-style clipping/noise at comparable update
distortion, not against strong record-level differential privacy.
