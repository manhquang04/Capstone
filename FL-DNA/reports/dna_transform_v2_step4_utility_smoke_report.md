# DNA Transform v2 — Step 4 utility smoke report

**Date:** 2026-09-16  
**Status:** STEP 4 COMPLETE — DEVELOPMENT UTILITY SMOKE PASS  
**Scope:** development-scale FL utility only. No attack, no RQ1 calibration, no
confirmatory execution.

## 1. Đã làm gì

Ran a small paired FL utility smoke test for DNA Transform v2 before any
security evaluation.

Frozen smoke design:

- dataset: PaySim development-scale sample;
- `MAX_ROWS=100000`;
- `NUM_ROUNDS=10`;
- `LOCAL_EPOCHS=1`;
- `FL_NUM_CLIENTS=3`;
- seeds: `[101, 202, 303]`;
- methods:
  - baseline;
  - `v2_ratio0p95_eta0p01`;
  - `v2_ratio0p9_eta0p01`.

The goal was only to detect obvious utility collapse. This is not a
confirmatory RQ2 non-inferiority experiment.

## 2. File/config thay đổi

New scripts:

- `experiments/run_fraud_fl_dna_transform_v2.py`
- `experiments/run_dna_transform_v2_step4_utility_smoke.py`
- `experiments/analyze_dna_transform_v2_step4_utility_smoke.py`

New artifacts:

- `artifacts/dna_transform_v2/step4_utility_smoke_20260916/smoke_manifest.json`
- `artifacts/dna_transform_v2/step4_utility_smoke_20260916/smoke_summary.json`
- per-job `metrics.json`, `stdout.log`, `stderr.log`, and `job_record.json`

New analysis outputs:

- `results/dna_transform_v2/step4_utility_smoke_20260916/paired_utility_rows.csv`
- `results/dna_transform_v2/step4_utility_smoke_20260916/utility_smoke_summary.json`

New report:

- `reports/dna_transform_v2_step4_utility_smoke_report.md`

Updated:

- `../PROJECT.md`

No v1 file was modified.

## 3. Lệnh thực sự đã chạy

```bash
python3 -m py_compile \
  experiments/run_fraud_fl_dna_transform_v2.py \
  experiments/run_dna_transform_v2_step4_utility_smoke.py \
  experiments/analyze_dna_transform_v2_step4_utility_smoke.py

PYTHONPATH=. .venv-phase1/bin/python -m pytest -q \
  tests/test_transform_defense_v2.py

PYTHONPATH=. .venv-phase1/bin/python \
  experiments/run_dna_transform_v2_step4_utility_smoke.py \
  --workers 3 \
  --output-dir artifacts/dna_transform_v2/step4_utility_smoke_20260916

PYTHONPATH=. .venv-phase1/bin/python \
  experiments/analyze_dna_transform_v2_step4_utility_smoke.py \
  --run-dir artifacts/dna_transform_v2/step4_utility_smoke_20260916 \
  --output-dir results/dna_transform_v2/step4_utility_smoke_20260916
```

## 4. Kết quả execution

```text
jobs:    9
success: 9
failed:  0
```

The predefined smoke gate was:

```text
For primary v2_ratio0p95_eta0p01:
mean delta F1 >= -0.05 and mean delta AUC >= -0.01
```

## 5. Kết quả utility

Paired deltas are computed against the baseline with the same seed/split.

| Method | n seeds | Mean ΔF1 | Mean ΔAUC | Mean ΔPR-AUC | Mean final-round v2 transform ms | Mean final-round aggregate relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2_ratio0p95_eta0p01` | 3 | `+0.0075` | `+0.00023` | `-0.00431` | `187.85` | `0.2135` |
| `v2_ratio0p9_eta0p01` | 3 | `-0.0159` | `-0.00182` | `-0.01085` | `190.94` | `0.2847` |

Per-seed primary config results:

| Seed | Baseline F1 | v2 0.95/0.01 F1 | ΔF1 | Baseline AUC | v2 0.95/0.01 AUC | ΔAUC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 0.2667 | 0.2667 | 0.0000 | 0.9744 | 0.9766 | +0.00228 |
| 202 | 0.5263 | 0.5263 | 0.0000 | 0.9582 | 0.9573 | -0.00095 |
| 303 | 0.6296 | 0.6522 | +0.0225 | 0.9906 | 0.9899 | -0.00064 |

## 6. Artifact-run ID and hashes

Artifact-run ID:

```text
artifacts/dna_transform_v2/step4_utility_smoke_20260916
```

Analysis output:

```text
results/dna_transform_v2/step4_utility_smoke_20260916
```

Hashes:

```text
experiments/run_fraud_fl_dna_transform_v2.py
14cac4d1e45174387f16f322bc929393394e1d998040313bd429a42554770184

experiments/run_dna_transform_v2_step4_utility_smoke.py
9adb0c64397937b04b375bdce78d988326d53ce1b2dd598a306ca079a6f65af4

experiments/analyze_dna_transform_v2_step4_utility_smoke.py
dd31b5ebcad467ce546954886a23d91ed9228db42812273d6bc0da4f4605f9fd

artifacts/dna_transform_v2/step4_utility_smoke_20260916/smoke_manifest.json
d9dcc4e0291f43adab64061cf12c008abb899c206e28a373ff3f4170a9257adc

artifacts/dna_transform_v2/step4_utility_smoke_20260916/smoke_summary.json
389c7409214056c46904e948d093cc14c1ed21e58be1adf1593b777acac7ffa7

results/dna_transform_v2/step4_utility_smoke_20260916/utility_smoke_summary.json
ce38bceedbbd239c48a7a4faeb65cefb2e7e791215166601fa9a79ccc4548ed4
```

## 7. Gate đạt/chưa đạt

| Gate | Status | Reason |
| --- | --- | --- |
| All smoke jobs completed | PASS | 9/9 success |
| v1 code untouched | PASS | separate v2 runner only |
| Primary v2 0.95/0.01 smoke utility | PASS | mean ΔF1 `+0.0075`, mean ΔAUC `+0.00023` |
| v2 0.90/0.01 sensitivity | CAUTION | mean ΔF1 `-0.0159`, mean ΔAUC `-0.00182` |
| Security/RQ1 evidence | NOT RUN | belongs to Step 5 after this gate |

## 8. Kết luận

Step 4 passes for development continuation: the conservative v2 configuration
`compression_ratio=0.95`, `quantization_eta=0.01` did not show obvious utility
collapse in this small development smoke.

This does not establish non-inferiority, deployment readiness, or privacy. It
only permits Step 5 development calibration to start, if supervisor approves.

The more lossy `0.90/0.01` configuration is weaker in this smoke and should be
treated cautiously unless a later protocol explicitly justifies testing it.

## 9. Bước tiếp theo được phép

Step 5 may begin as development work only:

- write a v2-specific RQ1 development-calibration protocol/amendment;
- calibrate a DP distortion-matched comparator specifically for v2 0.95/0.01;
- run attacker development gates before any confirmatory target set is created.

No v2 confirmatory/security claim is authorized by Step 4 alone.
