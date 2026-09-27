# Báo cáo confirmatory execution RQ1/RQ2 — 2026-09-13

**Trạng thái:** `COMPLETE — SINGLE EXECUTION CONSUMED`  
**RQ3:** `HELD`, không chạy  
**Rerun/replacement/tuning:** không  
**Target/seed ngoài registry khóa:** không

## 1. Đã làm gì

1. Ghi phê duyệt multiplicity trước execution: DNA Transform vs Baseline là
   giả thuyết chính duy nhất của RQ2 ở alpha 0.05; lossless DNA là sanity check
   riêng. `n=21` giữ nguyên vì DNA Transform/F1 đã quyết định cỡ mẫu.
2. Xác minh target RQ1 SHA-256 `caee0024...` và seed-contract RQ2 SHA-256
   `2015b606...`, chạy lại 34 test trước outcomes.
3. Chạy đúng một batch RQ1: 39 target × raw/DNA/DP, mỗi group là một job độc
   lập; 117/117 SUCCESS, 0 stderr, không retry.
4. Chạy đúng một batch RQ2: 21 seed × Baseline/lossless/Transform/DP-context;
   84/84 SUCCESS, 0 stderr, không retry.
5. Chạy analysis đã định trước, gate theo branch cho RQ1, exact sign-test,
   Holm secondary family, paired confidence intervals và non-inferiority rule
   cho RQ2.

## 2. Kết quả RQ1

Tất cả attacker branch vượt Prior và Zero-update controls:

| Branch | Prior wins / 39 | p | Zero wins / 39 | p | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Raw | 32 | 0.0000351 | 34 | 0.00000121 | PASS |
| DNA Transform | 33 | 0.00000715 | 34 | 0.00000121 | PASS |
| DP distortion-matched | 37 | 1.42e-9 | 36 | 1.80e-8 | PASS |

Primary paired contrast `MSE_DNA - MSE_DP`:

- wins/losses/ties: `19/20/0`, effective `n=39`;
- win probability: `0.4872`, exact 95% CI `[0.3242, 0.6522]`;
- one-sided exact sign-test: `p=0.62537`;
- mean MSE difference: `-242.42`, paired 95% CI `[-523.19, 38.36]`.

PSNR/SSIM secondary paired differences:

- `PSNR_DNA - PSNR_DP`: 95% CI `[0.5108, 2.4751]`; a positive difference
  means DNA reconstruction was descriptively better, not more resistant.
- `SSIM_DNA - SSIM_DP`: 95% CI `[-0.0218, 0.1272]`.
- Predeclared secondary Holm family: all four adjusted p-values are `1.0`;
  no secondary directional hypothesis rejects.

**RQ1 conclusion:** all required branch gates passed, but the confirmatory test
does not show DNA Transform protecting more effectively than distortion-matched
clipping/noise. This is a valid negative result, not proof that the methods are
equivalent.

## 3. Kết quả RQ2

Absolute mean final metrics:

| Method | Mean F1 | Mean AUC-ROC |
| --- | ---: | ---: |
| Baseline | 0.74089 | 0.992837 |
| DNA lossless | 0.74510 | 0.992821 |
| DNA Transform | 0.74232 | 0.992446 |
| DP distortion-matched, contextual | 0.67651 | 0.991884 |

Primary `DNA Transform - Baseline`, `n=21`:

| Endpoint | Mean paired delta | SD | Paired 95% CI | Margin | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| F1 | +0.001426 | 0.020989 | [-0.008128, 0.010980] | -0.020 | PASS |
| AUC-ROC | -0.000391 | 0.001567 | [-0.001104, 0.000323] | -0.005 | PASS |

Both lower confidence bounds are above the approved negative margins.
Therefore DNA Transform **establishes non-inferiority for both F1 and AUC-ROC**
under the frozen endpoint rule.

Lossless sanity check, reported outside the primary hypothesis family:

- F1 delta `+0.004211`, 95% CI `[-0.003789, 0.012211]`;
- AUC delta `-0.0000158`, 95% CI `[-0.000279, 0.000248]`.

Both intervals include zero. The float32 DNA encoder's bit-exact array
round-trip invariant passes the frozen automated test. The confirmatory run
artifacts did not store per-update maximum absolute error, so run-specific
bit equality cannot be reconstructed retrospectively and is reported as an
artifact limitation rather than inferred.

The contextual distortion-matched DP branch had F1 delta `-0.06439`, 95% CI
`[-0.07767, -0.05111]`, and AUC delta `-0.000952`, CI
`[-0.001842, -0.0000631]`. It is not utility-matched and is not a primary RQ2
hypothesis.

## 4. Runtime, artifacts and deviation

- RQ1 run ID: `rq1/confirmatory_run_20260913`; 117 jobs, 5.758 machine-hours,
  median 79.84 s/job, range 57.71–455.07 s.
- RQ2 run ID: `rq2/confirmatory_run_20260913`; 84 jobs, 11.548 machine-hours,
  median 498.14 s/job, range 264.52–568.03 s.
- RQ2 parent orchestration process lost its final aggregate write after all
  children completed. Audit found 84 metrics, 84 SUCCESS records, return code
  0 for every child and 0 non-empty stderr. `execution_summary_rebuilt.json`
  was generated only from immutable job records; no training was rerun.

## 5. Lệnh chính đã chạy

```text
PYTHONPATH=. .venv-phase1/bin/python -u experiments/run_rq1_confirmatory.py --workers 9
PYTHONPATH=. .venv-phase1/bin/python experiments/analyze_rq1_confirmatory.py
PYTHONPATH=. .venv-phase1/bin/python experiments/analyze_rq1_secondary_frozen_family.py
PYTHONPATH=. .venv-phase1/bin/python -u experiments/run_rq2_confirmatory.py --workers 8
PYTHONPATH=. .venv-phase1/bin/python experiments/rebuild_rq2_execution_summary.py
PYTHONPATH=. .venv-phase1/bin/python experiments/analyze_rq2_confirmatory.py
PYTHONPATH=. .venv-phase1/bin/python experiments/create_confirmatory_execution_manifest.py
PYTHONPATH=. .venv-phase1/bin/pytest -q tests
```

## 6. Gates và bước tiếp theo

- RQ1 artifact/completion gate: PASS.
- RQ1 raw/DNA/DP branch gates: PASS.
- RQ1 primary hypothesis: NOT REJECTED; claimed DNA advantage not supported.
- RQ2 artifact/completion gate: PASS.
- RQ2 F1 non-inferiority: PASS.
- RQ2 AUC-ROC non-inferiority: PASS.
- RQ2 combined endpoint rule: PASS.
- Resource ceilings: PASS for RQ1 and RQ2.
- RQ3: HELD; no benchmark executed.

The frozen RQ1/RQ2 confirmatory execution has been consumed. Only immutable
artifact verification, reporting and clearly labeled exploratory analyses are
allowed next; no rerun or parameter change is permitted because of outcomes.

## 7. Post-report wording note from Priority 3 DP accounting

Priority 3 DP accounting, completed on 2026-09-16, found that the
distortion-matched clipping/noise comparator used in the RQ1/RQ2 confirmatory
context has finite but extremely large epsilon under the approved update-level
Gaussian/RDP accountant.  Therefore the DP branch should be described as
`DP-style full-client-update clipping/noise with weak formal accounting` at
comparable update distortion, not as strong record-level Differential Privacy.
This note is interpretive only and does not alter any locked confirmatory
result.
