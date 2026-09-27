# DNA Transform v2 — Step 2 implementation report

**Date:** 2026-09-16  
**Status:** STEP 2 COMPLETE — UNIT TESTS PASS  
**Scope:** standalone v2 transform implementation and formula-level unit tests.

## 1. Đã làm gì

Implemented DNA Transform v2 in a new module:

```text
dna_encoder/transform_defense_v2.py
```

No v1 code was modified.  The existing v1 module remains:

```text
dna_encoder/transform_defense.py
```

The v2 implementation follows the Step 1 mathematical design:

```text
R_s = sqrt(n/k) · P_s · H · D_s
q = Q_delta(R_s u)
u_hat = R_s^T q
```

where:

- `H` is a normalized fast Walsh-Hadamard transform over padded power-of-two
  length `n`;
- `D_s` is a deterministic DNA-seeded sign flip;
- `P_s` samples `k` coordinates without replacement;
- `Q_delta` uses stochastic unbiased quantization;
- reconstruction lifts the sketch back into the original update shape.

## 2. File/config thay đổi

New files:

- `dna_encoder/transform_defense_v2.py`
- `tests/test_transform_defense_v2.py`
- `reports/dna_transform_v2_step2_implementation_report.md`

Updated:

- `../PROJECT.md`

Unchanged v1 hash:

```text
dna_encoder/transform_defense.py
407d126e0c0d5d7bb1903a763bc7ff536e6aa8f60f296eddd4bbed3217c1a181
```

## 3. Lệnh thực sự đã chạy

```bash
python3 -m py_compile \
  dna_encoder/transform_defense_v2.py \
  tests/test_transform_defense_v2.py

PYTHONPATH=. .venv-phase1/bin/python -m pytest -q \
  tests/test_transform_defense_v2.py
```

## 4. Kết quả

```text
6 passed in 0.27s
```

Unit tests cover:

1. deterministic output for identical seed/config;
2. shape preservation after reconstruction;
3. rank loss when `k < d`;
4. nonzero reconstruction error for a lossy sketch;
5. empirical unbiasedness of unquantized projection;
6. empirical unbiasedness of stochastic quantization;
7. invalid-config rejection.

## 5. Hashes

```text
dna_encoder/transform_defense_v2.py
a73f060b049cbdcf13ffae108e0a6b963aea9f2efabff5fd8f0922bf2878b629

tests/test_transform_defense_v2.py
a2ff22e46b9248ea6192954db93c03adc5621c6075f3d4ad8797e74c34d5f5e9
```

## 6. Gate đạt/chưa đạt

| Gate | Status | Reason |
| --- | --- | --- |
| Separate v2 module | PASS | implemented in `transform_defense_v2.py` |
| v1 untouched | PASS | v1 hash unchanged |
| Formula-level tests | PASS | 6/6 tests passed |
| Rank-loss mechanical property | PASS at unit level | materialized matrix rank `< d` for `k < d` |
| Utility evidence | NOT RUN | belongs to Step 4 |
| Security evidence | NOT RUN | belongs to Step 5 after utility passes |

## 7. Điều chưa thể kết luận

Passing Step 2 does not mean v2 is useful or private.  It only means the
standalone mathematical transform and reconstruction code matches the Step 1
contract at unit-test level.

No FL utility, attack audit, RQ1 calibration or confirmatory experiment has
been run for v2.

## 8. Bước tiếp theo được phép

Step 3 may begin next:

- audit projection rank and condition/mechanical properties;
- run a Level-2-equivalent known-seed reconstruction test;
- verify that reconstruction error is much larger than the v1 Level-2
  near-zero error (`~3.4e-08`).

Do not proceed to utility or RQ1 security calibration until Step 3 is reported.
