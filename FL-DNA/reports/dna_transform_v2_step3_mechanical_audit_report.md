# DNA Transform v2 — Step 3 mechanical audit report

**Date:** 2026-09-16  
**Status:** STEP 3 COMPLETE — MECHANICAL GATE PASS  
**Scope:** rank/conditioning diagnostics and known-seed reconstruction error only.

## 1. Đã làm gì

Ran a mechanical audit for DNA Transform v2 on development artifacts only:

- checked whether the v2 projection is rank-deficient as designed;
- measured known-seed Level-2-equivalent reconstruction error after sketching
  and lifting;
- compared the observed reconstruction error qualitatively against the v1
  Level-2 near-zero inversion error (`~3.4e-08`);
- preserved the v1 implementation unchanged.

This step did not run FL utility, gradient-inversion security calibration,
confirmatory target generation, or RQ1/RQ2/RQ3 experiments.

## 2. File/config thay đổi

New files:

- `experiments/audit_transform_v2_mechanics.py`
- `artifacts/dna_transform_v2/step3_mechanics_20260916_v2/mechanical_audit_report.json`
- `artifacts/dna_transform_v2/step3_mechanics_20260916_v2/group_reconstruction_errors.csv`
- `artifacts/dna_transform_v2/step3_mechanics_20260916_v2/tensor_rank_and_error_detail.csv`
- `artifacts/dna_transform_v2/step3_mechanics_20260916_v2/small_matrix_rank_checks.csv`
- `artifacts/dna_transform_v2/step3_mechanics_20260916_v2/summary.csv`
- `reports/dna_transform_v2_step3_mechanical_audit_report.md`

Updated:

- `../PROJECT.md`

Unchanged v1/v2 implementation hashes:

```text
dna_encoder/transform_defense.py
407d126e0c0d5d7bb1903a763bc7ff536e6aa8f60f296eddd4bbed3217c1a181

dna_encoder/transform_defense_v2.py
a73f060b049cbdcf13ffae108e0a6b963aea9f2efabff5fd8f0922bf2878b629
```

## 3. Development data used

Only the development target file below was used:

```text
artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/development_gate_targets.pt
SHA-256: 7ad42895965b0cc0d349c9d61fd39642eaf8c1e8b866a22100e55330b32e183b
```

The audit read corresponding development raw-update artifacts from:

```text
artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/
  dna_level1_forward_attack_development_gate_c4_r8_i600_lr0p1_nonneg0p001/
```

No post-hoc/confirmatory target set was used.

## 4. Lệnh thực sự đã chạy

```bash
python3 -m py_compile experiments/audit_transform_v2_mechanics.py

PYTHONPATH=. .venv-phase1/bin/python \
  experiments/audit_transform_v2_mechanics.py \
  --output-dir artifacts/dna_transform_v2/step3_mechanics_20260916_v2
```

An initial attempt failed before producing scientific output because of a
reporting-label mismatch (`0p90` vs `0p9`). The label was corrected and the
successful run was written to the fresh `_v2` output directory above. No dataset
or post-hoc artifact was modified.

## 5. Kết quả

All audited configurations were rank-deficient at the tensor level and produced
large known-seed reconstruction error.

| Config | k/d | eta | Mean relative L2 error | Median relative L2 error | Min | Max | Mean rank-retained upper fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2_ratio0p95_eta0` | 0.95 | 0.00 | 0.3003 | 0.3311 | 0.1996 | 0.3420 | 0.9570 |
| `v2_ratio0p95_eta0p01` | 0.95 | 0.01 | 0.3004 | 0.3312 | 0.1997 | 0.3421 | 0.9570 |
| `v2_ratio0p95_eta0p02` | 0.95 | 0.02 | 0.3004 | 0.3313 | 0.1996 | 0.3422 | 0.9570 |
| `v2_ratio0p9_eta0` | 0.90 | 0.00 | 0.3411 | 0.3468 | 0.3064 | 0.3707 | 0.9134 |
| `v2_ratio0p9_eta0p01` | 0.90 | 0.01 | 0.3412 | 0.3468 | 0.3065 | 0.3707 | 0.9134 |
| `v2_ratio0p9_eta0p02` | 0.90 | 0.02 | 0.3413 | 0.3469 | 0.3066 | 0.3707 | 0.9134 |

The observed relative L2 errors (`~0.30–0.34`) are many orders of magnitude
larger than the v1 Level-2 near-exact inversion error (`~3.4e-08`). This is the
expected mechanical evidence that v2 is genuinely lossy, unlike v1.

## 6. Artifact-run ID and hashes

Artifact-run ID:

```text
artifacts/dna_transform_v2/step3_mechanics_20260916_v2
```

Hashes:

```text
experiments/audit_transform_v2_mechanics.py
cdced025a78e16b9fac347c3300e1d21fcfd167ba739477d27d28c1aa23c84e7

artifacts/dna_transform_v2/step3_mechanics_20260916_v2/mechanical_audit_report.json
8d890bfb08f429b8895ff19715ea383d7780a467642adae63db78df8ad5e2e96

artifacts/dna_transform_v2/step3_mechanics_20260916_v2/summary.csv
58e56f9e6a6eddabe83e00112c2d5e9de1d7287ef9603bce32139d8afcecab80
```

## 7. Gate đạt/chưa đạt

| Gate | Status | Reason |
| --- | --- | --- |
| v1 code untouched | PASS | v1 hash unchanged |
| v2 projection rank loss observed | PASS | all audited configs report rank loss |
| Known-seed reconstruction not near-exact | PASS | relative L2 error around `0.30–0.34` |
| Error much larger than v1 Level 2 | PASS | `0.30–0.34` vs `~3.4e-08` |
| Utility evidence | NOT RUN | belongs to Step 4 |
| Security/RQ1 evidence | NOT RUN | belongs to Step 5 after utility passes |

## 8. Điều chưa thể kết luận

Step 3 only proves the intended mechanical difference from v1: v2 loses
information even when the seed/projection is known. It does not prove that v2
preserves model utility, improves privacy under attack, or is deployable.

The reconstruction error is intentionally large enough to establish lossiness,
but that same lossiness may damage F1/AUC. That must be checked before any
security evaluation.

## 9. Bước tiếp theo được phép

Step 4 may begin next:

- run v2 in the FL pipeline at development scale;
- measure F1/AUC against baseline and v1;
- stop if utility collapses before running any v2 security/RQ1 calibration.

Do not proceed to Step 5 or any confirmatory v2 experiment until Step 4 utility
is reported.
