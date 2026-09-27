# DNA Transform v2 — Step 5 development calibration report

**Date:** 2026-09-16  
**Status:** STEP 5 COMPLETE — DEVELOPMENT CALIBRATION STOPS BEFORE CONFIRMATORY  
**Scope:** v2-specific RQ1 development calibration only. No confirmatory target
was created.

## 1. Đã làm gì

Executed the Step 5 development calibration authorized by:

```text
protocols/amendments/2026-09-16_dna_transform_v2_step5_development_calibration.md
```

The frozen v2 configuration was:

```text
compression_ratio = 0.95
quantization_eta = 0.01
base_seed = 20260916
```

Two development-only checks were run:

1. DP distortion calibration against v2 lifted-update distortion.
2. v2 attacker development gate against Prior and Zero-update controls.

No post-hoc/confirmatory target was used and no v2 confirmatory target was
created.

## 2. File/config thay đổi

New amendment:

- `protocols/amendments/2026-09-16_dna_transform_v2_step5_development_calibration.md`

New scripts:

- `experiments/calibrate_rq1_dna_transform_v2_distortion.py`
- `experiments/run_phase4_dna_v2_development_gate.py`

New artifacts:

- `artifacts/dna_transform_v2/step5_distortion_calibration_20260916/`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_v2_development_gate_development_gate_ratio0p95_eta0p01_r4_i600_lr0p1_nonneg0p001/`

New report:

- `reports/dna_transform_v2_step5_development_calibration_report.md`

Updated:

- `../PROJECT.md`

## 3. Lệnh thực sự đã chạy

Validation:

```bash
python3 -m py_compile \
  experiments/calibrate_rq1_dna_transform_v2_distortion.py \
  experiments/run_phase4_dna_v2_development_gate.py \
  experiments/run_fraud_fl_dna_transform_v2.py

PYTHONPATH=. .venv-phase1/bin/python -m pytest -q \
  tests/test_transform_defense_v2.py
```

Distortion calibration:

```bash
PYTHONPATH=. .venv-phase1/bin/python \
  experiments/calibrate_rq1_dna_transform_v2_distortion.py \
  --output-dir artifacts/dna_transform_v2/step5_distortion_calibration_20260916
```

Attacker development gate:

```bash
PYTHONPATH=. .venv-phase1/bin/python \
  experiments/run_phase4_dna_v2_development_gate.py \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --target-file development_gate_targets.pt \
  --restarts 4 \
  --iterations 600 \
  --attack-lr 0.1 \
  --compression-ratio 0.95 \
  --quantization-eta 0.01 \
  --v2-base-seed 20260916
```

## 4. Development data

Only the Phase-4 development target was used:

```text
artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/development_gate_targets.pt
SHA-256: 7ad42895965b0cc0d349c9d61fd39642eaf8c1e8b866a22100e55330b32e183b
```

No post-hoc, confirmatory, replication, or Priority-2 clean target was used.

## 5. DP distortion calibration result

V2 DNA target median relative-L2:

```text
0.3312435936
```

Frozen grid:

```text
[0.00025, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
```

Selected nearest candidate:

| noise multiplier | DP median relative-L2 | DNA median relative-L2 | relative distance | within 5% tolerance |
| ---: | ---: | ---: | ---: | --- |
| `0.001` | `0.3123367286` | `0.3312435936` | `0.0570784321` | false |

Result:

```text
DP_DISTORTION_MATCHED_V2 = NOT_MATCHED
```

The nearest grid point missed the frozen tolerance by a small amount
(`5.71%` vs allowed `5%`), but the protocol does not allow silently changing
the grid or tolerance after seeing the calibration result.

## 6. Attacker development-gate result

The v2 development attacker optimized directly against the lifted full update:

```text
argmin_x L_balanced(delta(x), lifted_v2_update)
```

Clarification added after supervisor review: this was not the v1 Level-1
surrogate-realization attacker that searches over `M_r`, but it also was not a
fully v2-structured sketch/JL/quantization attacker. It was a raw-style
attacker against the server-lifted full update. Therefore the gate failure below
is a measurement-tool failure/boundary and must not be interpreted as privacy
evidence for v2.

Budget:

```text
groups = 8 development groups
restarts = 4
iterations = 600
attack_lr = 0.1
nonnegative_lambda = 0.001
```

Gate result:

| Control | Wins / n | Mean difference | Median difference | one-sided sign p | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Prior | `4/8` | `-1976.9244` | `-6.6132` | `0.6367` | FAIL |
| Zero-update | `4/8` | `173.2859` | `-9.8524` | `0.6367` | FAIL |

Overall:

```text
target_gate = false
```

Mean v2 defense relative-L2 in the gate run:

```text
0.3003622172
```

## 7. Artifact-run IDs and hashes

Distortion calibration:

```text
artifacts/dna_transform_v2/step5_distortion_calibration_20260916
```

Attacker development gate:

```text
artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/
  dna_v2_development_gate_development_gate_ratio0p95_eta0p01_r4_i600_lr0p1_nonneg0p001
```

Hashes:

```text
protocols/amendments/2026-09-16_dna_transform_v2_step5_development_calibration.md
f79a856f1336248967a9466ff6b11159f796d788ea6da53128da187288b39bd0

experiments/calibrate_rq1_dna_transform_v2_distortion.py
831fc2aff4427c11599f6bd052576fb4e9ba13cd384f6d7496f90010a24b3885

experiments/run_phase4_dna_v2_development_gate.py
018c34f03c10254c2a86fd40c3e80ef5c3ec6e011b41435cbc95eccc19934cbb

artifacts/dna_transform_v2/step5_distortion_calibration_20260916/calibration_report.json
8d817934fc52e2295e9c8d55eb3ec290405a27f77fcb374317faf66b0078d539

artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_v2_development_gate_development_gate_ratio0p95_eta0p01_r4_i600_lr0p1_nonneg0p001/dna_v2_development_gate_report.json
736b17ef309631b93049126881ad759f5d3ff6a02ebbcb31ec9980ff8017db65
```

## 8. Gate đạt/chưa đạt

| Gate | Status | Reason |
| --- | --- | --- |
| Amendment written before Step 5 results | PASS | amendment timestamped before calibration/gate runs |
| v1 untouched | PASS | v2-only scripts added |
| DP distortion matched within 5% | FAIL | nearest multiplier `0.001` has relative distance `5.71%` |
| v2 attacker beats Prior | FAIL | `4/8`, `p=0.6367` |
| v2 attacker beats Zero-update | FAIL | `4/8`, `p=0.6367` |
| v2 confirmatory target allowed | NO | blocked by both DP match and attacker gate |

## 9. Kết luận

Step 5 does not authorize v2 confirmatory RQ1 work.

The v2 mechanism remains mechanically lossy and passed the small utility smoke,
but the current RQ1 development pipeline is not yet valid for v2:

1. no frozen DP comparator matched v2 distortion within tolerance;
2. the attacker branch did not pass the required Prior/Zero controls on
   development data, and the attacker used here is not yet v2-structured.

This is a valid negative development result. It should be reported as a
methodological boundary, not hidden or overridden by changing parameters after
the fact.

## 10. Bước tiếp theo được phép

Only a new supervisor-approved amendment may continue v2 RQ1 work. Reasonable
options would be:

- a predeclared local interpolation/grid amendment for DP distortion matching;
- a separate attacker-design amendment for v2, because the current raw-style
  attacker against lifted updates failed development controls;
- or stopping v2 RQ1 security evaluation and reporting v2 as mechanically lossy
  but not yet supported by a valid inversion-evaluation pipeline.

No v2 confirmatory target set or privacy claim is authorized from this Step 5
run.
