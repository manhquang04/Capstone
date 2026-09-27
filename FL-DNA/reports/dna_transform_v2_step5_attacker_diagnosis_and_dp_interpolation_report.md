# DNA Transform v2 — Step 5 attacker diagnosis and DP interpolation report

**Date:** 2026-09-16  
**Status:** ATTACKER VALIDITY UNRESOLVED; DP DISTORTION MATCH FOUND  
**Scope:** development-only diagnosis and DP calibration. No confirmatory target
was created.

## 1. Đã làm gì

This report clarifies two separate issues from Step 5:

1. whether the failed v2 attacker development gate can be interpreted as
   privacy evidence;
2. whether `DP_DISTORTION_MATCHED_V2 = NOT_MATCHED` can be resolved by a
   predeclared local interpolation grid.

## 2. Chẩn đoán attacker Step 5

The Step-5 v2 attacker was not the original v1 Level-1 attacker that searches
over v1 linear realizations `M_r`.

Specifically, it did not use:

```text
L_balanced(M_r delta(candidate), transmitted_v1_update)
```

and it did not search over v1 block permutation/attenuation realizations.

However, it also was not a fully v2-structured attacker. It did not optimize in
the low-dimensional JL/sketch space, did not model the projection as the main
attack operator, and did not use a compressed-sensing-style reconstruction
strategy. Instead, it used a raw-style objective against the server-lifted full
update:

```text
argmin_x L_balanced(delta(x), lift(Q(R_s u_raw)))
```

So the correct interpretation is:

```text
The Step-5 FAIL means the current raw-style lifted-update attacker is not an
effective measurement tool for v2 on the development gate.
```

It must not be interpreted as:

```text
DNA Transform v2 protects gradients.
DNA Transform v2 is stronger than v1.
DNA Transform v2 has passed a privacy/security evaluation.
```

The v2 privacy status remains unresolved until a v2-appropriate attacker is
designed and itself passes reasonable Prior/Zero controls, or until a separate
technical argument justifies why such an attacker cannot be made effective.

## 3. Existing attacker-gate result, restated

The raw-style lifted-update attacker result remains:

| Control | Wins / n | one-sided sign p | Gate |
| --- | ---: | ---: | --- |
| Prior | `4/8` | `0.6367` | FAIL |
| Zero-update | `4/8` | `0.6367` | FAIL |

Because the attacker is not v2-structured, these failures are a measurement
failure/boundary, not evidence of v2 privacy.

## 4. DP local-grid amendment

Before running any interpolation result, the following amendment was written:

```text
protocols/amendments/2026-09-16_dna_transform_v2_dp_local_grid_interpolation.md
```

It authorized a development-only local grid around the previous nearest point
`noise_multiplier=0.001`:

```text
[0.00105, 0.00110, 0.00115]
```

This grid was selected using only distortion-calibration information, not
privacy/attack outcomes.

## 5. DP local-grid result

V2 DNA target:

```text
median relative-L2 = 0.3312435936
```

Local grid:

| noise multiplier | DP median relative-L2 | relative distance | within 5% tolerance |
| ---: | ---: | ---: | --- |
| `0.00105` | `0.3279535493` | `0.993%` | true |
| `0.00110` | `0.3435703945` | `3.721%` | true |
| `0.00115` | `0.3591872390` | `8.436%` | false |

Selection rule chooses the minimum absolute distance, tie-breaking by smaller
multiplier:

```text
DP_DISTORTION_MATCHED_V2 = 0.00105
```

This resolves the distortion-matching issue for development planning only. It
does not resolve the attacker-validity issue.

## 6. Lệnh thực sự đã chạy

```bash
PYTHONPATH=. .venv-phase1/bin/python \
  experiments/calibrate_rq1_dna_transform_v2_distortion.py \
  --output-dir artifacts/dna_transform_v2/step5_distortion_calibration_local_grid_20260916 \
  --multipliers 0.00105 0.00110 0.00115
```

## 7. Artifact-run IDs and hashes

DP local-grid artifact:

```text
artifacts/dna_transform_v2/step5_distortion_calibration_local_grid_20260916
```

Hashes:

```text
protocols/amendments/2026-09-16_dna_transform_v2_dp_local_grid_interpolation.md
6fe663650d11f58db90334cc89bbb95b5e9c5345951e7e26d66e30f92065b35c

artifacts/dna_transform_v2/step5_distortion_calibration_local_grid_20260916/calibration_report.json
ed4eade4a9b436431664491e5fa03d956f3ef5fea80a51f7c5524e3b1d1270d7

artifacts/dna_transform_v2/step5_distortion_calibration_local_grid_20260916/grid_summary.csv
c8d8619d80ff0fd2233a6a38f5813affb54de40310e1d5e42878aaa0533f6f4e
```

## 8. Gate đạt/chưa đạt

| Item | Status | Reason |
| --- | --- | --- |
| Clarify attacker structure | PASS | raw-style lifted-update attacker, not v1 `M_r`, not v2-structured |
| Interpret attacker failure as privacy evidence | FORBIDDEN | attacker validity unresolved |
| DP local-grid amendment before run | PASS | amendment written before interpolation |
| DP distortion match | PASS | `0.00105`, relative distance `0.993%` |
| v2 confirmatory target allowed | NO | attacker validity still unresolved |

## 9. Bước tiếp theo được phép

Only the DP distortion issue is resolved. Before any v2 RQ1 confirmatory work,
the next required step is a new attacker-design amendment for v2.

Reasonable attacker directions include:

- optimize directly in sketch space against `Q(R_s delta(x))`;
- use a differentiable straight-through approximation for quantization;
- test a compressed-sensing-style reconstruction objective;
- run a small development gate and require it to beat Prior and Zero controls
  before any privacy comparison is interpreted.

Until then, all v2 privacy conclusions must be labeled:

```text
unresolved; no validated v2-appropriate attacker yet
```
