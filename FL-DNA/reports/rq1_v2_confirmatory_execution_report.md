# RQ1-v2 confirmatory execution report

**Date:** 2026-09-16  
**Status:** VIỆC 3 COMPLETE — SINGLE CONFIRMATORY EXECUTION FINISHED  
**Scope:** RQ1-v2 `dna_transform_v2 ratio=0.95 eta=0.01` vs
`DP_DISTORTION_MATCHED_V2 noise_multiplier=0.00105`.

## 1. Đã làm gì

Ran the single authorized RQ1-v2 confirmatory execution on the frozen target
set:

```text
protocols/config/rq1_v2_confirmatory.json
artifacts/rq1_v2/confirmatory_freeze_20260916/rq1_v2_confirmatory_targets.pt
```

Frozen design:

```text
target draw = 176 groups
records_per_group = 4
fraud_records_per_group = 1
DNA v2 = compression_ratio 0.95, quantization_eta 0.01
attacker = compressed-sensing/IHT, sparsity_fraction 0.20
DP v2 comparator = clipping/noise MC, clip_norm 100, noise_multiplier 0.00105, MC samples 100
tie_threshold_mse = 0.01953125
primary contrast = feature_mse(DNA v2) - feature_mse(DP v2)
```

The run used the source-disjoint target hash:

```text
87153ed0550254e92bfd0ea2eacbaec840b31d887d421e4513001224976b064f
```

## 2. Lệnh đã chạy

Official execution:

```bash
PYTHONPATH=. .venv-phase1/bin/python \
  experiments/run_rq1_v2_confirmatory.py \
  --config protocols/config/rq1_v2_confirmatory.json \
  --output-dir artifacts/rq1_v2/confirmatory_run_20260916 \
  --workers 9
```

The official runner produced all 528 expected job records (`176 groups × 3
branches`) but five jobs failed before optimization due to numeric
capture-consistency assertions. Before aggregate analysis, a technical replay
amendment was written:

```text
protocols/amendments/2026-09-16_rq1_v2_confirmatory_numeric_replay.md
```

Authorized replay jobs:

```text
raw group 116
raw group 165
dna_v2 group 116
dp_v2 group 116
dp_v2 group 165
```

Replay used the same target/group/seed/config/optimizer/attacker settings, with
only the capture consistency tolerance relaxed to float32-compatible
`atol=1e-5`, `rtol=5e-3`. All five replay jobs succeeded.

Analysis:

```bash
PYTHONPATH=. .venv-phase1/bin/python \
  experiments/analyze_rq1_v2_confirmatory.py \
  --config protocols/config/rq1_v2_confirmatory.json \
  --run-dir artifacts/rq1_v2/confirmatory_run_20260916 \
  --output-dir results/rq1_v2/confirmatory_20260916
```

## 3. Gate đạt/chưa đạt

All branch gates passed.

| Branch | Prior wins / n | Prior p | Zero wins / n | Zero p | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Raw | 139 / 176 | 2.1131e-15 | 145 / 176 | 3.9531e-19 | PASS |
| DNA v2 | 137 / 176 | 2.8031e-14 | 144 / 176 | 1.8101e-18 | PASS |
| DP v2 | 131 / 176 | 3.0704e-11 | 138 / 176 | 7.8237e-15 | PASS |

Primary contrast validity: **VALID**, because both DNA-v2 and DP-v2 branches
passed their own controls. Raw also passed as contextual validation.

## 4. Kết quả chính kèm uncertainty

Primary paired sign test:

```text
DNA-v2 wins over DP-v2 = 85 / 176
losses = 91 / 176
ties = 0
win probability = 0.48295
exact 95% CI = [0.40713, 0.55936]
one-sided exact sign-test p = 0.70106
reject H0 at alpha=0.05 = false
```

Main feature-MSE effect:

```text
mean(feature_mse_DNA_v2 - feature_mse_DP_v2) = -494.3714
95% CI = [-739.7331, -249.0098]
```

Because higher feature-MSE means worse reconstruction, the mean feature-MSE
direction favors the DP-v2 comparator, not DNA-v2. The paired sign-test also
does not show a DNA-v2 advantage.

Secondary pseudo-image metrics:

```text
DNA_v2 - DP_v2 PSNR 95% CI = [1.3847, 2.5633]
DNA_v2 - DP_v2 SSIM 95% CI = [0.0927, 0.2079]
```

Because higher PSNR/SSIM means better reconstruction, these secondary metrics
also do not support a DNA-v2 protection advantage.

Branch-level selected reconstruction metrics:

| Branch | Mean feature-MSE | Median feature-MSE | Mean PSNR | Median PSNR | Mean SSIM | Median SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 1607.9708 | 103.0098 | 22.0781 | 21.4285 | 0.5430 | 0.5516 |
| DNA v2 | 1659.5557 | 86.4683 | 22.1025 | 21.2400 | 0.5345 | 0.5254 |
| DP v2 | 2153.9271 | 167.6468 | 20.1285 | 19.5015 | 0.3842 | 0.4169 |

## 5. DP accounting context

For `noise_multiplier=0.00105`, `clip_norm=100`, update-level Gaussian/RDP
diagnostic accounting remains extremely weak:

```text
single release, delta=1e-5, add/remove convention: epsilon = 4.5808e5
single release, delta=1e-5, replace-one convention: epsilon = 1.8232e6
```

This is not a strong record-level DP guarantee. The RQ1-v2 conclusion is
therefore stated against a distortion-matched DP-style full-update
clipping/noise comparator with very weak formal privacy accounting.

## 6. Artifact-run ID và hashes

| Artifact | Path / SHA-256 |
| --- | --- |
| Run directory | `artifacts/rq1_v2/confirmatory_run_20260916` |
| Analysis directory | `results/rq1_v2/confirmatory_20260916` |
| Replay summary | `0a7dc7afc14735fb73ccc6251cd699f21a2abff9e7a8812f1344a5cd8bf1e1e4` |
| Analysis summary | `b9fdf723e876455b67140f9214fb696702644beb2bd858c9b3634815ee67faef` |
| Paired metrics CSV | `35a45a025842b9f968bfb6416ee5fa506a0876a6ea3b6f70f8c471263d9e180b` |
| DP accounting v2 JSON | `b604ddf2223f6d5ee919dfa35b803f97713e4cc05b1f65b57bdd47de77b17007` |
| Numeric replay amendment | `5d3827a2fd0273c10accb2f6f9d3b6d26d1ac7d3d2442c933ea813a040e00ce9` |
| Raw runner after replay amendment | `787dd7bcc5405c3a2b03b30478b75deb99f7e8e86c92f434fe0100db29a573ca` |
| DP runner after replay amendment | `a8bf8eeba8fc217e880f7897282c0cff7c7adf2c9aecd79e56bbfcf5c5d24b97` |
| DNA-v2 IHT runner after replay amendment | `20f69767e5b29f71d5829c62045b72475a2b5efda98e27937dee200131da3e52` |

## 7. Kết luận theo protocol đã khóa

RQ1-v2 confirmatory result:

```text
DNA Transform v2 did not protect against gradient inversion more effectively
than the distortion-matched DP-style clipping/noise comparator in this frozen
confirmatory test.
```

This was a valid negative result: the attacker passed all branch controls, the
primary contrast was analyzable, and the single confirmatory execution was not
rerun or retuned after seeing results.

## 8. Bước tiếp theo được protocol cho phép

No rerun of this RQ1-v2 confirmatory design is permitted based on outcome.

Allowed next steps are reporting/interpretation only, or a separately approved
new protocol for a materially different question or mechanism. RQ3 remains
separate from this report.
