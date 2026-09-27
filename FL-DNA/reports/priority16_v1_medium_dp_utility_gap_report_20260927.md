# Priority 16 v1-medium DP utility gap report

**Date:** 2026-09-27
**Status:** COMPLETE
**Scope:** RQ2-style utility non-inferiority check for the exact v1-medium distortion-matched DP comparator added in Priority 16. No paper files were edited.

## Rationale

Priority 16 added a third head-to-head DNA-vs-DP comparator row for Transform
v1-medium:

- Transform v1-medium: `mix_ratio=0.10`, `keep_ratio=0.85`,
  `shrink_factor=0.40`;
- distortion-matched DP comparator: `clip_norm=100.0`,
  `noise_multiplier=0.000315`.

The epsilon for this comparator was already covered by Priority 3, but the
RQ2-style utility non-inferiority follow-up had only been run for the two
Priority 14 comparators (`0.0004` and `0.00105`). This report supplies the
missing paired utility result for `dp_v1_medium_0p000315`.

## Protocol

Wrote and froze before execution:

- Amendment: `protocols/amendments/2026-09-27_priority16_v1_medium_dp_utility_gap_protocol.md`
- Config: `protocols/config/rq2_priority16_v1_medium_dp_utility_confirmatory.json`

Frozen utility rule:

- paired non-inferiority, DP comparator minus unprotected FL baseline;
- alpha `0.05`, 95% paired CI;
- F1 non-inferiority margin `0.02`;
- AUC-ROC non-inferiority margin `0.005`;
- both endpoints must pass;
- required seed count `52`, matching the RQ2-v2 / Priority 14 DP utility precedent.

The test reused the already frozen unprotected baseline from RQ2-v2
confirmatory:

```text
artifacts/rq2_v2/confirmatory_20260916/
```

Baseline checksum verification:

| Baseline artifact | Expected SHA-256 | Observed status |
| --- | --- | --- |
| `protocols/config/rq2_v2_confirmatory.json` | `da0ca7cebb8bd394ec2c44a593fb357757938df7695f6793f670aa77a0987224` | MATCH |
| `artifacts/rq2_v2/confirmatory_20260916/execution_summary.json` | `aefbea99842877ed614a6b95f30663cdb72570f41c3ccf7a8d113ffd180f6cac` | MATCH |

## Command

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_rq2_priority14_dp_utility.py \
  --config protocols/config/rq2_priority16_v1_medium_dp_utility_confirmatory.json \
  --output-dir artifacts/rq2_priority16_v1_medium_dp_utility/confirmatory_20260927 \
  --workers 9

PYTHONPATH=. .venv-phase1/bin/python experiments/analyze_rq2_priority14_dp_utility.py \
  --config protocols/config/rq2_priority16_v1_medium_dp_utility_confirmatory.json \
  --run-dir artifacts/rq2_priority16_v1_medium_dp_utility/confirmatory_20260927 \
  --output-dir results/rq2_priority16_v1_medium_dp_utility/confirmatory_20260927
```

Execution summary: `52/52` DP jobs succeeded, `0` failed.

## Results

Contrast: `FL_DP_COMPARATOR_MINUS_FL_BASELINE`.

| DP comparator | Endpoint | n | Mean paired Δ(DP − Baseline) | SD | Median Δ | 95% CI | Margin | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| DP 0.000315 (v1-medium comparator) | F1 | 52 | -0.099293 | 0.033804 | -0.096468 | [-0.108704, -0.089882] | -0.020000 | FAIL |
| DP 0.000315 (v1-medium comparator) | AUC-ROC | 52 | -0.002362 | 0.001466 | -0.002441 | [-0.002770, -0.001954] | -0.005000 | PASS |

Primary utility conclusion under the frozen endpoint rule:

- `dp_v1_medium_0p000315`: FAIL overall, because F1 non-inferiority fails.

## DP accounting note

This exact comparator is already covered by Priority 3 as
`rq1_medium_distortion_matched`:

| Comparator | Scenario | δ | Adjacency | ε |
| --- | --- | ---: | --- | ---: |
| v1-medium DP 0.000315 | one release | 1e-5 | add/remove | 5,054,286.08156432 |
| v1-medium DP 0.000315 | 50 releases / same client | 1e-5 | add/remove | 252,060,366.41251898 |

These are update-level Gaussian-mechanism diagnostics only. They are not
record-level DP guarantees because the implementation clips full client/model
updates, not per-example gradients.

## Artifacts

| Artifact | SHA-256 |
| --- | --- |
| `protocols/amendments/2026-09-27_priority16_v1_medium_dp_utility_gap_protocol.md` | `0536f557c50693b96625156becff7e4c27ff07ae5f87ee7de4c94e3ecf08ad68` |
| `protocols/config/rq2_priority16_v1_medium_dp_utility_confirmatory.json` | `bfd3fbf3539a7b429127f2ec90cd9b347b0a8e998087a867a385d9f150de1d63` |
| `experiments/run_rq2_priority14_dp_utility.py` | `74dffd8c9c3d82c408b87a5de5f99cb548994bddb4d3f73f7aaa7fb9476257ec` |
| `experiments/analyze_rq2_priority14_dp_utility.py` | `289bfe4bd4729ba83fbba4be13b433ff2d950535bc191da356acfa788ae3efd0` |
| `artifacts/rq2_priority16_v1_medium_dp_utility/confirmatory_20260927/execution_manifest.json` | `f235cffe838906539cd17865cb90db4ca4df2c3c47e3eca3f243391959328683` |
| `artifacts/rq2_priority16_v1_medium_dp_utility/confirmatory_20260927/execution_summary.json` | `2de0e3f42b5b499bace106d1a7debe256949a452e6d478672ea54aa864f00545` |
| `results/rq2_priority16_v1_medium_dp_utility/confirmatory_20260927/summary.json` | `274245ecff2f6a4c9503d8b3ba051d09f716d298db957c777857a0f4c9891377` |
| `results/rq2_priority16_v1_medium_dp_utility/confirmatory_20260927/per_seed.csv` | `877c0b6d57d478d2c89eeb023b0a671da7672a62dce0c438d2cdc5b59b5494ad` |

## Final check

This report did not edit `main.tex` or any file under `Latex/`.
