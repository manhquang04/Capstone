# RQ3 optimized implementation supplemental benchmark

Date: 2026-09-16

Status: `COMPLETE`

Overall decision after optimization: `NOT_ACCEPTABLE`

This report is supplemental. It does not replace the frozen reference-implementation RQ3 reports.

## Amendment

Pre-run amendment:

- `protocols/amendments/2026-09-16_rq3_optimized_implementation_benchmark.md`

The amendment was written before profiling, optimization, or rerunning the RQ3 matrix. It retained Bundle B thresholds, network profiles, warm-up/measured repetitions, method order seed `271828`, and the single-thread-per-process policy.

## Commands run

Pre-optimization profiling:

```bash
FL-DNA/.venv-phase1/bin/python - <<'PY'
# cProfile wrapper around representative RQ3 v1/v2 transform workloads
PY
```

Optimized benchmark:

```bash
.venv-phase1/bin/python experiments/run_rq3_benchmark.py \
  --config protocols/config/rq3_v2_benchmark.json \
  --output artifacts/rq3/optimized_implementation_20260916
```

Analysis:

```bash
.venv-phase1/bin/python experiments/analyze_rq3_benchmark.py \
  --config protocols/config/rq3_v2_benchmark.json \
  --run artifacts/rq3/optimized_implementation_20260916 \
  --results results/rq3/optimized_implementation_20260916
```

## Artifacts

- Pre-optimization profile: `artifacts/rq3_optimization/profile_pre_20260916/`
- Post-optimization profile: `artifacts/rq3_optimization/profile_post_20260916/`
- Benchmark run ID: `artifacts/rq3/optimized_implementation_20260916`
- Analysis output: `results/rq3/optimized_implementation_20260916`
- Reference comparison summary: `artifacts/rq3_optimization/optimized_vs_reference_summary.json`

## Code changes

Two implementation-only optimizations were made:

1. `dna_encoder/transform_defense.py`
   - Avoids materializing large binary/DNA strings for v1 seed derivation.
   - Computes the same DNA-derived seed directly from float32 bytes.
   - Equivalence checked against the previous string path for multiple block sizes.

2. `dna_encoder/transform_defense_v2.py`
   - Vectorizes each FWHT stage via NumPy reshape operations instead of a Python loop over every block.
   - Equivalence checked against a local reference loop for power-of-two sizes up to 1024.

No DNA formula, frozen transform parameter, network profile, threshold, or benchmark repetition count was changed.

## Pre-optimization hotspots

Representative profile over 20 synthetic RQ3 iterations:

| Mechanism | Elapsed | Main hotspot |
|---|---:|---|
| v1 | 16.215 s | `dna_mapper.binary_to_dna` and string joins/generators |
| v2 | 18.298 s | `transform_defense_v2._fwht_normalized` Python loop |

Top cumulative-time functions before optimization:

| Mechanism | Function | Cumulative time |
|---|---|---:|
| v1 | `_transform` | 16.199 s |
| v1 | `transform_update_array` | 16.179 s |
| v1 | `binary_to_dna` | 12.119 s |
| v1 | string `join` | 7.995 s |
| v1 | `float32_array_to_binary` | 1.979 s |
| v2 | `_fwht_normalized` | 16.469 s |
| v2 | `transform_update_array_v2` | 9.417 s |
| v2 | `_serialize_v2` | 9.152 s |
| v2 | `_deserialize_v2` | 9.132 s |
| v2 | `reconstruct_update_array_v2` | 8.641 s |

## Post-optimization profile

Representative profile over the same 20 synthetic RQ3 iterations:

| Mechanism | Before | After | Ratio after/before |
|---|---:|---:|---:|
| v1 | 16.215 s | 2.981 s | 0.184 |
| v2 | 18.298 s | 2.072 s | 0.113 |

This confirms that the intended bottlenecks were substantially reduced before the official optimized benchmark was run.

## Full optimized RQ3 decision

Analyzer output:

```json
{
  "overall_decision": "NOT_ACCEPTABLE",
  "profile_decisions": {
    "LAN": "NOT_ACCEPTABLE",
    "BROADBAND": "NOT_ACCEPTABLE",
    "CONSTRAINED": "NOT_ACCEPTABLE"
  },
  "cells": 27
}
```

## Reference vs optimized comparison

Reference run: `results/rq3/v2_transport_benchmark_20260916`

Optimized run: `results/rq3/optimized_implementation_20260916`

| Metric | Method | Reference max p95 | Optimized max p95 | Optimized/reference |
|---|---|---:|---:|---:|
| Client encode/serialize seconds | v1 | 0.2681 | 0.1002 | 0.374 |
| Client encode/serialize seconds | v2 | 0.1909 | 0.0204 | 0.107 |
| Server decode/aggregate seconds | v1 | 0.0033 | 0.0035 | 1.056 |
| Server decode/aggregate seconds | v2 | 1.7190 | 0.1367 | 0.080 |
| End-to-end seconds | v1 | 4.6621 | 2.9337 | 0.629 |
| End-to-end seconds | v2 | 5.9438 | 2.8137 | 0.473 |
| Payload bytes total | v1 | 514580 | 514580 | 1.000 |
| Payload bytes total | v2 | 677629.55 | 677629.55 | 1.000 |
| Client peak allocation bytes | v1 | 1180303 | 1180303 | 1.000 |
| Client peak allocation bytes | v2 | 1913686.6 | 1913902.6 | 1.000 |
| Server peak allocation bytes | v1 | 968496 | 968496 | 1.000 |
| Server peak allocation bytes | v2 | 1396734 | 1396790 | 1.000 |

## Failing Bundle B criteria after optimization

| Method | Failing criterion | Range observed | Threshold | Failed cells |
|---|---|---:|---:|---:|
| v1 | End-to-end round overhead p95 fraction | 0.410 to 5.578 | 0.25 | 9/9 |
| v1 | Peak client memory overhead fraction | 0.624 to 0.876 | 0.25 | 9/9 |
| v2 | End-to-end round overhead p95 fraction | 0.358 to 2.211 | 0.25 | 9/9 |
| v2 | Peak client memory overhead fraction | 2.042 to 2.995 | 0.25 | 9/9 |
| v2 | Peak server memory overhead fraction | 0.443 to 1.932 | 0.25 | 9/9 |

## Interpretation

The optimized code substantially improves transform CPU time, especially for v2 server decode/aggregate. However, Bundle B is still not met.

The remaining failures are not explained by the original Python string/FWHT bottlenecks alone:

- Payload size is unchanged by implementation optimization. This is expected because the wire formats did not change.
- End-to-end overhead remains above the frozen 25% p95 threshold in all cells.
- Memory-overhead fraction remains above threshold, especially for v2, because the v2 sketch/reconstruction path still materializes arrays and metadata needed for the lossy projection transport format.

Conclusion: the reference implementation was inefficient, but optimizing the obvious bottlenecks is not enough to make v1 or v2 deployment-cost ACCEPTABLE under Bundle B. The residual NOT_ACCEPTABLE decision appears tied to structural costs of the current transport mechanisms and accepted memory-overhead threshold, not merely to poor implementation of the hottest loops.

## Gate status

- 27/27 optimized benchmark cells completed.
- Analyzer completed.
- Overall gate: `NOT_ACCEPTABLE`.
- No confirmatory workload beyond the listed optimized RQ3 benchmark was run.
