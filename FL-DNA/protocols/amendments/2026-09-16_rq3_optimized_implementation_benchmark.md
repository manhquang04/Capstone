# Amendment: RQ3 optimized implementation supplemental benchmark

Status: `AUTHORIZED_FOR_PROFILING_AND_TECHNICAL_OPTIMIZATION`

Written at: 2026-09-16T16:06:42Z, before profiling, optimizing, or rerunning the RQ3 deployment-cost matrix.

Scope: supplemental RQ3 benchmark after implementation optimization. This amendment does not replace or alter the frozen RQ3 conclusions for the reference implementation.

## Motivation

The frozen RQ3 conclusion for DNA Transform v1 and v2 is based on the reference implementation. This supplemental run asks whether the NOT_ACCEPTABLE deployment-cost result is a property of avoidable implementation overhead or of the mechanism's structure under the already approved Bundle B thresholds.

## Frozen acceptance criteria retained

This amendment does not change:

- Bundle B acceptance thresholds.
- Network profiles.
- Warm-up and measured repetition counts.
- Method order seed `271828`.
- Decision rule from `protocols/config/rq3_pre_pilot.yaml`.
- `torch.set_num_threads(1)` and single-thread-per-process policy.

## Required sequence

1. Profile `dna_encoder/transform_defense.py` and `dna_encoder/transform_defense_v2.py` before editing either implementation.
2. Record the top five hotspots for v1 and v2 in `reports/rq3_optimized_implementation_report_20260916.md`.
3. Apply only technical optimizations that preserve the mathematical transform and frozen parameters.
4. Do not cache or reuse seed-derived state if doing so changes the frozen `deterministic_per_round_client_seed` behavior.
5. Rerun the full RQ3 benchmark matrix with the same cell structure, warm-up/measured repetitions, and method order seed.
6. Compare old reference implementation results with optimized results for overhead, client encode/serialize time, server decode/aggregate time, memory, and payload expansion.

## Interpretation rule

If the optimized implementation still fails Bundle B, report NOT_ACCEPTABLE and identify the observed structural bottlenecks where supported by profiling evidence. Do not retune Bundle B thresholds or redefine ACCEPTABLE after seeing optimized results.
