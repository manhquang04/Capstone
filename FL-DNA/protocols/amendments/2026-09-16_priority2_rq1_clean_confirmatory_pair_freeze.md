# Amendment: Priority 2 RQ1 clean confirmatory pair freeze

**Date:** 2026-09-16  
**Supervisor:** Manh Quang  
**Status:** APPROVED FOR SINGLE PRIORITY-2 RQ1 CONFIRMATORY EXECUTION  
**Scope:** RQ1 Priority 2 clean confirmation only.

## Locked pair

The supervisor selects the following pair for the new clean confirmatory run:

```text
DNA point: dna_conservative
DNA config: block_size=256, mix_ratio=0.08, keep_ratio=0.88, shrink_factor=0.45
DP-style point: dp_0.00025
DP config: full-client-update clipping/noise, clip_norm=100.0, noise_multiplier=0.00025, MC samples=100
```

## Selection rationale locked before target creation

The pair is selected because it is the only pair with complete real attack
evidence available before Priority 2 began:

- original Group 2 conservative run;
- conservative replication 1;
- conservative replication 2;
- conservative replication 3.

Those four source-disjoint historical draws existed before the Priority 2
Pareto/proxy sweep was run.  The Priority 2 Pareto/proxy sweep is therefore
context only and is not the basis for choosing this pair.  This rule is written
before creating the new Priority 2 confirmatory target set to avoid
outcome-dependent cherry-picking.

The historical evidence remains post-hoc/exploratory for the new clean design
and does not count as a result of this new confirmatory run.

## Locked statistical design

The design inherits the Priority 2 power lock:

```text
p0 = 0.50
p1 = 0.65
alpha = 0.05
target power = 0.80
planned tie rate = 0.10
dropout rate = 0.05
required effective non-tied n = 69
rejection threshold at n=69 = at least 42 wins
initial target draw = 81
maximum_feasible_target_count = 150
```

If the actual number of non-tied targets differs from 69, the analyzer must
report the exact one-sided sign-test p-value for the actual non-tied count,
while still reporting the planned `42/69` threshold for traceability.

## Confirmatory target and execution constraints

- Create exactly one new target set with 81 groups.
- Each group has `records_per_group=4`, `fraud_records_per_group=1`, and
  `local_steps=1`.
- The target set must be source-disjoint with all previous target sets,
  including development, original Group 2, all Group 3 replications, the
  scope-boundary screen and all historical/post-hoc targets.
- Historical/post-hoc target files may be inspected only for source identifiers
  during disjointness checking.
- Run the three branches exactly once: raw, DNA conservative and DP-style
  `noise_multiplier=0.00025`.
- Use separate branch gates against Prior and Zero-update controls.
- Do not rerun, redraw or tune if the result is unfavorable.

## DP wording required in the conclusion

The DP-style comparator has epsilon approximately `8.019e6` for `delta=1e-5`,
add/remove neighboring updates and a single release under the approved
update-level Gaussian/RDP accountant.  The conclusion must therefore say that
the comparison is against weak-accounting DP-style full-client-update
clipping/noise at comparable update distortion, not against strong record-level
Differential Privacy.

## Locked seeds

```text
target_generation_seed = 338413135
dna_run_seed = 1151115843
dp_defense_seed = 314159265
```

Seed derivations:

```text
target_generation_seed = first 31-bit value from SHA256('PRIORITY2-RQ1-CLEAN|TARGETS|2026-09-16')
dna_run_seed = first 31-bit value from SHA256('PRIORITY2-RQ1-CLEAN|DNA-SURROGATE|2026-09-16')
```

## Implementation hashes at freeze time

```text
experiments/run_rq1_variant_confirmatory.py      d848a78f863189bd179a435a92ea7f5931fd365925f840a48c459878c1a7e5b0
experiments/analyze_rq1_variant_confirmatory.py  3febc5a33366a33decec6925298dcd695165602a8dc026209338c51fb9b27d43
experiments/create_phase4_source_disjoint_targets.py  ec63eb09c5d91741def37bc35c911ab0cbeb053d82fd4289bd03afb6b4059cdf
experiments/verify_rq1_target_disjointness.py    32e844161e3cd97afb06c6e9ea2d88342777ce2bc75775a0814fe9ac884671b0
attacks/inversion_metrics.py                     6ad0b2581d35ec804311dbecf7b682dd1a8e447ccc61db001b97f917eb2aa8c2
attacks/pseudo_image.py                          6273e59d73cd17b4c8812c46776356defbcf090dea9c2a1bdcca94498ff8993a
datasets/creditcard.csv                          16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b
```
