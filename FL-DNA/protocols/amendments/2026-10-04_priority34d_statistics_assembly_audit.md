# P34D statistics assembly audit, before confirmatory analysis

Add a pure, new assembly module consuming independent stage-audit endpoints.
It does not load targets, run models or modify frozen scientific drivers.
Each image arm must have all 39 unique target indices and identical source IDs
within each paired target. BN eligible checkpoints must likewise have exactly
39 pairs. Missing scheduled observations are errors, not NOT_ASSESSABLE.
Only preregistered qualification/calibration failures reserve p=1 slots.
Pooled comparisons require all three eligible complete strata; no subset pool.

The existing 76-test registry and combined 211=135+76 family remain unchanged.
Image primary endpoint is PSNR (higher is better); BN endpoint is standardized
batch-mean MSE (lower is better), with direction labels referring to numeric
DNA-minus-comparator. SSIM and MSE image intervals are descriptive. Report
median paired effects rather than a difference of unpaired medians.

SciPy binomtest independently verifies raw exact one-sided sign p values.
A separate vectorized NumPy cumulative-max implementation verifies both
Holm adjustments. Heterogeneity uses exploratory Kruskal-Wallis on per-stratum
paired effects, outside both confirmatory families; constant effects are
explicitly undefined, never substituted with a confirmatory p value.
Synthetic-only preflight precedes any real endpoint access. No completion
claim may follow until all stage, manifest and no-live-worker audits pass.
