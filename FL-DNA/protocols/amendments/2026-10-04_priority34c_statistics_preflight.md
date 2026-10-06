# Priority34C — fixed-family statistics implementation preflight

Before synthetic statistics tests or confirmatory outcomes. The primary family
is exactly72 directional tests:3datasets x3instrument/batch cells x2DNA arms
x2DP comparators x2directions. Primary feature accuracy: smaller DNA-minus-DP
means poorer DNA recovery. Use exact binomial sign tails with zero differences
as ties, both directions separately. Gates/calibration failures reserve p1
for both directions; no family shrinkage. Qualified but missing required
confirmatory observations fail analysis, not a p1 substitute.

For each complete pair require exactly39 finite paired values. Report arm
medians, paired effect median and order-statistic interval ranks13/27 (zero-based
indices12/26), wins/losses/ties/effective n, raw p and fixed-family Holm p.
Holm uses ascending raw p, stable original order for ties, cumulative maximum
of (72-rank)*p capped1. Independent recomputation must use another binomial
implementation and separately implemented Holm, not call the producer.

This implements already-registered tests, not a new scientific family or
outcome-based choice. No targets or reconstruction outputs are consumed by
the synthetic preflight. Qualification frozen sources remain unchanged.
