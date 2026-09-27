# Amendment — RQ1 conservative replication 3 and pooled analysis

**Timestamp:** 2026-09-14T00:00:00+07:00  
**Status:** PRE-RUN FROZEN / supervisor-authorized  
**Supervisor:** Manh Quang  
**Scope:** RQ1 conservative only; PaySim/creditcard.csv only.

## 1. Purpose

The supervisor authorizes exactly one additional source-disjoint draw for the
RQ1 conservative DNA-vs-DP distortion-matched comparison. This amendment is
written before target generation, before attack execution, and before observing
any replication 3 outcome.

The motivation is to clarify the instability already observed across the
existing conservative batches:

- original Group 2 conservative confirmatory run: `19/39`, `p=0.6254`;
- Group 3 replication 1: `26/39`, `p=0.0266`;
- Group 3 replication 2: `26/39`, `p=0.0266`;
- pooled over these three completed batches: `71/117`, `p=0.01304`, which is
  statistically significant but below the pre-specified practical-effect
  reference `p1=0.70`.

## 2. Final draw rule

Replication 3 is the third and final additional draw for RQ1 conservative at
confirmatory-quality. After this replication, no further target sets may be
drawn for this exact RQ1 conservative DNA-vs-DP distortion-matched comparison,
regardless of whether replication 3 or the pooled analysis is positive,
negative, borderline, or heterogeneous.

No result from replication 3 may trigger a replication 4, a changed sample
size, a changed tie threshold, or a changed attacker/defense configuration.

## 3. Replication 3 frozen design

Replication 3 uses the same design as Group 3 replications 1 and 2:

- Target count: `39` groups.
- Records per group: `4`.
- Fraud records per group: `1`.
- Target set: newly generated and source-disjoint from all previous targets,
  including the original Group 2 conservative target, Group 3 replications 1
  and 2, and post-hoc targets such as `level1_fresh_final_targets.pt`.
- DNA Transform: conservative (`block_size=256`, `mix=0.08`, `keep=0.88`,
  `shrink=0.45`).
- DP comparator: distortion-matched clipping/noise-MC with `clip_norm=100`,
  `noise_multiplier=0.00025`, `mc_noise_samples=100`.
- Attack: unchanged from replications 1 and 2 (`iterations=600`,
  `restarts=8`, `dna_candidates=4`, `learning_rate=0.1`,
  `nonnegative_lambda=0.001`, maximum 9 concurrent processes,
  `torch_num_threads=1`).
- Primary per-batch test: exact one-sided sign test on DNA wins over DP using
  the frozen tie threshold `0.0390625`.

Frozen seeds:

- Target generation seed: `655946849`, derived from
  `SHA256('RQ1-GROUP3-REPLICATION-3|TARGET|2026-09-14')`.
- DNA run seed: `656257992`, derived from
  `SHA256('RQ1-GROUP3-REPLICATION-3|DNA|2026-09-14')`.

## 4. Pooled analysis plan

After replication 3 finishes, run a pooled analysis across exactly four
batches:

1. the original conservative Group 2 confirmatory run;
2. Group 3 replication 1;
3. Group 3 replication 2;
4. Group 3 replication 3.

The pooled primary statistic is an exact one-sided sign test over the summed
wins and losses from all four batches. If all batches have zero ties as before,
the nominal total is `n=156`. If ties occur in replication 3, use the actual
non-tied total and report the tie count explicitly.

The pooled analysis must also report a heterogeneity check across batches. The
planned check is a two-sided chi-square test of the 4x2 table of wins/losses
across batches, with the test statistic equivalent to Cochran's Q for
independent binary batch summaries under equal within-batch weights. If any
expected cell count is too small, report the chi-square result as descriptive
and add an exact/permutation sensitivity if an implementation is available
without changing attacker outputs.

## 5. Interpretation rule

This pooled analysis is exploratory/post-hoc because it was not part of the
original RQ1 protocol. It does not replace or reverse the official decision of
any individual confirmatory run, and it must not be used to rewrite the
original Group 2 conservative conclusion.

The pooled result may be used only as a stability/sensitivity summary for the
thesis narrative:

- whether the conservative RQ1 signal is stable or heterogeneous across draws;
- whether the pooled win probability is statistically above `0.50`;
- whether the pooled estimate reaches the pre-specified practical-effect
  reference `p1=0.70`.

No attacker, DNA, DP, target-count, or tie-threshold parameter may be tuned
after seeing replication 3.
