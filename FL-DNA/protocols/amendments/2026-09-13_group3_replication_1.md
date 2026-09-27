# Group 3 replication protocol 1 — frozen before target/seed materialization

**Timestamp:** 2026-09-13T14:55:00+07:00  
**Status:** PRE-RUN FROZEN / supervisor-authorized Group 3 scope  
**Dataset:** PaySim/creditcard.csv only; no second dataset

This is the first of exactly two independently reported conservative
replications selected before observing either replication outcome.

## RQ1 family

- DNA: conservative (`block=256`, `mix=.08`, `keep=.88`, `shrink=.45`).
- Comparator: distortion-matched clipping/noise-MC (`clip=100`,
  `noise_multiplier=.00025`, 100 MC samples).
- Primary test: one-sided exact sign test, `p0=.50`, alpha `.05`, frozen tie
  threshold `.0390625`; 39 targets, rejection boundary inherited from the
  approved conservative exact-binomial design.
- Target generation seed: `2086313950`; DNA run seed: `1307020394`.
- One target set, four records/one fraud per group, source-disjoint from every
  old target. Raw/DNA/DP gates remain branch-specific.

## RQ2 family

- Same 21-seed paired non-inferiority design and methods as the original
  conservative confirmation: baseline, lossless, transform, and contextual
  distortion-matched DP.
- Seed list is deterministically derived from
  `SHA256('RQ2-GROUP3-REPLICATION-1|SEED|2026-09-13|i')`, excludes all
  development/original-confirmatory seeds, and is frozen in the machine config.
- Primary DNA Transform alpha remains `.05`; F1 margin `.02`, AUC margin
  `.005`, both endpoints must pass. Lossless remains a technical sanity check.

No result from replication 1 may change replication 2, trigger a rerun, or
change the optional FL-variation scope. Maximum concurrency is 9 and every
child retains `torch.set_num_threads(1)`.
