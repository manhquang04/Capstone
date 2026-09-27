# Group 3 replication protocol 2 — frozen before target/seed materialization

**Timestamp:** 2026-09-13T14:55:00+07:00  
**Status:** PRE-RUN FROZEN / supervisor-authorized Group 3 scope  
**Dataset:** PaySim/creditcard.csv only; no second dataset

This is the second and final independently reported conservative replication
selected before observing either replication outcome.

## RQ1 family

- DNA: conservative (`block=256`, `mix=.08`, `keep=.88`, `shrink=.45`).
- Comparator: distortion-matched clipping/noise-MC (`clip=100`,
  `noise_multiplier=.00025`, 100 MC samples).
- Primary test: one-sided exact sign test, `p0=.50`, alpha `.05`, frozen tie
  threshold `.0390625`; 39 targets under the approved conservative design.
- Target generation seed: `1689123716`; DNA run seed: `171253797`.
- One target set, four records/one fraud per group, source-disjoint from every
  old target. Raw/DNA/DP gates remain branch-specific.

## RQ2 family

- Same 21-seed paired non-inferiority design and methods as the original
  conservative confirmation: baseline, lossless, transform, and contextual
  distortion-matched DP.
- Seed list is deterministically derived from
  `SHA256('RQ2-GROUP3-REPLICATION-2|SEED|2026-09-13|i')`, excludes all seeds
  used by development, original confirmation, and replication 1, and is frozen
  in the machine config.
- Primary DNA Transform alpha remains `.05`; F1 margin `.02`, AUC margin
  `.005`, both endpoints must pass. Lossless remains a technical sanity check.

Replication 2 is run once regardless of replication 1's outcome. No rerun or
parameter adaptation is permitted. Maximum concurrency is 9 and every child
retains `torch.set_num_threads(1)`.
