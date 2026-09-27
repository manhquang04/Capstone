# Amendment — Pre-pilot administrative synchronization and seed plumbing

**Amendment ID:** `AMD-2026-09-12-PRE-PILOT-SYNC`  
**Authorized by user:** 2026-09-12  
**Scientific values changed:** none  

## Scope

This amendment synchronizes protocol headers, checklists and approval tables
with the already approved pre-pilot YAML files and supervisor decision package.
It also fixes seed plumbing before any Stage 1 outcome is observed.

The approved alpha, power, effect threshold, margins, acceptance thresholds,
grids, repetitions and resource ceilings are unchanged.

## Technical clarification for RQ2

`FL_RUN_SEED` is the paired protocol seed. For each seed in
`[101, 202, 303, 404, 505, 606, 707, 808]`, it controls dataset subsampling and
train/validation/test split, client partition, model initialization and loader
order. Every method within that seed must receive the identical `FL_RUN_SEED`.
Split and partition checksums are stored separately per seed. A development
seed may not appear in the later confirmatory seed list.

## Track-specific RQ3 clarification

Selection and locking of `network_emulation_tool` is deferred to Stage 1 and
must occur before the RQ3 pilot benchmark. It does not block RQ1 artifact audit,
RQ1 calibration or the RQ2 shared development batch. It does block the RQ3
pilot and official RQ3 benchmark until recorded in an RQ3 amendment/config.

## Integrity rule

The freeze manifest preserves the hashes supplied for the pre-sync approved
documents and separately records hashes after this administrative amendment.
Post-sync hashes are authoritative for execution. Any later modification
requires a new amendment and manifest entry.
