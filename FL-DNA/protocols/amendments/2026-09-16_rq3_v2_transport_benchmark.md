# Amendment: RQ3 DNA Transform v2 transport benchmark

**Date:** 2026-09-16  
**Status:** FROZEN FOR EXECUTION

This is a standalone 27-cell extension of the completed v1 RQ3 benchmark:
three mandatory network profiles (LAN, Broadband, Constrained) times client
scales 3/5/10 times three methods. Each cell compares `RAW_FLOAT32`, existing
`DNA_TRANSFORM_TRANSPORT` (v1), and `DNA_TRANSFORM_V2_TRANSPORT`. Lossless DNA
is not re-run because its frozen official cells already exist and the purpose
is the incremental v1-v2 transport comparison.

**Pre-execution correction, 2026-09-16:** The original wording incorrectly
called this a nine-cell matrix while specifying three methods, three profiles,
and three scales. The arithmetic-correct count is 27. The first invocation was
terminated after writing only environment/cell-order metadata and before a cell
result existed; its partial directory is retained as an aborted audit artifact
and is excluded from analysis. This correction changes no threshold, method,
network profile, implementation setting, or scientific selection.

Use the existing network-emulation tool, profiles, warm-up 10, measured 50,
cell isolation/order discipline, and Bundle B criteria without modification.
v2 uses ratio 0.95 and eta 0.01. Apply Bundle B independently to v2 and report
`ACCEPTABLE`, `NOT_ACCEPTABLE`, or `INCONCLUSIVE`; a result does not revise
the v1 conclusion. No parallel benchmark workload is allowed during this timing
study, and `torch.set_num_threads(1)` remains unchanged.
