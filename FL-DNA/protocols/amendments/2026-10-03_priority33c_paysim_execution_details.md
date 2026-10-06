# Priority33c PaySim execution details — before PaySim workloads

Supplement to the immutable execution annex. Source-disjoint group draw seed
333650, native local-step capture seed333660 with P24 derive_seed namespace;
DP reconstruction release seed333670+101*targetID+sum(ord(method)). Training
per-tensor DP noise seed=pairedSeed*100000+3*roundIndex+clientIndex, CPU only.
DataLoader workers0; optimizer, batches, partition and seed schedule unchanged.
Native baseline/v1/v2 seeds270201..270216; full-state transform callsites original.
Record per-round transmitted/aggregate minimum variance and final CPU evaluation.
No historical MPS utility target accepted as a new CPU target.

Use the P24 v1 transmitted-vector estimator (direct matching) and P33a/P25
key-holder sketch-space least squares for v2. Do not reuse P27's oracle
ground-truth estimator choice. v1 receipts contain kind/q only; v2 kind/q/key
metadata, no encoder diagnostic statistics. Capture fixed reference checkpoint
and four-record one-step scope unchanged; not multi-step client reconstruction.

Any BN/nonfinite per-target DP release makes the ENTIRE corresponding paired
comparison NOT_ASSESSABLE. Save all other jobs, no outlier or seed exclusion.
Qualification failures stop only dependent PaySim BN contrasts; image stages
remain authorized. No valid CPU utility target means no BN confirmatory targets
need creation for that comparator. Keep all gate failures and earlier artifacts.
