# P34D independent utility audit (administrative, before completed-output access)

The running scientific utility sources, grids, paired seeds, training budget,
matching thresholds and conditional extension rule remain unchanged. A new
auditor will run only after UTILITY_COMPLETE and no live P34D scientific
workers. It will independently regenerate the job registry, hash every receipt
output, recompute paired validation deltas and brackets, verify the conditional
extension trigger, and reload every final checkpoint for validation-only
inference. This is not training or recovery replay.

Accuracy is independently counted with argmax in batches of 512 (the existing
validation batch, not training or recovery batch); exact counts
must agree with stored accuracy within absolute 1e-12. Probe arrays must be
finite, positive, length 94 and have p95 clips agreeing within absolute 1e-12.
Probe gradient regeneration is not performed by this auditor: frozen probe
arrays/initial states and their receipts are verified. This limitation must
remain explicit. No outcomes are used to select tolerances or alter science.

The output is a stage-only audit, never a P34D COMPLETE claim. Failure produces
a new timestamped administrative receipt. Synthetic tests use only fabricated
accuracy vectors; no unfinished scientific results or targets are accessed.
