# P33c technical interruption: contiguous image input

Written before correction/replay. The initial image reconstruction pool stopped
before producing any per-target result: the P31 tensor loader permutes NHWC to
NCHW without materializing contiguous storage. LeNetZhu's read-only forward uses
view() after convolution, which rejects the resulting strides on CPU. The
preserved traceback identifies gradient_dict(), before optimization, not an
adaptive-loss or scientific gate failure.

Correction is a new wrapper that materializes the normalized image tensor with
contiguous() before model evaluation. Values, labels, source IDs, dtype, CPU,
seeds, objectives, 4800 iterations, TV, restarts and all frozen budgets remain
unchanged. External sources and every earlier driver/freeze/artifact are kept.
No RNG call is added. Regression tests check exact values, gradients and RNG
state, including a real frozen CIFAR target. Existing targets/calibrations are
reused; no redraw or utility rerun. No completed image results existed to replay.

A new remaining-stage supervisor verifies the original execution freeze and a
separate correction freeze; it starts only after no original workers remain.
It runs corrected image reconstruction, analysis and final checks only, skipping
the completed utility and PaySim BN stages. Old failure markers/logs remain as
historical interruption evidence, not active failures of the corrected attempt.
Original and correction hashes must both pass the final audit.

PaySim resolved separately: all 16 baseline jobs passed, all 16 v1 and 16 v2
full-state CPU utility jobs failed BN/numerical guards. Its dependent BN
utility-matched comparisons are NOT_ASSESSABLE, not evidence of privacy.
