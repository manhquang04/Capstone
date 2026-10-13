# Priority 35B B3 independent metric audit — 2026-10-11

Additive post-launch infrastructure, written before inspecting protected quality.
No measurement, target, decision, tolerance used in measurement, statistic,
Holm-144 family, or scientific interpretation is changed.

The independent scorer uses NumPy continuous postprocessing and a separately
constructed logical-feature cost matrix followed by SciPy Hungarian assignment.
It deliberately preserves the native **detailed-score denominator convention**:
each absent feature type contributes one to the denominator. This is verified
against synthetic all-continuous and all-categorical cases, not silently fixed.
It retains inclusive continuous tolerance endpoints and categorical string
equality from the frozen reference. Dataset decoding remains shared with the
frozen adapter and is not claimed as an independently implemented decoder.

Accuracy and assignment must match exactly. Independent float64 MSE reduction
may differ by absolute 1e-12, fixed here before inspecting actual metric errors;
the final audit must report its maximum difference and stop on any excess.
Bounded MSE is checked exactly from stored MSE and the paired mean/mode score.
The tolerance is only for cross-library reduction auditing; measurements, ties,
intervals, sign tests and Holm adjustment use unchanged stored values.

Synthetic tests check reference agreement, permutation, clamping without input
mutation, missing-type denominator, boundary behavior and corruption rejection.
This scorer is a building block, not a full finalizer or COMPLETE claim.
