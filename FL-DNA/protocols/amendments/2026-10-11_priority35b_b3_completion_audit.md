# Ratio/pooling audit — 2026-10-11, additive post-launch infrastructure

Written before protected-arm quality analysis; no measurement or rule changes.
The new completion-audit building block independently recomputes the ratio
division, standardized reconstruction and plain double-precision objective.
For sketch uploads, deterministic frozen LSMR decoding is replayed only to check
the saved solution/receipt/residual, not to run a new measurement or optimizer.
Exact agreement is required. Native member objectives use the already frozen
objective auditor; pooling separately redoes Hungarian alignment and median over
30 members, retaining the frozen categorical-softmax helper and decoder.
No true target rows enter objective/pooling selection. This checks the observable
minimum-member reference, not alignment to truth. Synthetic tests include
ratio receipt/objective and permuted-batch pooling corruption rejection.

These modules are not an independently implemented decoder/LSMR/softmax and
must not be described as such. Full analysis serialization, figure/report QA and
final integration/seals are still pending; no COMPLETE is emitted by this file.
