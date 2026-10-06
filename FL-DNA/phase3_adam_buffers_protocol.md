# Full-client Adam buffer development

This is development on the original client 2 (73,862 records), not new held-out
evidence. The fixed budget is 100 iterations per baseline/zero control, one
paired initialization matching the earlier categorical run. The only attack
change is inclusion of differentiable BatchNorm running statistics in the
equal-tensor MSE objective. Numeric/free and categorical/softmax representation,
local Adam, train-mode BatchNorm/Dropout, model checkpoint, batches and RNG stay
fixed. Target metrics never select the attack candidate.

Native Adam replay must pass before attack. Best candidates must pass objective
recomputation after reload. Report reconstruction MSE against the unoptimized
prior and zero control; objective decrease is not reconstruction success.

Run: artifacts/phase3_followup/adam_buffers_v6/adam_buffers.
No held-out evaluation is authorized by a positive development result alone as
statistical confirmation: any later evaluation needs a locked, source-disjoint
protocol and adequate independent units. Earlier confirmation datasets have
already been inspected and cannot be used as fresh confirmatory evidence.

Phase 3 closure requires evidence on the intended Adam update setting. A
negative development outcome is a valid research outcome and must not cause
automatic budget extensions or repeated testing until a threshold is crossed.
