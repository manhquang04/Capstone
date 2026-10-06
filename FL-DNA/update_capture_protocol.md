# Phase 3 capture and replay

## Full-client path

`experiments/run_phase3_full_client.py` invokes the existing native training
function after one FedAvg warmup on the same split/scaler. A recorder wraps the
real DataLoader, appends loader-local row identifiers, and records all batches.
It verifies each local record occurs exactly once. These identifiers together
with the split seed, dataset hash, ordered evaluator tensors and file checksums
identify the target set; they are not claimed to be source CSV row numbers.

`attacker_capture.pt` contains pre-local state, raw/transmitted delta, batch sizes,
RNG and configuration. `evaluator_targets.pt` holds features and labels separately.
Although labels are stored with evaluation data, the known-label protocol grants
them to the optimization routine. Original features only enter reporting.

Replay checks every parameter and buffer against native training before attack.
The attack then unrolls every batch for every objective evaluation. All dummy
records participate. Best-so-far includes initialization; no reconstruction metric
selects candidates or stopping. Each saved candidate is loaded and its objective
recomputed. A checksum manifest covers outputs; incomplete runs are rejected by
the summary script. CPU peak RSS covers the whole process, not exclusive tensor
allocation or device VRAM. Timings include all restart work.

The full-client budget is fixed at 100 iterations, 3 restarts, Adam attack LR 0.05,
with baseline, zero-update and paired prior. No tuning on the resulting errors is
performed. This is a development evaluation of one partition, not an independent
population estimate. The summary selects restarts by each method's own objective
and separately reports paired per-restart errors. Objectives across methods are
not a measure of leakage.

## Earlier bounded development implementation

Load the saved Phase 2 checkpoint and four development vectors. Clone the model,
run native torch.optim.Adam with learning rate 0.001, betas (0.9, 0.999), epsilon
1e-8, zero weight decay and fresh state. Save pre-local state, all state deltas,
transmitted baseline delta, source row IDs, labels, data, batch order and CPU RNG.
Capture files contain evaluator data and must not be treated as attacker access.

Replay uses functional_call with cloned buffers and differentiable Adam moments.
The order sqrt(v) / sqrt(1-beta2**t) matters in float32; moving the bias correction
inside sqrt caused measurable replay error. At zero variance the simulator uses
zero square-root derivative while preserving the forward value. This convention
has not been validated as an adaptive attack against nonsmooth points.

Before inversion, compare each full state delta with atol=2e-7, rtol=2e-4.
Current verified run has exact equality. A failed check aborts inversion.
Minimize the equally weighted mean of per-parameter mean squared delta errors.
Use paired normal initialization for baseline and zero-update, with the unchanged
initialization as prior. Include iteration zero in objective-only candidate
selection. Reload candidate and recompute objective before reporting.

The CPU reference uses one thread. Full-client batch counts, original loader
shuffle replay, larger unrolling, memory cost and independent targets remain
unvalidated. No full-client equivalence is asserted by this capture.
