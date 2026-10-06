# Phase 3: local-update threat models

## Full-client protocol

The full-client runner uses the existing PaySim loader on a stratified 500,000-row
subset, three clients, batch size 1024 and one complete local epoch. It creates a
one-round FedAvg warmup checkpoint using the same split and scaler. Native
`train_local_model` generates the observed delta. All records of a client are
optimized together; no minibatch gradient substitutes for the transmitted delta.

| Information | Attacker access | Reason / limitation |
|---|---|---|
| Pre-local parameters and BN running state | Yes | Server's global checkpoint |
| Individual parameter delta | Yes | Unprotected baseline server observation |
| Final BN buffer deltas | Captured, unused in objective | Attack may leave information unexploited |
| Full local-set size and batch sizes | Yes | Strong known-size scenario |
| Ordered labels of all records | Yes | Additional privileged assumption |
| Exact batch order | Yes | Additional privileged assumption |
| CPU RNG before local training | Yes | Dropout realization-known attacker; not an ordinary server assumption |
| Adam hyperparameters and initial state | Yes | Fresh optimizer per client in repository |
| Scaler and feature schema | Yes | Fixed preprocessing metadata |
| Features / source identities | Evaluator only | Never used for candidate selection |
| Observations | One local epoch after warmup | No multi-round or late-training claim |

DataLoader uses zero workers for auditable capture, with its seeded shuffle
unchanged. The model is in train mode, with original BatchNorm and Dropout.
Labels and batch memberships are fixed by assumption. Reported errors use this
known ordering; no unordered reconstruction or Hungarian matching claim is made.
The baseline and zero-update control share each initialization and budget. Prior
is that initialization without optimization. Zero-update is an algorithmic
control, not an optimized distribution-informed no-update attacker.

Capture validation for DNA is separate from attack evaluation. Secure aggregation
does not expose an individual delta in the ordinary server threat model; these
individual-delta attacks do not evaluate that scenario. No DP guarantee is made.

## Earlier four-record development pilot

The implemented development protocol observes a raw parameter delta after actual
PyTorch Adam training. It uses four existing Phase 2 records, including both
labels, as development data. It is not a full research-client experiment.

| Information | Attacker access | Assumption |
|---|---|---|
| Pre-local parameters and BN buffers | Yes | Server checkpoint |
| Raw local delta | Yes | Baseline individual-client observation |
| Number of records | Four | Privileged known-size setting |
| Labels and order | Yes | Additional strong assumption |
| Batch indices | Yes | Additional strong assumption |
| Dropout RNG state | Yes | Privileged realization-known setting |
| Adam state | Fresh zero moments | Matches current client optimizer lifecycle |
| Preprocessing/schema | Fixed existing pipeline | Assumed known |
| Ground-truth features | Evaluator only | Never candidate selection |
| Rounds observed | One local training event | No multi-round claim |

The model remains in train mode, including BatchNorm and Dropout. One-step uses
batch size four; two-step uses two ordered batches of size two. The objective
uses parameter deltas only. BN buffer deltas are captured and replay-checked but
are not used as an additional attack signal. Reconstruction order is assumed
known; no unordered-record matching claim is made.

Secure aggregation and DNA defense evaluation are outside this pilot.
