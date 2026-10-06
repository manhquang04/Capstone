# Priority 32: authorized numerical-failure diagnosis

Date: 2026-10-02. Status: FROZEN / AUTHORIZED (diagnosis only).
Human authorization: user answered "có" to diagnosis followed by a repair
amendment before rerunning. The original design and all earlier outputs remain
unchanged. This document does not authorize an unexamined scientific repair.

## Frozen diagnostic replay

Replay exactly BAF / DNA v1 conservative / seed 321001, the first documented
failure, using the original execution_freeze.json training configuration,
prepared data, client partition, initialization, seed derivation, 50 rounds,
batch 1024 and one Torch thread. Import the original train_job unchanged.
Output is isolated under artifacts/priority32_multidataset/diagnosis_20261002.
This replay is diagnostic, not a new confirmatory result or replacement.

Instrumentation records the minimum/negative count of every BatchNorm
running_var after aggregation, and saves the final model state before
validation. It must not change a tensor, RNG draw, forward/backward step or
aggregation. Re-evaluate the saved model on validation only with per-layer
forward hooks to locate the first nonfinite activation. No test scoring,
clamping, imputation, seed exclusion, optimizer/config changes or early stop.
Any exception is preserved along with the evidence and SHA-256 hashes.

## Repair decision

After diagnosis, write a separate repair amendment specifying the evidenced
cause, exact change, whether it changes the transmitted protocol or only an
implementation error, rerun scope, pairing and treatment of the original
189 attempts. Do not silently call a change to BatchNorm handling a technical
replay. If the proposed repair materially changes the scientific mechanism,
present the choice to the user before execution. No incomplete-sample CI or
noninferiority claim is authorized; Priority 33 remains deferred.
