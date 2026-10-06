# Priority 32b additional read-only backend diagnosis

Frozen before running these additional diagnostic forwards. The original P32b
amendment, runner, seeds and all 39/51 training jobs remain unchanged.
Motivation already observed: four completed registered MPS jobs have reproduced
50-round stored metrics exactly despite negative BN variance; two matched P32
CPU jobs have nonfinite final probabilities. This is not confirmatory evidence
selection, calibration or an attempt to repair any result.

After all six registered checkpoints exist, evaluate ALL six unchanged final
checkpoints on the SAME first1024 rows of the P32 prepared PaySim validation
array, once on CPU and once on native MPS (if available). This isolates inference
backend without training/data/seed changes. Scores on this diagnostic input are
not RQ2 metrics and do not replace recorded validation/test outputs. Copying the
model/state to each device is permitted; never clamp or change any tensor.
Record logits/probability finite counts and paired output differences only where
both outputs are finite. Also a fixed minimal eval-mode BN operator test uses
input [[1,1,1],[2,2,2]], running_mean [0,0,0], running_var [-.25,0,.25],
weight [1,1,1], bias [0,0,0], epsilon1e-5, trainingFalse on both devices.
Compare finite patterns against the defining (x-mean)/sqrt(var+epsilon) formula.
No new training, parameter choices, RNG draws, hypothesis tests or utility claims.
Torch threads1. Preserve all existing states and outputs, save new probe outputs
and hashes in artifacts/priority32b_bn_reconciliation/backend_probe/.
These diagnostics can establish backend dependence/invalid-domain behavior, not
the utility bias relative to an unrun, valid-BN counterfactual training experiment.
