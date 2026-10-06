# Phase 2 diagnostic protocol

Implementation: experiments/run_phase2_diagnostic.py. Full execution record and limitations: phase2_report.md.

Observe one known-label sample gradient at an eval-mode FraudMLP checkpoint after one FedAvg warm-up round. Attacker knows weights, BatchNorm buffers, feature representation and label. This is not a full client-update attack.

Optimize a standard-normal dummy with Adam (lr 0.05), tensor-mean gradient MSE, L2=0, 300 steps, three seeded restarts. Include initialization as a candidate. Select the minimum attacker objective per method, never reconstruction error. Pair baseline and zero-gradient initial tensors by exact assertion. Prior is each unoptimized initialization.

Development evidence: Phase 1 two targets, both L2 settings, final budget 300. Evaluation: 50k subset/new checkpoint, five fraud and five non-fraud validation targets. The verified run replays source-row sampling and asserts disjointness with development IDs. Dataset checksum is saved. This is a technical validation repeat of the prior pilot, not an independent blind evaluation. No hyperparameter changes were made after observing evaluation results.

Run-scoped artifacts preserve vectors, checkpoint checksum, scaler, seeds and per-step loss components including iteration 0. Reload checks verify all vector values/shapes/dtypes and recompute match/regularization/total from the saved candidate. Strict categorical validity is tested. Phase 2 gate is complete with pilot limitations; no reconstruction-success threshold has been calibrated (TBD). Read phase2_report.md for the scope of transition to Phase 3.
