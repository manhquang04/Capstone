# Priority 32b: BN variance reconciliation

Date 2026-10-02. Status FROZEN / diagnostic execution authorized by the user.
Written before any new replay. Not a new confirmatory utility experiment.
No prior result, report, source runner, dataset, Latex or external_defenses edit.
P33 remains deferred until this diagnostic report and checks are finished.

## Fixed sequence and jobs

1. Static code/config/artifact audit, including ALL stored registered metrics
   for null/nonfinite endpoints. No outcome-based seed selection.
2. Replay the first three seeds in each original frozen manifest order:
   v1 1984441809,653660776,650446433; v2 94400035,1530452029,1646028911.
   Call original run_fraud_fl_dna_transform[v2].main with exactly its original
   environment/config, original native device selection and loader defaults.
   Hooks only read states/probabilities, never draw RNG or alter tensors.
   50 rounds, no early stopping, retain every round validation/test evidence.
   Compare all non-timing stored core metrics, including null patterns:
   report bit equality and max absolute error; numerical reproduction gate
   tolerance 1e-8 (confusion counts must match exactly). A failed replay gate
   is reported, not silently tuned/retried.
3. Replay the same six method-specific seeds with the original P32 PaySim
   full-state pipeline, original prepared cache, CPU, same training budgets.
   Hook every aggregate and final probability arrays; capture failures without
   excluding a seed. Keep training all 50 rounds even if running variance
   becomes negative (the original loss guard remains); report any unavoidable
   earlier original exception. Negative state is not itself a stop gate.
4. One-factor swaps relative to P32, ALL six fixed method/seed combinations:
   (a) registered data bundle (cap-before-split and per-seed preprocessing);
   (b) registered device selection;
   (c) registered per-round validation/test iteration cadence;
   (d) registered loader worker count/persistence (same generators/data);
   (e) registered transform seed derivation (v1 identical: algebraic no-op
       documented without redundant run; v2 changes root label only).
   Data bundle is an explicit coupled-factor swap, not evidence separating
   cap/order/split/scaling. Its decomposition is fixed in advance: if the
   bundle changes the negative-variance outcome for any seed, run ALL six
   seeds for two additional data swaps: fixed-seed320032 cap-before-split;
   per-run-seed cap-after-split. Train-only scaling/imputation in both. These
   isolate per-run split realization from cap placement. If numeric missing
   count is zero, median scope is an algebraic no-op, no redundant replay.
   Feature/model/optimizer/epochs/batch/transform-before-FedAvg differences
   that are statically absent are not fabricated as experimental factors.
5. If no one-factor swap explains an observed discrepancy, run the fixed
   ALL-factors registered configuration through the original main route;
   use its already completed registered replay rather than recompute. Report
   unresolved interaction; no unregistered factorial search.

Mandatory workload: 12 base +24 one-factor jobs +3 v2-seed-label jobs=39.
Conditional data-decomposition: 12 more, maximum51 training jobs. Four
concurrent processes, Torch threads1, no early stopping. Estimated 1–3 hours;
original native device/loader process overhead may increase this estimate.
Every job result validates frozen identity/hashes before resume skip.
All failed/interrupted/replayed outputs retained in a new P32b namespace.

## Instrumentation and evidence

Read-only aggregate hook logs minimum/negative count/nonfinite count for each
BN running_var per round, plus local raw/transformed variance minima, when
available without altering original code. Read-only model forward hooks log
finite logits/probabilities during validation/test, without extra forwards or
loader iterations. RNG states before/after hook reads must be identical.
No guards that replace scores, clamping, imputation, optimizer reset changes,
seed exclusions, or altered metric semantics. Final states and output arrays
may be copied after existing computations. Hash code/config/data and output.

## Answer rule and limits

Registered numerical validity requires finite actual validation/test outputs
and reproduced stored core metrics on each replay; process returncode alone
does not suffice (evaluate_model catches metric ValueError as null AUC).
Report negative variance even when sigmoid yields finite values, including
possible zero BN weights or backend effects. Audit all stored rounds for null
AUC/PR-AUC; three replay seeds cannot establish absence of negative variance
in every unreplayed seed. Do not extrapolate BAF diagnosis to PaySim.
One-factor causality is conditional on these seeds and pipeline; report
interactions and unresolved factors plainly. No hypothesis tests or new RQ2
claim. Final report contains exact diffs, run table, reproduction errors,
source hashes, all disclosures, py_compile and git diff --check.
