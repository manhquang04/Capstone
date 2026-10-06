# Priority 30 protocol amendment: native defense adapters and no-balance PaySim ablation

Date written: 2026-09-30 13:25:45 +07

Status: FROZEN / AUTHORIZED

Scope: This priority continues the frozen Priority 29 design.  The native
instruments, E1/E2/E3 definitions, RQ4 answer rule, Holm family, and compute
reduction order from
`protocols/amendments/2026-09-30_priority29_native_audit.md` remain unchanged.
Priority 29 qualified the native IMAGE and TABULAR positive-control
instruments but stopped before defense verdicts because the available
`external_defenses/adaptive/` adapters did not satisfy the fail-closed
defense-not-identity requirement.

No defense result has been inspected before this amendment.

No `Latex/` edits are allowed.  Everything under `external_defenses/` remains
read-only.  Official code may be imported, or code may be copied into
`experiments/priority30_native_defenses/` with attribution and commit hash, but
no file under `external_defenses/` may be modified.  Missing dependencies are
reported; no package may be installed into any venv.

Threading: every process must keep `torch.set_num_threads(1)`.

Disclosure: every run, including interrupted, resumed, failed, discarded, and
replayed runs, is listed in the report.

## Part A: native defense adapters

Adapters must be built and tested before any defense evaluation is emitted.
The report generator remains fail-closed: if a defense adapter fails any
mandatory check, its E1/E2/E3 cell is `NOT_ASSESSABLE`.

Mandatory checks for each adapter:

1. Defense-not-identity: with published/frozen hyperparameters, the transmitted
   gradient or model forward pass differs measurably from the undefended path.
2. Lossless sanity for its adaptive attacker where an identity/lossless setting
   exists.  For v2 key-holder attacks this means least-squares through the
   sketch operator, not the naive lift.
3. Server-knowledge receipt: any key, seed, hash, mask, stochastic sample, or
   metadata required for decoding/aggregation/attack is declared in
   `server_knowledge` and delivered to the adaptive attacker when the threat
   model says the server knows it.
4. Decoy correctness and positive-control gates inherited from Priority 29.

Adapters and frozen porting choices:

- PRECODE: use the official variational bottleneck from
  `external_defenses/precode/src/VariationalBottleneck.py` (commit
  `c66adc4cdd62993139eafebc1b57fc25b0694874`) as a model-level module in the
  forward pass.  For LeNet-Zhu and Adult FullyConnected, insert the bottleneck
  immediately before the final classifier.  The stochastic forward pass is used
  when computing gradients.  A pass-through update adapter is invalid.
- Soteria: use the official representation-sensitivity mechanism from
  `external_defenses/soteria/GS_attack/reconstruct_image.py` (commit
  `23cf90e9e5cb41d5dc45e7540ef63a4a0ca0a8ca`).  Compute sensitivity
  client-side from the model and data and mask columns of the final classifier
  gradient.  Frozen pruning rate: 80% for IMAGE (matching the GS-style defense
  note) and 40% for TABULAR unless the official architecture lacks the required
  representation layer, in which case report `NOT_ASSESSABLE`.
- Gradient pruning: use per-tensor lowest-magnitude pruning following the
  retained ATS implementation referenced in the DLG porting notes.  Frozen
  sparsity: 70% (the strongest value discussed in the reproduction notes).
- ATS: IMAGE only.  Apply the official searched policy `3-1-7` to the training
  input before gradient computation.  Score reconstructions against the raw
  image as requested by Priority 30, while also recording the original-paper
  transformed-image metric.
- Count-Sketch: use FetchSGD-style fixed hash Count-Sketch with 5 rows,
  500,000 columns, seed 21, and registered trainable-gradient tensor order.
  Evaluate both server-knowledge variants: hashes known and hashes unknown.  If
  hashes are unknown, the adaptive attack must report unavailable rather than
  inventing a sketch matrix.
- DNA v1 conservative and DNA v2 0.95/0.01: apply to the transmitted gradient.
  v2 adaptive key-holder attacker must use least-squares through the sketch
  operator.

For each defense, compare against one published number or qualitative claim
from the original paper/porting notes where possible.  If the domain/model does
not match the original paper, label the comparison `not comparable`.

## Part B: Priority 29 E1/E2/E3 matrix

For every `(defense, domain, qualified native instrument)` cell whose Part A
adapter passes all mandatory checks, run exactly the Priority 29 frozen matrix:

- E1 original-style official non-adaptive attacker and paper metric;
- E2 adaptive attacker matched to server knowledge, primary comparison against
  no defense;
- E3 DP comparator:
  1. paper-documented unclipped Gaussian noise at the paper sigma;
  2. properly clipped DP at matched L2 distortion;
  3. TABULAR only utility-matched DP if E2 survives and the compute estimate
     remains under the ceiling.

Confirmatory target count remains n=39 per emitted cell.  Fresh target IDs must
be source-disjoint from Priority 29 n=8/n=24 and from all prior project target
pools.  Overlap is checked in code.

Holm correction and the RQ4 verdict rule are unchanged from Priority 29.

## Part C: PaySim no-balance-feature RQ2 ablation

This is descriptive sensitivity analysis only and does not replace any frozen
RQ2 conclusion.

Remove these six balance-derived features from the PaySim pipeline:

- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`
- source/origin balance-difference feature
- destination balance-difference feature

Run paired RQ2-style training replicates for:

- unprotected baseline;
- Transform v1 conservative;
- Transform v2 0.95/eta 0.01.

Contract:

- 16 paired replicates;
- same RQ2 high-level contract unless the feature-removal data loader requires
  a documented adapter;
- report absolute F1, AUC-ROC, PR-AUC with and without balance features;
- report paired transform-vs-baseline deltas for F1, AUC-ROC, PR-AUC;
- descriptive only, no non-inferiority gate replacement.

## Compute estimate and frozen reduction

Native IMAGE official attacks are roughly 9-16 seconds per attack.  Adult
TabLeak native attacks are roughly 70-100 seconds per batch.  A full defense x
domain x E1/E2/E3 matrix plus utility-matched Adult DP and 16-replicate PaySim
ablation can exceed four days.

Reduction is frozen before execution:

1. Build and run adapter unit/sanity tests first.  Do not run defense E1/E2/E3
   cells for adapters that fail.
2. Run IMAGE cells before TABULAR cells if both cannot fit in the current pass.
3. Drop E3(iii) utility-matched DP unless E2 survives for a tabular defense and
   remaining projected wall time stays under four days.
4. Drop ATS if native IMAGE non-ATS cells already push runtime over the ceiling.
5. Run Part C after adapter checks and emitted E-cells; if Part C cannot finish
   under the current pass, report it as queued with no partial scientific
   conclusion unless full 16 paired replicates complete.

## Deliverable

Report path:
`reports/priority30_native_audit_report_20260930.md`

The report contains adapter tests, original-paper comparisons, emitted E1/E2/E3
tables with Holm correction, RQ4 verdicts, reconstruction grids, Part C ablation
table if completed, SHA-256 hashes, and all deviations.

