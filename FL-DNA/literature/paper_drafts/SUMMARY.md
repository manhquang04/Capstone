# Paper-draft work summary

## Completed outputs

- `published_claims.md` records the exact source-level privacy claims,
  attacker/threat model, metric, setting, and hyperparameters for Soteria,
  PRECODE, gradient pruning, ATS, and FedSKETCH Count-Sketch. It separately
  records *Privacy For Free* because it is a wireless-channel-noise DP method,
  not a Count-Sketch method; combining the two would be a false attribution.
- `S1_independent_check.md` independently audits the finished S1 image records.
- `audit_methods.tex` is a 784-word, result-free IEEE-style methods draft.
  `audit_tables.tex` provides empty RQ4 and per-domain E1/E2/E3 table
  skeletons.
- `data_ethics.tex` provides the requested approximately 200-word statement.

## S1 independent-check finding

S1 is internally consistent: 546 result records comprise 351 `ran` and 195
`NOT_ASSESSABLE`; all nine emitted ran cells have the complete unique target set
`0--38`; target-to-CIFAR-index mappings agree across cells; result fields agree
with paths; and every summary row agrees with its JSON metrics. The recomputed
mean JSON PSNR is 14.907343312837888 dB, agreeing with the report.

The important discrepancy is missing paired evidence, rather than a numerical
inconsistency. S1 contains no per-target `undefended` results, and the report
does not provide them for the S1 target pool. Consequently, exact paired win
counts and one-sided sign tests against the undefended attack cannot be
recomputed. The independent check explicitly labels this comparison **NOT
RECOMPUTABLE**. It should be resolved by preserving and citing per-target
undefended results for the identical 39 targets, rather than by treating a
different target pool or aggregate as paired evidence.

Recomputed metrics from 8-bit saved PNGs differ slightly from the pre-save
tensor JSON metrics (maximum absolute difference 0.0154699102; 190 of 351
records exceed 0.001 in at least one metric). This is consistent with PNG
quantization, not a report/CSV mismatch. Final numeric tables should use the
unquantized JSON metrics, with saved reconstructions retained for visual and
persistence checks.

## Unverified or unavailable evidence

- The Soteria full original paper was unavailable through the source used for
  this work; its table row uses the original abstract and marks inaccessible
  architecture, batch-size, attacker-detail, and hyperparameter fields
  **UNVERIFIED**.
- FedSKETCH's conditional Count-Sketch DP theorem is not an evaluated
  reconstruction-attack result. *Privacy For Free* does not establish a
  Count-Sketch claim.
- PaySim's additional dataset-page/Kaggle terms and CIFAR-10 explicit license
  terms remain unverified in the underlying dataset documentation.
- No RQ4 result is inserted in the Methods/tables. Per the frozen rule, such a
  verdict requires paired E2 undefended comparisons and DP comparator evidence
  with the specified Holm correction.

## Compliance

All writes are confined to `literature/paper_drafts/`. No S2 path was read,
listed, or modified. No training, attack, or dataset download was run; metric
recomputation used one CPU thread.
