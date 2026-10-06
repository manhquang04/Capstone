# Phase 4 Protocol: DNA Transform Adaptive Evaluation

## 1. Purpose

Phase 4 evaluates whether DNA Transform adds measurable protection beyond the
validated FL baseline under an adaptive attacker. It answers RQ1 and RQ2 only
inside scopes where the baseline attack first passes its controls.

Phase 4 is not a linear checklist. Each stage has a gate. If the adaptive
baseline attack fails controls, DNA evaluation is not opened. If implementation
changes occur, prior diagnostic conclusions must be revalidated before
continuing.

## 2. Lessons Carried From Phase 3

Phase 3 established three constraints for Phase 4:

1. Reconstruction behavior is class-dependent. Fraud records show consistent
   signal; non-fraud records do not reliably beat prior controls.
2. A partially improved result is not a pass. Passing only one control, such as
   prior but not zero-update, remains a failed gate.
3. Scope changes matter. A pass at a smaller scope does not authorize claims at
   a larger scope.

These constraints are part of the Phase 4 protocol, not optional commentary.

## 3. Threat Model

### 3.1 Official Attacker

The official Phase 4 attacker:

- knows the model architecture and pre-local checkpoint;
- knows preprocessing and feature schema;
- knows the local optimizer configuration;
- observes the transmitted update for the evaluated method;
- does not receive oracle class-separated gradients;
- does not use ground truth reconstruction error for candidate selection;
- selects candidates only by the attacker objective;
- is evaluated separately for fraud and non-fraud records.

The official attacker must not assume oracle fraud/non-fraud labels unless a
separate threat-model variant explicitly grants labels.

### 3.2 Diagnostic Oracle Variants

Oracle-label or class-decomposed variants may be used only for diagnosis. They
must be labeled as diagnostic or attacker-favorable analysis, not as the
server-side FL threat model.

If oracle diagnostics are reported, they must be separated from the official
adaptive-attack table.

### 3.3 Level-B Architecture-Uncertain Attacker

The Level-B attacker is an attacker-knowledge variant used after the bounded
Phase 4 evaluation. It keeps the same data access as the official no-oracle
attacker but removes exact architecture and optimizer hyperparameter knowledge.

The Level-B attacker knows:

- the task is binary tabular classification with a feedforward MLP family;
- the feature schema, preprocessing, input dimension, and output dimension;
- either a flattened transmitted update vector or a documented public projection
  of the transmitted update;
- the local optimizer family is Adam;
- the number of local steps and target grouping rule used in the evaluated
  scope;
- the DNA Transform algorithm and public DNA hyperparameters when attacking a
  DNA-transformed update.

The Level-B attacker does not know:

- the exact hidden-layer count or hidden widths;
- whether BatchNorm and Dropout are present;
- the exact focal-loss `alpha` and `gamma`;
- the exact local Adam learning rate, betas, or epsilon.

If the attacker does not know the exact architecture, it also cannot load or use
the exact pre-local checkpoint tensor-by-tensor. A Level-B experiment must
therefore choose and report one of two checkpoint assumptions before runtime:

```text
checkpoint_mode = true_checkpoint_disclosed
checkpoint_mode = surrogate_checkpoint
```

`true_checkpoint_disclosed` is internally inconsistent unless the checkpoint is
published in an architecture-hidden representation. If exact checkpoint tensors,
keys, or shapes are available, the architecture is disclosed and the setting is
not Level-B. `surrogate_checkpoint` means each candidate architecture starts
from a documented public initialization or a separately trained public surrogate
checkpoint. The surrogate source, seed, and training data access must be locked
before development. Ground-truth target records must not be used to fit or
select the surrogate checkpoint.

The following candidate grid is locked before any Level-B experiment. It may be
changed only by starting a new development/evaluation cycle:

```text
architecture_id  hidden_widths        batch_norm  dropout
B1               64,32                false       0.0,0.0
B2               128,64               false       0.0,0.0
B3               128,64,32            true        0.3,0.2,0.1
B4               256,128,64           true        0.3,0.2,0.1
B5               128,128,64,32        true        0.3,0.2,0.2,0.1
B6               64,64,32             true        0.2,0.2,0.1
```

The locked local Adam replay grid is:

```text
local_lr         betas          eps
0.0005           0.9,0.999      1e-8
0.0010           0.9,0.999      1e-8
0.0020           0.9,0.999      1e-8
0.0050           0.9,0.999      1e-8
```

The loss grid is:

```text
loss_id          alpha          gamma
focal_low        0.90           2.0
focal_main       0.95           2.0
focal_steep      0.95           1.5
weighted_bce     n/a            n/a
```

This creates a maximum coarse grid of `6 x 4 x 4 = 96` model/replay/loss
combinations. A full R8/I600 attack over all combinations is not allowed as the
first run because it would be a compute-heavy grid search, not a controlled
evaluation.

### 3.4 Level-B' Narrow-Architecture-Uncertain Attacker

Level-B' is a narrower attacker-knowledge ablation introduced after the broad
Level-B coarse screen failed to retain the true architecture. It asks a smaller
question: if the attacker knows the general shape of the deployed MLP, but not
the exact hidden widths, can the projected-observation attack still identify a
usable architecture candidate?

The Level-B' attacker knows:

- the model family is a feed-forward MLP for the same binary tabular task;
- there are exactly three hidden layers;
- BatchNorm and Dropout are used after each hidden layer, matching the deployed
  architectural pattern;
- the input/output dimensions, preprocessing schema, local step count, local
  optimizer family, and feature validity constraints already available in Mức A;
- the loss family and local optimizer hyperparameters are fixed to the Mức A
  values for this ablation: Adam with `lr = 0.001, betas = (0.9, 0.999),
  eps = 1e-8`, and focal loss with `alpha = 0.95, gamma = 2.0`.

The Level-B' attacker does not know the exact hidden width of each layer. The
width search is locked before execution:

```text
layer_1_widths = {102, 128, 154}
layer_2_widths = {51, 64, 77}
layer_3_widths = {26, 32, 38}
batch_norm = true
dropout = {0.3, 0.2, 0.1}
```

This gives 27 width combinations. The center point `(128, 64, 32)` equals the
deployed bounded Phase 4 architecture, so Level-B' assumes a strong public prior
over a common/default architecture template rather than complete ignorance. This
must be reported explicitly with any result. Level-B' is therefore weaker than
Mức A and narrower than broad Level-B; it is useful only as an architecture-
uncertainty sensitivity check.

Level-B' reuses the Level-B projected-observation path:

```text
observation_mode = public_projection_matrix
projection_distribution = signed_hadamard_jl
projection_objective_normalization = l2_normalize_projected_update
checkpoint_mode = surrogate_checkpoint
```

Candidate ranking uses the same complexity-adjusted objective, but
`reference_parameter_count` is recomputed as the median parameter count of the
27 locked Level-B' width combinations. The coarse-screening sanity gate is:

```text
R2/I200 coarse budget
true width tuple = (128, 64, 32)
pass condition = true width tuple ranks in the top three width tuples by
                 complexity_adjusted_objective
```

If this gate fails, stop the Level-B/Level-B' branch and report that the
projected gradient-matching screen is not reliable under architecture
uncertainty. Do not run the full development or fresh evaluation for Level-B'.
If it passes, continue with the same development/fresh-set discipline used in
Mức A and broad Level-B.

Level-B is not valid if the transmitted artifact exposes exact state-dict keys
and tensor shapes. In that case the architecture is effectively disclosed by the
observation channel, and the evaluation falls back to the official architecture-
known attacker. A Level-B run must therefore define one of the following before
runtime:

```text
observation_mode = flattened_update_vector
observation_mode = public_projection_matrix
```

For `flattened_update_vector`, candidate updates whose parameter count differs
from the observed vector length are rejected before optimization. For
`public_projection_matrix`, all candidate updates are projected to the same
public observation dimension before objective evaluation. The projection matrix
seed and output dimension must be locked before development.

The default Level-B path for this project is:

```text
observation_mode = public_projection_matrix
checkpoint_mode = surrogate_checkpoint
```

This path makes the setting meaningfully different from the architecture-known
Phase 4 attacker. It is also a different and weaker attacker model, so its
results must be compared against Mức A only as an attacker-knowledge ablation,
not as a replacement for the official bounded Phase 4 conclusion.

The public projection is fixed before development:

```text
projection_output_dim = 4096
projection_seed = 202609104091
projection_distribution = signed_hadamard_jl
padding_rule = zero_pad_or_truncate_to_projection_input_dim
projection_input_dim = 131072
projection_objective_normalization = l2_normalize_projected_update
```

The transmitted update is first flattened in a deterministic canonical order,
then zero-padded to `projection_input_dim` before projection. Candidate
updates from every architecture use the same padding rule and the same public
projection matrix. The receiver observes only the 4096-dimensional projected
update. It does not observe the unprojected length, state-dict keys, tensor
names, or tensor shapes. If any of those are visible, the experiment is not
Level-B.

The projection input dimension is a protocol constant. It is larger than the
largest architecture in the locked grid (`max_parameter_count = 45697`) and is
not equal to any candidate architecture size. It is also a power of two, so a
structured signed-Hadamard projection can be applied without materializing a
dense matrix. This avoids making the largest candidate a special no-padding
case. The projection matrix dimensions are fixed by this public constant and the
public output dimension, not by the true architecture. This prevents the
observation vector length from revealing the true parameter count. The
projection is public; its role is to remove architectural side-channel
information from the observation, not to act as a secret defense.

The Level-B objective compares L2-normalized projected updates. Raw projected
norms, coordinate statistics, and sparsity diagnostics are still logged for
sanity checks, but raw norm is not used for candidate selection. This prevents a
candidate from being accepted or rejected only because its surrogate checkpoint
produces a different update scale before any reconstruction optimization.

Surrogate checkpoints are also fixed before development:

```text
surrogate_checkpoint_source = deterministic_public_initialization
surrogate_init_seed = 202609107337
surrogate_initialization = PyTorch default module initialization
```

Each candidate architecture initializes from its own deterministic public seed.
No tensor value, scale statistic, running buffer, or optimizer state is copied
from the true checkpoint. If a separately trained public surrogate checkpoint is
used in a later run, its training data, seed, utility, and selection rule must be
documented before development; target records and evaluation records must not be
used to fit or select it.

Before the runtime probe, run a projection sanity check:

1. project the true bounded Phase 4 update using the public projection;
2. project candidate updates from B1, B3, and B4 using their surrogate
   checkpoints, where B1 is the smallest/padding-heavy endpoint, B4 is the
   largest grid endpoint, and B3 is the true bounded Phase 4 architecture;
3. compare projected norm, coordinate mean/std, and sparsity diagnostics.

This check is diagnostic only. It must confirm that all projected updates live
on comparable numeric scales. If one architecture is trivially identified by
projection scale alone, keep the L2-normalized projection objective above and
run the sanity check again. If the normalized projection still exposes a
trivial architecture signature, stop and redesign the observation model before
running the Level-B grid.

Level-B candidate selection must use only attacker-visible objectives:

- gradient/update matching objective against the transmitted update;
- a complexity-normalized version of that objective;
- validity penalties that use preprocessing/schema constraints;
- logged compute budget and convergence diagnostics.

It must not use raw-update error, ground-truth reconstruction error, true model
architecture, true local optimizer hyperparameters, or class labels to choose a
candidate. If oracle information is used to analyze why a candidate failed, that
analysis is diagnostic only and must not decide the selected Level-B result.

Because Level-B searches across model families, selecting the lowest unadjusted
objective can bias the attack toward larger or easier-to-optimize architectures.
Every candidate must therefore report both:

```text
raw_objective = update_matching_loss + validity_penalty
complexity_adjusted_objective =
    raw_objective
    * (1 + 0.05 * log(parameter_count / reference_parameter_count))
```

`reference_parameter_count` is the median parameter count of the locked
candidate architecture grid. It is computed from the grid itself, not from the
true model architecture. If the candidate has fewer parameters than the
reference, the log term is clipped at zero; smaller models are not rewarded
purely for being small. Candidate ranking for the Level-B grid uses
`complexity_adjusted_objective`. The raw objective is still reported so the
effect of the complexity adjustment can be audited.

The Level-B search procedure is two-stage:

0. Runtime probe: run one representative grid cell with the coarse budget and
   record wall-clock time, memory if available, and projected full-grid cost.
   Do not start the full grid if the projection is not acceptable for the
   current compute budget.
1. Coarse-screening sanity check: include the true bounded Phase 4 architecture
   in the candidate grid, run a small development-only check, and verify that it
   is not discarded by the coarse budget. This check may use knowledge that the
   architecture is true only to decide whether R2/I200 has enough resolution for
   screening. "The true architecture reaches the top three" means that B3,
   using its best `(local Adam, loss)` configuration under
   `complexity_adjusted_objective`, ranks among the top three architectures
   after each architecture is represented by its own best configuration. It must
   not choose the final Level-B candidate or final result.
2. Coarse screening on development targets: run the full locked grid with a
   reduced budget of R2/I200 only if the sanity check above passes. Select the
   top three `(architecture, local Adam, loss)` combinations by
   `complexity_adjusted_objective`.
3. If the true architecture fails to reach the top three in the sanity check,
   increase the coarse budget once to R4/I300 and repeat the sanity check. If it
   still fails, stop and report the coarse screen as not reliable instead of
   running the full grid.
4. Full development attack: rerun only those top three combinations with the
   locked Phase 4 budget, R8/I600. The best candidate is again chosen only by
   `complexity_adjusted_objective`.

If the full development attack passes both controls, create a new source-
disjoint fresh target set and run one single-shot final evaluation. If it fails,
do not tune on the fresh set. Return to development diagnostics.

## 4. Scope Definition

In this protocol, "same scope" means the exact combination of:

- update source: sample gradient, bounded local update, 16-record local update,
  or full-client local update;
- optimizer and number of local steps;
- target class group: fraud, non-fraud, or all records;
- target selection rule and source split;
- model checkpoint and preprocessing metadata;
- attacker knowledge: no oracle label, oracle label, known seed, unknown seed;
- defense configuration and random realization assumptions;
- controls, metrics, restarts, and attack budget.

DNA may be evaluated only in the exact `(scope, class group, threat model)` where
the baseline adaptive attack has passed both controls. For example, a pass on
`fraud / 16-record / no-oracle-label` does not open DNA evaluation for
`fraud / full-client`, `non-fraud / 16-record`, or an oracle-label attacker.

## 5. Required Gates

### Gate A: Baseline Effectiveness

Before any DNA evaluation in a scope, the baseline adaptive attack must beat both:

1. its paired unoptimized prior;
2. its paired zero-update or no-update control.

The gate requires all of the following for both controls:

- negative mean reconstruction delta;
- negative median reconstruction delta;
- one-sided sign-test p-value below the locked threshold;
- candidate selected by attacker objective only;
- no hidden target overlap with development data.

Passing only one of the two controls is a failed gate. It must be reported as
partial improvement, not as readiness for DNA evaluation.

### Gate B: Implementation Integrity

Before a result is accepted, the runner must verify:

- native local training replay matches the observed update where replay is part
  of the protocol;
- saved candidate reload reproduces the recorded best objective;
- paired branches share the intended initialization and target data;
- random seeds and defense seeds are logged;
- failed jobs are recorded rather than silently dropped.

### Gate C: Revalidation After Changes

If the attack objective, transform path, optimizer replay, BatchNorm handling,
target sampling, or defense implementation changes, the following sanity checks
must be rerun before continuing:

- baseline gate on the smallest relevant scope;
- class-split reconstruction summary;
- class-gradient contribution diagnostic;
- feature-spread diagnostic if class-wise conclusions are affected.

Old conclusions remain archived, but they cannot be silently mixed with the new
implementation.

## 6. DNA Transform Specification Before Attack

Before implementing or running an adaptive DNA attack, document the actual code
behavior:

- seed derivation and whether seeds are known to the attacker;
- block size and block ordering;
- permutation, masking, retention, mixing, and shrinkage rules;
- which operations are deterministic and which are random;
- whether the server needs to invert any part of the transform;
- which steps are lossless and which intentionally lose information;
- whether the transform is linear for a fixed seed and block realization.

Do not infer privacy from the words "permutation", "attenuation", "DNA", or
"masking". The exact numeric transform must be inspected.

If the fixed-realization transform is linear or nearly linear, attempt direct
inversion or rank/conditioning analysis before building a more expensive
optimization attack.

## 7. Adaptive Attacker Levels

Adaptive evaluation proceeds in levels. A higher level is attempted only after
the lower-level baseline gate is meaningful in the same scope.

### Level 0: Non-adaptive Reference

Attack the transmitted update without modeling DNA Transform. This is a
reference only. Failure here does not prove DNA protection.

### Level 1: Algorithm-aware, Seed-unknown

The attacker knows the DNA Transform algorithm and hyperparameters but not the
specific random realization or seed. The protocol must state how randomness is
handled: expectation, restarts over candidate seeds, surrogate transform, or a
documented approximation.

### Level 2: Algorithm-aware, Seed-known

The attacker knows the transform seed or exact random realization when the
threat model permits it. This is the strongest standard adaptive setting for the
current prototype.

If Level 2 fails, the conclusion must first check whether the baseline attack in
the same scope passed controls. If baseline also fails, DNA protection is not
established.

## 8. Methods and Controls

Minimum methods:

- FL Baseline;
- DNA lossless;
- DNA Transform;
- random retention;
- top-k retention;
- clipping/noise.

Additional methods may be included only if their utility and compute budgets are
available and comparable.

DNA lossless is a representation baseline. If it is bit-exact, it should not be
presented as a privacy defense.

Secure Aggregation changes what the server observes. Individual-update inversion
under Secure Aggregation is only valid as pre-aggregation leakage analysis unless
the protocol explicitly evaluates aggregate-level inversion.

## 9. Ablation Plan

DNA Transform ablations should remove or vary one mechanism at a time:

- no mixing;
- no retention or attenuation;
- no shrinkage;
- alternate block size;
- arithmetic non-DNA equivalent where possible;
- fixed seed versus logged dynamic seed replay where relevant.

Each ablation reports:

- utility;
- reconstruction metrics;
- overhead;
- class-wise results;
- whether the baseline gate for that scope was open.

Do not use update cosine similarity as a substitute for reconstruction metrics.

## 10. Metrics

Primary reconstruction metrics for tabular data:

- feature-wise MSE;
- mean and median record MSE;
- class-wise MSE;
- feature-wise MAE in scaled space and raw units where available;
- categorical argmax accuracy;
- one-hot validity.

Secondary metrics:

- cosine similarity;
- Pearson correlation;
- sign-match ratio;
- pseudo-image PSNR and SSIM, clearly labeled as pseudo-image diagnostics.

Metrics must be reported separately for fraud and non-fraud. A global mean may
be shown only as a secondary summary.

## 11. Tuning and Evaluation Split

Development data may be used to choose:

- attack objective variant;
- attack learning rate;
- iteration budget;
- number of restarts;
- at most one or two DNA candidate configurations.

The final evaluation set must be source-disjoint from development targets. If a
configuration is changed after seeing evaluation results, a new development and
evaluation cycle is required.

## 12. Stop Rules

Stop and diagnose instead of expanding when:

- baseline passes only prior or only zero-update;
- baseline wins a majority but p-value misses the locked gate;
- results improve only in aggregate but fail by class;
- an adaptive DNA attack fails but the same-scope baseline also fails;
- implementation changes affect objective, replay, transform, or sampling;
- repeated failures have the same pattern.

The response to these cases is not to increase sample count until significance
appears. The next step is mechanism diagnosis.

## 13. Reporting Rules

The Phase 4 report must separate:

- official no-oracle adaptive attack results;
- oracle-label or class-decomposed diagnostics;
- Secure Aggregation true server-side results;
- Secure Aggregation pre-aggregation leakage analysis;
- development results;
- final evaluation results.

Every table must state the scope, class group, attacker knowledge, defense
configuration, controls, and sample count.

Claims must be limited to scopes where gates pass.

## 14. Expected Artifacts

Phase 4 should produce:

- `phase4_protocol.md`;
- `phase4_report.md`;
- `artifacts/phase4/<run_id>/protocol_lock.json`;
- `artifacts/phase4/<run_id>/baseline_gate.json`;
- `artifacts/phase4/<run_id>/adaptive_attack_results.json`;
- `artifacts/phase4/<run_id>/defense_comparison.csv`;
- `artifacts/phase4/<run_id>/ablation_results.csv`;
- `artifacts/phase4/<run_id>/selected_configs.json`;
- saved original, initial, and reconstructed vectors;
- per-record metrics;
- runtime and failure logs;
- manifest/checksums where feasible.

## 15. Phase 4 Decision

### Proceed to Phase 5

Proceed only if DNA Transform reduces reconstruction relative to baseline and
simple controls in an open scope while keeping utility and compute within the
locked criteria.

### Stop or Redesign DNA

Stop or redesign if random retention, top-k retention, clipping/noise, or another
simple method matches or exceeds DNA under comparable utility and compute.

### Report Mixed Evidence

If DNA helps only on fraud, only under oracle diagnostics, or only at a bounded
scope, report exactly that. Do not generalize to full-client FL.

### Inconclusive

If baseline attack does not pass controls in the evaluated scope, report the
defense comparison as not opened or inconclusive.
