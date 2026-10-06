# Priority 28 protocol amendment: audit of published FL defenses with four checks

Date written: 2026-09-30

Status: FROZEN / AUTHORIZED

Scope: This is a new, pre-registered extension audit. It does not modify any
earlier RQ1/RQ2/RQ3 result, artifact, report, or conclusion. No file under
`Latex/` may be edited by this priority.

External code boundary: `external_defenses/` is read-only. Priority 28 may
import it, or copy small adapter code into `experiments/priority28/` with
attribution, but must not edit `external_defenses/`.

Threading: every worker process must keep `torch.set_num_threads(1)`.

Disclosure rule: every run, including interrupted, resumed, replayed,
discarded, or failed runs, must be listed in the final report.

No early stopping: development/pilot/confirmatory target sets and runs that are
pre-specified below are not dropped because partial results are unfavorable.
When a protocol says "stop this instrument if the gate fails", stopping is a
pre-registered rule, not outcome-driven early stopping.

## Compute estimate and reduction

The unreduced design covers 7 defenses, 2 domains, at least 4 instrument cells,
3 evaluation modes, utility matching grids for tabular E3, and CIFAR attacks at
batch 1 and batch 4 with 2,000-4,000 optimization iterations. Based on previous
Priority 23/27 timings, the unreduced matrix is expected to exceed five wall
days on this machine even with worker parallelism.

The following reduction is therefore frozen before any run:

1. E3 DP-comparator runs are executed only for defense/domain/instrument cells
   that survive E2. Cells that do not survive E2 are labeled
   `DOES_NOT_SURVIVE_E2`; E3 is not run for them.
2. CIFAR batch-4 qualification is deferred unless CIFAR batch-1 qualifies and
   the projected remaining wall time is below five days. The primary CIFAR
   qualification setting for Priority 28 is FedSGD batch 1.
3. Utility grids for E3 use 6 points by default. A one-time extension is allowed
   only when the grid is not bracketed and the corresponding defense survived
   E2.
4. ATS is evaluated last. If Steps 0-2 for all other defenses already exceed the
   compute ceiling, ATS is reported as `DEFERRED_BY_FROZEN_COMPUTE_REDUCTION`.

## Defenses and domains

Defenses:

- Soteria, using client-side representation scores from the model and local
  data and pruning the final classifier gradient columns.
- PRECODE, implemented at model level as a variational bottleneck in the
  forward pass. A pass-through update adapter is not a valid PRECODE defense.
- Gradient pruning, using per-tensor bottom-magnitude pruning of trainable
  gradients.
- Count-Sketch aggregation, FetchSGD-style. It is audited as an aggregation
  sketch, not as a defense with a published privacy claim.
- DNA Transform v1 conservative, with mix `0.08`, keep `0.88`, shrink `0.45`.
- DNA Transform v2, with compression ratio `0.95`, quantization eta `0.01`.
- ATS, CIFAR only, using the published policy implementation when feasible.

Domains:

- PaySim tabular MLP with BatchNorm, matching the RQ2 model family.
- CIFAR-10 LeNet-style CNN, approximately 137k trainable parameters.

## Step 0: mandatory harness checks

The Priority 27 harness is extended with the following fail-closed checks. The
report generator must refuse to emit a result for any defense/attack cell that
fails a mandatory check.

1. Lossless sanity:
   - every adaptive attacker must recover the input within tolerance when the
     corresponding defense is in identity/lossless mode and the attacker gets
     exactly the declared server knowledge;
   - v2 key-holder recovery must use a least-squares solve through the sampled
     sketch map, not the naive `R_s^T q` lift;
   - tolerances are `1e-5` max absolute error for tabular toy fixtures and
     `1e-4` image MSE for image toy fixtures.
2. Defense-not-identity:
   - with published hyperparameters, the defense must measurably change the
     transmitted update or model forward pass;
   - PRECODE must be represented by a real variational bottleneck forward path;
   - Soteria must compute representation scores client-side;
   - ATS must transform training inputs.
3. Server-knowledge consistency:
   - if decoding or aggregation uses secret `X`, then `X` must be declared in
     `server_knowledge`;
   - the adaptive attacker must receive the same declared knowledge.
4. Decoy correctness:
   - a decoy reconstruction must be another target's reconstruction scored
     against the current target, never against the decoy target.
5. Positive control:
   - before any defense test, the undefended attack must beat the Prior, the
     decoy, and data-free baselines;
   - tabular data-free baseline: training population mean;
   - CIFAR baselines: constant gray image and CIFAR training-set mean image;
   - qualification must pass at n=8 and n=24 by exact one-sided sign test,
     alpha `0.05`.

Unit tests are mandatory for all five checks.

## Step 1: instrument qualification

### Tabular instruments

T1: closed-form BN-statistics attack. Estimand is the first-layer input batch
mean recoverable from transmitted BatchNorm running statistics. This is a
qualified instrument only for statistics that include transmitted BN buffers.

T2: TabLeak-style gradient attack on trainable parameters, tested in two
settings only if the setting materially differs from Priority 27:

- one local Adam-step setting, matching the protocol optimizer;
- one FedSGD step setting, the literature-standard secondary setting.

Priority 27 found no gradient-only tabular cell qualified. Priority 28 therefore
does not repeat the identical Priority 27 settings. It may run a T2 retest only
if the code path differs materially because a published defense requires a
different forward path, such as PRECODE.

### CIFAR instrument

I1: cosine + total-variation gradient attack in the literature-standard FedSGD
setting, known label, batch 1. Frozen budget:

- 3,000 optimization iterations;
- 4 restarts;
- Adam attacker learning rate `0.05`;
- TV weight `1e-4`;
- restart selection by attacker-observable objective.

If batch-1 qualification passes and compute remains below the ceiling, batch 4
uses the same budget and gate as secondary.

Failed instrument rule: if an instrument fails its n=8 or n=24 positive-control
gate, that instrument is reported as `not qualified` and no defense test is run
with it.

## Step 2: evaluations for qualified cells

For each qualified `(defense, domain, instrument)` cell, draw fresh,
source-disjoint confirmatory targets with n=39. Development and pilot target
sets must also be fresh and source-disjoint. Overlap checks must compare source
IDs in code, not only target bundle names.

E1 original-style:

- use the non-adaptive attacker and metric described in the corresponding
  original defense paper or reproduction notes;
- question: whether the qualitative published conclusion ("the defense
  protects") is reproduced.

E2 four checks:

- positive control must already have passed;
- use the adaptive attacker matched to declared server knowledge;
- score in input space against raw input and data-free baselines;
- record DP-comparator specification if a DP baseline appears in the cell;
- primary test: defended reconstruction error > undefended reconstruction error
  by exact one-sided sign test.

E3 tabular DP comparator:

- run only if the corresponding cell survives E2;
- compare against:
  1. the paper's own Gaussian-noise baseline as documented in
     `external_defenses/REPRODUCTION_REPORT.md`, with no invented clipping;
  2. per-tensor DP noise with each tensor clip `C_t` set to the 95th percentile
     norm and composed RDP accounting;
  3. FedBN DP: BN statistics kept local, DP applied only to trainable
     parameters, with an upward grid extension if needed to bracket F1;
- utility matching uses 16 paired RQ2-style replicates unless reduced by an
  E2 failure. Matching rule follows Priority 25b: choose the largest sigma whose
  mean `Delta_F1` remains at least the defense's mean `Delta_F1 - 0.005`.

## Step 3: verdict rule

Per defense, RQ4 verdict:

- `SURVIVES`: under the adaptive attacker, the defense significantly reduces
  reconstruction relative to no defense, and it is not significantly worse than
  utility-matched FedBN DP.
- `DOES_NOT_SURVIVE`: otherwise. The report must name the check that flips or
  blocks the conclusion.
- `NOT_ASSESSABLE`: no qualified instrument in that domain.

Count-Sketch/FetchSGD has no reproduced published privacy conclusion in
`external_defenses/REPRODUCTION_REPORT.md`; its verdict is therefore framed as
an audit of a compression/aggregation mechanism, not as overturning a privacy
claim unless E1 supplies one.

Holm correction:

- E2 defended-vs-none p-values are adjusted across the Priority 28 E2 family.
- E3 defense-vs-DP p-values are adjusted across the Priority 28 E3 family.
- CIFAR and tabular are reported separately as descriptive subfamilies in
  addition to the study-wide counts.

## Deliverables

The final report is:

`reports/priority28_defense_audit_report_20260930.md`

It must include exact commands, SHA-256 hashes of every new file, Step 0 unit
test results, qualification tables, E1/E2/E3 tables with Holm correction, the
RQ4 verdict table, example CIFAR reconstructions and tabular rows when a cell is
run, a list of deviations from original papers, and per-target CSVs for every
reported test.
