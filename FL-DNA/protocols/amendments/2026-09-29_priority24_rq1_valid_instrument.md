# Amendment — Priority 24: valid-instrument RQ1 test

Date written: 2026-09-29  
Status: FROZEN / AUTHORIZED FOR REDUCED PRIORITY 24 EXECUTION  
Scope: new pre-registered experiment to answer the original RQ1 where possible; no edits under `Latex/`.

## Original RQ1 retained verbatim

RQ1. Does DNA-based update transformation reduce gradient-inversion reconstruction quality, measured via attack success under an exact one-sided sign test on reconstruction MSE, more than a distortion- or utility-matched DP baseline?

## Why earlier RQ1 evidence is not reused

The earlier RQ1 evidence is not used as confirmatory evidence for this
valid-instrument test because Priorities 21--23 found:

- the DP-vs-transform head-to-head statistic was an update-space residual, not
  input-space reconstruction quality;
- that update-space residual included a DP noise floor;
- old DP multipliers were calibrated on BN-contaminated vectors rather than the
  attacked/trainable vectors;
- the tabular optimization attacker failed an undefended positive control;
- the CIFAR Zero-update control was degenerate in the original image-domain
  gate;
- the utility-matched DP baseline was not available for the same valid
  instrument/comparator chain.

Priority 24 therefore runs a new valid-instrument chain instead of reusing those
results.

## Compute estimate and reduced plan

Full requested scope (T1, T2, T3, I1; distortion and utility matching; n=8 ->
n=24 -> confirmatory; utility grid of 5--7 multipliers x 8 FL replicates) is
estimated to exceed the approximately 3-day budget:

- Utility-matched fraud-domain DP grid alone: at least 5 multipliers x 8
  paired FL replicates. Based on existing RQ2 runs, one full 50-round method
  job takes roughly 9--10 minutes wall time on this machine; paired baseline/DP
  grids would require many machine-hours and multi-day wall time when run under
  the single-thread-per-process discipline.
- T2/T3 upgraded TabLeak-style optimization requires designing and validating a
  new tabular optimizer. Priority 22 already showed the existing tabular
  optimizer is not a valid positive-control instrument, so a bounded quick fix
  would risk another invalid instrument.
- The CIFAR repaired attacker is secondary-domain evidence and Priority 23
  showed it is target-specific but weak against gray/mean-image controls.

Reduced-but-valid plan for this amendment:

1. Fully execute fraud-domain T1 (BN-statistics channel) as the only
   confirmatory RQ1 instrument in this Priority.
2. Use distortion-matched DP only. This still answers the registered "distortion-
   or utility-matched" RQ1 because distortion-matched is one of the two
   pre-registered comparator families.
3. Defer T2/T3 upgraded optimization, utility-matched DP grid, and I1
   confirmatory expansion to later amendments if supervisor explicitly allocates
   additional compute and attacker-development time.

## Step 1 — Instrument qualification

Instrument T1: closed-form BN-statistics recovery of the 4-record batch mean
from the first BatchNorm `running_mean` channel. This is explicitly a
transmitted-update/statistics reconstruction channel, not gradient-only
reconstruction.

Data:

- PaySim fraud domain.
- 4-record bounded local update with exactly 1 fraud record per group.
- Fresh source-disjoint target sets generated separately for n=8 development,
  n=24 pilot, and n=39 confirmatory.

Gate:

- Score per-feature-standardized input-space MSE against the true 4-record
  batch mean.
- Prior control: RQ2-training population feature mean.
- Decoy control: recovery from another fresh target group's undefended
  `running_mean` signal, evaluated against the current target's true mean.
- T1 qualifies only if recovery beats both Prior and decoy under the exact
  one-sided sign test at alpha 0.05 at n=8 and again at n=24.
- No tuning or bounded retry is authorized for T1. If it fails either gate,
  RQ1 is unanswerable in the fraud domain under this reduced plan.

## Step 2 — DP comparators calibrated on the attacked vector

Attacked vector for T1: first-BN `network.1.running_mean` update.

Transforms:

- v1 conservative: `mix=0.08`, `keep=0.88`, `shrink=0.45`;
- v2: `compression_ratio=0.95`, `quantization_eta=0.01`, decoded with
  server-held seed/metadata.

DP comparator:

- Gaussian mechanism on the same BN `running_mean` vector.
- Clip norm is set to at least the 95th percentile of undefended BN-vector
  norms on development captures.
- Noise multiplier is chosen so median DP L2 distortion on the attacked vector
  matches the transform's median L2 distortion on the development captures.
- The chosen clip norm and multipliers are frozen after development
  calibration and before pilot/confirmatory DP-vs-DNA scoring.
- Report RDP epsilon at delta `1e-5`, update-level add/remove adjacency, one
  release and 50 releases.

Utility-matched DP is not run in this reduced Priority 24 scope.

## Step 3 — Confirmatory DNA-vs-DP tests

Only if T1 qualifies at n=8 and n=24:

- Draw fresh source-disjoint confirmatory targets, draw n=39.
- For each target and each transform comparator, compute standardized
  input-space MSE of the BN recovered mean under:
  - none,
  - DNA transform,
  - distortion-matched DP,
  - Prior,
  - decoy.
- A DNA win means `MSE_DNA > MSE_DP + tie_band`; a DP win means
  `MSE_DP > MSE_DNA + tie_band`.
- Tie band is fixed from deterministic replay stability on development targets.
  If replay discrepancy is exactly zero, tie band is zero.
- Primary test: exact one-sided sign test for "DNA wins more than half the
  non-tied targets".
- Also compute exact one-sided sign test in the opposite direction and effect
  sizes including median MSE ratio.

Multiplicity:

- Holm correction across all confirmatory DNA-vs-DP tests in this amendment.
- In the reduced plan, there are two tests: v1-conservative vs DP and v2 vs DP.

## Step 4 — Pre-registered RQ1 answer rule

- "Yes": DNA significantly beats DP after Holm correction under at least one
  distortion-matched fraud-domain T1 test, and DP does not significantly beat
  DNA under any qualified fraud-domain T1 test.
- "No": DP significantly beats DNA on any qualified fraud-domain T1 test, or
  there is no significant difference in either direction on any qualified
  fraud-domain test.
- "Unanswerable": T1 fails qualification and no other fraud-domain instrument
  qualifies in this amendment.

## Reporting

Report exact commands, target/provenance hashes, calibration values, raw
win/loss/tie counts, exact p-values, Holm-adjusted p-values, per-target CSVs,
and a few tabular example reconstructions.
