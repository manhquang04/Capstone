# Amendment — Priority 25: RQ1 extension with utility-matched DP and instrument audit

Date written: 2026-09-29  
Status: FROZEN / AUTHORIZED FOR REDUCED PRIORITY 25 EXECUTION  
Scope: new pre-registered extension to Priority 24; no edits under `Latex/`.

## Original RQ1 retained verbatim

RQ1. Does DNA-based update transformation reduce gradient-inversion reconstruction quality, measured via attack success under an exact one-sided sign test on reconstruction MSE, more than a distortion- or utility-matched DP baseline?

Priority 24 remains unchanged.  Its T1 distortion-matched results are included
in the final combined answer rule but are not rerun, replaced, or rescored.

## Motivation

Priority 24 answered the original RQ1 as `NO_NOT_SHOWN` under a reduced,
valid-instrument plan:

- qualified instrument: T1 BN-statistics batch-mean recovery;
- comparator family: distortion-matched DP only;
- v1 conservative: DNA 16 / DP 23, Holm p(DNA > DP) = 0.9002;
- v2 0.95/0.01: DNA 24 / DP 15, Holm p(DNA > DP) = 0.1996.

Two gaps remain:

1. the utility-matched DP comparator named in RQ1 has not been run for a valid
   instrument;
2. T1 uses transmitted BN statistics rather than gradients alone, and no
   gradient-only fraud-domain instrument has passed a positive control.

Priority 25 addresses gap 1 directly and records a bounded diagnostic for gap
2.  It does not reinterpret or modify frozen results from earlier priorities.

## Compute estimate and authorized reduction

The full requested Priority 25 scope includes utility-matched DP, T2, T3, and
CIFAR I1.  Based on existing RQ2 and attacker runtimes, the full plan is
estimated to exceed about three days of wall-clock time even with worker-level
parallelism:

- Utility grid, full request: 6 multipliers x 8 paired FL replicates plus norm
  probes and analysis.  With 50-round PaySim FL jobs at roughly 9--10 minutes
  each and `--workers 9`, this is feasible but non-trivial.
- T2 TabLeak-style gradient-only attack requires new implementation and
  validation.  A positive-control chain at n=8, n=24, optional retry, and
  confirmatory n=39 across multiple branches would dominate the runtime and
  cannot be treated as a safe quick add-on.
- T3 repeats the same expensive attack in a secondary FedSGD setting.
- CIFAR I1 is secondary-domain evidence and was already shown to require
  careful positive controls in Priorities 21--23.

The authorized fallback order in the user prompt is applied as follows:

1. Drop Part 3 (CIFAR DP comparator) from this execution.
2. Drop T3 from this execution.
3. Shrink Part 1 grid to four pre-frozen log-spaced multipliers.
4. Execute Part 2a Adam-sign diagnostic only.  T2 is not used as an RQ1
   instrument unless a later amendment freezes and validates a real
   TabLeak-style implementation.  Therefore the gradient-only answer for this
   Priority can only be "not answerable with the gradient-only attackers
   evaluated" unless a separately authorized T2 gate is run later.

This reduced scope still closes the utility-matched-DP gap for the already
qualified fraud-domain T1 instrument.

## Part 1 — Utility-matched DP for T1

### Mechanism

Use the existing full-client-update Gaussian mechanism applied to every
floating tensor in the transmitted state, including BatchNorm buffers, matching
the RQ2 DP code path:

```text
privacy/dp_engine.py::apply_dp_to_local_state
```

The whole floating update is clipped to norm `C`, then independent Gaussian
noise `N(0, (sigma * C)^2)` is added per floating coordinate.

### Norm probe and clip norm

Before any utility grid run, run exactly two development training replicates
to collect per-client, per-round full transmitted-update norms under the
unprotected FL training path:

- norm-probe seeds: `[2501001, 2501002]`;
- training setup: 500k rows, 50 rounds, 3 clients, one local epoch, batch 1024,
  focal loss alpha 0.95 gamma 2.0;
- torch threads: one per process.

Set:

```text
C = 1.01 * empirical 95th percentile of collected full-update norms
```

After the norm probe, write a machine-readable config freezing `C` and the
utility grid before any DP utility grid job is run.

### Utility grid

Use four log-spaced multipliers:

```text
[1e-6, 3e-6, 1e-5, 3e-5]
```

Rationale: previous full-update DP utility evidence showed that
`noise_multiplier=1e-5` is close to utility-neutral while `0.000315` and above
cause large F1 degradation.  The grid is therefore focused on the plausible
utility-matching region and is frozen before execution.

### Development replicates

Use exactly eight new paired seeds, disjoint from known RQ2 confirmatory and
pilot seeds:

```text
[2501101, 2501102, 2501103, 2501104, 2501105, 2501106, 2501107, 2501108]
```

For every seed, run a paired unprotected baseline and all four DP grid points.
No seed may be dropped after seeing results.

### Matching rule

For each transform `T` in `{v1 conservative, v2 0.95/0.01}`:

```text
sigma_T = largest grid sigma whose development mean ΔF1 >= (RQ2_mean_delta_F1(T) - 0.005)
```

The frozen RQ2 mean deltas are:

- v1 conservative: `+0.0014`, so the threshold is `-0.0036`;
- v2 0.95/0.01: `-0.0117`, so the threshold is `-0.0167`.

If no grid point satisfies the rule, use the smallest sigma and report this.
Also report ΔAUC, ΔPR-AUC, and RDP ε for one release and 50 releases at
δ=`1e-5`, update-level add/remove adjacency.

### Confirmatory T1 use

Only after the utility grid selects `sigma_T`:

1. Generate fresh source-disjoint PaySim T1 development targets, n=8.
2. Re-verify T1 qualification against Prior and decoy on that development set.
3. Generate fresh source-disjoint PaySim confirmatory targets, n=39.
4. For each confirmatory target, apply:
   - none;
   - DNA transform;
   - utility-matched DP using the selected `C` and `sigma_T`.
5. Run T1 on BN statistics and score per-feature-standardized input-space MSE.
6. Test DNA-vs-DP_utility in both directions with an exact one-sided sign test.
7. Tie band comes from deterministic development replay; if replay discrepancy
   is zero, tie band is zero.

No early stopping is authorized.

## Part 2 — Gradient-only fraud-domain evidence

### Part 2a diagnostic

Run a descriptive Adam-step diagnostic on eight fresh development targets:

- scope: 4 records, 1 fraud, one local Adam step from the fixed capture path;
- report cosine similarity and sign agreement between the observed Adam local
  update and `-lr * sign(gradient)`;
- report whether gradient magnitude is recoverable from the Adam step.

This diagnostic does not qualify an instrument and is not an RQ1 test.

### T2 status in this reduced amendment

No T2 confirmatory or gate run is authorized here.  A future T2 attempt must
write a separate amendment freezing a concrete TabLeak-style implementation,
including objective, softmax relaxation, ensemble pooling rule, restarts,
iterations, runtime estimate, and positive-control gates.

Therefore Priority 25's gradient-only statement is pre-registered as:

```text
not answerable with the gradient-only attackers evaluated
```

unless a later separate amendment executes and qualifies T2.

## Part 3 and T3 status

Part 3 CIFAR DP comparator and T3 FedSGD attacker are dropped from this
execution under the compute-reduction rule.  They may be revisited only under
new amendments.

## Combined RQ1 answer rule

Primary family for this reduced Priority 25 report:

1. Priority 24 T1 x distortion-matched:
   - v1 conservative DNA>DP p = `0.9002045665401965`;
   - v2 DNA>DP p = `0.09979543345980349`;
   - v1 conservative DP>DNA p = `0.16839181759496574`;
   - v2 DP>DNA p = `0.9459354892969714`.
2. Priority 25 T1 x utility-matched:
   - v1 conservative;
   - v2 0.95/0.01.

Apply Holm separately to:

- all DNA>DP p-values in this primary family;
- all DP>DNA p-values in this primary family.

Answer:

- "Yes": at least one DNA>DP test is significant after Holm and no DP>DNA
  test is significant after Holm.
- "No": otherwise.  Report whether the reason is significant DP superiority
  or no significant difference.

Gradient-only statement:

- "answered with a qualified gradient-only instrument" only if T2 qualifies in
  a separately authorized run;
- otherwise "not answerable with the gradient-only attackers evaluated."

## Reporting

Write `reports/priority25_rq1_extension_report_20260929.md` with:

- exact commands;
- SHA-256 hashes for all new artifacts;
- compute reduction rationale;
- utility grid table with ΔF1, ΔAUC, ΔPR-AUC and ε;
- T1 qualification table;
- utility-matched T1 confirmatory tables;
- Holm-adjusted combined primary-family p-values;
- Adam-sign diagnostic;
- plain combined RQ1 answer and gradient-only statement.

Disclose every run, including failed, interrupted, resumed, or discarded runs.

