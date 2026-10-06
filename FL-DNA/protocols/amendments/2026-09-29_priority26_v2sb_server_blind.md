# Amendment — Priority 26: Transform v2-SB server-blind aggregation

Date written: 2026-09-29  
Status: FROZEN / AUTHORIZED FOR REDUCED PRIORITY 26 EXECUTION  
Scope: new mechanism and new registered question; no edits under `Latex/`.

## Registered question

RQ1-v2SB. Against an honest-but-curious server that does not hold the sketch key, and assuming no collusion between the server and any client, does server-blind sketch aggregation (v2-SB) reduce reconstruction quality of individual client updates, measured under an exact one-sided sign test on input-space reconstruction MSE, more than a distortion- or utility-matched DP baseline?

## Naming and relation to earlier mechanisms

Transform v2-SB is not a new transform family and must not be called v3.  It
uses the v2 sketch and quantization parameters:

- compression ratio `k/d = 0.95`;
- quantization eta `0.01`.

Only the protocol/key ownership changes:

- clients share a 256-bit round key `K`;
- clients sketch with the same `R_K`;
- the server aggregates sketches linearly without decoding and never receives
  `K`;
- clients decode the aggregate.

This does not change Priority 24/25 answers for Transform v1/v2.

## Compute estimate and authorized reduction

Full requested scope includes utility, privacy, cross-round attacks, T2 arms,
and RQ3-style cost.  To keep the execution under approximately three days and
avoid unqualified attacker claims, the requested fallback order is applied:

1. Drop Part 4 cost benchmark.
2. Drop A4 cross-round attack.
3. Drop all T2-based arms.

The reduced execution keeps:

- Part 0 key-space audit;
- Part 1 v2-SB module and unit tests;
- Part 2 RQ2-style utility confirmatory if implementation passes tests;
- Part 3 T1-only privacy tests:
  - A1 seed-leak control with `K`;
  - A2 key-less direct attack using identity/padded sketch and random-surrogate lifts;
  - A3 invariant-statistics ridge regressor trained on development sketches.

If Part 2 utility runtime is interrupted or exceeds practical wall-clock
constraints, report the partial run and do not replace/drop seeds.

## Part 0 — key-space audit

Before running v2-SB privacy tests, report:

- v1 effective seed space from `dna_encoder/transform_defense.py`;
- v2 effective seed space from `dna_encoder/transform_defense_v2.py`;
- `privacy/seed_manager.py::derive_seed` output range;
- measured time per candidate v2 decode of the first BN `running_mean`
  tensor;
- projected time to enumerate 2^32 candidates for v2 and 2^31 candidates for
  v1, single process and with 9 processes;
- seed-verification criterion:
  - for v2 BN tensor, reconstruct using a candidate seed and score
    standardized T1 batch-mean MSE against the true mean;
  - compare true seed vs 1,000 random wrong seeds on 8 development targets.

No full brute force is authorized.

## Part 1 — v2-SB implementation

New module only:

```text
dna_encoder/transform_defense_v2_server_blind.py
```

No existing transform module may be modified for v2-SB.

Keying:

- round key is exactly 32 bytes from `secrets.token_bytes(32)` in real use;
- tests may use deterministic 32-byte test keys;
- randomness is derived by SHAKE-256 over domain-separated encodings of:
  `K`, round number, tensor index, client id where relevant, and purpose;
- no `% 2**31` or `% 2**32` truncation is allowed in the v2-SB path.

Unit tests:

1. linearity: `R^T(sum_i w_i q_i)` equals `sum_i w_i R^T(q_i)` to tolerance;
2. one-client v2-SB lift matches v2-style lift for the same materialized
   projection and quantization draw where applicable;
3. server aggregator object has no access to `K`;
4. key-space: key length is 256 bits and no seed truncation is used.

## Part 2 — RQ2-style utility

If Part 1 tests pass, run paired non-inferiority vs unprotected baseline:

- 500k rows;
- 50 rounds;
- 3 clients;
- 1 local epoch;
- batch 1024;
- margins: F1 `0.02`, AUC `0.005`;
- n = 52 fresh seeds, disjoint from prior RQ2 seeds;
- workers up to 9, one thread per process;
- also report PR-AUC descriptively.

Because v2-SB is linearly decoded by clients before applying FedAvg, utility is
expected to be close to Transform v2, but this is an empirical confirmatory
test and no early stopping is allowed.

## Part 3 — T1-only privacy vs DP

Threat model:

- honest-but-curious server;
- server sees every individual sketch `q_i` and aggregate sketch `q_agg`;
- server knows the transform family and hyperparameters;
- server does not know `K`;
- no collusion.

T1 instrument:

- closed-form BN-statistics batch-mean recovery from `network.1.running_mean`;
- standardized input-space MSE;
- Prior: population mean;
- decoy: another target's reconstruction under the same attack branch.

Targets:

- fresh PaySim targets, 4 records per group, 1 fraud;
- n=8 development for A1/A2/A3 gates and calibration;
- n=39 confirmatory for the strongest Level-1 attack if any Level-1 attack
  beats Prior and decoy at n=8.

Attacks:

- A1 seed-leak control: attacker is given `K`, decodes the BN sketch, and runs
  T1.  Gate must beat Prior and decoy at n=8; this demonstrates instrument
  validity when the key leaks.
- A2 key-less direct attack:
  - identity/padded sketch: put observed sketch entries into the first
    positions of a 128-dimensional BN delta and run T1;
  - random surrogate lift: use a fixed public random surrogate key to lift the
    sketch and run T1.
- A3 invariant-statistics attack:
  - train a ridge regressor on development targets with fresh random keys;
  - features are key-invariant sketch statistics: norm, mean, standard
    deviation, absolute mean, quantiles, histogram counts, and pairwise
    differences where available;
  - target is the true 13-dimensional batch mean;
  - evaluate on confirmatory targets only if A3 beats Prior and decoy at n=8.

Comparators:

- DP distortion-matched for T1: Gaussian noise on the same BN running-mean
  vector, calibrated on development captures to match median v2-SB decoded
  distortion.
- DP utility-matched: Priority 25 v2 selected full-update DP multiplier
  `sigma=3e-5`, clip norm `259.0841131896973`, applied to the full transmitted
  update, then T1 reads the BN running-mean tensor.
- SecAgg descriptive baseline: server sees no individual update.  For T1 this
  is reported as unanswerable/hidden individual signal; no superiority claim is
  made.

Confirmatory tests:

- For each Level-1 attack that qualified at n=8, compare v2-SB vs
  DP-distortion and v2-SB vs DP-utility at n=39.
- Primary direction: v2-SB MSE larger than DP MSE.
- Opposite direction also reported.
- Holm correction within this RQ1-v2SB family.

Answer rule:

"Yes": v2-SB > DP is significant after Holm for at least one matching on the strongest pre-specified Level-1 attack (the attack with the lowest v2-SB reconstruction MSE), AND DP > v2-SB is not significant for any test.

"No": otherwise.

Always state these conditions:

- `K` is secret from the server;
- no collusion;
- fresh `K` every round;
- v2-SB is not differentially private if `K` leaks;
- update/sketch norms leak and are reported.

