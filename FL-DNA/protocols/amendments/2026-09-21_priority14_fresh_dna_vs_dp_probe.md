# Amendment: Priority 14 fresh clean-vector DNA-vs-DP probe

**Date:** 2026-09-21  
**Status:** DEVELOPMENT PROBE AUTHORIZED  
**Scope:** cheap n=8 probe only; no n=24 or confirmatory escalation in this amendment.

## Motivation

Priority 13 replayed two previously null DNA-vs-DP comparisons after excluding
BatchNorm buffers. The replay used candidates that had been optimized under the
old contaminated objective, then rescored them on trainable parameters only. The
zero-threshold replay showed DP beating DNA for every stored candidate pair, but
that could reflect frozen candidates that were optimized for the wrong vector
rather than the behavior of a clean-vector attacker.

Priority 14 therefore runs the cheapest fresh check first: optimize the attacker
directly on the clean vector from the start, using fresh n=8 target sets.

## Corrected vector definition

For every branch and every loss:

- include only trainable model parameters (`model.named_parameters()` with
  `requires_grad=True`);
- exclude all buffers from `model.named_buffers()`, including
  `running_mean`, `running_var`, `num_batches_tracked`, and any other
  non-trainable buffers;
- apply this definition before constructing DNA/DP defended signals and before
  computing attacker matching loss.

This is the same corrected vector convention used by Priority 12.

## Probe cells

Two cells are tested independently:

1. PaySim, DNA Transform v1 stronger vs distortion-matched DP.
   - DNA v1 stronger: `mix_ratio=0.12`, `keep_ratio=0.82`,
     `shrink_factor=0.35`.
   - DP matched: clipping/noise Monte Carlo with `clip_norm=100`,
     `noise_multiplier=0.0004`, `mc_noise_samples=100`.
2. PaySim, DNA Transform v2 ratio0.95/eta0.01 vs distortion-matched DP-v2.
   - DNA v2: `compression_ratio=0.95`, `quantization_eta=0.01`.
   - DP matched: clipping/noise Monte Carlo with `clip_norm=100`,
     `noise_multiplier=0.00105`, `mc_noise_samples=100`.

Each cell uses a new n=8 target set, source-disjoint from every previous target
pool, including Priority 12 and Priority 13 artifacts.

## Attacker generation

Use one simple attacker generation only:

- hard balance-difference parameterization;
- known target labels as in the original PaySim bounded-update attacker;
- balanced-tensor update matching objective;
- no GEN_COSINE_TV and no GEN_IDLG_STYLE;
- 4 restarts per branch;
- 600 optimization iterations;
- Adam learning rate `0.1`;
- nonnegative penalty `0.001`.

This keeps the probe comparable to the original RQ1 variant/IHT-family
structure while avoiding newer literature-style attackers.

## Head-to-head decision rule

This is a DNA-vs-DP head-to-head probe, not an attack-success-vs-control gate.

For each target group:

1. optimize a candidate directly against the clean-vector DNA-defended signal;
2. optimize a candidate directly against the clean-vector DP-defended signal;
3. select the best objective candidate per branch;
4. compute clean-vector defended-space MSE for each selected branch;
5. define:

```text
D_i = MSE_DNA_i - MSE_DP_i
```

A DNA win means `D_i > 0`: the DNA-defended signal had larger clean-vector
MSE than the DP-defended signal for the same group. A DP win means `D_i < 0`.
Exact `D_i == 0` is an exact tie. The n=8 probe reports zero-threshold
descriptive sign counts and exact one-sided sign p-values in both directions.

## Escalation rule

- If n=8 shows the same qualitative pattern as Priority 13 replay
  (DP wins overwhelmingly), supervisor may authorize an n=24 pilot and then
  confirmatory power analysis in a separate amendment.
- If n=8 is near 50/50 or otherwise not clear, stop and report that the
  Priority 13 44/44 and 176/176 pattern may have been tied to frozen candidates
  optimized under the old contaminated objective.

No escalation is authorized by this amendment.

## Invariants

- Do not change `torch.set_num_threads(1)`.
- Do not reuse any old target set.
- Do not tune any parameter after observing probe outcomes.
- Do not run n=24 or confirmatory from this amendment.

