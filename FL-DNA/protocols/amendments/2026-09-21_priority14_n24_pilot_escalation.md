# Amendment: Priority 14 n=24 development pilot escalation

**Date:** 2026-09-21  
**Status:** DEVELOPMENT PILOT AUTHORIZED — no confirmatory execution authorized.  
**Scope:** n=24 escalation for the two Priority 14 clean-vector DNA-vs-DP cells that showed DP wins 8/8 at n=8.

## Motivation

Priority 14's fresh n=8 clean-vector probe optimized the original raw-lift
attacker directly on trainable parameters only, using the corrected
`_nonnegative_penalty(latent, meta)` helper. Both cells reproduced the Priority
13 direction at the smallest probe size:

- PaySim v1 stronger vs distortion-matched DP: DNA wins 0, DP wins 8.
- PaySim v2 ratio0.95/eta0.01 vs distortion-matched DP-v2: DNA wins 0, DP wins 8.

Because prior project history includes small-sample optimism at n=8/n=24 in
other attacker families, this amendment authorizes an intermediate n=24
development pilot only. It does not authorize jumping directly to confirmatory
execution.

## Frozen vector and attacker definition

All Priority 14 n=24 jobs must keep the exact corrected definition from the n=8
probe:

- use only trainable parameters (`model.named_parameters()` with
  `requires_grad=True`);
- exclude all model buffers, including BatchNorm `running_mean`,
  `running_var`, `num_batches_tracked`, and any other non-trainable buffer;
- construct both DNA and DP defended signals from that trainable-only vector;
- compute attacker loss and final head-to-head MSE on that same trainable-only
  vector;
- use the original raw-lift/harddiff attacker with balanced-tensor objective,
  known labels, Adam learning rate `0.1`, 600 optimization iterations, 4
  restarts per branch, and `0.001 * _nonnegative_penalty(latent, meta)`.

No GEN_COSINE_TV, GEN_IDLG_STYLE, parameter retuning, or objective retuning is
authorized by this amendment.

## Authorized cells

Run exactly one n=24 pilot for each of the following cells:

1. PaySim DNA Transform v1 stronger vs distortion-matched DP.
   - DNA v1 stronger: `mix_ratio=0.12`, `keep_ratio=0.82`,
     `shrink_factor=0.35`.
   - DP matched: clipping/noise Monte Carlo with `clip_norm=100`,
     `noise_multiplier=0.0004`, `mc_noise_samples=100`.
2. PaySim DNA Transform v2 ratio0.95/eta0.01 vs distortion-matched DP-v2.
   - DNA v2: `compression_ratio=0.95`, `quantization_eta=0.01`.
   - DP matched: clipping/noise Monte Carlo with `clip_norm=100`,
     `noise_multiplier=0.00105`, `mc_noise_samples=100`.

## Target generation and data firewall

For each cell, create a new n=24 target set:

- PaySim;
- four records per group;
- one fraud record per group;
- source-disjoint from every existing target pool under `artifacts/`, including
  Priority 14 n=8 generic-penalty and corrected-penalty target sets.

The run report must record the target file path, target SHA-256, provenance
file path, dataset SHA-256, and maximum overlap with existing target artifacts.
The pilot must not run if the source-disjointness gate fails.

## Head-to-head decision rule

Use the same zero-threshold descriptive rule as Priority 14 n=8:

```text
D_i = MSE_DNA_i - MSE_DP_i
```

- DNA win: `D_i > 0`;
- DP win: `D_i < 0`;
- exact tie: `D_i == 0`.

Report exact one-sided sign-test p-values in both directions. This is still a
development pilot, not a confirmatory result.

## Escalation after n=24

- If n=24 still shows DP wins overwhelmingly in either cell, compute the
  confirmatory power analysis for that cell and draft a separate confirmatory
  amendment, but do not create confirmatory targets and do not run
  confirmatory execution in this turn.
- If n=24 no longer shows a strong DP-win pattern, stop that cell and report it
  as possible small-sample instability rather than a confirmed effect.

## Invariants

- Do not change `torch.set_num_threads(1)`.
- Do not tune after seeing n=24 results.
- Do not reuse any candidate or target from Priority 14 n=8.
- Do not run confirmatory execution under this amendment.
