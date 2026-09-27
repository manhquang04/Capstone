# Amendment: Priority 14 confirmatory escalation after n=24 pilot

**Date:** 2026-09-21  
**Status:** FROZEN/AUTHORIZED — exactly one confirmatory execution authorized per listed cell.  
**Authorization:** Supervisor approval received in chat on 2026-09-21, before confirmatory target generation or execution.  
**Scope:** confirmatory escalation for the two Priority 14 clean-vector DNA-vs-DP cells after n=24 development pilot.

## Triggering evidence

The Priority 14 n=24 development pilot, authorized by
`protocols/amendments/2026-09-21_priority14_n24_pilot_escalation.md`,
showed the same direction as the n=8 probe for both cells:

| Cell | DNA wins | DP wins | Exact ties | DP one-sided p |
|---|---:|---:|---:|---:|
| PaySim v1 stronger vs DP 0.0004 | 0 | 24 | 0 | 5.960464477539063e-08 |
| PaySim v2 ratio0.95/eta0.01 vs DP 0.00105 | 0 | 24 | 0 | 5.960464477539063e-08 |

This amendment records the confirmatory power calculation required by the n=24
escalation amendment and authorizes exactly one confirmatory run for each listed
cell.

## Frozen design

The tested cells must remain identical to the n=24 development pilot:

1. PaySim DNA Transform v1 stronger vs distortion-matched DP.
   - DNA v1 stronger: `mix_ratio=0.12`, `keep_ratio=0.82`,
     `shrink_factor=0.35`.
   - DP matched: clipping/noise Monte Carlo with `clip_norm=100`,
     `noise_multiplier=0.0004`, `mc_noise_samples=100`.
2. PaySim DNA Transform v2 ratio0.95/eta0.01 vs distortion-matched DP-v2.
   - DNA v2: `compression_ratio=0.95`, `quantization_eta=0.01`.
   - DP matched: clipping/noise Monte Carlo with `clip_norm=100`,
     `noise_multiplier=0.00105`, `mc_noise_samples=100`.

Both cells must keep:

- trainable-parameter-only vector definition;
- all buffers excluded;
- original raw-lift/harddiff attacker;
- balanced-tensor objective;
- 4 restarts per branch;
- 600 optimization iterations;
- Adam learning rate `0.1`;
- `0.001 * _nonnegative_penalty(latent, meta)`;
- no GEN_COSINE_TV, no GEN_IDLG_STYLE, and no parameter retuning.

## Power analysis

Power analysis artifact:

`artifacts/priority14_fresh_probe/confirmatory_power_p1_0p70_20260921/rq1_power_analysis.json`

Inputs:

- exact one-sided binomial sign test;
- p0 = 0.50;
- p1 = 0.70, inherited from the pre-existing RQ1/Priority 6 minimum meaningful
  win-probability convention and not refit from the n=24 pilot;
- alpha = 0.05;
- target power = 0.80;
- tie rate = 0.0 for the primary calculation;
- dropout rate = 0.0 for the primary calculation.

Result:

- required effective non-tied n = 37;
- rejection threshold = at least 24 DP wins among 37 non-tied groups for the
  DP-advantage direction;
- actual alpha = 0.04943587479647249;
- actual power at p1 = 0.8070956916527874;
- initial draw count = 37 under the zero-tie/zero-dropout primary calculation.

The sensitivity table in the artifact gives draw counts under nonzero tie/dropout
assumptions; for example, with a 5% dropout rate and zero tie rate, draw count
would be 39.

Following project convention for a small infrastructure buffer, draw count is
frozen at `39` target groups for each cell. The primary rejection threshold is
still evaluated on the non-tied count; with 37 non-tied groups, the rejection
threshold is at least 24 DP wins.

## Data firewall

Each confirmatory target set must be newly generated and source-disjoint from all
previous PaySim targets, including:

- Priority 14 n=8 targets, both generic-penalty technical artifacts and
  corrected-penalty artifacts;
- Priority 14 n=24 development-pilot targets;
- Priority 12 and Priority 13 artifacts;
- all earlier development, post-hoc, confirmatory and replication targets under
  `artifacts/`.

The confirmatory run must not reuse any candidate, target, restart, or output
from n=8 or n=24 development.

## Execution boundary

This amendment authorizes only:

- creating two fresh confirmatory target sets, one for each listed cell, with
  draw count 39 and source-disjointness gate PASS;
- running exactly one confirmatory execution per listed cell using the frozen
  settings above;
- reporting the result without retuning or repeating.

This amendment does **not** authorize:

- any second confirmatory run if the result is unfavorable;
- changing p1, alpha, attacker parameters, DNA parameters, DP parameters, or the
  trainable-only vector definition after observing confirmatory results;
- technical replay except for clear infrastructure/runtime failure before a
  scientific result is obtained, documented by a separate pre-replay amendment.
