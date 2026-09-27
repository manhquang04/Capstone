# Amendment: v1-medium Priority-14-style DNA-vs-DP head-to-head

**Date:** 2026-09-27  
**Status:** PRE-REGISTERED / AUTHORIZED ESCALATION CHAIN  
**Scope:** close the v1 configuration-alignment evidence gap without editing the paper.

## Rationale

The paper currently combines two v1 facts from different configurations:

- RQ2 utility non-inferiority used DNA Transform v1 **medium**
  (`mix_ratio=0.10`, `keep_ratio=0.85`, `shrink_factor=0.40`);
- Priority 14 RQ1 head-to-head DNA-vs-DP used DNA Transform v1 **stronger**
  (`mix_ratio=0.12`, `keep_ratio=0.82`, `shrink_factor=0.35`).

This amendment freezes a fresh Priority-14-style head-to-head test for the v1
medium configuration so the evidentiary chain can state whether the same v1
configuration that passed utility also loses to the distortion-matched DP
comparator.

## Frozen comparator

Use the existing v1-medium distortion-matched DP comparator from
`protocols/config/rq1_medium_confirmatory.json`:

| Field | Value |
| --- | ---: |
| DNA block size | 256 |
| DNA mix ratio | 0.10 |
| DNA keep ratio | 0.85 |
| DNA shrink factor | 0.40 |
| DNA run seed | 681958327 |
| DP clip norm | 100.0 |
| DP noise multiplier | 0.000315 |
| DP MC noise samples | 100 |
| DP defense seed | 314159265 |

The prior medium calibration reports:

- DNA median relative-L2 distortion: `0.09875763649729574`;
- DP median relative-L2 distortion: `0.09838606969571326`;
- relative mismatch: `0.0037624108348589245`;
- calibration gate: `PASS`.

No new DP calibration is authorized by this amendment.

## Frozen attacker/methodology

Reuse the Priority 14 clean-vector DNA-vs-DP head-to-head methodology:

- vector scope: trainable parameters only; model buffers excluded;
- attacker: simple balanced-tensor raw-lift/harddiff attacker;
- candidate optimization directly against the defended signal;
- penalty: corrected `0.001 * _nonnegative_penalty(latent, meta)`;
- restarts: 4 per group;
- iterations: 600 per restart;
- attack learning rate: 0.1;
- target scope: PaySim, 4 records/group, 1 fraud/group, 1 local Adam step.

For each group, select the best DNA restart by DNA MSE and the best DP restart
by DP MSE, exactly as in Priority 14.

Define:

```text
D_i = MSE_DNA_i - MSE_DP_i
```

- DP win: `D_i < 0` (attacker MSE is lower against DP-defended update);
- DNA win: `D_i > 0`;
- exact tie: `D_i == 0`.

For continuity with the older v1-family reporting, additionally report the
historical tie-threshold decomposition with `tie_threshold_mse = 0.0390625`.
However, the Priority-14-style zero-threshold decomposition is the primary
head-to-head check because Priority 14 showed that the old threshold can hide
all trainable-vector differences.

## Frozen statistical framework

- exact one-sided sign test;
- `p0 = 0.50`;
- `p1 = 0.70`;
- `alpha = 0.05`;
- power target `0.80`;
- required non-tied `n = 37`;
- confirmatory draw `n = 39`, following the same small project buffer used in
  Priority 14.

Escalation chain:

1. development probe: `n=8`;
2. pilot: `n=24`, only if the `n=8` stage does not fail decisively;
3. confirmatory: `n=39`, only if the `n=24` pilot preserves the DP-win pattern.

Do not skip a stage.  Do not stop early because a result is favourable.  Do not
rerun a stage because a result is unfavourable.

## Data firewall

Every target set created under this amendment must be newly generated and
source-disjoint from all existing PaySim target pools, including all RQ1
development/post-hoc/confirmatory pools, Priority 6-15 pools, and old
v1-medium targets.  Reuse of any existing target pool is forbidden.

## Accounting

Do not recompute the v1-medium epsilon unless needed for verification.  Cite
Priority 3 for this exact comparator:

- one release, add/remove, `delta=1e-5`: epsilon `5.054e6`;
- 50 releases/same client, add/remove, `delta=1e-5`: epsilon `2.521e8`.

The precise machine-readable values are in
`artifacts/dp_accounting/priority3_20260916_v2/dp_update_accounting.csv`.

## Reporting

For every stage actually run, report:

- target artifact path and SHA-256;
- provenance path, source-disjointness maximum overlap, and SHA-256;
- run artifact path and SHA-256;
- DNA wins, DP wins, exact ties, non-tied n;
- exact one-sided p-values for both DNA advantage and DP advantage;
- mean and median `D_i`;
- historical threshold decomposition at `0.0390625`.

No edits to `main.tex` or any file under `Latex/` are authorized here.
