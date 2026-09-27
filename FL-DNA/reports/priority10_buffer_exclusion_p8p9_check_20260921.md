# Priority 10 buffer-exclusion sensitivity check for Priority 8/9

**Date:** 2026-09-21  
**Scope:** descriptive replay from frozen artifacts only. No new target set, no new
attacker optimization, no protocol change, and no change to frozen RQ1 conclusions
were performed.

## Question

Apply the same BatchNorm-buffer-exclusion sensitivity check used for the two
Priority 6 confirmatory cells to:

1. Priority 9: PaySim / `v1_stronger` / `GEN_IDLG_STYLE` confirmatory, n=39.
2. Priority 8: IEEE-CIS / `v2_ratio0p9_eta0p01` / `GEN_COSINE_TV` development
   gate, n=8.

The recomputation excludes BatchNorm buffers (`running_mean`, `running_var`,
`num_batches_tracked`) and keeps only trainable parameter keys.

## Important metric note

The frozen gate reports selected restarts by attacker objective and then evaluated
feature-MSE. Feature-MSE has no BatchNorm keys to remove. Therefore this check keeps
the frozen report's selected group/restart structure and recomputes an update-space
MSE on the defended update sub-vector containing trainable parameters only.

The new MSE scale is update-space MSE and is not directly comparable to feature-MSE.
The win/loss sign-test logic is unchanged: negative `baseline_minus_control` means
baseline is closer than the control.

## Data sources

| Cell | Target artifact | Run artifact |
|---|---|---|
| Priority 9 PaySim `v1_stronger` / `GEN_IDLG_STYLE` | `artifacts/priority9_v1_family/stronger_confirmatory_targets_20260917/paysim_priority9_stronger_confirmatory_targets.pt` | `artifacts/priority9_v1_family/stronger_confirmatory_gate_20260917` |
| Priority 8 IEEE-CIS `v2_ratio0p9_eta0p01` / `GEN_COSINE_TV` | `artifacts/priority8_v2_ratio0p90/ieee_n8_targets_20260917/ieee_priority6_bundle.pt` | `artifacts/priority8_v2_ratio0p90/ieee_n8_gate_20260917` |

The full replay output was saved as:

```text
artifacts/priority10_diagnostics/buffer_exclusion_p8p9_20260921.json
```

## Keys retained after buffer exclusion

The recomputation retained 14 trainable parameter keys:

```text
network.0.weight
network.0.bias
network.1.weight
network.1.bias
network.4.weight
network.4.bias
network.5.weight
network.5.bias
network.8.weight
network.8.bias
network.9.weight
network.9.bias
network.12.weight
network.12.bias
```

Excluded keys include BatchNorm buffers such as `*.running_mean`,
`*.running_var`, and `*.num_batches_tracked`.

## Summary table

| Cell | Metric space | Control | Wins / non-ties | Mean diff | Median diff | One-sided exact sign p | Gate pass |
|---|---|---:|---:|---:|---:|---:|---|
| Priority 9 PaySim `v1_stronger` / `GEN_IDLG_STYLE` | frozen report feature-MSE | Prior | 27 / 39 | -2582.348835 | -112.925603 | 0.011851351235 | PASS |
| Priority 9 PaySim `v1_stronger` / `GEN_IDLG_STYLE` | frozen report feature-MSE | Zero | 29 / 39 | -2504.597900 | -120.828715 | 0.001688923956 | PASS |
| Priority 9 PaySim `v1_stronger` / `GEN_IDLG_STYLE` | trainable-update MSE, BN buffers excluded | Prior | 38 / 39 | -2.000936e-07 | -1.763625e-07 | 7.275957614e-11 | PASS |
| Priority 9 PaySim `v1_stronger` / `GEN_IDLG_STYLE` | trainable-update MSE, BN buffers excluded | Zero | 0 / 39 | 4.257245e-07 | 4.359765e-07 | 1.000000000 | FAIL |
| Priority 8 IEEE-CIS `v2_ratio0p9_eta0p01` / `GEN_COSINE_TV` | frozen report feature-MSE | Prior | 6 / 8 | -10.786748 | -1.123243 | 0.144531250 | FAIL |
| Priority 8 IEEE-CIS `v2_ratio0p9_eta0p01` / `GEN_COSINE_TV` | frozen report feature-MSE | Zero | 6 / 8 | -10.982827 | -1.064386 | 0.144531250 | FAIL |
| Priority 8 IEEE-CIS `v2_ratio0p9_eta0p01` / `GEN_COSINE_TV` | trainable-update MSE, BN buffers excluded | Prior | 6 / 8 | -1.726127e-08 | -1.065677e-08 | 0.144531250 | FAIL |
| Priority 8 IEEE-CIS `v2_ratio0p9_eta0p01` / `GEN_COSINE_TV` | trainable-update MSE, BN buffers excluded | Zero | 0 / 8 | 3.292587e-07 | 3.277836e-07 | 1.000000000 | FAIL |

## Overall gate comparison

| Cell | Frozen report overall gate | Buffer-excluded trainable-update overall gate |
|---|---|---|
| Priority 9 PaySim `v1_stronger` / `GEN_IDLG_STYLE` | PASS | FAIL |
| Priority 8 IEEE-CIS `v2_ratio0p9_eta0p01` / `GEN_COSINE_TV` | FAIL | FAIL |

## Procedure used

For each group, I used the same selected `baseline_restart` and `zero_restart`
recorded in the frozen `priority6_gate_report.json`.

For PaySim `v1_stronger`:

- Re-loaded the frozen target group.
- Re-created the observed update with the same local seed.
- Recomputed defended update vectors for:
  - selected baseline reconstruction,
  - selected zero-update reconstruction,
  - four prior initializations.
- Averaged the four prior trainable-update MSE values, matching the frozen report's
  prior aggregation structure.

For IEEE-CIS `v2_ratio0p9_eta0p01`:

- Re-loaded the frozen IEEE bundle.
- Re-created the observed update with the same local seed and local learning rate.
- Recomputed defended sketch vectors for:
  - selected baseline reconstruction and attacker labels,
  - selected zero-update reconstruction and attacker labels,
  - four prior initializations decoded from the same per-restart initialization
    seed rule.
- Averaged the four prior trainable-update MSE values, matching the frozen report's
  prior aggregation structure.

## Check status

- No target set was created.
- No attacker optimization was run.
- No frozen artifact was modified.
- No RQ1 conclusion was changed in this report.

