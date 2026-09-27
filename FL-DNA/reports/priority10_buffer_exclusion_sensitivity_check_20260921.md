# Priority 10 buffer-exclusion sensitivity check

**Date:** 2026-09-21  
**Scope:** descriptive sensitivity analysis on frozen Priority 6 confirmatory
artifacts only. No new target set, no new attacker run, no protocol change, and no
reinterpretation of frozen RQ1 conclusions were performed.

## Question

For the two main confirmed Priority 6 cells:

1. PaySim / `v1_medium` / `GEN_IDLG_STYLE` / n=39
2. IEEE-CIS / `v2_ratio0p95_eta0p01` / `GEN_COSINE_TV` / n=39

what happens if the win/loss sign-test is recomputed after excluding BatchNorm
buffers (`running_mean`, `running_var`, `num_batches_tracked`) and keeping only
trainable parameter keys?

## Important metric note

The frozen Priority 6 gate report selected restarts by attacker objective and then
reported sign tests on reconstruction feature-MSE. Feature-MSE itself has no
BatchNorm keys to remove. Therefore this sensitivity check keeps the same selected
group/restart structure from the frozen report, but recomputes an update-space MSE
on the defended update sub-vector containing only trainable parameter keys.

The new MSE magnitudes are update-space MSE values and are not directly comparable
in scale to the original feature-MSE values. The sign-test structure is the same:
negative `baseline_minus_control` means the baseline attack reconstruction is closer
than the control.

## Data sources

| Cell | Target artifact | Run artifact |
|---|---|---|
| PaySim `v1_medium` / `GEN_IDLG_STYLE` | `artifacts/priority6_sota_attackers/confirmatory_paysim_v1_idlg_targets_20260917/paysim_priority6_v1_idlg_confirmatory_targets.pt` | `artifacts/priority6_sota_attackers/confirmatory_paysim_v1_idlg_run_20260917` |
| IEEE-CIS `v2_ratio0p95_eta0p01` / `GEN_COSINE_TV` | `artifacts/priority6_sota_attackers/confirmatory_ieee_v2_cosine_targets_20260917/ieee_priority6_bundle.pt` | `artifacts/priority6_sota_attackers/confirmatory_ieee_v2_cosine_run_20260917` |

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
| PaySim `v1_medium` / `GEN_IDLG_STYLE` | frozen report feature-MSE | Prior | 29 / 39 | -1636.697748 | -268.806299 | 0.001688923956 | PASS |
| PaySim `v1_medium` / `GEN_IDLG_STYLE` | frozen report feature-MSE | Zero | 32 / 39 | -1820.119876 | -506.423711 | 0.000035127392 | PASS |
| PaySim `v1_medium` / `GEN_IDLG_STYLE` | trainable-update MSE, BN buffers excluded | Prior | 37 / 39 | -2.006682e-07 | -1.766924e-07 | 1.420630724e-09 | PASS |
| PaySim `v1_medium` / `GEN_IDLG_STYLE` | trainable-update MSE, BN buffers excluded | Zero | 0 / 39 | 4.511585e-07 | 4.515640e-07 | 1.000000000 | FAIL |
| IEEE-CIS `v2_ratio0p95_eta0p01` / `GEN_COSINE_TV` | frozen report feature-MSE | Prior | 36 / 39 | -18.220055 | -1.866349 | 1.804437488e-08 | PASS |
| IEEE-CIS `v2_ratio0p95_eta0p01` / `GEN_COSINE_TV` | frozen report feature-MSE | Zero | 36 / 39 | -18.155714 | -2.010708 | 1.804437488e-08 | PASS |
| IEEE-CIS `v2_ratio0p95_eta0p01` / `GEN_COSINE_TV` | trainable-update MSE, BN buffers excluded | Prior | 31 / 39 | -3.035272e-08 | -3.012013e-08 | 0.000147038438 | PASS |
| IEEE-CIS `v2_ratio0p95_eta0p01` / `GEN_COSINE_TV` | trainable-update MSE, BN buffers excluded | Zero | 0 / 39 | 3.016186e-07 | 3.096211e-07 | 1.000000000 | FAIL |

## Overall gate comparison

| Cell | Frozen report overall gate | Buffer-excluded trainable-update overall gate |
|---|---|---|
| PaySim `v1_medium` / `GEN_IDLG_STYLE` | PASS | FAIL |
| IEEE-CIS `v2_ratio0p95_eta0p01` / `GEN_COSINE_TV` | PASS | FAIL |

## Procedure used

For each group, I used the same selected `baseline_restart` and `zero_restart`
recorded in the frozen `priority6_gate_report.json`.

For PaySim:

- Re-loaded the frozen target group.
- Re-created the observed update with the same local seed.
- Recomputed defended update vectors for:
  - selected baseline reconstruction,
  - selected zero-update reconstruction,
  - four prior initializations.
- Averaged the four prior trainable-update MSE values, matching the frozen report's
  prior-aggregation structure.

For IEEE-CIS:

- Re-loaded the frozen IEEE bundle.
- Re-created the observed update with the same local seed and local learning rate.
- Recomputed defended sketch vectors for:
  - selected baseline reconstruction and attacker labels,
  - selected zero-update reconstruction and attacker labels,
  - four prior initializations decoded from the same per-restart initialization
    seed rule.
- Averaged the four prior trainable-update MSE values, matching the frozen report's
  prior-aggregation structure.

In both cells, the recomputed MSE was:

```text
mean((candidate_defended[key] - control_or_target_signal[key])^2)
```

after concatenating only retained trainable parameter keys.

## Check status

- No target set was created.
- No attacker optimization was run.
- No frozen artifact was modified.
- No RQ1 conclusion was changed in this report.

