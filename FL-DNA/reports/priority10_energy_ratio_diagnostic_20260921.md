# Priority 10 final-layer energy-ratio diagnostic

**Date:** 2026-09-21  
**Type:** descriptive analysis only. No new target set, no attacker run, no
protocol amendment, and no confirmatory interpretation.

## Question

For real saved/reconstructible local updates, what fraction of total update
energy is in the final classifier layer of `models/fraud_mlp.py`?

The final classifier layer is the actual `nn.Linear(32, 1)` in the current
model:

- `network.12.weight`: 32 values;
- `network.12.bias`: 1 value;
- total final-layer values: 33.

Note: `network.9` in the current `FraudMLP` is `BatchNorm1d(32)`, not the final
classifier. This diagnostic uses the 33-value final linear layer requested by
the analysis question.

## Method

For each update sample:

```text
ratio = sum(update[network.12.weight, network.12.bias]^2)
        / sum(all floating update tensor values^2)
```

The denominator is the full floating update/state-delta vector present in the
artifact path used by the attack/evaluation pipeline, including floating BatchNorm
state tensors when those are present in the saved update dict.

The per-sample table with every artifact path is:

`results/priority10_energy_ratio/per_sample_energy_ratio.csv`

The machine-readable summary is:

`results/priority10_energy_ratio/summary.json`

## Data used

| Dataset | Samples | Source type | Artifact path(s) |
|---|---:|---|---|
| PaySim | 40 | stored observed update | `artifacts/rq1_v2/confirmatory_run_20260916/dna_v2/.../baseline.pt` files, one unique source group each; exact paths in CSV |
| PaySim | 24 | recomputed update from saved target artifact | `artifacts/priority6_sota_attackers/targets_paysim_n24/paysim_priority6_n24_targets.pt` |
| PaySim | 24 | recomputed update from saved target artifact | `artifacts/priority9_v1_family/stronger_n24_targets_20260917/paysim_priority9_stronger_n24_targets.pt` |
| IEEE-CIS | 8 | stored observed update | `artifacts/priority5_ieee_cis/phase53_rq1_development_gate_20260916_rerun1/development/group_{0..7}/restart_0/baseline.pt` |
| IEEE-CIS | 24 | recomputed update from saved bundle artifact | `artifacts/priority6_sota_attackers/targets_ieee_n24/ieee_priority6_bundle.pt` |
| IEEE-CIS | 8 | recomputed update from saved bundle artifact | `artifacts/priority10_v3/ieee_n8_targets_20260921/ieee_priority6_bundle.pt` |

The recomputed rows use only saved artifact bundles/targets and deterministic
local-update simulation; no new rows, targets, or attacker jobs were created.

## Results

| Dataset | n | Final-layer params | Total floating update values | Final-layer parameter fraction | Mean energy ratio | Median energy ratio | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PaySim | 88 | 33 | 13,057 | 0.252737995% | 0.000001496% | 0.0000003346% | 0.00000000001310% | 0.00004063% |
| IEEE-CIS | 40 | 33 | 34,049-34,561 | 0.095770508% mean | 0.00000002330% | 0.000000009776% | 0.0000000002592% | 0.00000008795% |

Same values as raw ratios:

| Dataset | n | Mean ratio | Median ratio | Min ratio | Max ratio |
|---|---:|---:|---:|---:|---:|
| PaySim | 88 | 1.4964216169e-08 | 3.3456083138e-09 | 1.3102520602e-13 | 4.0632254913e-07 |
| IEEE-CIS | 40 | 2.3295316038e-10 | 9.7764617970e-11 | 2.5922019471e-12 | 8.7954907653e-10 |

## Reproducibility

Command:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority10_energy_ratio_diagnostic.py
```

Script:

`experiments/priority10_energy_ratio_diagnostic.py`

SHA-256:

| File | SHA-256 |
|---|---|
| `experiments/priority10_energy_ratio_diagnostic.py` | `c2356356950de1a29db0c3083da6c3b156cc4cc3334260a47a21badb74760dbd` |
| `results/priority10_energy_ratio/per_sample_energy_ratio.csv` | `124012705e820003e35a6a8638cb73a2223a898a0877fd86e065a12df38360c3` |
| `results/priority10_energy_ratio/summary.json` | `dff8ace17bfb82e58bd4822c9bed02ffaa39c76bb380ee9d668d97a10d5cd03d` |

## Checks

- `py_compile`: PASS for `experiments/priority10_energy_ratio_diagnostic.py`.
- No target set was created.
- No attacker was run.
- No original artifact was overwritten.
- `torch.set_num_threads(1)` is set in the diagnostic script.
