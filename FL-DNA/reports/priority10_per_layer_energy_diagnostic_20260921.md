# Priority 10 per-layer energy diagnostic

**Date:** 2026-09-21  
**Type:** descriptive analysis only. No amendment, no new target set, no
attacker run, and no hypothesis test.

## Question

How is squared-update energy distributed across the four linear layers of
`models/fraud_mlp.py`?

Layer keys used:

- `network.0.weight` + `network.0.bias`: input -> 128;
- `network.4.weight` + `network.4.bias`: 128 -> 64;
- `network.8.weight` + `network.8.bias`: 64 -> 32;
- `network.12.weight` + `network.12.bias`: 32 -> 1.

The denominator is the full floating update/state-delta vector present in the
artifact update dict, including floating BatchNorm state tensors when present.

## Data

Same artifact sample set as `priority10_energy_ratio_diagnostic_20260921.md`:

- PaySim: `n=88`;
- IEEE-CIS: `n=40`.

Per-sample output:

`results/priority10_per_layer_energy/per_sample_layer_energy.csv`

Machine-readable summary:

`results/priority10_per_layer_energy/summary.json`

## PaySim

| Layer | Params | Mean energy % | Median energy % |
|---|---:|---:|---:|
| input -> 128 (`network.0`) | 1,792 | 0.0000565093% | 0.0000118746% |
| 128 -> 64 (`network.4`) | 8,256 | 0.0003007767% | 0.0000686717% |
| 64 -> 32 (`network.8`) | 2,080 | 0.0000863323% | 0.0000197228% |
| 32 -> 1 (`network.12`) | 33 | 0.0000014964% | 0.0000003346% |

Raw ratios:

| Layer | Mean ratio | Median ratio |
|---|---:|---:|
| input -> 128 (`network.0`) | 5.6509278726e-07 | 1.1874617530e-07 |
| 128 -> 64 (`network.4`) | 3.0077671378e-06 | 6.8671699985e-07 |
| 64 -> 32 (`network.8`) | 8.6332347347e-07 | 1.9722757196e-07 |
| 32 -> 1 (`network.12`) | 1.4964216169e-08 | 3.3456083138e-09 |

## IEEE-CIS

| Layer | Params | Mean energy % | Median energy % |
|---|---:|---:|---:|
| input -> 128 (`network.0`) | 22,784 | 0.0000037851% | 0.0000017801% |
| 128 -> 64 (`network.4`) | 8,256 | 0.0000047697% | 0.0000020092% |
| 64 -> 32 (`network.8`) | 2,080 | 0.0000013540% | 0.0000005633% |
| 32 -> 1 (`network.12`) | 33 | 0.0000000233% | 0.0000000098% |

Raw ratios:

| Layer | Mean ratio | Median ratio |
|---|---:|---:|
| input -> 128 (`network.0`) | 3.7850972782e-08 | 1.7801339278e-08 |
| 128 -> 64 (`network.4`) | 4.7696725729e-08 | 2.0092489991e-08 |
| 64 -> 32 (`network.8`) | 1.3540495005e-08 | 5.6326997914e-09 |
| 32 -> 1 (`network.12`) | 2.3295316038e-10 | 9.7764617970e-11 |

## Reproducibility

Command:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority10_per_layer_energy_diagnostic.py
```

SHA-256:

| File | SHA-256 |
|---|---|
| `experiments/priority10_per_layer_energy_diagnostic.py` | `19994ca22506d4b2883da2cbb71f02dd1cebd70155f646d3bb97f900fda48e42` |
| `results/priority10_per_layer_energy/per_sample_layer_energy.csv` | `70cfe413124744c14b8f53ea8eacf1a5c8cb96e8a551e1edfa1a58e2b4ccf59d` |
| `results/priority10_per_layer_energy/summary.json` | `6c8e69aad37c160c95dc3773108391c47b20375a117b9c1cebf7f97f11cd4d0b` |

## Checks

- `py_compile`: PASS for `experiments/priority10_per_layer_energy_diagnostic.py`.
- No target set was created.
- No attacker was run.
- No original artifact was overwritten.
- `torch.set_num_threads(1)` is set in the diagnostic script.
