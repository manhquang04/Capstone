# Priority 10 energy-ratio verification

**Date:** 2026-09-21  
**Type:** descriptive verification only. No amendment, no new target set, no
attacker run, and no hypothesis test.

## Purpose

The previous per-layer Linear-only table summed to far below 100%. This report
verifies the full key-level energy accounting for the same sample set:

- PaySim: `n=88`;
- IEEE-CIS: `n=40`.

## What the analyzed update is

The update vector analyzed in these artifacts is not a raw one-step
parameter-gradient vector. It is the output of `attacks/local_update.py::simulate`:

```text
full model state delta after local Adam replay
```

It includes:

- trainable parameter deltas for all Linear and BatchNorm affine parameters;
- floating BatchNorm running-stat buffer deltas:
  `running_mean` and `running_var`;
- excludes integer BatchNorm counters from energy sums:
  `num_batches_tracked`.

The integer keys excluded from energy accounting were:

- `network.1.num_batches_tracked`;
- `network.5.num_batches_tracked`;
- `network.9.num_batches_tracked`.

## 100% accounting check

| Dataset | Samples | Min total % | Max total % | Max absolute error from 1.0 |
|---|---:|---:|---:|---:|
| PaySim | 88 | 99.99999999999997 | 100.00000000000003 | 2.220446049250313e-16 |
| IEEE-CIS | 40 | 99.99999999999997 | 100.00000000000003 | 3.3306690738754696e-16 |

So the energy accounting is complete for all floating update keys. The small
error is floating-point summation error.

## PaySim full key-level energy

| Key | Kind | Numel | Mean % | Median % | Min % | Max % |
|---|---|---:|---:|---:|---:|---:|
| `network.1.running_var` | BatchNorm buffer | 128 | 99.745532206 | 99.853321940 | 95.630346453 | 99.998816555 |
| `network.1.running_mean` | BatchNorm buffer | 128 | 0.245489277 | 0.140660452 | 0.001183330 | 4.177133887 |
| `network.5.running_var` | BatchNorm buffer | 64 | 0.005014525 | 0.001855811 | 0.000000074 | 0.067776769 |
| `network.9.running_var` | BatchNorm buffer | 32 | 0.001787592 | 0.000271298 | 0.000000016 | 0.065104623 |
| `network.5.running_mean` | BatchNorm buffer | 64 | 0.001341622 | 0.000323789 | 0.000000019 | 0.033889086 |
| `network.9.running_mean` | BatchNorm buffer | 32 | 0.000371197 | 0.000068539 | 0.000000002 | 0.013389108 |
| `network.4.weight` | parameter | 8192 | 0.000300734 | 0.000068653 | 0.000000003 | 0.007900428 |
| `network.8.weight` | parameter | 2048 | 0.000086310 | 0.000019717 | 0.000000001 | 0.002352855 |
| `network.0.weight` | parameter | 1664 | 0.000056494 | 0.000011874 | 0.000000000 | 0.001575076 |
| `network.1.weight` | parameter | 128 | 0.000005036 | 0.000001126 | 0.000000000 | 0.000131740 |
| `network.1.bias` | parameter | 128 | 0.000005036 | 0.000001126 | 0.000000000 | 0.000131738 |
| `network.5.weight` | parameter | 64 | 0.000002748 | 0.000000618 | 0.000000000 | 0.000073877 |
| `network.5.bias` | parameter | 64 | 0.000002748 | 0.000000618 | 0.000000000 | 0.000073875 |
| `network.12.weight` | parameter | 32 | 0.000001450 | 0.000000324 | 0.000000000 | 0.000039401 |
| `network.9.bias` | parameter | 32 | 0.000001450 | 0.000000324 | 0.000000000 | 0.000039384 |
| `network.9.weight` | parameter | 32 | 0.000001449 | 0.000000324 | 0.000000000 | 0.000039389 |
| `network.12.bias` | parameter | 1 | 0.000000046 | 0.000000010 | 0.000000000 | 0.000001231 |
| `network.4.bias` | parameter | 64 | 0.000000043 | 0.000000011 | 0.000000000 | 0.000000745 |
| `network.8.bias` | parameter | 32 | 0.000000022 | 0.000000004 | 0.000000000 | 0.000000308 |
| `network.0.bias` | parameter | 128 | 0.000000015 | 0.000000001 | 0.000000000 | 0.000000209 |

## IEEE-CIS full key-level energy

| Key | Kind | Numel | Mean % | Median % | Min % | Max % |
|---|---|---:|---:|---:|---:|---:|
| `network.1.running_var` | BatchNorm buffer | 128 | 99.989532621 | 99.994959521 | 99.924880004 | 99.999199420 |
| `network.1.running_mean` | BatchNorm buffer | 128 | 0.010369065 | 0.004942648 | 0.000764128 | 0.074903030 |
| `network.5.running_var` | BatchNorm buffer | 64 | 0.000038057 | 0.000019684 | 0.000000469 | 0.000164245 |
| `network.5.running_mean` | BatchNorm buffer | 64 | 0.000021797 | 0.000009576 | 0.000000239 | 0.000077090 |
| `network.9.running_var` | BatchNorm buffer | 32 | 0.000018841 | 0.000007766 | 0.000000215 | 0.000094849 |
| `network.9.running_mean` | BatchNorm buffer | 32 | 0.000009396 | 0.000005454 | 0.000000133 | 0.000036184 |
| `network.4.weight` | parameter | 8192 | 0.000004754 | 0.000002003 | 0.000000052 | 0.000018443 |
| `network.0.weight` | parameter | 22656 | 0.000003775 | 0.000001778 | 0.000000037 | 0.000015116 |
| `network.8.weight` | parameter | 2048 | 0.000001348 | 0.000000561 | 0.000000015 | 0.000005197 |
| `network.1.weight` | parameter | 128 | 0.000000080 | 0.000000034 | 0.000000001 | 0.000000302 |
| `network.1.bias` | parameter | 128 | 0.000000080 | 0.000000034 | 0.000000001 | 0.000000302 |
| `network.5.weight` | parameter | 64 | 0.000000043 | 0.000000019 | 0.000000000 | 0.000000168 |
| `network.5.bias` | parameter | 64 | 0.000000043 | 0.000000019 | 0.000000000 | 0.000000168 |
| `network.9.weight` | parameter | 32 | 0.000000023 | 0.000000009 | 0.000000000 | 0.000000085 |
| `network.12.weight` | parameter | 32 | 0.000000023 | 0.000000009 | 0.000000000 | 0.000000085 |
| `network.9.bias` | parameter | 32 | 0.000000023 | 0.000000009 | 0.000000000 | 0.000000085 |
| `network.4.bias` | parameter | 64 | 0.000000015 | 0.000000007 | 0.000000000 | 0.000000061 |
| `network.0.bias` | parameter | 128 | 0.000000010 | 0.000000007 | 0.000000000 | 0.000000051 |
| `network.8.bias` | parameter | 32 | 0.000000006 | 0.000000003 | 0.000000000 | 0.000000019 |
| `network.12.bias` | parameter | 1 | 0.000000001 | 0.000000000 | 0.000000000 | 0.000000003 |

## Files

Full per-sample/key table:

`results/priority10_energy_verification/per_sample_key_energy.csv`

Per-sample 100% sum checks:

`results/priority10_energy_verification/sample_energy_sum_checks.csv`

Per-key summary:

`results/priority10_energy_verification/key_energy_summary.csv`

Machine-readable summary:

`results/priority10_energy_verification/summary.json`

## SHA-256

| File | SHA-256 |
|---|---|
| `experiments/priority10_energy_ratio_verification.py` | `906df937cac00e944839f0f4f0d37f7e56263aaf3fb27915811d43d3d841bbd0` |
| `results/priority10_energy_verification/per_sample_key_energy.csv` | `50798809c11232998fe3e54e2b3ab379fee9cabcaee748f0c54f5c996fe867f1` |
| `results/priority10_energy_verification/key_energy_summary.csv` | `6b35b5034f74d3cb835339223bcdbc68d9cd73b754c1e9231029da93c7d5b1f7` |
| `results/priority10_energy_verification/sample_energy_sum_checks.csv` | `9180391d651497605df356855b0c80df5c8980b5f58f046c05f9020b706e1e28` |
| `results/priority10_energy_verification/summary.json` | `f841b20bc638f5338b97391fa4515db00f678c2c761b06565685981b089f321e` |

## Checks

- `py_compile`: PASS for `experiments/priority10_energy_ratio_verification.py`.
- No target set was created.
- No attacker was run.
- No original artifact was overwritten.
- `torch.set_num_threads(1)` is set in the verification script.
