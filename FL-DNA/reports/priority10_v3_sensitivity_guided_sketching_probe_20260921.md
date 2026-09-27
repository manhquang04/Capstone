# Priority 10 v3 sensitivity-guided sketching probe

**Date:** 2026-09-21  
**Scope:** IEEE-CIS n=8 development-only probe.  
**Status:** STOPPED AFTER THE AUTHORIZED n=8 GATE. No n=24 escalation and no
confirmatory run were performed.

## Purpose

This probe tested a new candidate mechanism, DNA Transform v3, without changing
any frozen RQ1/RQ2/RQ3 conclusion.

The design question was:

> Can v3, a non-uniform sketch that compresses the pre-registered sensitive
> final classifier layer more strongly than the rest of the update, reach
> resistance comparable to uniform v2 ratio 0.90 while strongly compressing
> only 33 parameters?

## Pre-run amendment

The following amendment was written before target generation or execution:

`protocols/amendments/2026-09-21_priority10_v3_sensitivity_guided_sketching_probe.md`

It authorized exactly one n=8 IEEE-CIS development gate and explicitly did not
authorize n=24 escalation or confirmatory execution.

## Mechanism tested

Defense label:

`v3_sensitive_final_layer_ratio0p50_rest0p95_eta0p01`

Transform v3 is a wrapper around the frozen v2 SRHT plus quantization primitive:

- sensitive block: final `FraudMLP` classifier layer, `network.9.weight` plus
  `network.9.bias`;
- sensitive block size: `32 + 1 = 33` parameters;
- sensitive block v2 config: `compression_ratio=0.50`, `quantization_eta=0.01`;
- rest block v2 config: `compression_ratio=0.95`, `quantization_eta=0.01`;
- seed derivation: existing deterministic v2-style per block;
- server lift: existing v2 lift per block, then unpack back into original tensor
  positions.

The sensitive block was defined before execution from the iDLG motivation that
the final classifier layer gradient carries label information. No data-driven
layer selection was used.

## Implementation

New/changed files:

- `dna_encoder/transform_defense_v3.py`: new blockwise wrapper around v2.
- `tests/test_transform_defense_v3.py`: mechanical tests for the 33-parameter
  split, determinism, block-specific ratios, and reconstruction shapes.
- `experiments/run_priority6_sota_style_attackers.py`: added the v3 defense
  label and extended IEEE-CIS target-firewall discovery to include all existing
  `ieee_priority6_bundle.pt` artifacts, including Priority 8.

The frozen v2 implementation file `dna_encoder/transform_defense_v2.py` was not
modified.

## Target generation

Command:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority6_sota_style_attackers.py \
  prepare-ieee \
  --output artifacts/priority10_v3/ieee_n8_targets_20260921 \
  --seed 2026092101 \
  --groups 8 \
  --max-rows 50000 \
  --amendment protocols/amendments/2026-09-21_priority10_v3_sensitivity_guided_sketching_probe.md
```

Data firewall:

- historical IEEE target files checked: `6`;
- maximum overlap: `0`;
- warmup/target overlap: `0`;
- disjointness gate: `PASS`.

Historical target files checked:

- `artifacts/priority5_ieee_cis/phase53_rq1_development_gate_20260916/development_targets.pt`;
- `artifacts/priority5_ieee_cis/phase53_rq1_development_gate_20260916_rerun1/development_targets.pt`;
- `artifacts/priority6_sota_attackers/confirmatory_ieee_v2_cosine_targets_20260917/ieee_priority6_bundle.pt`;
- `artifacts/priority6_sota_attackers/targets_ieee/ieee_priority6_bundle.pt`;
- `artifacts/priority6_sota_attackers/targets_ieee_n24/ieee_priority6_bundle.pt`;
- `artifacts/priority8_v2_ratio0p90/ieee_n8_targets_20260917/ieee_priority6_bundle.pt`.

Target bundle SHA-256:

`7ae97c24377a1b5fe8a1b3cc6e7eb9f38a2456ccd014746e0522afec1d8c4f02`

## Gate command

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority6_sota_style_attackers.py \
  execute \
  --dataset ieee \
  --bundle artifacts/priority10_v3/ieee_n8_targets_20260921/ieee_priority6_bundle.pt \
  --output artifacts/priority10_v3/ieee_n8_gate_20260921 \
  --seed 2026092102 \
  --workers 8 \
  --groups 8 \
  --defenses v3_sensitive_final_layer_ratio0p50_rest0p95_eta0p01 \
  --generations GEN_COSINE_TV \
  --amendment protocols/amendments/2026-09-21_priority10_v3_sensitivity_guided_sketching_probe.md
```

Completed jobs: `32/32`.

## Results

| Control | Wins/non-ties | Mean Delta MSE | Median Delta MSE | One-sided exact sign-test p | Gate |
|---|---:|---:|---:|---:|---|
| Prior | 8/8 | -11.9146802744 | -5.2493190136 | 0.00390625 | PASS |
| Zero-update | 8/8 | -11.9648454902 | -5.3769181514 | 0.00390625 | PASS |

Overall development gate: **PASS**.

Selected per-group deltas:

| Group | Baseline MSE | Prior MSE | Zero MSE | Baseline-Prior | Baseline-Zero |
|---:|---:|---:|---:|---:|---:|
| 0 | 167.1099454220 | 179.3664323987 | 178.8464096904 | -12.2564869767 | -11.7364642684 |
| 1 | 168.2263197196 | 177.8055711888 | 177.8504873961 | -9.5792514692 | -9.6241676765 |
| 2 | 2208.3593979662 | 2267.8123134598 | 2268.7568837027 | -59.4529154936 | -60.3974857365 |
| 3 | 2.8397367807 | 3.8070075379 | 3.5744883605 | -0.9672707571 | -0.7347515797 |
| 4 | 32.1485313119 | 40.3451881170 | 40.5636609697 | -8.1966568051 | -8.4151296578 |
| 5 | 5.5396759975 | 7.3968915134 | 7.3603670193 | -1.8572155159 | -1.8206910217 |
| 6 | 1.5377618305 | 2.2434257857 | 2.1891291667 | -0.7056639551 | -0.6513673361 |
| 7 | 8.2484234994 | 10.5504047215 | 10.5871301443 | -2.3019812221 | -2.3387066450 |

## Interpretation

The authorized n=8 development gate shows that `GEN_COSINE_TV` breaks this v3
candidate under the pre-registered gate: it beats both Prior and Zero-update in
all 8 non-tied groups.

Per the amendment, this stops the v3 probe. The result supports the second
pre-specified interpretation: compressing only the final classifier layer more
strongly is not sufficient against `GEN_COSINE_TV` in this IEEE-CIS scope. A
plausible reason is that this attacker optimizes global cosine agreement across
the defended update representation, rather than relying only on the final-layer
label-recovery channel emphasized by iDLG.

This is still development-only evidence. It does not alter any frozen RQ1/RQ2/RQ3
conclusion, and no v3 utility or confirmatory claim is made.

## Artifact hashes

| File | SHA-256 |
|---|---|
| `protocols/amendments/2026-09-21_priority10_v3_sensitivity_guided_sketching_probe.md` | `94c2124846e39ddb81e4af33af6dcb15a32ece81c8e098d644002165a3ea7d61` |
| `dna_encoder/transform_defense_v3.py` | `e04bac525b0f16d80810f929452050e078130daacd2d893c992a1ebf0ac7d1c7` |
| `experiments/run_priority6_sota_style_attackers.py` | `0780fa2d4aa9a8c56a49b58910b55ab8de22e32d553ee8ffc8d83efa3622e9f0` |
| `ieee_n8_targets_20260921/ieee_priority6_bundle.pt` | `7ae97c24377a1b5fe8a1b3cc6e7eb9f38a2456ccd014746e0522afec1d8c4f02` |
| `ieee_n8_targets_20260921/ieee_target_firewall.json` | `b65c97f4c0c1b4296c445460673be00196111250dc1f28db5f9f37e674086955` |
| `ieee_n8_gate_20260921/priority6_gate_report.json` | `596bc82727ea31216bd182cc5d040d7f433e8bb79545b66a3f5d5a90a6d83a3d` |

## Checks

- `torch.set_num_threads(1)`: preserved in worker and manifest.
- Source-disjoint target set: PASS.
- Development n=8 gate: completed.
- n=24 escalation: not run.
- Confirmatory: not run.
- No post-hoc target set was used except for source-disjointness checks.
