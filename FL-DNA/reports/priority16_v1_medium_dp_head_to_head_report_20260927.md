# Priority 16 — v1-medium DNA-vs-DP head-to-head gap closure

Date: 2026-09-27  
Status: COMPLETE  
Scope: PaySim / Transform v1-medium / Priority-14-style DNA-vs-DP head-to-head comparison  

## Purpose

This run closes the configuration-mismatch gap identified during paper review:

- RQ1 head-to-head Table `tab:dpvsdna` previously used Transform v1-stronger.
- RQ2 utility non-inferiority previously used Transform v1-medium.
- This report tests whether the same v1-medium configuration used in RQ2 also loses head-to-head against its distortion-matched DP comparator under the Priority 14 methodology.

No files under `Latex/` were touched.

## Pre-run protocol amendment

Amendment:

- `protocols/amendments/2026-09-27_v1_medium_priority14_style_dp_head_to_head.md`
- SHA-256: `281ec1f05fce291264b9ed869fe92bd4d77e305920fd0eeaf71e98949169dabf`

Runner:

- `experiments/priority16_v1_medium_dna_vs_dp_probe.py`
- SHA-256: `1a12aecd274c6fc9b2fca86df58405cca202e7e6b62dcb96e6362a257789c152`

## Frozen configuration

Transform v1-medium:

- `block_size = 256`
- `mix_ratio = 0.10`
- `keep_ratio = 0.85`
- `shrink_factor = 0.40`
- `dna_run_seed = 681958327`

Distortion-matched DP comparator:

- `clip_norm = 100.0`
- `noise_multiplier = 0.000315`
- `noise_std = 0.0315`
- `mc_noise_samples = 100`
- `defense_seed = 314159265`

Comparator source:

- `protocols/config/rq1_medium_confirmatory.json`
- `reports/priority3_dp_accounting_report.md`
- `artifacts/dp_accounting/priority3_20260916_v2/dp_update_accounting.csv`

The comparator matches the existing RQ1 medium distortion-matched calibration: median relative-L2 mismatch `0.0037624108348589245` (< 5% tolerance).

## DP accounting for this exact comparator

Already covered by Priority 3, so no new accounting run was needed.

Convention: add/remove adjacency, `delta = 1e-5`, sensitivity ratio 1.

| Scenario | Releases | Epsilon | Optimal alpha |
|---|---:|---:|---:|
| Single release | 1 | 5,054,286.08156432 | 1.00151 |
| Same client, 50 releases | 50 | 252,060,366.41251898 | 1.00021 |

Interpretation note: this DP comparator is distortion-matched, but has extremely large epsilon and therefore should not be described as strong formal DP.

## Commands run

### n=8 development

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority16_v1_medium_dp_head_to_head/n8_targets_20260927 \
  --output-name paysim_priority16_v1_medium_n8_targets.pt \
  --groups 8 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092701 \
  --purpose 'Priority 16 v1-medium Priority14-style DNA-vs-DP n8 development probe'

PYTHONPATH=. .venv-phase1/bin/python experiments/priority16_v1_medium_dna_vs_dp_probe.py \
  --cell v1_medium_vs_dp_0p000315 \
  --target artifacts/priority16_v1_medium_dp_head_to_head/n8_targets_20260927/paysim_priority16_v1_medium_n8_targets.pt \
  --output artifacts/priority16_v1_medium_dp_head_to_head/n8_probe_20260927 \
  --seed 2026092702 --groups 8 --restarts 4 --workers 8 \
  --amendment protocols/amendments/2026-09-27_v1_medium_priority14_style_dp_head_to_head.md
```

### n=24 pilot

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority16_v1_medium_dp_head_to_head/n24_targets_20260927 \
  --output-name paysim_priority16_v1_medium_n24_targets.pt \
  --groups 24 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092703 \
  --purpose 'Priority 16 v1-medium Priority14-style DNA-vs-DP n24 pilot'

PYTHONPATH=. .venv-phase1/bin/python experiments/priority16_v1_medium_dna_vs_dp_probe.py \
  --cell v1_medium_vs_dp_0p000315 \
  --target artifacts/priority16_v1_medium_dp_head_to_head/n24_targets_20260927/paysim_priority16_v1_medium_n24_targets.pt \
  --output artifacts/priority16_v1_medium_dp_head_to_head/n24_pilot_20260927 \
  --seed 2026092704 --groups 24 --restarts 4 --workers 8 \
  --amendment protocols/amendments/2026-09-27_v1_medium_priority14_style_dp_head_to_head.md
```

### n=39 confirmatory

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_targets_20260927 \
  --output-name paysim_priority16_v1_medium_confirmatory_targets.pt \
  --groups 39 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092705 \
  --purpose 'Priority 16 v1-medium Priority14-style DNA-vs-DP confirmatory n39'

PYTHONPATH=. .venv-phase1/bin/python experiments/priority16_v1_medium_dna_vs_dp_probe.py \
  --cell v1_medium_vs_dp_0p000315 \
  --target artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_targets_20260927/paysim_priority16_v1_medium_confirmatory_targets.pt \
  --output artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_run_20260927 \
  --seed 2026092706 --groups 39 --restarts 4 --workers 8 \
  --amendment protocols/amendments/2026-09-27_v1_medium_priority14_style_dp_head_to_head.md
```

## Data firewall

All three target sets were generated fresh and source-disjoint from existing project target pools. The provenance files report `max_overlap_with_existing_targets = 0`.

| Stage | Target file | SHA-256 | Provenance SHA-256 |
|---|---|---|---|
| n=8 | `artifacts/priority16_v1_medium_dp_head_to_head/n8_targets_20260927/paysim_priority16_v1_medium_n8_targets.pt` | `151a004056727ea706dfb0ee194afb0f85d34d6deea870db3428a391fa3f67e5` | `4fcec75f75b05bfd5c826c28ea493fcaa5615fca57ec5fd5d75ccb40cbc2898a` |
| n=24 | `artifacts/priority16_v1_medium_dp_head_to_head/n24_targets_20260927/paysim_priority16_v1_medium_n24_targets.pt` | `1f4129871a703578a1f95e031b875801b6468322e6aa69edf0b3226bcbcdb33f` | `ecaa671f353f774335aacdfc3f58341114dbb484313bd323fc0308af52a2f02e` |
| n=39 | `artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_targets_20260927/paysim_priority16_v1_medium_confirmatory_targets.pt` | `645fe7ff62403e0bce81a3fc6cbed4d857306d36719cc2e3bad740acb73297fe` | `90f990bbed9f0fb8b47ac77e0718b89842d9445c703af4d8ca5cd8bc6d151566` |

Dataset checksum reported by all three target provenance files:

- `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`

## Results

Definition:

- `D = MSE(attacker, DNA-defended update) - MSE(attacker, DP-defended update)`.
- DP wins when `D < 0`, meaning the attacker reconstruction is closer to the DP-defended update than to the DNA-defended update.
- Exact p-values are one-sided exact sign tests against `p0 = 0.5`.

### Primary Priority-14-style strict comparison

This uses strict MSE ordering / zero threshold, matching the original Priority 14 head-to-head implementation.

| Stage | DNA wins | DP wins | Exact ties | Non-tied n | DP-advantage p-value | DNA-advantage p-value | Mean D | Median D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n=8 development | 0 | 8 | 0 | 8 | 0.00390625 | 1.0 | -0.001012266038510287 | -0.0010110999722448394 |
| n=24 pilot | 0 | 24 | 0 | 24 | 5.960464477539063e-08 | 1.0 | -0.0010028280051420102 | -0.0010036643399969044 |
| n=39 confirmatory | 0 | 39 | 0 | 39 | 1.8189894035458565e-12 | 1.0 | -0.0010026436558672064 | -0.0010047588902870148 |

Confirmatory result: DP wins 39/39 under the strict Priority-14-style head-to-head comparison.

### Historical tie-threshold sensitivity

The amendment also records the legacy v1-family `tie_threshold_mse = 0.0390625`. Because the observed per-pair differences are approximately `0.001`, all pairs fall inside this legacy tie band.

| Stage | DNA wins | DP wins | Ties within ±0.0390625 | Non-tied n | DP-advantage p-value |
|---|---:|---:|---:|---:|---:|
| n=8 development | 0 | 0 | 8 | 0 | N/A |
| n=24 pilot | 0 | 0 | 24 | 0 | N/A |
| n=39 confirmatory | 0 | 0 | 39 | 0 | N/A |

This sensitivity table is reported for transparency. It does not change the strict Priority-14-style result above.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `artifacts/priority16_v1_medium_dp_head_to_head/n8_probe_20260927/priority14_probe_report.json` | `1d0a44c57819cb27c3bd3687269fed9f8c2f03131e636e7af9c49c2c75cc6672` |
| `artifacts/priority16_v1_medium_dp_head_to_head/n24_pilot_20260927/priority14_probe_report.json` | `2e436a24d87d865b12c0604f23e24f75db986373677263cf0360d43bdabe22f3` |
| `artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_run_20260927/priority14_probe_report.json` | `0d4832e5f688cc4834528d0e07f6807d734b7c8c7903774a2d97409e3c1fa943` |

## Gate / conclusion for this gap

The full escalation chain was completed without technical failure:

1. n=8 development: DP wins 8/8.
2. n=24 pilot: DP wins 24/24.
3. n=39 confirmatory: DP wins 39/39.

Therefore, under the same strict head-to-head Priority 14 methodology used for v1-stronger, Transform v1-medium also loses decisively to its distortion-matched DP comparator.

This closes the specific evidentiary gap that no single v1 configuration had both:

- existing RQ2 utility evidence for v1-medium, and
- RQ1 head-to-head DP comparison evidence for the same v1-medium configuration.

## Notes for paper use

- The DP comparator is distortion-matched, not strong formal DP: epsilon is approximately `5.054e6` for one release and `2.521e8` for 50 same-client releases.
- If the paper uses the strict Priority-14-style Table II convention, the appropriate confirmatory result is `DP wins 39/39, p = 1.8189894035458565e-12`.
- If discussing the older `tie_threshold_mse = 0.0390625`, note explicitly that this threshold classifies all Priority 16 differences as ties because the observed `D` values are around `-0.001`.
