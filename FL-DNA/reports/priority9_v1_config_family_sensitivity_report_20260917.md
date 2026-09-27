# Priority 9 Report: DNA Transform v1 Config-Family Sensitivity on PaySim

**Date:** 2026-09-17  
**Dataset:** PaySim (`datasets/creditcard.csv`)  
**Attacker:** `GEN_IDLG_STYLE`  
**Scope:** 4 records/group, 1 fraud record/group, 1 client step  
**Status:** Complete for PaySim v1 conservative/medium/stronger family.

## Executive result

`GEN_IDLG_STYLE` is not limited to the previously confirmed v1-medium setting:

- `v1_conservative` failed the initial n=8 development gate and was not
  escalated.
- `v1_medium` retains the existing Priority 6 confirmatory PASS.
- `v1_stronger` passed n=8, passed n=24, and then passed the one-shot n=39
  confirmatory run.

This means the confirmed v1 vulnerability covers medium and stronger, but the
lighter conservative setting did not meet the precommitted gate in this run.
The conservative failure is not evidence of safety; it only marks the boundary
of this attacker/config test.

## Protocol and amendments

- Development amendment:
  `protocols/amendments/2026-09-17_priority9_v1_config_family_sensitivity.md`
- Confirmatory escalation amendment for stronger:
  `protocols/amendments/2026-09-17_priority9_v1_stronger_confirmatory.md`

The stronger confirmatory amendment was written after the precommitted n=24
pilot passed and before the n=39 confirmatory target set was generated. It kept
`p1=0.70`, required non-tied n=37, draw n=39, and authorized exactly one
confirmatory run.

## Commands run

Conservative n=8 target:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority9_v1_family/conservative_n8_targets_20260917 \
  --output-name paysim_priority9_conservative_n8_targets.pt \
  --groups 8 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026091811 \
  --purpose 'Priority 9 PaySim v1_conservative GEN_IDLG_STYLE n8 development gate'
```

Conservative n=8 gate:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority6_sota_style_attackers.py execute \
  --dataset paysim \
  --target artifacts/priority9_v1_family/conservative_n8_targets_20260917/paysim_priority9_conservative_n8_targets.pt \
  --output artifacts/priority9_v1_family/conservative_n8_gate_20260917 \
  --seed 2026091812 --workers 8 --groups 8 \
  --defenses v1_conservative --generations GEN_IDLG_STYLE \
  --amendment protocols/amendments/2026-09-17_priority9_v1_config_family_sensitivity.md
```

Stronger n=8 target/gate:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority9_v1_family/stronger_n8_targets_20260917 \
  --output-name paysim_priority9_stronger_n8_targets.pt \
  --groups 8 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026091821 \
  --purpose 'Priority 9 PaySim v1_stronger GEN_IDLG_STYLE n8 development gate'

PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority6_sota_style_attackers.py execute \
  --dataset paysim \
  --target artifacts/priority9_v1_family/stronger_n8_targets_20260917/paysim_priority9_stronger_n8_targets.pt \
  --output artifacts/priority9_v1_family/stronger_n8_gate_20260917 \
  --seed 2026091822 --workers 8 --groups 8 \
  --defenses v1_stronger --generations GEN_IDLG_STYLE \
  --amendment protocols/amendments/2026-09-17_priority9_v1_config_family_sensitivity.md
```

Stronger n=24 target/gate:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority9_v1_family/stronger_n24_targets_20260917 \
  --output-name paysim_priority9_stronger_n24_targets.pt \
  --groups 24 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026091823 \
  --purpose 'Priority 9 PaySim v1_stronger GEN_IDLG_STYLE n24 development pilot'

PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority6_sota_style_attackers.py execute \
  --dataset paysim \
  --target artifacts/priority9_v1_family/stronger_n24_targets_20260917/paysim_priority9_stronger_n24_targets.pt \
  --output artifacts/priority9_v1_family/stronger_n24_gate_20260917 \
  --seed 2026091824 --workers 8 --groups 24 \
  --defenses v1_stronger --generations GEN_IDLG_STYLE \
  --amendment protocols/amendments/2026-09-17_priority9_v1_config_family_sensitivity.md
```

Stronger confirmatory target/gate:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority9_v1_family/stronger_confirmatory_targets_20260917 \
  --output-name paysim_priority9_stronger_confirmatory_targets.pt \
  --groups 39 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026091825 \
  --purpose 'Priority 9 PaySim v1_stronger GEN_IDLG_STYLE confirmatory p1_0p70 one-shot'

PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority6_sota_style_attackers.py execute \
  --dataset paysim \
  --target artifacts/priority9_v1_family/stronger_confirmatory_targets_20260917/paysim_priority9_stronger_confirmatory_targets.pt \
  --output artifacts/priority9_v1_family/stronger_confirmatory_gate_20260917 \
  --seed 2026091826 --workers 8 --groups 39 \
  --defenses v1_stronger --generations GEN_IDLG_STYLE \
  --amendment protocols/amendments/2026-09-17_priority9_v1_stronger_confirmatory.md
```

## Summary table for paper

| v1 config | Parameters | Highest stage reached | Prior wins | Prior p | Zero wins | Zero p | Gate status | Interpretation |
|---|---:|---|---:|---:|---:|---:|---|---|
| Conservative | mix=0.08, keep=0.88, shrink=0.45 | n=8 development | 6/8 | 0.14453125 | 6/8 | 0.14453125 | FAIL | Not escalated; no safety claim |
| Medium | mix=0.10, keep=0.85, shrink=0.40 | n=39 confirmatory, Priority 6 | 29/39 | 0.0016889239559532143 | 32/39 | 0.000035127392038702965 | PASS | Confirmed vulnerability |
| Stronger | mix=0.12, keep=0.82, shrink=0.35 | n=39 confirmatory, Priority 9 | 27/39 | 0.011851351235236507 | 29/39 | 0.0016889239559532143 | PASS | Confirmed vulnerability |

## Stage-by-stage Priority 9 results

### v1-conservative

Initial n=8 development gate:

- Prior: 6/8 wins, p=0.14453125, mean difference=-1381.8061417588833,
  median difference=-195.6530208500523.
- Zero-update: 6/8 wins, p=0.14453125,
  mean difference=-2922.78509679691,
  median difference=-237.77517740396874.
- Overall: FAIL.

Per protocol, this branch stopped at n=8.

### v1-stronger

Initial n=8 development gate:

- Prior: 7/8 wins, p=0.03515625, mean difference=-6043.443174031052,
  median difference=-5617.945203885221.
- Zero-update: 7/8 wins, p=0.03515625,
  mean difference=-7390.194793356139,
  median difference=-6134.434554845777.
- Overall: PASS; escalated to n=24.

n=24 development pilot:

- Prior: 20/24 wins, p=0.000771939754486084,
  mean difference=-2272.1867015020944,
  median difference=-176.26516760813485.
- Zero-update: 19/24 wins, p=0.003305375576019287,
  mean difference=-2988.244544728095,
  median difference=-309.01801145977964.
- Overall: PASS; separate confirmatory amendment written before n=39 target
  generation.

n=39 confirmatory:

- Prior: 27/39 wins, p=0.011851351235236507,
  mean difference=-2582.348835044493,
  median difference=-112.92560287958473.
- Zero-update: 29/39 wins, p=0.0016889239559532143,
  mean difference=-2504.597899877654,
  median difference=-120.82871476647404.
- Overall: PASS.

## Data firewall and provenance

All newly generated PaySim target sets reported `max_overlap_with_existing_targets=0`
in their provenance files. Dataset checksum was stable:

`16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`

| Artifact | Groups | Max overlap | SHA-256 |
|---|---:|---:|---|
| `conservative_n8_targets/...pt` | 8 | 0 | `05fb04535491facf0a2aa6452b123fbde5a69c899a5f286bb61c6e70c6064ce1` |
| `conservative_n8_targets/...provenance.json` | 8 | 0 | `fb9dca8771a34717a3ad24aa9636e1c52f007156ba1c94d4ea2f93144bbb7b75` |
| `stronger_n8_targets/...pt` | 8 | 0 | `9b0987e8af166d1d8aac9b5a21f37343df26f2f4d15e6e666524c5f351ad35f2` |
| `stronger_n8_targets/...provenance.json` | 8 | 0 | `a697372131cfe248afbc26d1cc27af8b1b5f9f720cb096a1059f124442241beb` |
| `stronger_n24_targets/...pt` | 24 | 0 | `36c53230b1a7f7547055c215d4c54b8f6bc9cf5c2384b316e16a5f4f74fe4652` |
| `stronger_n24_targets/...provenance.json` | 24 | 0 | `51bfe65c102b69e686d03383884e64c5e98048eb76ee20b15d9632df4d283a32` |
| `stronger_confirmatory_targets/...pt` | 39 | 0 | `798fa736e2748f06a3e0d579f8042185da36cd6d943fb1d3b9c4433af9e14696` |
| `stronger_confirmatory_targets/...provenance.json` | 39 | 0 | `97dc3c2c008ac3db19c664458bb0c01fcd2af88cbf304d24c2b9917313f8c143` |

Gate report hashes:

- Conservative n=8 gate JSON:
  `ca41877b103d0186aaa3c3d58222e3b474d5e6037874f3f27d43c14d3c5c4314`
- Stronger n=8 gate JSON:
  `e6f668ae896ef425f1269d8fd8104d04f5dab4a5cb3f1caa67f29fed585f4bf7`
- Stronger n=24 gate JSON:
  `ee301e257c17fedae50d5ca16ac3f1459cc15dbfbadebbf8cb95e39f3536942c`
- Stronger n=39 confirmatory gate JSON:
  `f40ae1ccf76d440f1d938615f99d39c0ce0e7df4b7cb0cab181c146f71a14211`

Priority 6 medium reference hashes:

- Medium confirmatory target:
  `1df22f43699ac4ae3f207ff06224aeb062667130d82d36f2c967259b67b1d159`
- Medium confirmatory gate JSON:
  `aae52a1d50a76a8ebce916b60dc8fff479a48d46b95de473ceecc30e41495a06`

## Implementation note

The runner was extended to expose frozen v1 config names without changing the
underlying v1 formula:

- `v1_conservative`
- `v1_medium`
- `v1_stronger`

Runner hash after the extension:
`f2361bb245bd2c3b46e570d3eaeb582f9f78262c969a615b8b24639bc6c19985`

## Final conclusion

For PaySim, `GEN_IDLG_STYLE` has now confirmed attacks against two v1 settings:
medium and stronger. The lighter conservative setting did not pass the n=8
development gate in this precommitted run and therefore was not escalated.

The v1 family conclusion for paper tables should therefore be phrased as:

> The published-loss attacker confirmed vulnerability for v1-medium and
> v1-stronger on PaySim, while v1-conservative did not pass development gate
> under the same attacker and scope. This bounds, rather than eliminates, the
> v1-family vulnerability claim.
