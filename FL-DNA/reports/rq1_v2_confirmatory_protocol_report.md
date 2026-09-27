# RQ1-v2 confirmatory protocol report

**Date:** 2026-09-16  
**Status:** VIỆC 1 COMPLETE — PROTOCOL FROZEN BEFORE TARGET GENERATION  
**Scope:** RQ1 confirmatory design for DNA Transform v2.

## 1. Đã làm gì

Created the confirmatory protocol amendment before creating any RQ1-v2
confirmatory target:

```text
protocols/amendments/2026-09-16_rq1_v2_confirmatory_protocol.md
```

Added v2-specific confirmatory runner and analyzer:

```text
experiments/run_rq1_v2_confirmatory.py
experiments/analyze_rq1_v2_confirmatory.py
```

Frozen mechanism:

```text
DNA Transform v2:
  compression_ratio = 0.95
  quantization_eta = 0.01
  base_seed = 20260916

Attacker:
  compressed-sensing/IHT
  sparsity_fraction = 0.20
  iht_iterations = 80
  iht_step_size = 1.0
  restarts = 4
  attack_iterations = 600
  attack_lr = 0.1
  nonnegative_lambda = 0.001

Comparator:
  DP_DISTORTION_MATCHED_V2 noise_multiplier = 0.00105
  clip_norm = 100.0
  mc_noise_samples = 100
```

## 2. Power analysis

The v2 design does not reuse the v1 `p1=0.65` or `p1=0.70` by default.

Frozen practical effect:

```text
p1 = 0.60
```

Rationale: v2 is a new lossy mechanism; the smallest practical effect worth
detecting is DNA v2 beating the v2 distortion-matched DP-style comparator in
60% of non-tied targets. The observed n=24 development win rate is not used as
the expected effect size.

Exact-binomial result:

```text
required_non_tied_n = 158
rejection_minimum_wins = 90
actual_alpha = 0.04723726761096572
actual_power = 0.8056549358366878
planned_tie_rate = 0.05
planned_dropout_rate = 0.05
initial_target_draw = 176
```

Power artifact:

```text
artifacts/dna_transform_v2/confirmatory_power_p1_0p60_20260916/rq1_power_analysis.json
```

## 3. Tie threshold

V2-specific numerical-stability study:

```text
artifacts_checked = 192
observed_max_absolute_replay_discrepancy = 0.001953125
safety_factor = 10
tie_threshold_mse = 0.01953125
```

Tie-threshold artifact:

```text
artifacts/dna_transform_v2/confirmatory_tie_threshold_20260916/tie_threshold_report.json
```

## 4. Lệnh đã chạy

```bash
python3 experiments/rq1_power_analysis.py \
  --p1 0.60 \
  --maximum-n 300 \
  --tie-rate 0.05 \
  --dropout-rate 0.05 \
  --output-dir artifacts/dna_transform_v2/confirmatory_power_p1_0p60_20260916

PYTHONPATH=. .venv-phase1/bin/python experiments/study_rq1_tie_threshold.py \
  --artifact-dirs dna_v2_iht_attack_dna_v2_iht_expanded_development_ratio0p95_eta0p01_s0p2_iht80_step1_r4_i600_lr0p1_nonneg0p001 \
  --floor 1e-12 \
  --safety-factor 10 \
  --output artifacts/dna_transform_v2/confirmatory_tie_threshold_20260916/tie_threshold_report.json

python3 -m py_compile \
  experiments/run_rq1_v2_confirmatory.py \
  experiments/analyze_rq1_v2_confirmatory.py \
  experiments/run_phase4_dna_v2_iht_attack.py

PYTHONPATH=. .venv-phase1/bin/python -m pytest -q tests/test_transform_defense_v2.py
```

## 5. Artifact IDs and hashes

| Artifact | SHA-256 |
| --- | --- |
| protocol amendment | `c3d218d681db0256c90b11f4917366853b0574957107ada7535b325bb0498a0a` |
| v2 runner | `01733b999133dfd02ef3d754c3851c17992c2866fef71f98c7d5316c2c3457a7` |
| v2 analyzer | `5d62e5095113c025e43512a6653d5c7c23a1470e1331f1de18693c3865c245a9` |
| IHT attacker runner | `930565f1ee46b9691a911b41d6d6b04948d53d38ecf0ad24c118b0330e6fb576` |
| power report | `1092debb522c542516a528263e7287582c3c42e808e11c066885971ffb09bad8` |
| tie threshold report | `9c5d8706becca5c63adb3aebcb4096435fce88277ad91ffab6bbcfaeaf692224` |

## 6. Gate đạt/chưa đạt

| Gate | Status | Reason |
| --- | --- | --- |
| Protocol before target generation | PASS | no RQ1-v2 confirmatory target created before amendment |
| Attacker config frozen | PASS | IHT-0.20 exactly as n=24 development validation |
| DP comparator frozen | PASS | `noise_multiplier=0.00105` |
| V2 tie threshold separate from v1 | PASS | `0.01953125` from v2 development artifacts |
| Power analysis v2-specific | PASS | `p1=0.60`, not copied from v1 |
| Implementation validation | PASS | py_compile and v2 unit tests pass |

## 7. Bước tiếp theo

Proceed to Việc 2: create one source-disjoint RQ1-v2 confirmatory target set
with `176` groups. Do not run confirmatory attacks until target hashes,
source-overlap verification, and machine-readable config are frozen.
