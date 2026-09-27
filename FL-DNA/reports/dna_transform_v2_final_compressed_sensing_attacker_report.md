# DNA Transform v2 — Final compressed-sensing attacker report

**Date:** 2026-09-16  
**Status:** FINAL V2 ATTACKER ATTEMPT FAILED DEVELOPMENT GATE — NO V2 CONFIRMATORY AUTHORIZED  
**Scope:** development-only final attacker generation for DNA Transform v2. No
confirmatory target was created.

## 1. Đã làm gì

Wrote the final-attacker amendment before coding or running results:

```text
protocols/amendments/2026-09-16_dna_transform_v2_final_compressed_sensing_attacker.md
```

Implemented a compressed-sensing-style attacker:

```text
experiments/run_phase4_dna_v2_iht_attack.py
```

This is the final v2 attacker family authorized by supervisor. It does not use
the v1 realization attacker or the previous sketch-space STE objective.

## 2. Attacker design

The attacker assumes the v2 measurement operator is known under the same
algorithm-known threat-model convention used elsewhere in the project:

```text
R_s = sqrt(n / k) P_s H D_s
```

It treats quantization as bounded/additive measurement noise rather than using a
straight-through estimator. The attacker first reconstructs a sparse full-update
proxy with Iterative Hard Thresholding:

```text
u_{t+1/2} = u_t + step_size R_s^T(q_obs - R_s u_t)
u_{t+1}   = H_K(u_{t+1/2})
```

Then it runs the usual hard-diff data inversion stage against the recovered
update proxy:

```text
argmin_x L_balanced(delta(x), u_iht)
```

## 3. Development data and fixed config

Development target:

```text
artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/development_gate_targets.pt
```

Dataset SHA-256 recorded by all three run reports:

```text
16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b
```

V2 config:

```text
compression_ratio = 0.95
quantization_eta = 0.01
base_seed = 20260916
```

No post-hoc, replication, clean Priority-2, or confirmatory target was used.

## 4. Variants attempted

All variants used:

```text
groups = 8 development groups
iht_iterations = 80
iht_step_size = 1.0
restarts = 4
attack_iterations = 600
attack_lr = 0.1
nonnegative_lambda = 0.001
```

The frozen amendment allowed three sparsity variants:

| Variant | Sparsity fraction | Description |
| --- | ---: | --- |
| IHT-0.05 | 0.05 | strongest sparsity prior |
| IHT-0.10 | 0.10 | middle sparsity prior |
| IHT-0.20 | 0.20 | weakest sparsity prior among attempted variants |

## 5. Development-gate results

Gate rule:

```text
mean difference < 0
median difference < 0
one-sided sign-test p < 0.05
```

Result summary:

| Variant | Prior wins/n | Prior p | Zero wins/n | Zero p | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| IHT-0.05 | 4/8 | 0.6367 | 5/8 | 0.3633 | FAIL |
| IHT-0.10 | 5/8 | 0.3633 | 6/8 | 0.1445 | FAIL |
| IHT-0.20 | 5/8 | 0.3633 | 6/8 | 0.1445 | FAIL |

No compressed-sensing/IHT variant passed both Prior and Zero-update controls.

## 6. Interpretation

This is the third controlled attacker generation attempted for DNA Transform v2:

1. raw-lift attacker against lifted v2 update;
2. sketch-space STE attacker;
3. compressed-sensing/IHT attacker.

None passed the development controls on the n=8 development pilot.

Correct final wording:

```text
After three controlled attacker generations (raw-lift, sketch-space STE,
compressed-sensing/IHT), no effective v2 attacker was found on the development
pilot. This is meaningful evidence that v2 is harder to reconstruct within the
attempted scope, but it is not a formal proof of security.
```

Forbidden wording:

```text
DNA Transform v2 is proven secure.
DNA Transform v2 is confirmed safer than v1.
No attacker can invert v2.
```

Because no attacker passed the controls, v2 is not eligible for confirmatory RQ1
target creation under the current protocol.

## 7. Artifact-run IDs and hashes

Amendment:

```text
protocols/amendments/2026-09-16_dna_transform_v2_final_compressed_sensing_attacker.md
SHA-256: 46bb5fd1ef81d504b4009b7ca8f016f5016ef8a2ea775509ce50c8710c272e13
```

Runner:

```text
experiments/run_phase4_dna_v2_iht_attack.py
SHA-256: 410ae5e525fa9f000dcddd02d9bdf7fea752491ad59f50ef6434bc61da1b939e
```

Run reports:

```text
IHT-0.05:
artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/
  dna_v2_iht_attack_development_gate_ratio0p95_eta0p01_s0p05_iht80_step1_r4_i600_lr0p1_nonneg0p001/
  dna_v2_iht_attack_report.json
SHA-256: fb3c0720a376d9fbe81e140b02d8ca87c6a30954fb99f43726a2d18ff604ae54

IHT-0.10:
artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/
  dna_v2_iht_attack_development_gate_ratio0p95_eta0p01_s0p1_iht80_step1_r4_i600_lr0p1_nonneg0p001/
  dna_v2_iht_attack_report.json
SHA-256: 748551051ae0f125181c7c14eabb665c1adf91f282adf0438a00a29534606696

IHT-0.20:
artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/
  dna_v2_iht_attack_development_gate_ratio0p95_eta0p01_s0p2_iht80_step1_r4_i600_lr0p1_nonneg0p001/
  dna_v2_iht_attack_report.json
SHA-256: e2001cb29d572d4aff366d7d984c6ad31d0425d03c02723e0e3f8d6bd1ceb754
```

## 8. Commands actually run

Validation:

```bash
python3 -m py_compile experiments/run_phase4_dna_v2_iht_attack.py
PYTHONPATH=. .venv-phase1/bin/python -m pytest -q tests/test_transform_defense_v2.py
```

Development gate, IHT-0.05:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_phase4_dna_v2_iht_attack.py \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --target-file development_gate_targets.pt \
  --sparsity-fraction 0.05 \
  --iht-iterations 80 \
  --iht-step-size 1.0 \
  --restarts 4 \
  --iterations 600 \
  --attack-lr 0.1 \
  --compression-ratio 0.95 \
  --quantization-eta 0.01 \
  --v2-base-seed 20260916
```

Development gate, IHT-0.10:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_phase4_dna_v2_iht_attack.py \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --target-file development_gate_targets.pt \
  --sparsity-fraction 0.10 \
  --iht-iterations 80 \
  --iht-step-size 1.0 \
  --restarts 4 \
  --iterations 600 \
  --attack-lr 0.1 \
  --compression-ratio 0.95 \
  --quantization-eta 0.01 \
  --v2-base-seed 20260916
```

Development gate, IHT-0.20:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_phase4_dna_v2_iht_attack.py \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --target-file development_gate_targets.pt \
  --sparsity-fraction 0.20 \
  --iht-iterations 80 \
  --iht-step-size 1.0 \
  --restarts 4 \
  --iterations 600 \
  --attack-lr 0.1 \
  --compression-ratio 0.95 \
  --quantization-eta 0.01 \
  --v2-base-seed 20260916
```

## 9. Gate status

| Gate | Status | Reason |
| --- | --- | --- |
| Amendment before code/results | PASS | amendment written before compressed-sensing runs |
| Compressed-sensing attacker implemented | PASS | IHT reconstruction plus hard-diff inversion |
| Post-hoc data untouched | PASS | only `development_gate_targets.pt` used |
| Development controls beaten | FAIL | no variant passed both Prior and Zero |
| V2 confirmatory target authorized | NO | no validated v2 attacker |

## 10. Bước tiếp theo được phép

No further v2 attacker variants are authorized under the current supervisor
decision. The v2 attacker line should stop here unless supervisor explicitly
opens a new research direction later.

The current official v2 privacy status is:

```text
No effective attacker was found after three controlled development attacker
generations. This is meaningful but non-confirmatory evidence of reconstruction
difficulty, not a formal security proof and not a claim that v2 is validated
safe.
```
