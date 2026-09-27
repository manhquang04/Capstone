# DNA Transform v2 — Structured attacker development report

**Date:** 2026-09-16  
**Status:** DEVELOPMENT ATTACKER NOT VALIDATED — NO V2 CONFIRMATORY AUTHORIZED  
**Scope:** development-only attacker design for DNA Transform v2. No
confirmatory target was created.

## 1. Đã làm gì

Wrote the attacker-design amendment before coding or running results:

```text
protocols/amendments/2026-09-16_dna_transform_v2_structured_attacker_development.md
```

Implemented a v2 sketch-space attacker:

```text
experiments/run_phase4_dna_v2_sketch_space_attack.py
```

This attacker optimizes against the observed v2 sketch rather than the lifted
full update:

```text
argmin_x L_balanced(STE_Q(R_s delta(x)), q_obs)
```

where quantization uses a straight-through estimator:

```text
STE_Q(z) = z + (Q(z) - z).detach()
```

This is a genuine v2-structured attacker attempt, unlike the earlier raw-style
lifted-update diagnostic.

## 2. Development data and fixed v2 config

Development target:

```text
artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/development_gate_targets.pt
```

V2 config:

```text
compression_ratio = 0.95
quantization_eta = 0.01
base_seed = 20260916
```

No post-hoc, replication, clean Priority-2, or confirmatory target was used.

## 3. Variants attempted

All variants used:

```text
groups = 8 development groups
restarts = 4
iterations = 600
init_mode = standard
nonnegative_lambda = 0.001
STE quantization = true
```

| Variant | Description | learning rate | L1 update lambda |
| --- | --- | ---: | ---: |
| A1 | sketch-space STE | 0.1 | 0 |
| A2 | sketch-space STE | 0.03 | 0 |
| B1 | sketch-space STE + update L1 prior | 0.03 | 1e-5 |
| B2 | sketch-space STE + update L1 prior | 0.03 | 1e-4 |

## 4. Development-gate results

Gate rule:

```text
mean difference < 0
median difference < 0
one-sided sign-test p < 0.05
```

Result summary:

| Variant | Prior wins/n | Prior p | Zero wins/n | Zero p | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| A1 `lr=0.1` | 6/8 | 0.1445 | 3/8 | 0.8555 | FAIL |
| A2 `lr=0.03` | 4/8 | 0.6367 | 5/8 | 0.3633 | FAIL |
| B1 `lr=0.03,l1=1e-5` | 6/8 | 0.1445 | 5/8 | 0.3633 | FAIL |
| B2 `lr=0.03,l1=1e-4` | 6/8 | 0.1445 | 5/8 | 0.3633 | FAIL |

No attempted v2-structured attacker passed both Prior and Zero controls.

## 5. Interpretation

This is stronger evidence than the previous raw-style lifted-update failure
that the current v2 attack pipeline is not yet effective. However, it is still
not a proof of security.

Correct wording:

```text
Within the attempted development scope, no v2-structured attacker was found
that passes the required controls.
```

Forbidden wording:

```text
DNA Transform v2 is secure.
DNA Transform v2 protects better than v1.
No attacker can invert v2.
```

The privacy status of v2 remains unresolved, but the project now has a
methodologically valid negative attacker-development result.

## 6. Artifact-run IDs and hashes

Amendment:

```text
protocols/amendments/2026-09-16_dna_transform_v2_structured_attacker_development.md
SHA-256: 5f4fdf114d3b53915fbb40d4c235ec9f9533f0462988d44cb90c19edf1d00bed
```

Runner:

```text
experiments/run_phase4_dna_v2_sketch_space_attack.py
SHA-256: c4b3cac16a2726f1c59b289eed9becf78419c8c781a72110753c75bc24fd6cc7
```

Run reports:

```text
A1:
artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/
  dna_v2_sketch_attack_development_gate_ratio0p95_eta0p01_r4_i600_lr0p1_nonneg0p001_l10_ste/
  dna_v2_sketch_attack_report.json
SHA-256: 50ad2f4a319e98bc919a72ca0401e1c366de0c8111ab2a444b29eff7615dc4a1

A2:
artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/
  dna_v2_sketch_attack_development_gate_ratio0p95_eta0p01_r4_i600_lr0p03_nonneg0p001_l10_ste/
  dna_v2_sketch_attack_report.json
SHA-256: f7ff86db5719cca69d4b15307c427bd6131172567e008f231ca4cff51506b9cc

B1:
artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/
  dna_v2_sketch_attack_development_gate_ratio0p95_eta0p01_r4_i600_lr0p03_nonneg0p001_l11e-05_ste/
  dna_v2_sketch_attack_report.json
SHA-256: 7e9faae70e21b1a911295f693cec6b4b948637eda1bfac2822e9e48907f86579

B2:
artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/
  dna_v2_sketch_attack_development_gate_ratio0p95_eta0p01_r4_i600_lr0p03_nonneg0p001_l10p0001_ste/
  dna_v2_sketch_attack_report.json
SHA-256: 0310893d7783a45b85d478d44dd9a50cda08c20f6ca45101265713b9750599ce
```

## 7. Gate status

| Gate | Status | Reason |
| --- | --- | --- |
| Amendment before code/results | PASS | amendment written before structured-attacker runs |
| v2-structured objective implemented | PASS | sketch-space STE objective |
| Development controls beaten | FAIL | no variant passed both Prior and Zero |
| Confirmatory target authorized | NO | no validated v2 attacker |

## 8. Next allowed step

No v2 confirmatory RQ1 work is authorized.

Possible next actions require explicit supervisor approval:

- stop v2 privacy evaluation and report unresolved attacker status;
- time-box a second-generation attacker with a more principled compressed
  sensing objective;
- reduce Step-5 scope to diagnosing why sketch-space objective improves Prior
  more often than Zero but cannot pass both controls.
