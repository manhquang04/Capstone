# Priority 7 CIFAR-10 v2 GEN_COSINE_TV n=24 development report

**Date:** 2026-09-17  
**Scope:** development escalation only, not confirmatory.  
**Cell:** CIFAR-10 / minimal LeNet / DNA Transform v2
`ratio0.95 eta0.01` / `GEN_COSINE_TV` / scope 4 images.

## Protocol amendment

Before target generation or execution, the following amendment was written:

`protocols/amendments/2026-09-17_priority7_cifar_v2_cosine_n24_escalation.md`

It escalates exactly one cell:

- `v2_ratio0p95_eta0p01__GEN_COSINE_TV`.

It explicitly does not escalate:

- `v1_medium__GEN_IDLG_STYLE`;
- `v1_medium__GEN_RAW_LIFT`;
- `v2_ratio0p95_eta0p01__GEN_RAW_LIFT`.

Rationale: `v2_ratio0p95_eta0p01__GEN_COSINE_TV` was the only cell that passed
strongly and consistently at both previous image-domain development scopes
(`1 image` and `4 images`, both `8/8` vs Prior and Zero-update).

## Target generation

Created a new CIFAR-10 n=24 target set:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority7_image_domain_gate.py \
  prepare-targets \
  --output artifacts/priority7_image_domain/n24_scope4_v2_cosine_targets_20260917 \
  --seed 2026091796 \
  --groups 24 \
  --images-per-group 4
```

Data firewall:

- historical CIFAR image-domain manifests checked: `2`;
- overlap with `minimal_scope1_targets_20260917`: `0`;
- overlap with `scope4_targets_20260917`: `0`;
- maximum overlap: `0`;
- disjointness gate: `PASS`.

Target artifact:

- path:
  `artifacts/priority7_image_domain/n24_scope4_v2_cosine_targets_20260917/cifar10_priority7_targets.pt`;
- SHA-256:
  `7e8790d303e21ef3601561af2b6ef2cddee2925a9e3e285d2a0f45caa95a7e7e`.

## Command run

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority7_image_domain_gate.py \
  execute \
  --target artifacts/priority7_image_domain/n24_scope4_v2_cosine_targets_20260917/cifar10_priority7_targets.pt \
  --output artifacts/priority7_image_domain/n24_scope4_v2_cosine_gate_20260917 \
  --seed 2026091797 \
  --groups 24 \
  --restarts 4 \
  --iterations 25 \
  --attack-lr 0.05 \
  --local-lr 0.01 \
  --workers 1 \
  --cells v2_ratio0p95_eta0p01__GEN_COSINE_TV
```

Runtime: `360` seconds.  
Completed jobs: `96/96`.  
No multiprocessing was used.

## Results

| Control | Wins/non-ties | Mean ΔMSE | Median ΔMSE | One-sided exact sign-test p | Gate |
|---|---:|---:|---:|---:|---|
| Prior | 24/24 | -0.0190448228 | -0.0184335996 | 5.960464477539063e-08 | PASS |
| Zero-update | 24/24 | -0.0191510966 | -0.0188477624 | 5.960464477539063e-08 | PASS |

Overall development gate: **PASS**.

## Interpretation

The n=24 pilot preserves the strong n=8 signal for the CIFAR-10 minimal-LeNet
image-domain setting:

- scope 1 image: `8/8` vs both controls;
- scope 4 images: `8/8` vs both controls;
- scope 4 n=24: `24/24` vs both controls.

This is still development evidence only. It does not by itself authorize or
constitute confirmatory image-domain RQ1 evidence, and it does not alter any
frozen tabular PaySim/IEEE-CIS RQ1/RQ2/RQ3 conclusion.

Per supervisor instruction, execution stops here. The next step, if approved,
is:

1. run exact-binomial power analysis using the Priority 6 method;
2. request quick nod if the selected `p1` differs from the historical `0.70`;
3. write a separate confirmatory amendment;
4. create a new source-disjoint CIFAR target set;
5. run confirmatory exactly once.

## Artifact hashes

| File | SHA-256 |
|---|---|
| `protocols/amendments/2026-09-17_priority7_cifar_v2_cosine_n24_escalation.md` | `6d168750681c18a68fe202c04ecd863a597e94c4320441f982e41bfb76523043` |
| `n24_scope4_v2_cosine_targets_20260917/cifar10_priority7_targets.pt` | `7e8790d303e21ef3601561af2b6ef2cddee2925a9e3e285d2a0f45caa95a7e7e` |
| `n24_scope4_v2_cosine_targets_20260917/target_manifest.json` | `5f347c087bfa8604507d49768f3bbc4f5c86af00e05e341233f2d576b3baf0c0` |
| `n24_scope4_v2_cosine_gate_20260917/priority7_image_gate_report.json` | `1828c7602a00fb53fddf9e9ceff54f8d2d0dd6b46787518b5eec3c5f67c4fc56` |

## Checks

- `torch.set_num_threads(1)`: preserved.
- Single-thread-per-process: preserved (`workers=1`).
- Post-hoc/tabular data: not touched.
- Confirmatory execution: not run.
- Scope beyond n=24: not run.

