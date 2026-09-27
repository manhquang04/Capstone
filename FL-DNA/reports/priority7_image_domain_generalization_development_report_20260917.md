# Priority 7 image-domain generalization development report

**Date:** 2026-09-17  
**Scope:** CIFAR-10 image-domain development-gate setup only. No confirmatory
run was authorized or executed.  
**Status:** DEVELOPMENT GATE COMPLETED after minimal-LeNet technical amendment.
No confirmatory run was authorized or executed.

## What was approved and frozen

Supervisor quick nod approved all four pending items in
`protocols/config/rq1_image_domain_pre_pilot.yaml` before any gate execution:

- LeNet-style CIFAR-10 CNN;
- SGD `lr=0.01`, `local_steps=1`;
- scope escalation `1 image -> 4 images`;
- attacker matrix:
  - `GEN_IDLG_STYLE` vs `v1_medium`;
  - `GEN_COSINE_TV` vs `v2_ratio0p95_eta0p01`;
  - `GEN_RAW_LIFT` vs both defenses.

The config was updated to:

- `status: DEVELOPMENT-GATE AUTHORIZED`;
- `execution_authorized: true`.

## CIFAR-10 provenance

The CIFAR-10 archive used by torchvision was downloaded from the public
BrainChip mirror:

`https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz`

Torchvision then verified and extracted the archive using its normal CIFAR-10
integrity checks. The actual log observed during dataset setup was:

```text
Using downloaded and verified file: datasets/cifar10/cifar-10-python.tar.gz
Extracting datasets/cifar10/cifar-10-python.tar.gz to datasets/cifar10
Files already downloaded and verified
```

Dataset hashes recorded in the pre-pilot config:

- archive SHA-256:
  `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`;
- directory manifest SHA-256:
  `44e3967e7fecd77443f81803d4f0e3cabe61d23538dab9ee5d96b17fa09189a5`.

## Real-image PSNR/SSIM adapter verification

`attacks/pseudo_image.py` is tabular-specific, so a separate real-image adapter
was added in `attacks/image_metrics.py`. Before any gate use, the adapter was
tested against `skimage.metrics` on simple image pairs.

Command:

```bash
.venv-phase1/bin/python -m pytest tests/test_image_metrics_adapter.py -q
```

Observed result:

```text
3 passed, 1 warning in 0.48s
```

The tests include:

- identical RGB images: `MSE=0`, `PSNR=inf`, `SSIM=1`;
- noisy random RGB images: adapter values match direct `skimage.metrics`;
- out-of-range reconstruction values are clipped to `[0,1]` before metrics.

## Target set generated

Created a new CIFAR-10 scope-1 development target set:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority7_image_domain_gate.py \
  prepare-targets \
  --output artifacts/priority7_image_domain/minimal_scope1_targets_20260917 \
  --seed 2026091792 \
  --groups 8 \
  --images-per-group 1
```

Target indices:

```text
1404, 6197, 12308, 21471, 25954, 32854, 33461, 43574
```

Data firewall:

- historical CIFAR image-domain manifests checked: `0`;
- maximum overlap: `0`;
- disjointness gate: `PASS`.

Target artifact:

- `artifacts/priority7_image_domain/minimal_scope1_targets_20260917/cifar10_priority7_targets.pt`
- SHA-256:
  `60a129d35913fc2a10b7d4aac4d941dcc2a2d42e077450da8b29d937ad834f92`.

## Technical replay amendment

The first implementation smoke exposed a mechanical mismatch between the old
tabular v1 surrogate and CIFAR-10 CNN gradients. The historical v1 surrogate
materialized a dense `256 x 256` matrix for every gradient block; this is
feasible for the tabular MLP but not for LeNet-scale CNN gradients.

Before any valid gate result was run or interpreted, the following amendment was
written:

`protocols/amendments/2026-09-17_priority7_image_domain_gate_technical_replay.md`

The amendment froze two technical changes:

1. replace dense v1 surrogate materialization with the mathematically
   equivalent sparse/blockwise application
   `((1 - mix) * block) + mix * shrink * block[permutation]`;
2. reduce the image development-gate optimizer budget to `iterations=25` for
   this screening run, with an explicit interpretation restriction that FAIL
   cannot be interpreted as privacy success.

## Minimal-LeNet timeout amendment

After the first technical replay still showed infeasible runtime for the wider
LeNet-style CNN, supervisor authorized one final technical narrowing:

`protocols/amendments/2026-09-17_priority7_image_domain_minimal_lenet_timeout.md`

The image-domain model was reduced from the initially approved wider LeNet-style
CNN to a minimal LeNet-5-style variant while preserving CIFAR-10's original
`3 x 32 x 32` input resolution.

Parameter counts:

| Model | Trainable parameters |
|---|---:|
| Original wider LeNet-style CNN (`32/64`, `4096->384->192->10`) | 1,702,794 |
| Minimal LeNet-5-style CNN (`6/16`, `1024->120->84->10`) | 136,886 |

This change is a technical feasibility change only. It does not change CIFAR
targets, DNA v1/v2 formulas, attacker loss definitions, sign-test gates, or any
frozen tabular RQ1/RQ2/RQ3 conclusion.

## Commands attempted

Initial all-cell gate attempt before technical replay:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority7_image_domain_gate.py \
  execute \
  --target artifacts/priority7_image_domain/minimal_scope1_targets_20260917/cifar10_priority7_targets.pt \
  --output artifacts/priority7_image_domain/minimal_scope1_gate_20260917 \
  --seed 2026091794 \
  --groups 8 \
  --restarts 4 \
  --iterations 300 \
  --attack-lr 0.05 \
  --local-lr 0.01 \
  --workers 8
```

Outcome: process stalled before completing any job because the dense v1
surrogate materialization was computationally infeasible for CNN gradients.
This attempt produced no valid gate result.

Sparse-surrogate smoke:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority7_image_domain_gate.py \
  execute \
  --target artifacts/priority7_image_domain/minimal_scope1_targets_20260917/cifar10_priority7_targets.pt \
  --output artifacts/priority7_image_domain/debug_scope1_gate_smoke_sparse_20260917 \
  --seed 2026091794 \
  --groups 1 \
  --restarts 1 \
  --iterations 1 \
  --attack-lr 0.05 \
  --local-lr 0.01 \
  --workers 1 \
  --cells v1_medium__GEN_RAW_LIFT
```

Outcome: completed mechanically. This was a one-iteration technical smoke only,
not a development-gate result.

Full scope-1 replay attempts:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority7_image_domain_gate.py \
  execute \
  --target artifacts/priority7_image_domain/minimal_scope1_targets_20260917/cifar10_priority7_targets.pt \
  --output artifacts/priority7_image_domain/minimal_scope1_gate_20260917_fork \
  --seed 2026091794 \
  --groups 8 \
  --restarts 4 \
  --iterations 25 \
  --attack-lr 0.05 \
  --local-lr 0.01 \
  --workers 8
```

and:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority7_image_domain_gate.py \
  execute \
  --target artifacts/priority7_image_domain/minimal_scope1_targets_20260917/cifar10_priority7_targets.pt \
  --output artifacts/priority7_image_domain/minimal_scope1_gate_20260917_w2 \
  --seed 2026091794 \
  --groups 8 \
  --restarts 4 \
  --iterations 25 \
  --attack-lr 0.05 \
  --local-lr 0.01 \
  --workers 2
```

Outcome: multiprocessing workers were terminated abruptly, consistent with
memory/runtime pressure from concurrent second-order CNN inversion. These
attempts produced no valid gate result.

Sequential runtime probe:

```bash
time PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority7_image_domain_gate.py \
  execute \
  --target artifacts/priority7_image_domain/minimal_scope1_targets_20260917/cifar10_priority7_targets.pt \
  --output artifacts/priority7_image_domain/debug_scope1_gate_25iter_seq_20260917 \
  --seed 2026091794 \
  --groups 1 \
  --restarts 1 \
  --iterations 25 \
  --attack-lr 0.05 \
  --local-lr 0.01 \
  --workers 1 \
  --cells v1_medium__GEN_RAW_LIFT
```

Outcome: still running after more than 5.5 minutes for a single
group/restart/cell, then manually stopped as a runtime probe. The full planned
scope-1 matrix would require 128 such jobs. No valid gate result was produced.

Minimal-LeNet prioritized cell probe:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority7_image_domain_gate.py \
  execute \
  --target artifacts/priority7_image_domain/minimal_scope1_targets_20260917/cifar10_priority7_targets.pt \
  --output artifacts/priority7_image_domain/minimal_lenet_probe_v1_idlg_20260917 \
  --seed 2026091794 \
  --groups 8 \
  --restarts 4 \
  --iterations 25 \
  --attack-lr 0.05 \
  --local-lr 0.01 \
  --workers 1 \
  --cells v1_medium__GEN_IDLG_STYLE
```

Outcome: completed successfully in 666 seconds. The prioritized cell produced a
valid scope-1 development-gate result, so the remaining pre-authorized scope-1
cells were run.

Remaining scope-1 cells:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority7_image_domain_gate.py \
  execute \
  --target artifacts/priority7_image_domain/minimal_scope1_targets_20260917/cifar10_priority7_targets.pt \
  --output artifacts/priority7_image_domain/minimal_lenet_remaining_scope1_20260917 \
  --seed 2026091794 \
  --groups 8 \
  --restarts 4 \
  --iterations 25 \
  --attack-lr 0.05 \
  --local-lr 0.01 \
  --workers 1 \
  --cells \
  v2_ratio0p95_eta0p01__GEN_COSINE_TV \
  v1_medium__GEN_RAW_LIFT \
  v2_ratio0p95_eta0p01__GEN_RAW_LIFT
```

Outcome: completed successfully in 835 seconds.

Because `v2_ratio0p95_eta0p01__GEN_COSINE_TV` passed the scope-1 gate, a fresh
scope-4 CIFAR target set was created:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority7_image_domain_gate.py \
  prepare-targets \
  --output artifacts/priority7_image_domain/scope4_targets_20260917 \
  --seed 2026091793 \
  --groups 8 \
  --images-per-group 4
```

Scope-4 data firewall:

- historical CIFAR image-domain manifests checked: `1`;
- overlap with scope-1 target set: `0`;
- maximum overlap: `0`;
- disjointness gate: `PASS`;
- target SHA-256:
  `12d56777b65cb906556b109694ac845aaa6aa628cfcf08c735e2ad55b8a74eef`.

Scope-4 expansion for the passing cell:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority7_image_domain_gate.py \
  execute \
  --target artifacts/priority7_image_domain/scope4_targets_20260917/cifar10_priority7_targets.pt \
  --output artifacts/priority7_image_domain/minimal_lenet_scope4_v2_cosine_20260917 \
  --seed 2026091795 \
  --groups 8 \
  --restarts 4 \
  --iterations 25 \
  --attack-lr 0.05 \
  --local-lr 0.01 \
  --workers 1 \
  --cells v2_ratio0p95_eta0p01__GEN_COSINE_TV
```

Outcome: completed successfully in 133 seconds.

## Gate result

Valid image-domain development-gate results are now available for the minimal
LeNet-5-style CNN. These are development-only results, not confirmatory results.

Scope-1 results:

| Cell | Prior wins/non-ties | Prior p | Zero wins/non-ties | Zero p | Overall |
|---|---:|---:|---:|---:|---|
| `v1_medium__GEN_IDLG_STYLE` | 4/8 | 0.63671875 | 5/8 | 0.36328125 | FAIL |
| `v2_ratio0p95_eta0p01__GEN_COSINE_TV` | 8/8 | 0.00390625 | 8/8 | 0.00390625 | PASS |
| `v1_medium__GEN_RAW_LIFT` | 4/8 | 0.63671875 | 3/8 | 0.85546875 | FAIL |
| `v2_ratio0p95_eta0p01__GEN_RAW_LIFT` | 3/8 | 0.85546875 | 5/8 | 0.36328125 | FAIL |

Scope-4 expansion result for the only scope-1 passing cell:

| Cell | Prior wins/non-ties | Prior p | Zero wins/non-ties | Zero p | Overall |
|---|---:|---:|---:|---:|---|
| `v2_ratio0p95_eta0p01__GEN_COSINE_TV` | 8/8 | 0.00390625 | 8/8 | 0.00390625 | PASS |

Interpretation:

- The CIFAR-10 image-domain feasibility blocker was resolved by reducing the
  CNN architecture.
- Under the minimal LeNet-5-style CNN, `GEN_COSINE_TV` against DNA Transform v2
  passed development gates at both 1-image and 4-image scopes.
- `GEN_IDLG_STYLE` and `GEN_RAW_LIFT` did not pass their authorized scope-1
  cells.
- These results are development-gate evidence only. They do not authorize or
  constitute confirmatory image-domain RQ1 evidence, and they do not alter the
  frozen tabular RQ1/RQ2/RQ3 conclusions.

## Artifact run IDs and hashes

Key hashes:

| File | SHA-256 |
|---|---|
| `protocols/config/rq1_image_domain_pre_pilot.yaml` | `af05b7fe703c0e07dc7d95f8b6b8721ad49751ceb7fbf70e071f78dc01e3dc8c` |
| `protocols/amendments/2026-09-17_priority7_image_domain_gate_technical_replay.md` | `44432ac243bb947f46f16a1ffcab1e310d1d182755269d7257e9dddb27a8f6a3` |
| `protocols/amendments/2026-09-17_priority7_image_domain_minimal_lenet_timeout.md` | `79fa0f531289ac7bdf5ef0dd7ce4d5d773053057f0cc495fae4bb5a640f4f4f5` |
| `attacks/image_metrics.py` | `7d636904d0f08d2fca406bcaba16810683f001c7717e05ef103306ea4f4b79a9` |
| `tests/test_image_metrics_adapter.py` | `1b4e4385ee360d6d65642261d4b50ba228ee5a1a2a646432960eb3177d1cc983` |
| `experiments/run_priority7_image_domain_gate.py` | `904890af8df1c82032fa0cbae358c9f4664b9bb5468a45d8e9a1c686f4894786` |
| `cifar10_priority7_targets.pt` | `60a129d35913fc2a10b7d4aac4d941dcc2a2d42e077450da8b29d937ad834f92` |
| `target_manifest.json` | `39c592ef3fe0be588a3c4f8f4f56f1440adca013806771bcf36313bd48b11be3` |
| `minimal_lenet_probe_v1_idlg_20260917/priority7_image_gate_report.json` | `c1dc1ad19d44c1a0a2baf9746703ca5510060ad3debb53b1e88b33021c8dc8be` |
| `minimal_lenet_remaining_scope1_20260917/priority7_image_gate_report.json` | `412a2fe61c54434b69d3db725b7f08fa73bffb60df04344c5885cb3602078260` |
| `scope4_targets_20260917/cifar10_priority7_targets.pt` | `12d56777b65cb906556b109694ac845aaa6aa628cfcf08c735e2ad55b8a74eef` |
| `scope4_targets_20260917/target_manifest.json` | `55c9961bc13afcc14e65d6a953b65456cab9a561bbfdab0b692af19b669e3993` |
| `minimal_lenet_scope4_v2_cosine_20260917/priority7_image_gate_report.json` | `1226c522b42f58a041b43c9711bfeeedc467a34ed2313096fd37e75c5022b41b` |

## Checks

- `py_compile` for `attacks/image_metrics.py`: PASS.
- `py_compile` for `experiments/run_priority7_image_domain_gate.py`: PASS.
- PSNR/SSIM adapter unit test: PASS (`3 passed`).
- CIFAR target source-disjointness: PASS.
- `torch.set_num_threads(1)`: preserved in runner and worker.
- Confirmatory execution: NOT RUN.
- Scope-4 expansion: RUN only for the one pre-authorized scope-1 passing cell,
  still development-only.

## Recommended next step

Before any further CIFAR image-domain gate attempt, supervisor should choose one
of the following explicitly:

1. authorize a much larger runtime budget for the current exact second-order CNN
   inversion gate;
2. authorize a lighter image-domain development gate, for example fewer
   restarts/groups or a first-order/smaller-gradient-subset screening objective;
3. amend the image-domain CNN architecture to a smaller LeNet variant and rerun
   the same provenance/target firewall checks;
4. defer image-domain generalization and report the current result as a
   controlled feasibility blocker rather than scientific evidence about privacy.
