# Priority 7 CIFAR-10 v2 GEN_COSINE_TV confirmatory report

**Date:** 2026-09-17  
**Status:** CONFIRMATORY RUN COMPLETED — PASS.  
**Cell:** CIFAR-10 / minimal LeNet / DNA Transform v2
`ratio0.95 eta0.01` / `GEN_COSINE_TV` / scope 4 images.

## Summary

The pre-registered CIFAR-10 image-domain confirmatory run passed both required
controls:

- Prior: `39/39` wins, exact one-sided sign-test
  `p = 1.8189894035458565e-12`;
- Zero-update: `39/39` wins, exact one-sided sign-test
  `p = 1.8189894035458565e-12`.

This is confirmatory image-domain evidence that the `GEN_COSINE_TV` attacker
successfully attacks DNA Transform v2 in the CIFAR-10 minimal-LeNet setting.
It is a domain-generalization finding and does not rewrite the frozen tabular
PaySim/IEEE-CIS reports, but it is important supplementary evidence that the
v2 weakness is not limited to tabular data.

## Pre-run amendment

The confirmatory amendment was written before target generation or execution:

`protocols/amendments/2026-09-17_priority7_cifar_v2_cosine_confirmatory.md`

Frozen choices:

- `p1 = 0.70`, inherited from the RQ1/Priority 6 protocol;
- required non-tied `n = 37`;
- draw `n = 39` target groups, with the same 2-group buffer convention used in
  Priority 6;
- no refit of `p1` from the n=24 development result;
- one confirmatory run only.

## Target generation

Command:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority7_image_domain_gate.py \
  prepare-targets \
  --output artifacts/priority7_image_domain/confirmatory_v2_cosine_targets_20260917 \
  --seed 2026091798 \
  --groups 39 \
  --images-per-group 4
```

Data firewall:

- historical CIFAR image-domain manifests checked: `3`;
- overlap with `minimal_scope1_targets_20260917`: `0`;
- overlap with `scope4_targets_20260917`: `0`;
- overlap with `n24_scope4_v2_cosine_targets_20260917`: `0`;
- maximum overlap: `0`;
- disjointness gate: `PASS`.

Target artifact:

- path:
  `artifacts/priority7_image_domain/confirmatory_v2_cosine_targets_20260917/cifar10_priority7_targets.pt`;
- SHA-256:
  `9c12af4cd51cece9bb15d9c0522719cf1b7b0ecbf610dcf4b5e78db7b7c74b22`.

## Confirmatory command

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority7_image_domain_gate.py \
  execute \
  --target artifacts/priority7_image_domain/confirmatory_v2_cosine_targets_20260917/cifar10_priority7_targets.pt \
  --output artifacts/priority7_image_domain/confirmatory_v2_cosine_gate_20260917 \
  --seed 2026091799 \
  --groups 39 \
  --restarts 4 \
  --iterations 25 \
  --attack-lr 0.05 \
  --local-lr 0.01 \
  --workers 1 \
  --cells v2_ratio0p95_eta0p01__GEN_COSINE_TV
```

Runtime: `577` seconds.  
Completed jobs: `156/156`.  
No multiprocessing was used.

## Results

| Control | Wins/non-ties | Mean ΔMSE | Median ΔMSE | One-sided exact sign-test p | Gate |
|---|---:|---:|---:|---:|---|
| Prior | 39/39 | -0.0167283340 | -0.0174395090 | 1.8189894035458565e-12 | PASS |
| Zero-update | 39/39 | -0.0163785405 | -0.0169549690 | 1.8189894035458565e-12 | PASS |

Overall confirmatory gate: **PASS**.

## Interpretation

The confirmatory result is stronger than the pre-registered threshold required
by the exact-binomial design. With `n=39` non-tied groups, the attacker beat
both controls in every group.

This supports the domain-generalization claim:

> In the CIFAR-10 minimal-LeNet image-domain setting, DNA Transform v2
> `ratio=0.95, eta=0.01` is vulnerable to the literature-style
> `GEN_COSINE_TV` gradient-inversion attacker.

This does not alter frozen PaySim/IEEE-CIS RQ1/RQ2/RQ3 results. It should be
reported as a supplementary confirmatory image-domain result showing that the
successful attack on v2 is not confined to tabular datasets.

## Artifact hashes

| File | SHA-256 |
|---|---|
| `protocols/amendments/2026-09-17_priority7_cifar_v2_cosine_confirmatory.md` | `4bb8ad6a3f970f320d21b70459c751478379744e067e5f63f734a3620a0f73f9` |
| `confirmatory_v2_cosine_targets_20260917/cifar10_priority7_targets.pt` | `9c12af4cd51cece9bb15d9c0522719cf1b7b0ecbf610dcf4b5e78db7b7c74b22` |
| `confirmatory_v2_cosine_targets_20260917/target_manifest.json` | `15338209c59550ae9adc18f9d6050083ca1c413d1b6f8225701db50817cb4c2f` |
| `confirmatory_v2_cosine_gate_20260917/priority7_image_gate_report.json` | `fd7befa0647ad0589e11d0b38c0214db02b1983b321d595504bd6366eacd634e` |

## Checks

- `torch.set_num_threads(1)`: preserved.
- Single-thread-per-process: preserved (`workers=1`).
- Source-disjoint target set: PASS.
- Confirmatory execution count: exactly one.
- No rerun after observing results.
- No other image-domain cell escalated in this confirmatory run.

