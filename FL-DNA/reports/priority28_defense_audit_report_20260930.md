# Priority 28 defense audit report

Date: 2026-09-30

Status: PARTIAL / GATED STOP

This report records the Priority 28 amendment, Step 0 harness hardening, and
instrument qualification outcomes. No `Latex/` file, earlier report, earlier
artifact, or `external_defenses/` file was modified.

## Pre-registration

Protocol amendment:
`protocols/amendments/2026-09-30_priority28_defense_audit.md`

SHA-256:
`3d6c942d50a5d85b3dfc47d2a3e89c20f1bf9cf1ae55b19c94ce0ee7741500fb`

The amendment froze the compute reduction before any Priority 28 run:
E3 only for cells that survive E2, CIFAR batch-1 before batch-4, 6-point
utility grids if E3 becomes available, and ATS last.

## Step 0 harness checks

Implemented changes:

- `experiments/harness/validators.py`
  - added data-free positive-control gating;
  - added lossless sanity records;
  - added defense-not-identity records;
  - added attacker knowledge-receipt records;
  - added strict `validate_priority28_for_report(...)`.
- `experiments/priority28/harness_checks.py`
  - added explicit least-squares decode helper;
  - added update-delta defense-not-identity helper.
- `tests/test_priority28_harness_checks.py`
  - checks least-squares is not the naive transpose lift;
  - checks update identity/change detection.
- `tests/test_priority27_harness.py`
  - extended with Priority 28 strict gate tests while preserving Priority 27 API.

Unit-test results:

| Test command | Result | Log |
| --- | --- | --- |
| `PYTHONPATH=. .venv-phase1/bin/python -m pytest tests/test_priority27_harness.py tests/test_priority28_harness_checks.py -q` | 11 passed | `artifacts/priority28_defense_audit/step0_20260930/harness_unit_tests.log` |
| `PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 external_defenses/.venv/bin/python -m unittest external_defenses.tests.test_adaptive -v` | 8 passed | `artifacts/priority28_defense_audit/step0_20260930/external_adaptive_unit_tests.log` |
| `PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 external_defenses/.venv/bin/python -m external_defenses.adaptive.smoke` | completed | `artifacts/priority28_defense_audit/step0_20260930/external_adaptive_smoke.json` |

Note: the external adaptive unit test file internally calls
`torch.set_num_threads(4)` in its own `setUp`. This was used only as a provenance
baseline for the external read-only code, not as a Priority 28 experimental
workload. Priority 28 experiment scripts written here set `torch.set_num_threads(1)`.

## Step 1 qualification

### CIFAR I1

Target:
`artifacts/priority28_defense_audit/cifar_i1_n8_targets_20260930/cifar10_priority7_targets.pt`

Target SHA-256:
`3d69a5bbb45c401995fa1a817606545de54c35533b470d1b0605a82ac25cbb3f`

Source-disjoint evidence:
`artifacts/priority28_defense_audit/cifar_i1_n8_targets_20260930/target_manifest.json`
reports `max_overlap = 0` against 4 prior CIFAR target manifests.

Command:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority28_cifar_positive_control.py \
  --target artifacts/priority28_defense_audit/cifar_i1_n8_targets_20260930/cifar10_priority7_targets.pt \
  --output artifacts/priority28_defense_audit/cifar_i1_n8_positive_control_20260930 \
  --seed 2026093001 --groups 8 --restarts 4 --iterations 3000 \
  --attack-lr 0.05 --tv-lambda 0.0001 --local-lr 0.01
```

Result:

| Gate | Wins / n | Exact p | Result |
| --- | ---: | ---: | --- |
| Own signal vs decoy | 8 / 8 | 0.00390625 | PASS |
| Own signal vs Prior | 6 / 8 | 0.14453125 | FAIL |
| Own signal vs gray image | 4 / 8 | 0.63671875 | FAIL |
| Own signal vs CIFAR mean image | 4 / 8 | 0.63671875 | FAIL |

Conclusion: CIFAR I1 failed the pre-registered n=8 positive-control gate. Per
amendment, CIFAR n=24, CIFAR defense tests, ATS, and CIFAR batch-4 were not run.
All CIFAR defense/domain cells are therefore `NOT_ASSESSABLE` in this Priority
28 pass.

### Tabular T1

Targets:

- n=8:
  `artifacts/priority28_defense_audit/t1_targets_20260930/paysim_priority28_t1_n8_targets.pt`
- n=24:
  `artifacts/priority28_defense_audit/t1_targets_20260930/paysim_priority28_t1_n24_targets.pt`

Target SHA-256:

- n=8:
  `7047a7b7ef5dc48f2ff40fd7178e3f75e1439143cbcf6c6707dbd0a2f22e48d5`
- n=24:
  `2b364dd513ffa06a9bc46792a67a1b4cd91ed794bbcd51e3b85f07986d34a59b`

Both provenance files report `max_overlap_with_existing_targets = 0`.

Command:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority28_t1_qualification.py \
  --n8-target artifacts/priority28_defense_audit/t1_targets_20260930/paysim_priority28_t1_n8_targets.pt \
  --n24-target artifacts/priority28_defense_audit/t1_targets_20260930/paysim_priority28_t1_n24_targets.pt \
  --output artifacts/priority28_defense_audit/t1_qualification_20260930 \
  --seed 2026093002
```

Result:

| Stage | Wins vs Prior | p | Wins vs decoy | p | Qualified |
| --- | ---: | ---: | ---: | ---: | --- |
| n=8 | 8 / 8 | 0.00390625 | 8 / 8 | 0.00390625 | YES |
| n=24 | 24 / 24 | 5.960464477539063e-08 | 24 / 24 | 5.960464477539063e-08 | YES |

Conclusion: tabular T1 qualifies.

## E1/E2/E3 status

No E1/E2/E3 confirmatory defense cell was emitted in this pass.

Reason:

- CIFAR has no qualified instrument after the n=8 positive-control gate.
- Tabular T1 qualifies, but it is specifically a BN-statistics channel
  instrument. The read-only external-defense adapters for Soteria, PRECODE,
  gradient pruning, ATS, and Count-Sketch are gradient/update or input/model
  mechanisms; they do not create a compatible transmitted BN-statistics server
  view for T1. Emitting an E2 result for those cells would violate the
  Priority 28 check that the attack and server view match the declared defense
  knowledge.
- PRECODE and ATS in particular cannot be treated as pass-through update
  adapters for an E2 claim. This is exactly why Step 0 added the
  defense-not-identity and model/input-level checks.

Accordingly, external published-defense cells in this pass are reported as
`NOT_ASSESSABLE`, not as `SURVIVES` or `DOES_NOT_SURVIVE`.

| Defense | Domain | Instrument status | E1 | E2 | E3 | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Soteria | PaySim | T1 qualified but not a compatible BN-view defense cell | not run | not run | not run | NOT_ASSESSABLE |
| PRECODE | PaySim | T1 qualified but PRECODE must be model-level; no compatible BN-view cell emitted | not run | not run | not run | NOT_ASSESSABLE |
| Gradient pruning | PaySim | T1 qualified but pruning is trainable-gradient only | not run | not run | not run | NOT_ASSESSABLE |
| Count-Sketch | PaySim | T1 qualified but sketch privacy claim is not published and BN-view cell unavailable | not run | not run | not run | NOT_ASSESSABLE |
| v1 conservative | PaySim | T1 qualified; not run in this pass | not run | not run | not run | PENDING |
| v2 0.95/0.01 | PaySim | T1 qualified; not run in this pass | not run | not run | not run | PENDING |
| Soteria | CIFAR | I1 failed n=8 positive control | not run | not run | not run | NOT_ASSESSABLE |
| PRECODE | CIFAR | I1 failed n=8 positive control | not run | not run | not run | NOT_ASSESSABLE |
| Gradient pruning | CIFAR | I1 failed n=8 positive control | not run | not run | not run | NOT_ASSESSABLE |
| Count-Sketch | CIFAR | I1 failed n=8 positive control | not run | not run | not run | NOT_ASSESSABLE |
| ATS | CIFAR | I1 failed n=8 positive control; ATS last by compute reduction | not run | not run | not run | NOT_ASSESSABLE |
| v1 conservative | CIFAR | I1 failed n=8 positive control | not run | not run | not run | NOT_ASSESSABLE |
| v2 0.95/0.01 | CIFAR | I1 failed n=8 positive control | not run | not run | not run | NOT_ASSESSABLE |

Study-wide conclusion for this partial Priority 28 pass: zero published
privacy conclusions were shown to survive the four checks, but this is because
the required compatible instruments/cells were not available after gating, not
because the defenses failed E2.

## New file hashes

| File | SHA-256 |
| --- | --- |
| `experiments/harness/validators.py` | `f4a575c20e78a13ed7c4df5e3e9bd73f962c550b948bb4e0c978812894e09580` |
| `experiments/harness/__init__.py` | `2b8f3d0a99198a91f109fa01a84ffa1bd1bca005743e53e7ee8f1cc0efc72f96` |
| `experiments/priority28/harness_checks.py` | `0ce7022b70c1ff40799f009b5ae8792d041335be1ee7d187ac6d1d8215c692a8` |
| `experiments/priority28_cifar_positive_control.py` | `c88423c2af54ecb7796e18d3481f09bf3d0ed810a3b525d9375c150efe5202b0` |
| `experiments/priority28_t1_qualification.py` | `b68d5fe8a3cf76e966efbd71196a83c6ee25a3fca5348ccd5f63347f8ee61eca` |
| `tests/test_priority28_harness_checks.py` | `5cefdcb1d31e36ae741b08479830e8c78ddb5ac81d48a4c63a41d1c55c61aa21` |
| `artifacts/priority28_defense_audit/step0_20260930/harness_unit_tests.log` | `23a7967b0e78f625b0ede644c4500d575e290131f962d80b182d2016db34c621` |
| `artifacts/priority28_defense_audit/step0_20260930/external_adaptive_unit_tests.log` | `4ca7a3c9fa3ff7b9db85ab8c90fd1332264dca303ecdce45778ab09462478a1f` |
| `artifacts/priority28_defense_audit/step0_20260930/external_adaptive_smoke.json` | `ec3301fe244be7d92a0ae84d7e8511ee40ec9ee8da9b3fc56be1d61ecdd1e836` |
| `artifacts/priority28_defense_audit/cifar_i1_n8_positive_control_20260930/cifar_positive_control.csv` | `0c30b9d79a50f4511ac81a3363fc17e5b58792f3d0b5d8831724aa56ccb69c5e` |
| `artifacts/priority28_defense_audit/cifar_i1_n8_positive_control_20260930/cifar_positive_control_summary.json` | `0f374b7f8a4ae2e1ffd96ee4cf5927b31bd6b08aa780109f93ff0304317fcc80` |
| `artifacts/priority28_defense_audit/t1_qualification_20260930/t1_qualification_summary.json` | `20184cff6af2d493a3d8e7a9595276883f62a2bf47153670b6878c5a7b4d1da4` |
| `artifacts/priority28_defense_audit/t1_qualification_20260930/t1_n8_qualification.csv` | `982531d382a40493758dea60aecec88f70951dbedc0433759b2501325cdf3c34` |
| `artifacts/priority28_defense_audit/t1_qualification_20260930/t1_n24_qualification.csv` | `c3b666416fead3cdc97a72bb9e3cfd28015425c1462fdef1d9e99477b2677684` |

## Commands run

```bash
PYTHONPATH=. .venv-phase1/bin/python -m pytest tests/test_priority27_harness.py tests/test_priority28_harness_checks.py -q
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 external_defenses/.venv/bin/python -m unittest external_defenses.tests.test_adaptive -v
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 external_defenses/.venv/bin/python -m external_defenses.adaptive.smoke
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority7_image_domain_gate.py prepare-targets --output artifacts/priority28_defense_audit/cifar_i1_n8_targets_20260930 --seed 2026093008 --groups 8 --images-per-group 1
PYTHONPATH=. .venv-phase1/bin/python experiments/priority28_cifar_positive_control.py --target artifacts/priority28_defense_audit/cifar_i1_n8_targets_20260930/cifar10_priority7_targets.pt --output artifacts/priority28_defense_audit/cifar_i1_n8_positive_control_20260930 --seed 2026093001 --groups 8 --restarts 4 --iterations 3000 --attack-lr 0.05 --tv-lambda 0.0001 --local-lr 0.01
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py artifacts/priority28_defense_audit/t1_targets_20260930 --output-name paysim_priority28_t1_n8_targets.pt --groups 8 --records-per-group 4 --fraud-per-group 1 --seed 20260930081 --purpose priority28_t1_n8_qualification
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py artifacts/priority28_defense_audit/t1_targets_20260930 --output-name paysim_priority28_t1_n24_targets.pt --groups 24 --records-per-group 4 --fraud-per-group 1 --seed 20260930241 --purpose priority28_t1_n24_qualification
PYTHONPATH=. .venv-phase1/bin/python experiments/priority28_t1_qualification.py --n8-target artifacts/priority28_defense_audit/t1_targets_20260930/paysim_priority28_t1_n8_targets.pt --n24-target artifacts/priority28_defense_audit/t1_targets_20260930/paysim_priority28_t1_n24_targets.pt --output artifacts/priority28_defense_audit/t1_qualification_20260930 --seed 2026093002
PYTHONPATH=. .venv-phase1/bin/python -m py_compile experiments/harness/validators.py experiments/harness/__init__.py experiments/priority28/harness_checks.py experiments/priority28_cifar_positive_control.py experiments/priority28_t1_qualification.py tests/test_priority27_harness.py tests/test_priority28_harness_checks.py
git diff --check -- experiments/harness/validators.py experiments/harness/__init__.py experiments/priority28/harness_checks.py experiments/priority28_cifar_positive_control.py experiments/priority28_t1_qualification.py tests/test_priority27_harness.py tests/test_priority28_harness_checks.py protocols/amendments/2026-09-30_priority28_defense_audit.md
```

## Final checks

- `py_compile`: PASS for all new/modified Priority 28 Python files.
- `pytest` harness tests: PASS, 11/11.
- `external_defenses` unit tests: PASS, 8/8; used as read-only provenance only.
- `git diff --check`: PASS for touched code/amendment files.
- Background workload: none left running after this report.

## Plain verdict

Priority 28 did not produce any valid E1/E2/E3 defense-survival claim in this
pass. The harness was hardened and tested; CIFAR failed the required
positive-control gate; tabular T1 qualified but is only a BN-statistics
instrument and did not provide a compatible way to audit the external
gradient-only published defenses without violating the server-view/knowledge
checks.

