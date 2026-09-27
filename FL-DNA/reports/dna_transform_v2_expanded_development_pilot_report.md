# DNA Transform v2 — Expanded development pilot report

**Date:** 2026-09-16  
**Status:** DEVELOPMENT ATTACKER VALIDATED AT N=24 — NO THEORETICAL-LIMIT STEP RUN  
**Scope:** development-only stability check for the compressed-sensing/IHT v2
attacker. No confirmatory target was created.

## 1. Đã làm gì

Wrote the pre-run amendment before creating the expanded development target set:

```text
protocols/amendments/2026-09-16_dna_transform_v2_expanded_development_pilot.md
```

The amendment locked:

- new independent development target size: `n=24`;
- `records_per_group=4`;
- `fraud_records_per_group=1`;
- attacker family: compressed-sensing/IHT;
- attacker variant: `sparsity_fraction=0.20`;
- no attacker hyperparameter tuning after seeing results.

Created one source-disjoint development target set:

```text
artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/
  dna_v2_iht_expanded_development_targets.pt
```

Ran the pre-selected IHT-0.20 attacker once on all 24 development groups:

```text
artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/
  dna_v2_iht_attack_dna_v2_iht_expanded_development_ratio0p95_eta0p01_s0p2_iht80_step1_r4_i600_lr0p1_nonneg0p001/
```

## 2. Technical replay amendment

The first execution stopped at group 17 before producing an aggregate gate
report because a Phase-3 `capture()` numeric consistency assertion was too
tight for one float32 state-dict element:

```text
max absolute difference = 1.9073486328125e-06
original tolerance = atol=2e-7, rtol=2e-4
```

Before resuming, a technical replay amendment was written:

```text
protocols/amendments/2026-09-16_dna_transform_v2_expanded_pilot_numeric_replay.md
```

The script was patched only for this IHT runner to first try the original
`capture()` and, on numeric assertion failure only, recompute the same capture
with a float32-compatible check:

```text
atol = 2e-6
rtol = 3e-4
```

No target, seed, attacker hyperparameter, v2 config, or
`torch.set_num_threads(1)` setting was changed. Completed restart artifacts
were skipped by existing `results.json` resume logic.

## 3. Source-disjointness

Independent verifier result:

```text
confirmatory_groups = 24
confirmatory_source_count = 96
historical_target_files_checked = 1046
load_failures = {}
max_overlap = 0
disjointness_gate = PASS
```

Despite the verifier field name `confirmatory_target`, this target is
development-only under the amendment.

## 4. Commands actually run

Create target:

```bash
PYTHONPATH=. .venv-phase1/bin/python \
  experiments/create_phase4_source_disjoint_targets.py \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --output-name dna_v2_iht_expanded_development_targets.pt \
  --groups 24 \
  --records-per-group 4 \
  --fraud-per-group 1 \
  --seed 2026091624 \
  --purpose dna_transform_v2_expanded_development_pilot_n24
```

Verify source-disjointness:

```bash
PYTHONPATH=. .venv-phase1/bin/python \
  experiments/verify_rq1_target_disjointness.py \
  --confirmatory-target artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_v2_iht_expanded_development_targets.pt \
  --output artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_v2_iht_expanded_development_source_overlap_matrix.json
```

Run expanded-pilot attacker:

```bash
PYTHONPATH=. .venv-phase1/bin/python \
  experiments/run_phase4_dna_v2_iht_attack.py \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --target-file dna_v2_iht_expanded_development_targets.pt \
  --sparsity-fraction 0.20 \
  --iht-iterations 80 \
  --iht-step-size 1.0 \
  --restarts 4 \
  --iterations 600 \
  --attack-lr 0.1 \
  --compression-ratio 0.95 \
  --quantization-eta 0.01 \
  --v2-base-seed 20260916 \
  --amendment protocols/amendments/2026-09-16_dna_transform_v2_expanded_development_pilot.md
```

Resume after technical replay amendment used the same command.

## 5. Kết quả kèm uncertainty/gate

The pre-locked gate rule was:

```text
mean difference < 0
median difference < 0
one-sided exact sign-test p < 0.05
```

For `n=24`, the first exact sign-test rejection count is `17/24`
(`p=0.0319573283`).

Observed result:

| Control | Wins/n | Sign-test p | Mean diff | Median diff | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Prior | 20/24 | 0.00077194 | -2897.8253 | -373.7514 | PASS |
| Zero-update | 20/24 | 0.00077194 | -1956.9827 | -241.9631 | PASS |
| Overall | — | — | — | — | PASS |

The attacker passes both development controls at n=24.

## 6. Interpretation

The n=8 failures of the compressed-sensing/IHT family were not stable under the
pre-locked independent n=24 development expansion. Under the supervisor's
decision rule, this means:

```text
The IHT-0.20 attacker is validated at development scale for DNA Transform v2.
Do not proceed to theoretical-limit analysis as the next step under this
branch.
```

This does not itself establish an RQ1 confirmatory result for v2. It only means
that v2 now has a development-valid attacker candidate. The next allowed step
is to write and freeze a full v2 confirmatory protocol, including target
generation, branch gates, and any comparator calibration required for the v2
scope.

## 7. Artifact-run IDs and hashes

| Artifact | SHA-256 |
| --- | --- |
| expanded-pilot amendment | `68c6bd1491e8baabdfa4611e0cbf3ac900cc038001704316b89a3a21697b7888` |
| numeric replay amendment | `3ced3befce2dbdf678e4261a1699be443ed07745f24fe8f422f00db5ab191b83` |
| IHT runner | `930565f1ee46b9691a911b41d6d6b04948d53d38ecf0ad24c118b0330e6fb576` |
| expanded development targets | `43b7cec7e925b94207684d6c683460538f0a25f6012189216a97fb3cda957bd3` |
| target provenance | `2c4ab90a3362860e83b9fc4ef3a5ba76ce61ce58cd4fe7783d27e69dc9ad666e` |
| source-overlap matrix | `ca56377d720fd7add2269a3afadd4668bbb3c1a9660b0aac31c6d0be9e29849b` |
| expanded IHT report | `3c9ebcf70518156fcf9db3cddb4321121fe68272bd4d8f185bdc53ebac789996` |

## 8. Gate đạt/chưa đạt

| Gate | Status | Reason |
| --- | --- | --- |
| Pre-run amendment before target/results | PASS | amendment written before target generation |
| Source-disjoint target | PASS | max overlap 0 across 1046 historical target files |
| Fixed attacker variant | PASS | only IHT-0.20 run |
| Prior control | PASS | 20/24, p=0.00077194 |
| Zero-update control | PASS | 20/24, p=0.00077194 |
| Overall development attacker validation | PASS | both controls passed |
| Theoretical-limit Việc B | NOT RUN | blocked by decision rule; attacker passed at n=24 |
| V2 confirmatory target | NOT CREATED | requires separate protocol/freeze |

## 9. Bước tiếp theo được phép

Do not run Việc B under this branch.

The next permitted action is a new supervisor-approved v2 confirmatory protocol:

- freeze the validated attacker config;
- freeze the v2 DP distortion-matched comparator or explicitly state its role;
- compute/freeze confirmatory target count and gate criteria;
- create a new source-disjoint confirmatory target set only after protocol
  approval.
