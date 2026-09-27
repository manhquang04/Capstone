# Priority 2 — RQ1 clean confirmatory execution

**Date:** 2026-09-16  
**Supervisor:** Manh Quang  
**Status:** COMPLETE — SINGLE CONFIRMATORY EXECUTION CONSUMED  
**Scope:** RQ1 clean confirmation, `dna_conservative` versus `dp_0.00025`

## 1. Đã làm gì

The supervisor selected the pair:

```text
dna_conservative vs dp_0.00025
```

The pair-selection rationale was frozen before target creation in:

```text
protocols/amendments/2026-09-16_priority2_rq1_clean_confirmatory_pair_freeze.md
```

The key anti-cherry-picking rule is:

```text
The pair was selected because it is the only pair with complete real attack
evidence available before Priority 2 began: original Group 2 conservative plus
conservative replications 1, 2 and 3.  The Priority 2 Pareto/proxy sweep is
context only and is not the basis for choosing the pair.
```

Then one new source-disjoint target set was generated and verified:

```text
artifacts/rq1/priority2_clean_freeze_20260916/rq1_priority2_clean_targets.pt
```

The official run executed exactly once with 3 branches:

- raw;
- DNA Transform conservative;
- DP-style full-client-update clipping/noise with `noise_multiplier=0.00025`.

No confirmatory target was redrawn and no attack/training parameter was changed
after target creation.

## 2. File/config thay đổi

New or updated protocol/config files:

- `protocols/amendments/2026-09-16_priority2_rq1_clean_confirmatory_pair_freeze.md`
- `protocols/config/rq1_priority2_clean_confirmatory.json`

New artifacts/results:

- `artifacts/rq1/priority2_clean_freeze_20260916/rq1_priority2_clean_targets.pt`
- `artifacts/rq1/priority2_clean_freeze_20260916/rq1_priority2_clean_targets.provenance.json`
- `artifacts/rq1/priority2_clean_freeze_20260916/source_overlap_matrix.json`
- `artifacts/rq1/priority2_clean_confirmatory_run_20260916/`
- `artifacts/rq1/priority2_clean_confirmatory_run_20260916/execution_summary_rebuilt.json`
- `results/rq1/priority2_clean_confirmatory_20260916/summary.json`
- `results/rq1/priority2_clean_confirmatory_20260916/paired_metrics.csv`
- `results/rq1/priority2_clean_confirmatory_20260916/selected_metrics.csv`

The original runner process did not leave `execution_summary.json`, but all
243 immutable `job_record.json` files existed and all had return code `0`.
`execution_summary_rebuilt.json` was therefore generated only from those job
records.  No attack job was rerun.

## 3. Lệnh thực sự đã chạy

```bash
PYTHONPATH=. .venv-phase1/bin/python \
  experiments/create_phase4_source_disjoint_targets.py \
  artifacts/rq1/priority2_clean_freeze_20260916 \
  --output-name rq1_priority2_clean_targets.pt \
  --groups 81 \
  --records-per-group 4 \
  --fraud-per-group 1 \
  --seed 338413135 \
  --purpose PRIORITY2_RQ1_CLEAN_CONFIRMATORY_SINGLE_DRAW

PYTHONPATH=. .venv-phase1/bin/python \
  experiments/verify_rq1_target_disjointness.py \
  --confirmatory-target artifacts/rq1/priority2_clean_freeze_20260916/rq1_priority2_clean_targets.pt \
  --output artifacts/rq1/priority2_clean_freeze_20260916/source_overlap_matrix.json

PYTHONPATH=. .venv-phase1/bin/python -u \
  experiments/run_rq1_variant_confirmatory.py \
  --config protocols/config/rq1_priority2_clean_confirmatory.json \
  --output-dir artifacts/rq1/priority2_clean_confirmatory_run_20260916 \
  --workers 9

PYTHONPATH=. .venv-phase1/bin/python \
  experiments/analyze_rq1_variant_confirmatory.py \
  --config protocols/config/rq1_priority2_clean_confirmatory.json \
  --run-dir artifacts/rq1/priority2_clean_confirmatory_run_20260916 \
  --output-dir results/rq1/priority2_clean_confirmatory_20260916
```

One technical setup attempt failed before writing a target because the target
directory did not exist.  The directory was then created and the same frozen
seed/config was rerun.  This was a pre-output filesystem error, not a
scientific rerun.

## 4. Target/source-disjointness result

| Item | Value |
| --- | ---: |
| Target groups | 81 |
| Source rows | 324 |
| Historical target files checked | 802 |
| Load failures | 0 |
| Max overlap | 0 |
| Disjointness gate | PASS |
| Target SHA-256 | `5f7cf97d3bd26ce951e80ab06d72ea43192b7dd0b78f6592ecc555b66fc8d6ca` |

The target set is source-disjoint with all previous development,
post-hoc/diagnostic, original Group 2, Group 3 replication and scope-boundary
target sets found under `artifacts/`.

## 5. Execution completeness

| Branch | Jobs | Success | Failed |
| --- | ---: | ---: | ---: |
| raw | 81 | 81 | 0 |
| DNA conservative | 81 | 81 | 0 |
| DP-style `0.00025` | 81 | 81 | 0 |
| total | 243 | 243 | 0 |

All job return codes were `0`; all per-job stderr logs were empty.

Runtime summary from job records:

```text
min seconds/job = 55.15
median seconds/job = 79.93
max seconds/job = 454.29
```

## 6. Branch gates

All three branches passed both Prior and Zero-update controls.

| Branch | Prior wins/n | Prior p | Zero wins/n | Zero p | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| raw | 66/81 | 4.31e-09 | 66/81 | 4.31e-09 | PASS |
| DNA conservative | 65/81 | 1.82e-08 | 71/81 | 9.00e-13 | PASS |
| DP-style `0.00025` | 67/81 | 9.45e-10 | 71/81 | 9.00e-13 | PASS |

The primary contrast is therefore valid.

## 7. Primary result

Frozen planning design:

```text
p0 = 0.50
p1 = 0.65
alpha = 0.05
required effective non-tied n = 69
planned rejection threshold at n=69 = at least 42 wins
```

Observed data:

```text
targets = 81
non-tied n = 81
ties = 0
DNA wins = 44
losses = 37
win probability = 0.5432
95% exact CI = [0.4287, 0.6544]
one-sided exact sign-test p = 0.2526
```

Because the actual non-tied count was 81, the exact rejection threshold at
alpha <= 0.05 is 49 wins.  The observed 44 wins does not reject H0.

For traceability to the frozen `42/69` planning rule, the first 69 ordered
groups contain 38 DNA wins, also below 42.

Primary conclusion:

```text
DNA Transform conservative does not establish greater reconstruction
resistance than DP-style clipping/noise at noise_multiplier=0.00025 in this
clean Priority 2 confirmatory run.
```

## 8. Direction and secondary metrics

The feature-MSE difference is defined as:

```text
D = feature_mse(DNA) - feature_mse(DP)
```

Observed:

```text
mean D = -472.68
95% t-CI = [-763.71, -181.64]
```

Thus the mean direction in this run is not merely non-significant for DNA; it
is descriptively opposite the desired DNA-advantage direction.

Secondary pseudo-image metrics likewise do not support a DNA privacy advantage:

```text
DNA-minus-DP PSNR CI = [0.0195, 1.3635]
DNA-minus-DP SSIM CI = [0.0191, 0.1419]
```

For PSNR/SSIM, higher values mean better reconstruction.  Positive DNA-minus-DP
values therefore indicate less reconstruction resistance for DNA than DP-style
clipping/noise on these secondary metrics.

## 9. DP epsilon wording

The DP-style comparator in this run is not strong record-level DP.

For `noise_multiplier=0.00025`, `clip_norm=100`, `delta=1e-5`, add/remove
neighboring updates and a single release, the approved Priority 3
Gaussian/RDP accountant gives:

```text
epsilon ≈ 8.019e6
```

Therefore the result should be worded as:

```text
DNA Transform conservative did not outperform a weak-accounting DP-style
full-client-update clipping/noise comparator at comparable update distortion.
```

It must not be worded as:

```text
DNA failed against strong formal record-level Differential Privacy.
```

## 10. Gate đạt/chưa đạt

| Gate | Status | Reason |
| --- | --- | --- |
| Pair-freeze before target creation | PASS | Amendment written before creating target |
| Target source-disjointness | PASS | max overlap 0 across 802 target files |
| Execution completeness | PASS | 243/243 job records SUCCESS |
| Branch validity | PASS | raw, DNA and DP each pass Prior and Zero controls |
| Primary RQ1 DNA advantage | NOT REJECTED | 44/81, p=0.2526 |
| Practical p1=0.65 target | NOT MET | CI upper bound 0.6544 is below/at p1=0.65 boundary and point estimate 0.5432 |
| DP formal-privacy wording | WEAK ONLY | epsilon approximately 8.019e6, no record-level DP claim |

## 11. Artifact/run ID

```text
artifacts/rq1/priority2_clean_freeze_20260916
artifacts/rq1/priority2_clean_confirmatory_run_20260916
results/rq1/priority2_clean_confirmatory_20260916
```

## 12. Amendment

```text
protocols/amendments/2026-09-16_priority2_rq1_clean_confirmatory_design.md
protocols/amendments/2026-09-16_priority2_rq1_clean_confirmatory_pair_freeze.md
```

## 13. Bước tiếp theo được phép

This single Priority 2 clean confirmatory execution has been consumed.  Do not
redraw targets, rerun the same comparison, alter the threshold or tune
parameters because of this result.

Allowed next steps are reporting, immutable artifact verification, or moving to
the next supervisor-approved priority under a new pre-run amendment/protocol.
