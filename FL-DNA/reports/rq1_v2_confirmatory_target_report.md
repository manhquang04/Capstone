# RQ1-v2 confirmatory target report

**Date:** 2026-09-16  
**Status:** VIỆC 2 COMPLETE — TARGET FROZEN AND SOURCE-DISJOINT  
**Scope:** target generation and machine-readable config for RQ1-v2
confirmatory execution.

## 1. Đã làm gì

Created exactly one RQ1-v2 confirmatory target set:

```text
artifacts/rq1_v2/confirmatory_freeze_20260916/rq1_v2_confirmatory_targets.pt
```

Target design:

```text
groups = 176
records_per_group = 4
fraud_records_per_group = 1
generation_seed = 2026091676
```

Created frozen config:

```text
protocols/config/rq1_v2_confirmatory.json
```

## 2. Source-disjointness

Independent verifier result:

```text
confirmatory_groups = 176
confirmatory_source_count = 704
historical_target_files_checked = 1047
load_failures = {}
max_overlap = 0
disjointness_gate = PASS
```

The checked files include the original v2 n=8 development target and the v2
n=24 expanded development target.

## 3. Lệnh đã chạy

Initial attempt failed before writing a target because the output directory did
not exist. Then the directory was created and the same seed/command was rerun.

```bash
mkdir -p artifacts/rq1_v2/confirmatory_freeze_20260916

PYTHONPATH=. .venv-phase1/bin/python \
  experiments/create_phase4_source_disjoint_targets.py \
  artifacts/rq1_v2/confirmatory_freeze_20260916 \
  --output-name rq1_v2_confirmatory_targets.pt \
  --groups 176 \
  --records-per-group 4 \
  --fraud-per-group 1 \
  --seed 2026091676 \
  --purpose rq1_v2_confirmatory_single_frozen_draw_p1_0p60

PYTHONPATH=. .venv-phase1/bin/python \
  experiments/verify_rq1_target_disjointness.py \
  --confirmatory-target artifacts/rq1_v2/confirmatory_freeze_20260916/rq1_v2_confirmatory_targets.pt \
  --output artifacts/rq1_v2/confirmatory_freeze_20260916/source_overlap_matrix.json
```

## 4. Artifact IDs and hashes

| Artifact | SHA-256 |
| --- | --- |
| target | `87153ed0550254e92bfd0ea2eacbaec840b31d887d421e4513001224976b064f` |
| target provenance | `514e3832846adf37b7966f3127712517851d3a69684824e60cc6de42704dd973` |
| source-overlap matrix | `4a08fbc49f663ca65cb0c38b4c593677a0775e56a68859ff7eee79abffe86d77` |

## 5. Gate đạt/chưa đạt

| Gate | Status | Reason |
| --- | --- | --- |
| Target created once after protocol freeze | PASS | one successful target file with fixed seed |
| Source-disjointness | PASS | max overlap 0 across 1047 historical target files |
| Fraud composition | PASS | every group has exactly one fraud row |
| Frozen config created | PASS | `rq1_v2_confirmatory.json` references target hash and protocol |
| Confirmatory attack run | NOT YET | authorized next, but not part of Việc 2 |

## 6. Bước tiếp theo

Proceed to Việc 3: run exactly one RQ1-v2 confirmatory execution using:

```text
protocols/config/rq1_v2_confirmatory.json
```

Do not regenerate target or modify the config after seeing results.
