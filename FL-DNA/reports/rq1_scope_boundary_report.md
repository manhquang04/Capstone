# RQ1 scope-boundary screening report

**Execution date:** 2026-09-14  
**Status:** `COMPLETE — STOPPED AT FIRST FAILING LEVEL`  
**Scope:** raw-branch coarse screening for attacker validity under larger
`records_per_group`; no DNA/DP confirmatory escalation was run.

## 1. Đã làm gì

1. Wrote the pre-run amendment locking the coarse screening protocol.
2. Created one source-disjoint development target set for
   `records_per_group = 8`, with `10` groups and
   `fraud_records_per_group = 1`.
3. Verified source-ID disjointness against all existing target artifacts using
   source identifiers only.
4. Ran the raw branch only with the reduced, pre-locked budget of `4` restarts
   per group.
5. Analyzed the prior and zero-update control gates.
6. Stopped immediately after `records_per_group = 8` failed, as required by the
   amendment.  No `records_per_group = 12` or `16` target was created or run.

## 2. File/config thay đổi

- `protocols/amendments/2026-09-14_rq1_scope_boundary_coarse_screening.md`
- `experiments/run_rq1_raw_scope_screening.py`
- `experiments/analyze_rq1_raw_scope_screening.py`
- `protocols/config/rq1_scope_boundary_screen_rpg8.json`

## 3. Lệnh thực sự đã chạy

```bash
python3 -m py_compile \
  experiments/run_rq1_raw_scope_screening.py \
  experiments/analyze_rq1_raw_scope_screening.py

PYTHONPATH=. .venv-phase1/bin/python \
  experiments/create_phase4_source_disjoint_targets.py \
  artifacts/rq1/scope_boundary_rpg8_freeze_20260914 \
  --output-name rq1_scope_boundary_rpg8_targets.pt \
  --groups 10 \
  --records-per-group 8 \
  --fraud-per-group 1 \
  --seed 1940948331 \
  --purpose rq1_scope_boundary_coarse_screening_rpg8

PYTHONPATH=. .venv-phase1/bin/python \
  experiments/verify_rq1_target_disjointness.py \
  --confirmatory-target artifacts/rq1/scope_boundary_rpg8_freeze_20260914/rq1_scope_boundary_rpg8_targets.pt \
  --output artifacts/rq1/scope_boundary_rpg8_freeze_20260914/source_overlap_matrix.json

PYTHONPATH=. .venv-phase1/bin/python \
  experiments/run_rq1_raw_scope_screening.py \
  --config protocols/config/rq1_scope_boundary_screen_rpg8.json \
  --output-dir artifacts/rq1/scope_boundary_rpg8_raw_screen_20260914 \
  --workers 9

PYTHONPATH=. .venv-phase1/bin/python \
  experiments/analyze_rq1_raw_scope_screening.py \
  --config protocols/config/rq1_scope_boundary_screen_rpg8.json \
  --run-dir artifacts/rq1/scope_boundary_rpg8_raw_screen_20260914 \
  --output-dir results/rq1/scope_boundary_rpg8_raw_screen_20260914
```

## 4. Kết quả từng mức kèm gate pass/fail

Gate rule was locked as: mean difference `< 0`, median difference `< 0`, and
one-sided exact sign-test `p < 0.05` for both prior and zero-update controls.

| records_per_group | groups | raw jobs | Prior wins / n | Prior p | Prior gate | Zero wins / n | Zero p | Zero gate | Decision |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| 8 | 10 | 10/10 success | 7/10 | 0.171875 | FAIL | 8/10 | 0.0546875 | FAIL | STOP |
| 12 | — | not run | — | — | — | — | — | — | blocked by stop rule |
| 16 | — | not run | — | — | — | — | — | — | blocked by stop rule |

For `records_per_group = 8`, the differences were directionally favorable on
mean and median, but neither exact sign-test crossed the locked alpha threshold:

- Prior control mean difference `-1100.4133`, median `-42.0448`, 95% t-CI for
  the mean `[-2314.5035, 113.6769]`.
- Zero-update control mean difference `-823.7900`, median `-168.1036`, 95% t-CI
  for the mean `[-1831.3023, 183.7223]`.

## 5. Ranh giới scope kết luận được

The current RQ1 raw attacker/evaluation method is confirmed by prior work at
the `4 records / 1 fraud / 1 local step` scope, but this coarse screening did
not establish validity at `8 records / 1 fraud / 1 local step` under the locked
raw-branch gate.

Therefore the supported boundary finding for this amendment is:

```text
confirmed scope = 4 records/group;
first failing screened scope = 8 records/group;
boundary: 4 < unsupported/failing transition <= 8 records/group
```

This is a valid negative result for the current method.  It is not evidence
that no improved attacker could work at 8 records/group; it only states that
the frozen current attacker/evaluation setup did not pass its raw validity gate
when non-fraud rows were increased from 3 to 7 per group.

## 6. Artifact/run ID

- Amendment: `protocols/amendments/2026-09-14_rq1_scope_boundary_coarse_screening.md`
- Config: `protocols/config/rq1_scope_boundary_screen_rpg8.json`
- Target freeze: `artifacts/rq1/scope_boundary_rpg8_freeze_20260914/`
- Raw run: `artifacts/rq1/scope_boundary_rpg8_raw_screen_20260914/`
- Results: `results/rq1/scope_boundary_rpg8_raw_screen_20260914/`

Important hashes:

| Artifact | SHA-256 |
| --- | --- |
| amendment | `e4c34d54bc02661cb0f5b4aae463973fb780ed9c9789f2d4cacfb427d5f9c676` |
| config | `521b963e5a420f5bc8c069aecab6d4dffff3b000c134370f86170d1b2c33c76a` |
| target | `6653f4d904f74cc46d95abf89eeab80995f8687afb8d8dfd199a7116e1f568a0` |
| source-overlap matrix | `30efe1230b20604d0e65fa028ffae7d8f59fa99ae0fdea1b256026fd6b1c9262` |
| execution manifest | `15f727143dfdc737c0a43209d81a62974d946a5663c7d0cd58a18cc6f3abd4bd` |
| execution summary | `61411734d2883881a93a162ecc521c95d47ce99f5e75412c6449bb0b8bb1d00d` |
| analysis summary | `294e541d1e6efc5481aa44077f1409b1f68aefea6084e94f8507143937569bbe` |
| raw group summary CSV | `91ea6cf633f600d5a5f1715c9044a8f20e38b7549e3e79dbc7e17f6e8b243dd4` |

## 7. Gate đạt/chưa đạt

| Gate | Status | Reason |
| --- | --- | --- |
| Pre-run amendment | PASS | Created before target generation and before observing outcomes |
| Source disjointness | PASS | 80 source rows, zero overlap against 791 existing target files |
| Execution completeness | PASS | 10/10 raw jobs succeeded |
| Prior control gate | FAIL | 7/10 wins, one-sided p `0.171875` |
| Zero-update control gate | FAIL | 8/10 wins, one-sided p `0.0546875` |
| Raw branch validity at 8 rows | FAIL | Both controls must pass; neither crossed alpha `0.05` |
| Stop rule compliance | PASS | Levels 12 and 16 were not created or run |

## 8. Amendment

The amendment was written before creating the `records_per_group = 8` target
set and before seeing any screening result.  No attacker parameter was changed
after observing outcomes.  No post-hoc target outcomes were inspected; old
target artifacts were used only for source-ID overlap checking.

## 9. Bước tiếp theo được phép

Under this amendment, no further action is authorized: `records_per_group = 12`
and `16` must remain unrun because `8` failed first.

Any future attempt to handle larger groups would need a new supervisor-approved
amendment explicitly framed as attacker redevelopment or a different evaluation
method, not a continuation of this locked coarse-screening ladder.
