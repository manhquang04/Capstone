# RQ2 DNA Transform v2 confirmatory report

**Date:** 2026-09-16  
**Status:** COMPLETE — DNA Transform v2 establishes non-inferiority for both RQ2 endpoints

## Đã làm gì

Supervisor approved raising the RQ2-v2 `maximum_feasible_seed_count` from 30
to 60 because the prior machine-hour ceiling has been removed and the v2 pilot
showed higher paired F1 variance than v1. The scientific rule was unchanged:
F1 margin 0.02, AUC-ROC margin 0.005, alpha 0.05, power 0.80, and both
endpoints must pass.

Created a frozen RQ2-v2 confirmatory config with 52 deterministic seeds. The
seed list is source-disjoint from the 8 pilot seeds and prior RQ2 historical
seeds found in frozen RQ2 config files. Then ran the single authorized
confirmatory execution: Baseline and DNA Transform v2
(`compression_ratio=0.95`, `quantization_eta=0.01`) on 500,000 rows, 50 rounds,
3 clients, one local epoch, focal loss.

## Lệnh thực sự đã chạy

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_rq2_v2_confirmatory.py \
  --config protocols/config/rq2_v2_confirmatory.json \
  --output-dir artifacts/rq2_v2/confirmatory_20260916 \
  --workers 9

PYTHONPATH=. .venv-phase1/bin/python experiments/analyze_rq2_v2_confirmatory.py \
  --config protocols/config/rq2_v2_confirmatory.json \
  --run-dir artifacts/rq2_v2/confirmatory_20260916 \
  --output-dir results/rq2_v2/confirmatory_20260916
```

## Kết quả kèm uncertainty

All 104 jobs completed successfully: 52 paired Baseline jobs and 52 paired DNA
Transform v2 jobs. The confirmatory analysis used the frozen paired
non-inferiority rule.

| Endpoint | n | Mean paired Δ(v2 - Baseline) | SD | Median Δ | 95% CI | Margin | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| F1 | 52 | -0.01171 | 0.02800 | -0.00690 | [-0.01951, -0.00392] | -0.02000 | PASS |
| AUC-ROC | 52 | -0.00198 | 0.00740 | -0.00013 | [-0.00404, 0.00008] | -0.00500 | PASS |

Interpretation: v2 reduces mean F1 and AUC slightly relative to Baseline, but
the lower 95% confidence bounds remain within the pre-approved
non-inferiority margins for both endpoints.

## Gate đạt/chưa đạt

| Gate | Status | Reason |
| --- | --- | --- |
| Amendment before execution | PASS | cap increase and single-execution rule frozen before running confirmatory |
| Seed count within raised cap | PASS | required n=52 <= maximum feasible n=60 |
| Seed disjointness | PASS | 52 confirmatory seeds are disjoint from the 8 v2 pilot seeds |
| Execution completion | PASS | 104/104 jobs succeeded, 0 failed |
| F1 non-inferiority | PASS | lower 95% CI = -0.01951 >= -0.02 |
| AUC non-inferiority | PASS | lower 95% CI = -0.00404 >= -0.005 |
| Primary RQ2-v2 rule | PASS | both mandatory endpoints passed |

## Artifact/run ID

- Amendment: `protocols/amendments/2026-09-16_rq2_v2_seed_cap_increase_and_confirmatory.md`
- Config: `protocols/config/rq2_v2_confirmatory.json`
- Seed contract: `artifacts/rq2_v2/confirmatory_freeze_20260916/per_seed_contract.json`
- Confirmatory run: `artifacts/rq2_v2/confirmatory_20260916/`
- Analysis: `results/rq2_v2/confirmatory_20260916/`

Hashes recorded after analysis:

- `protocols/config/rq2_v2_confirmatory.json`: `da0ca7cebb8bd394ec2c44a593fb357757938df7695f6793f670aa77a0987224`
- `artifacts/rq2_v2/confirmatory_freeze_20260916/per_seed_contract.json`: `92fe693b81ed12021262aa1b98dea28e7bd9fe862fd12908e9919ad00744fd15`
- `artifacts/rq2_v2/confirmatory_20260916/execution_summary.json`: `aefbea99842877ed614a6b95f30663cdb72570f41c3ccf7a8d113ffd180f6cac`
- `results/rq2_v2/confirmatory_20260916/summary.json`: `02b1c33620ddfcc6a4f6bee11968245216578ec8692cb88c01c6d713a9276b2d`
- `experiments/run_rq2_v2_confirmatory.py`: `027d38ec164ed84163c2eac97391336f54fe1a1d1a6f10bd0a44dea7e9769c72`
- `experiments/analyze_rq2_v2_confirmatory.py`: `828ee1e17f3c5ede19287f49e3e9cacfad654207304fbb7891bd634378b1b80f`

## Conclusion and next permitted step

Under the supervisor-approved raised seed cap and unchanged RQ2
non-inferiority rule, DNA Transform v2 **establishes non-inferiority** against
Baseline for both F1 and AUC-ROC at n=52.

This does not change the separate RQ3-v2 result: the current v2 transport
implementation remains `NOT_ACCEPTABLE` under Bundle B because of overhead and
server decode/aggregate cost. No additional RQ2-v2 confirmatory rerun is
authorized without a new amendment written before any new execution.
