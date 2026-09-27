# RQ2 DNA Transform v2 — pre-frozen pilot report

**Date:** 2026-09-16  
**Status:** COMPLETE — confirmatory execution NOT AUTHORIZED

## Đã làm gì

Ran the pre-frozen paired v2 pilot: Baseline and DNA Transform v2
(`compression_ratio=0.95`, `quantization_eta=0.01`) on eight source-disjoint
training seeds at the ordinary RQ2 scale (500,000 rows, 50 rounds, three
clients, one local epoch). All methods shared the same realization within each
seed.

## Lệnh thực sự đã chạy

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_rq2_v2_pilot.py \
  --output-dir artifacts/rq2_v2/pilot_20260916 --workers 8
PYTHONPATH=. .venv-phase1/bin/python experiments/analyze_rq2_v2_pilot.py \
  --run-dir artifacts/rq2_v2/pilot_20260916 \
  --output-dir results/rq2_v2/pilot_20260916
```

## Kết quả kèm uncertainty

The frozen analysis used one-sided paired non-inferiority power with alpha
0.05, power 0.80, the approved margins (F1 0.02; AUC 0.005), and a one-sided
95% chi-square upper bound for paired variance.

| Endpoint | Mean paired Δ(v2 − Baseline) | Pilot SD | 95% descriptive CI | Upper variance bound | Required n |
| --- | ---: | ---: | ---: | ---: | ---: |
| F1 | -0.01185 | 0.03181 | [-0.03845, 0.01474] | 0.0032680 | 52 |
| AUC-ROC | -0.00208 | 0.00486 | [-0.00614, 0.00198] | 0.00007616 | 21 |

Because both endpoints are mandatory, the required confirmatory size is 52.
This exceeds the pre-frozen maximum feasible seed count of 30.

## Gate đạt/chưa đạt

| Gate | Status | Reason |
| --- | --- | --- |
| Pilot execution | PASS | 16/16 Baseline/v2 jobs completed |
| Paired seed contract | PASS | all runs have a matching seed and final round 50 |
| Required n within cap | FAIL | F1 requires 52 > 30 |
| Confirmatory creation/execution | NOT AUTHORIZED | protocol forbids raising cap or changing margins after pilot |

## Artifact/run ID

- Amendment: `protocols/amendments/2026-09-16_rq2_v2_pilot_and_confirmatory.md`
- Pilot: `artifacts/rq2_v2/pilot_20260916/`
- Analysis: `results/rq2_v2/pilot_20260916/`

## Conclusion and next permitted step

This pilot does not establish or refute non-inferiority. Under the frozen
design it is **underpowered/precision-limited** for a v2 confirmatory claim.
No new v2 confirmatory seeds may be generated and no margin, endpoint rule, or
cap may be changed based on this result. A future study would require a new
supervisor-approved protocol amendment, written before any new pilot or
confirmatory run.
