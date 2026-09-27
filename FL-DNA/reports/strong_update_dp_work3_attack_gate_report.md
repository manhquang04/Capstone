# Strong update-DP exploratory study — Work 3: inversion resistance

**Date:** 2026-09-16  
**Status:** COMPLETE — branch gates FAIL; resistance inference unresolved

## Đã làm gì / lệnh thực sự đã chạy

Created a new eight-group, four-record/one-fraud development target. A source-ID-only verifier checked 1,577 prior target artifacts, maximum overlap zero. Then ran raw plus the clipping/noise Monte-Carlo attacker for each frozen multiplier (8 restarts, 600 iterations, 100 MC samples, fixed defense seed); all 40/40 jobs succeeded.

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_strong_update_dp_attack.py \
  --target artifacts/strong_update_dp/attack_dev_final_20260916/strong_dp_development_targets.pt \
  --output-dir artifacts/strong_update_dp/attack_dev_final_20260916/run --workers 9
PYTHONPATH=. .venv-phase1/bin/python experiments/analyze_strong_update_dp_attack.py \
  --run-dir artifacts/strong_update_dp/attack_dev_final_20260916/run \
  --output-dir results/strong_update_dp/attack_20260916
```

## Gate and feature-MSE

| Branch | Prior wins / 8 (p) | Zero wins / 8 (p) | Branch gate | Mean selected feature-MSE |
| --- | --- | --- | --- | ---: |
| Raw | 6 (0.1445) | 7 (0.0352) | FAIL | 86.30 |
| DP epsilon=100 | 1 (0.9961) | 2 (0.9648) | FAIL | 1086.22 |
| DP epsilon=50 | 2 (0.9648) | 2 (0.9648) | FAIL | 1280.42 |
| DP epsilon=10 | 2 (0.9648) | 1 (0.9961) | FAIL | 1857.92 |
| DP epsilon=1 | 1 (0.9961) | 2 (0.9648) | FAIL | 3724.93 |

Raw failed the Prior control, so no valid attacker gate was established. The increasing feature-MSE values are descriptive diagnostics only, not a validated reconstruction-resistance curve or a DP privacy claim.

## Artifact/run ID and gate

- Target SHA-256: `dc2d43901c0ae14447e42b97a62ff0660870c31acb50100621230519fc46ca89`
- Disjointness SHA-256: `b1cacda65416efe53c086b1e78f5c969ddf5261ee0b33b67833e88ce51085e59`
- Execution SHA-256: `16e664a7c8dcd8afbd5289e9835693d3f165787929447f196710a586bc185a86`
- Analysis SHA-256: `c609721dd3d57ad53a30712dd99800e97123bd51fa909cd3c00f0536f4105d23`
- Gate: FAIL due to raw Prior; no amendment or tuning follows from this result.
