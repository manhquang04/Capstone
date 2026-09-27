# Strong update-DP exploratory study — Work 2: utility

**Date:** 2026-09-16  
**Status:** COMPLETE — descriptive development result, not confirmatory

## Đã làm gì / lệnh thực sự đã chạy

Ran paired Baseline plus all four frozen DP points on seeds `[101,202,303]`,
`MAX_ROWS=100000`, 3 clients, one local epoch, and 10 rounds.

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_strong_update_dp_utility.py \
  --output-dir artifacts/strong_update_dp/utility_20260916 --workers 9
PYTHONPATH=. .venv-phase1/bin/python experiments/analyze_strong_update_dp_utility.py \
  --run-dir artifacts/strong_update_dp/utility_20260916 \
  --output-dir results/strong_update_dp/utility_20260916
```

All 15/15 jobs succeeded; the pipeline retains `torch.set_num_threads(1)`.

## Kết quả kèm uncertainty

| Target epsilon | Mean paired ΔF1 | Descriptive 95% normal CI | Mean paired ΔAUC |
| ---: | ---: | ---: | ---: |
| 100 | -0.4706 | [-0.6810, -0.2602] | -0.3656 |
| 50 | -0.4714 | [-0.6830, -0.2599] | -0.5492 |
| 10 | -0.4711 | [-0.6823, -0.2599] | -0.4138 |
| 1 | -0.4716 | [-0.6832, -0.2600] | -0.5623 |

These intervals are descriptive only (n=3). All predeclared levels show severe
utility loss; none was dropped, expanded, or retuned after the result.

## Artifact/run ID and gate

- Run: `artifacts/strong_update_dp/utility_20260916/`
- Manifest SHA-256: `929e4467cd5c01fe322a9ccdb7491f96870a9b2bafd5aa5c53a2eb807e1190ff`
- Execution summary SHA-256: `23d05ad1e5ae7dbc69579d3ef3f730520098a825ba6c3a2e13fae1df0d222af7`
- Analysis SHA-256: `4cc77bdb172354fcc627e21e8e28f3035ea5db4fef7b7305970324e8edf2c4d8`
- Execution gate: PASS (15/15 success). No utility-pass gate was defined.
