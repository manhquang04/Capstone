# Strong update-DP exploratory study — Work 1: inverse accounting

**Date:** 2026-09-16  
**Status:** COMPLETE — development/exploratory only

## Đã làm gì

The existing utility-matched extension (`noise_multiplier=0.00001`) has
epsilon `5.000e9` at delta `1e-5`, add/remove, one release. It is therefore
weaker than the existing distortion-matched point (`8.019e6`), despite its
negligible utility effect. Comparable utility is not meaningful formal
privacy here.

Inverted the frozen Priority-3 Gaussian RDP accountant for epsilon 100, 50,
10, and 1. The implementation clips a whole client update rather than
individual record gradients; these are update-level diagnostics, not
transaction-level DP claims.

## Convention and command

Primary convention: one release; delta `1e-5`; add/remove adjacency;
sensitivity ratio 1; L2 clip norm `C=100`. Replace-one (ratio 2) is reported
only as a sensitivity analysis. The selected multiplier is the smallest value
whose calculated epsilon is at most the predeclared target.

```bash
PYTHONPATH=. .venv-phase1/bin/python -m py_compile experiments/dp_strong_budget_inverse.py
PYTHONPATH=. .venv-phase1/bin/python experiments/dp_strong_budget_inverse.py \
  --output-dir artifacts/dp_accounting/strong_update_dp_inverse_20260916
```

## Kết quả

| Target epsilon | Noise multiplier | Noise SD | Attained epsilon | Replace-one epsilon |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.098664 | 9.866 | 100.000 | 302.734 |
| 50 | 0.158903 | 15.890 | 50.000 | 139.604 |
| 10 | 0.567897 | 56.790 | 10.000 | 23.101 |
| 1 | 4.900556 | 490.056 | 1.000 | 2.042 |

The landmarks span public Opacus examples at epsilon 50 and lower values 5/7.5,
and its guidance to target a small constant epsilon; they are illustrative,
not universal deployment thresholds. [Opacus privacy engine](https://opacus.ai/api/privacy_engine.html), [Opacus tutorial](https://opacus.ai/tutorials/building_image_classifier), [Opacus FAQ](https://opacus.ai/docs/faq).

## Artifact/run ID and gate

- Output: `artifacts/dp_accounting/strong_update_dp_inverse_20260916/`
- JSON SHA-256: `f9b0183c26227884feff0ac3907fbddc2f9293fca5d6fb000c1f6f4514517a18`
- Gate: PASS — every inverse result reaches its predeclared add/remove target.
