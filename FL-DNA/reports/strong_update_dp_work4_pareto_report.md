# Strong update-DP exploratory study — Work 4: three-axis comparison

**Date:** 2026-09-16  
**Status:** COMPLETE — exploratory synthesis

Higher feature-MSE denotes worse reconstruction. `UNRESOLVED` is deliberate: the new raw attacker failed its Prior gate, so the new attack-axis values cannot scientifically rank reconstruction resistance.

| Method / point | Update-level epsilon, delta=1e-5 | Utility ΔF1 | Reconstruction resistance (feature-MSE) |
| --- | ---: | ---: | --- |
| DNA v1 conservative | ~8.019e6 (weak) | ~0, historical | historical RQ1 evidence; not re-ranked here |
| DNA v2 | ~4.581e5 add/remove / ~1.823e6 replace-one (weak) | ~0, historical smoke | historical v2 evidence; not re-ranked here |
| DP utility-matched (`sigma=0.00001`) | ~5.000e9 (very weak) | ~0, historical | no new attack measurement |
| DP epsilon=100 (`sigma=0.098664`) | 100 add/remove | -0.4706 | UNRESOLVED; descriptive mean 1086.22 |
| DP epsilon=50 (`sigma=0.158903`) | 50 add/remove | -0.4714 | UNRESOLVED; descriptive mean 1280.42 |
| DP epsilon=10 (`sigma=0.567897`) | 10 add/remove | -0.4711 | UNRESOLVED; descriptive mean 1857.92 |
| DP epsilon=1 (`sigma=4.900556`) | 1 add/remove | -0.4716 | UNRESOLVED; descriptive mean 3724.93 |

## Conclusion

The missing utility/accounting context is now explicit: at the project's whole-update Gaussian convention, epsilon values 1–100 require noise orders of magnitude above prior matching points and cause severe development-scale F1/AUC loss. The experiment does not establish an attack comparison because the raw branch gate failed. No claim that DNA is superior or inferior to strong DP follows.

All epsilon values are update-level diagnostics, not transaction-level DP, because clipping is not per-example. No target, DNA/model parameter, or `torch.set_num_threads(1)` was modified after outcome inspection.

## Next permitted step

Any attempt to establish a valid strong-DP attack curve needs a separately approved development amendment addressing the raw Prior-gate failure. This target cannot become confirmatory and cannot be used to tune against outcomes.
