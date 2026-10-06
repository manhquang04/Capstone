# P34D report and audit annex

Written before confirmatory image access or statistical analysis. Primary family
remains fixed76, combined211=135+76. Use images PSNR and BN standardized MSE;
both directions, no new tests for SSIM/MSE or heterogeneity. Exact binomial sign
implementation independently checked by SciPy; Holm independently recomputed by
NumPy sorted/cumulative-max. Finite complete39pairs per cell,117 only if all
three eligible strata complete. One-based median ranks13/27 for39; exact
binomial ranks for117. No inference from missing/invalid/gated outputs.

Independent image scorer reloads raw truth/reconstruction arrays, checks IDs
and finite values, recomputes Hungarian MSE assignment BEFORE metric-only
pixel clipping (as original P33C), then independently computes PSNR,SSIM,MSE
and batch means. Qualification controls and BN payload decoder/calibration
must be reloaded independently before BN results accepted. Mean/std are from
immutable P32 training population. Validate every source/input/output receipt.

Report all model-specific and pooled medians, paired effects, raw/Holm76/Holm211
p, sigmas/clips/one-release accounting and calibration brackets. Independent
analytic Gaussian RDP upper bound for utility comparator uses original P31
single-release sensitivity convention: rho=r^2/(2sigma^2), epsilon=rho+
2sqrt(rho log(1/delta)), delta1e-5, r1 single or sqrt(L) per-tensor. This is
not a record-level or whole-training privacy guarantee and does not replace
the local epsilon10 mechanisms of P34B/C. Heterogeneity: effect medians/range
and exploratory Kruskal-Wallis distribution comparison, explicitly descriptive
and outside confirmatory families, no new superiority claim. All-identical
effects have no defined exploratory test, reported as such, never adjusted.

Before final claim require independently checked three stage freezes, all
scheduled outputs and complete39pairs, full immutable manifest, tests/compile/
diff and no-live audit. First report verification may need integration repairs;
preserve errors/amendment and use additive wrapper, never modify frozen science.
COMPLETE must not be written by incomplete audits. PROJECT snapshot then new
administrative note/seal, old evidence untouched. Paused P34B Goal unchanged.
