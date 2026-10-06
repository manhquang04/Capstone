# Phase 2 metric specification

Version: paysim_tabular_v1+joint_range_pseudo_image_v2. Implementations: attacks/tabular_metrics.py and attacks/inversion_metrics.py.

Feature MSE/MAE average squared/absolute differences across numeric and categorical coordinates. Per-feature normalized absolute error is also retained. Numeric inverse transform uses the training RobustScaler center and scale: raw = scaled * scale + center. Raw absolute error is reported per feature; an average mixing steps and monetary units is diagnostic only.

Transaction type uses argmax over type= columns (first maximum wins ties). Accuracy compares decoded types. Version strict_onehot_v2 requires every coordinate to be within 1e-6 of zero or one (rtol=0), and exactly one coordinate near one. Soft simplex [0.5, 0.5] fails. Prior v1 used only bounds/sum and is retained in old artifacts.

Balance consistency is evaluated after inverse scaling: orig difference = oldbalanceOrg - newbalanceOrig; destination difference = newbalanceDest - oldbalanceDest. Report absolute discrepancy against reconstructed derived features. It is not an optimizer constraint.

Pseudo-image: flatten, zero-pad to ceil(sqrt(d)) squared, reshape. Both arrays use their joint min/max; padding participates. PSNR uses image MSE and range 1, capped at 99 for error <=1e-12. SSIM is the existing global-statistics approximation, not sliding-window image SSIM. Constant unequal pairs no longer collapse to identical images. These are secondary metrics and are not comparable with earlier reference-only normalization artifacts.

Cosine and Pearson use vector dot products and centered dot products respectively. Existing implementation returns zero for zero denominators; treat these as undefined mathematically, not evidence of orthogonality. Nonfinite values are rejected. Equal flattened length is checked; original multidimensional shape semantics are not enforced by this evaluator.

Tests cover exact identity, unequal constants, nonfinite/length errors, categorical range violations, derived-feature errors, scaler-length errors, soft simplex rejection, source-row replay and attack best/history consistency. Dataset checksum and disjoint source target IDs are saved in the verified run. Undefined cosine/Pearson legacy zero convention remains explicitly documented above; do not interpret zero-denominator cases scientifically.
