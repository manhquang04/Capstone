# Class Feature Spread Diagnostic

This diagnostic measures whether non-fraud records are statistically tighter than fraud records under the same scaled feature representation used by the attack.

| Scope | Class | n | Mean centroid MSE | Median centroid MSE | Mean pairwise MSE | Mean feature variance |
|---|---|---:|---:|---:|---:|---:|
| train_reference_500k | non_fraud | 324581 | 100.779240 | 8.393242 | 299.863343 | 100.779240 |
| train_reference_500k | fraud | 419 | 4038.719079 | 1325.761277 | 8096.762172 | 4038.719079 |
| run_20260908T180405538992Z | non_fraud | 36 | 30.056516 | 4.235694 | 61.830547 | 30.056516 |
| run_20260908T180405538992Z | fraud | 12 | 2751.483347 | 932.430985 | 6003.236394 | 2751.483347 |
| scale_dev_20260908T191737994986Z | non_fraud | 144 | 76.543792 | 4.019673 | 154.158127 | 76.543792 |
| scale_dev_20260908T191737994986Z | fraud | 48 | 1055.848379 | 346.621365 | 2156.626476 | 1055.848379 |
| class_decomp_20260908T200353687099Z | non_fraud | 144 | 98.094265 | 7.236811 | 197.560478 | 98.094265 |
| class_decomp_20260908T200353687099Z | fraud | 48 | 3538.482891 | 1205.339318 | 7227.539522 | 3538.482891 |

Fraud/non-fraud ratios:

| Scope | Mean centroid ratio | Median centroid ratio | Mean pairwise ratio | Mean variance ratio |
|---|---:|---:|---:|---:|
| train_reference_500k | 40.074911 | 157.955794 | 27.001507 | 40.074911 |
| run_20260908T180405538992Z | 91.543655 | 220.136513 | 97.091756 | 91.543655 |
| scale_dev_20260908T191737994986Z | 13.794043 | 86.231225 | 13.989703 | 13.794043 |
| class_decomp_20260908T200353687099Z | 36.072271 | 166.556684 | 36.583934 | 36.072271 |

Interpretation: lower centroid and pairwise MSE means records in that class are closer to their class-level typical point. A low non-fraud spread makes an unoptimized prior more competitive.
