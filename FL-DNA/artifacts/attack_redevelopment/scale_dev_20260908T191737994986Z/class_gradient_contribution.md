# Class Gradient Contribution Diagnostic

Run: `/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/artifacts/attack_redevelopment/scale_dev_20260908T191737994986Z`

This diagnostic computes fraud and non-fraud gradient contributions on the locked confirmation targets. It does not run inversion and does not change attack gates.

| Mode | Groups | Fraud weighted norm > non-fraud | Mean weighted ratio | Median weighted ratio | Mean fraud cosine/full | Mean non-fraud cosine/full |
|---|---:|---:|---:|---:|---:|---:|
| eval | 12 | 12/12 | 30.172757 | 30.542365 | 0.997142 | -0.380780 |
| train | 12 | 12/12 | 138.640140 | 106.837831 | 0.039169 | -0.171198 |
| train_shared_bn | 12 | 12/12 | 124.319230 | 105.290608 | 0.999963 | 0.310647 |

Weighted ratio uses class-gradient norm multiplied by class count / group size, matching mean-reduction loss scale. A ratio above 1 indicates fraud contributes a larger weighted gradient norm than non-fraud despite fewer records.
`train_shared_bn` is the closest diagnostic to the attack simulator: it uses one full train-mode forward pass and splits the per-sample loss on the same BatchNorm/dropout graph.
