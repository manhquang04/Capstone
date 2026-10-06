# Class Gradient Contribution Diagnostic

Run: `/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/artifacts/attack_redevelopment/run_20260908T180405538992Z`

This diagnostic computes fraud and non-fraud gradient contributions on the locked confirmation targets. It does not run inversion and does not change attack gates.

| Mode | Groups | Fraud weighted norm > non-fraud | Mean weighted ratio | Median weighted ratio | Mean fraud cosine/full | Mean non-fraud cosine/full |
|---|---:|---:|---:|---:|---:|---:|
| eval | 12 | 10/12 | 35.791337 | 26.700656 | 0.830354 | 0.092965 |
| train_shared_bn | 12 | 12/12 | 118.325667 | 118.498655 | 0.999658 | 0.394465 |

Weighted ratio uses class-gradient norm multiplied by class count / group size, matching mean-reduction loss scale. A ratio above 1 indicates fraud contributes a larger weighted gradient norm than non-fraud despite fewer records.
`train_shared_bn` is the closest diagnostic to the attack simulator: it uses one full train-mode forward pass and splits the per-sample loss on the same BatchNorm/dropout graph.
