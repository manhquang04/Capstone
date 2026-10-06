# Class Gradient Contribution Diagnostic

Run: `/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/artifacts/phase3_closure/adam_ladder_balanced_objective_20260909T060144815783Z`

This diagnostic computes fraud and non-fraud gradient contributions on the locked confirmation targets. It does not run inversion and does not change attack gates.

| Mode | Groups | Fraud weighted norm > non-fraud | Mean weighted ratio | Median weighted ratio | Mean fraud cosine/full | Mean non-fraud cosine/full |
|---|---:|---:|---:|---:|---:|---:|
| eval | 10 | 8/10 | 22.204870 | 22.356134 | 0.766478 | 0.182282 |
| train_shared_bn | 10 | 10/10 | 127.325945 | 114.022751 | 0.999962 | 0.313449 |

Weighted ratio uses class-gradient norm multiplied by class count / group size, matching mean-reduction loss scale. A ratio above 1 indicates fraud contributes a larger weighted gradient norm than non-fraud despite fewer records.
`train_shared_bn` is the closest diagnostic to the attack simulator: it uses one full train-mode forward pass and splits the per-sample loss on the same BatchNorm/dropout graph.
