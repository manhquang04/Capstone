# Class Gradient Contribution Diagnostic

Run: `/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z`

This diagnostic computes fraud and non-fraud gradient contributions on the locked confirmation targets. It does not run inversion and does not change attack gates.

| Mode | Groups | Fraud weighted norm > non-fraud | Mean weighted ratio | Median weighted ratio | Mean fraud cosine/full | Mean non-fraud cosine/full |
|---|---:|---:|---:|---:|---:|---:|
| eval | 12 | 8/12 | 22.981437 | 17.924532 | 0.666452 | 0.163974 |
| train_shared_bn | 12 | 12/12 | 148.010976 | 121.111405 | 0.999958 | 0.425767 |

Weighted ratio uses class-gradient norm multiplied by class count / group size, matching mean-reduction loss scale. A ratio above 1 indicates fraud contributes a larger weighted gradient norm than non-fraud despite fewer records.
`train_shared_bn` is the closest diagnostic to the attack simulator: it uses one full train-mode forward pass and splits the per-sample loss on the same BatchNorm/dropout graph.
