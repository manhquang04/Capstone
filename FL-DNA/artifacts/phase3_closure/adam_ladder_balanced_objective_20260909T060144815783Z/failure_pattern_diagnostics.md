# Attack Failure Pattern Diagnostics

Run: `/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/artifacts/phase3_closure/adam_ladder_balanced_objective_20260909T060144815783Z`

## Class Split

| Class | Control | Wins | Mean delta | Median delta |
|---|---|---:|---:|---:|
| non_fraud | prior | 5/10 | -2.385704 | -0.437396 |
| non_fraud | zero | 5/10 | 0.361362 | 0.173888 |
| fraud | prior | 9/10 | -43.432379 | -12.966417 |
| fraud | zero | 8/10 | -51.711708 | -22.042208 |

Negative delta means the baseline attack has lower reconstruction MSE than the control.

## Top Tensor Terms

| Tensor | Kind | Mean term | Mean cosine loss | Mean relative MSE |
|---|---|---:|---:|---:|
| `network.0.bias` | parameter | 14.009899 | 1.034100 | 129.757989 |
| `network.4.bias` | parameter | 1.386426 | 1.049623 | 3.368031 |
| `network.0.weight` | parameter | 1.231956 | 0.998539 | 2.334173 |
| `network.1.weight` | parameter | 1.195896 | 0.993353 | 2.025429 |
| `network.4.weight` | parameter | 1.195226 | 0.991476 | 2.037504 |
| `network.5.bias` | parameter | 1.174044 | 0.977179 | 1.968643 |
| `network.1.bias` | parameter | 1.170086 | 0.971839 | 1.982472 |
| `network.5.weight` | parameter | 1.143320 | 0.951726 | 1.915940 |
| `network.8.bias` | parameter | 1.111343 | 0.908911 | 2.024319 |
| `network.8.weight` | parameter | 1.075999 | 0.894577 | 1.814212 |

Interpretation: these tensor terms describe where the selected reconstruction still fails to match the observed update. They are diagnostic only; they do not alter the locked confirmation result.
