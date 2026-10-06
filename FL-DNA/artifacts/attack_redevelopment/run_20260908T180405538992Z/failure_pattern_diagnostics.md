# Attack Failure Pattern Diagnostics

Run: `/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/artifacts/attack_redevelopment/run_20260908T180405538992Z`

## Class Split

| Class | Control | Wins | Mean delta | Median delta |
|---|---|---:|---:|---:|
| non_fraud | prior | 8/12 | 23.313853 | -1.644455 |
| non_fraud | zero | 8/12 | -5.244579 | -35.450519 |
| fraud | prior | 10/12 | -1441.369387 | -483.239062 |
| fraud | zero | 11/12 | -1787.567590 | -596.438704 |

Negative delta means the baseline attack has lower reconstruction MSE than the control.

## Top Tensor Terms

| Tensor | Kind | Mean term | Mean cosine loss | Mean relative MSE |
|---|---|---:|---:|---:|
| `network.4.bias` | parameter | 1.320045 | 1.059616 | 2.604282 |
| `network.0.bias` | parameter | 1.293587 | 1.111693 | 1.818935 |
| `network.8.bias` | parameter | 1.216159 | 0.985476 | 2.306835 |
| `network.0.weight` | parameter | 1.213188 | 0.977783 | 2.354056 |
| `network.1.weight` | parameter | 1.014479 | 0.843446 | 1.710332 |
| `network.1.bias` | parameter | 1.013989 | 0.843010 | 1.709782 |
| `network.4.weight` | parameter | 1.005319 | 0.835504 | 1.698147 |
| `network.8.weight` | parameter | 0.906785 | 0.754869 | 1.519161 |
| `network.5.weight` | parameter | 0.674090 | 0.561602 | 1.124879 |
| `network.5.bias` | parameter | 0.650862 | 0.542343 | 1.085187 |

Interpretation: these tensor terms describe where the selected reconstruction still fails to match the observed update. They are diagnostic only; they do not alter the locked confirmation result.
