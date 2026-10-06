# Attack Failure Pattern Diagnostics

Run: `/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/artifacts/attack_redevelopment/scale_dev_20260908T191737994986Z`

## Class Split

| Class | Control | Wins | Mean delta | Median delta |
|---|---|---:|---:|---:|
| non_fraud | prior | 0/12 | 100.409617 | 81.648257 |
| non_fraud | zero | 2/12 | 74.163816 | 47.520288 |
| fraud | prior | 11/12 | -729.984863 | -189.804291 |
| fraud | zero | 12/12 | -869.143462 | -292.470469 |

Negative delta means the baseline attack has lower reconstruction MSE than the control.

## Top Tensor Terms

| Tensor | Kind | Mean term | Mean cosine loss | Mean relative MSE |
|---|---|---:|---:|---:|
| `network.0.bias` | parameter | 1.192169 | 1.023003 | 1.691660 |
| `network.4.bias` | parameter | 1.154008 | 0.977549 | 1.764590 |
| `network.0.weight` | parameter | 1.127237 | 0.933180 | 1.940574 |
| `network.8.bias` | parameter | 1.073527 | 0.883622 | 1.899049 |
| `network.4.weight` | parameter | 1.053312 | 0.879291 | 1.740208 |
| `network.8.weight` | parameter | 0.964211 | 0.808962 | 1.552494 |
| `network.1.bias` | parameter | 0.921681 | 0.770901 | 1.507803 |
| `network.1.weight` | parameter | 0.907684 | 0.759259 | 1.484247 |
| `network.5.weight` | parameter | 0.771525 | 0.641433 | 1.300917 |
| `network.5.bias` | parameter | 0.733581 | 0.609865 | 1.237155 |

Interpretation: these tensor terms describe where the selected reconstruction still fails to match the observed update. They are diagnostic only; they do not alter the locked confirmation result.
