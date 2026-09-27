# Priority 5 — IEEE-CIS Fraud Detection progress

**Date:** 2026-09-16  
**Status:** Phase 5.1 COMPLETE; Phase 5.2 DEVELOPMENT BASELINE SMOKE COMPLETE; Phase 5.3 DEVELOPMENT GATE COMPLETE/FAILED  
**Scope:** dataset feasibility and small FL baseline only. No attack target was
created.

## Phase 5.1 — Feasibility survey

Dataset files present:

```text
datasets/ieee-fraud-detection/train_transaction.csv
datasets/ieee-fraud-detection/train_identity.csv
datasets/ieee-fraud-detection/test_transaction.csv
datasets/ieee-fraud-detection/test_identity.csv
```

Observed structure:

| File | Rows | Columns |
| --- | ---: | ---: |
| `train_transaction.csv` | 590,540 | 394 |
| `train_identity.csv` | 144,233 | 41 |
| `test_transaction.csv` | 506,691 | 393 |
| `test_identity.csv` | 141,907 | 41 |

Fraud label:

```text
train isFraud count = 20,663
train fraud rate = 0.0349900091
```

Identity coverage:

```text
transactions with identity row = 24.4239%
fraud rate with identity = 7.8470%
fraud rate without identity = 2.0939%
```

This means identity availability is itself correlated with fraud risk and must
be handled carefully. For the first baseline smoke, the pipeline intentionally
uses transaction-only features to avoid dropping ~75.6% of the training rows.

Candidate feature groups similar in role to PaySim:

- Amount/time/product: `TransactionAmt`, `TransactionDT`, `ProductCD`
- Card/payment: `card1`, `card2`, `card3`, `card4`, `card5`, `card6`
- Address/distance: `addr1`, `addr2`, `dist1`, `dist2`
- Email/domain: `P_emaildomain`, `R_emaildomain`
- Transaction count features: `C1`–`C14`
- Time-delta features: `D1`–`D15`
- Matching flags: `M1`–`M9`
- Optional identity/device features: `DeviceType`, `DeviceInfo`, selected `id_*`

Hard-constraint assessment:

```text
No explicit PaySim-like balance conservation columns were found.
```

IEEE-CIS has amount/card/time/domain/count/delta/device features, but not
`oldbalance/newbalance` identities analogous to PaySim
`balance_diff_orig/dest`. This is an important modeling difference for any
future attacker reparameterization.

Artifact:

```text
artifacts/priority5_ieee_cis/phase51_dataset_profile.json
```

## Phase 5.2 — Development-scale FL baseline smoke

Implemented a standalone IEEE-CIS baseline-smoke script:

```text
experiments/run_ieee_cis_fl_baseline_smoke.py
```

Smoke configuration:

```text
max_rows = 50,000
rounds = 5
clients = 3
seed = 20260916
model = FraudMLP
optimizer = Adam
loss = BCEWithLogitsLoss(pos_weight)
features = transaction-only subset from Phase 5.1
```

Final round metrics:

| Split | F1 | AUC-ROC | PR-AUC |
| --- | ---: | ---: | ---: |
| validation | 0.2434 | 0.7560 | 0.2070 |
| test | 0.2360 | 0.7673 | 0.1990 |

Per-round test AUC improved from `0.7203` at round 1 to `0.7673` at round 5,
so the development pipeline is learning a nontrivial signal. Utility is still
modest and would need stronger feature engineering/model tuning before any
serious cross-dataset claim.

Artifact:

```text
artifacts/priority5_ieee_cis/phase52_fl_baseline_smoke.json
```

## Commands run

```bash
.venv-phase1/bin/python <dataset profiling snippet>

python3 -m py_compile experiments/run_ieee_cis_fl_baseline_smoke.py

.venv-phase1/bin/python experiments/run_ieee_cis_fl_baseline_smoke.py \
  --max-rows 50000 \
  --rounds 5 \
  --clients 3 \
  --output artifacts/priority5_ieee_cis/phase52_fl_baseline_smoke.json
```

## Gate status

| Gate | Status | Reason |
| --- | --- | --- |
| Dataset structure read | PASS | transaction + identity files loaded |
| True fraud rate verified | PASS | `3.499%` |
| Identity coverage measured | PASS | `24.424%` |
| Feature subset selected | PASS | transaction-only subset for first baseline |
| PaySim-like hard constraints found | NO | no balance-conservation analogue |
| Baseline learns nontrivial signal | PASS | test AUC `0.7673` after 5 rounds |
| Attack target creation | NOT RUN | belongs to Phase 5.3 later |

## Next allowed step

Phase 5.3 is not started here. If approved later, the next IEEE-CIS task is a
coarse attacker screening at the smallest scope, using a target design that
accounts for the absence of PaySim-style balance constraints.

## Phase 5.3 — RQ1 raw attacker development gate

Implemented and ran the frozen IEEE-CIS RQ1 development gate:

```text
experiments/run_ieee_cis_rq1_development_gate.py
```

Scope:

```text
records_per_group = 4
fraud_records_per_group = 1
local_steps = 1
development_groups = 8
restarts = 4
iterations = 300
attack_lr = 0.05
attacker = label-free feature + soft-label optimization
```

Valid artifact:

```text
artifacts/priority5_ieee_cis/phase53_rq1_development_gate_20260916_rerun1/
```

The initial artifact
`artifacts/priority5_ieee_cis/phase53_rq1_development_gate_20260916/` stopped
with a PyTorch autograd graph-reuse error before producing gate results and is
excluded from interpretation.

Gate result:

| Control | Wins / non-ties | Mean delta | Median delta | One-sided sign p | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Prior | 4 / 8 | -0.05329 | -0.02094 | 0.63672 | FAIL |
| Zero-update | 6 / 8 | -0.14217 | -0.02752 | 0.14453 | FAIL |

Per the frozen amendment, the 8-record screen is not authorized because the
4-record raw gate failed. No IEEE-CIS DP calibration, DNA-vs-DP experiment, or
confirmatory target was created.

Detailed report:

```text
reports/priority5_ieee_cis_rq1_development_gate_report_20260916.md
```
