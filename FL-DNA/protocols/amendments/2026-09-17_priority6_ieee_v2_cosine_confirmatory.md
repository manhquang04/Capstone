# Amendment: Priority 6 IEEE-CIS v2 GEN_COSINE_TV Confirmatory Escalation

Status: PRE-RUN FROZEN
Timestamp: 2026-09-17T00:00:00+07:00
Scope: Confirmatory escalation for one Priority 6 cell only

## Cell

Dataset: IEEE-CIS Fraud Detection
Defense: DNA Transform v2 ratio0.95 eta0.01
Attacker generation: `GEN_COSINE_TV`

This amendment is independent from the PaySim v1 IDLG escalation. Results from one cell must not be used to alter the other cell's protocol.

## Development evidence motivating escalation

The Priority 6 development gates passed on two independent target sets:

- n=8 pilot: Prior 7/8, p=0.03515625; Zero-update 7/8, p=0.03515625.
- n=24 expansion: Prior 18/24, p=0.0113279223; Zero-update 18/24, p=0.0113279223.

The n=24 observed win probability was 18/24 = 0.75 against both Prior and Zero-update. This justifies escalation but is not treated as final evidence because previous attacker development pilots have shown small-sample optimism.

## Power analysis

Use the same RQ1 pre-pilot exact one-sided binomial sign-test method:

- p0 = 0.50
- p1 = 0.70, inherited from the original RQ1 minimum meaningful win-probability decision
- alpha = 0.05
- target power = 0.80
- tie_rate = 0.0 for the confirmatory target-count calculation
- dropout_rate = 0.05
- maximum_n = 150

Power artifact:

- `artifacts/priority6_sota_attackers/confirmatory_power_p1_0p70_20260917/rq1_power_analysis.json`

Frozen result:

- required effective non-tied n = 37
- initial target draw = 39
- rejection threshold = at least 24 wins among 37 non-tied groups
- actual alpha = 0.04943587479647249
- actual power at p1 = 0.8070956916527874

No new p1 is introduced in this amendment.

## Target and seed freeze

- Confirmatory target groups: 39
- records_per_group: 4
- fraud_records_per_group: 1
- target seed: 2026091782
- attack seed: 2026091784

The target set must be newly created and source-disjoint from all known IEEE-CIS target pools, including the Priority 5 development targets, the Priority 6 n=8 and n=24 development targets, and any prior IEEE-CIS target pools.

## Frozen attacker settings

Use the exact `GEN_COSINE_TV` settings from the n=24 expansion:

- 4 restarts per group
- 300 optimization steps
- lr = 0.05
- cosine matching plus public range/domain constraints only
- `lambda_tabular_tv = 0`
- no hyperparameter changes after target creation

## Decision rule

The confirmatory cell PASSes only if it passes both controls under the frozen gate:

- baseline reconstruction beats Prior with mean difference < 0, median difference < 0, and exact one-sided sign-test p < 0.05;
- baseline reconstruction beats Zero-update with mean difference < 0, median difference < 0, and exact one-sided sign-test p < 0.05.

If the confirmatory run PASSes, report this as a confirmed successful Priority 6 attacker escalation for this cell and discuss how it affects the interpretation of RQ1. Do not rewrite prior frozen reports; only add a new report.

If the confirmatory run FAILs, report this as small-sample optimism in the development pilots and keep the prior frozen RQ1 conclusions unchanged.

## One-shot commitment

Run this confirmatory target set exactly once. Do not repeat, extend, or retune after seeing results. Technical replay is allowed only for implementation/runtime failure, documented by a separate pre-replay amendment.
