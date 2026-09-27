# Amendment: Priority 6 PaySim v1 Medium GEN_IDLG_STYLE Confirmatory Escalation

Status: PRE-RUN FROZEN
Timestamp: 2026-09-17T00:00:00+07:00
Scope: Confirmatory escalation for one Priority 6 cell only

## Cell

Dataset: PaySim (`datasets/creditcard.csv`)
Defense: DNA Transform v1 medium
Attacker generation: `GEN_IDLG_STYLE`

This amendment is independent from the IEEE-CIS v2 cosine escalation. Results from one cell must not be used to alter the other cell's protocol.

## Development evidence motivating escalation

The Priority 6 development gates passed on two independent target sets:

- n=8 pilot: Prior 7/8, p=0.03515625; Zero-update 8/8, p=0.00390625.
- n=24 expansion: Prior 18/24, p=0.0113279223; Zero-update 22/24, p=0.0000179410.

The n=24 observed win probability against Prior was 18/24 = 0.75 and against Zero-update was 22/24 = 0.9167. These values justify escalation but are not treated as final evidence because previous attacker development pilots have shown small-sample optimism.

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
- target seed: 2026091781
- attack seed: 2026091783

The target set must be newly created and source-disjoint from all known PaySim target pools, including the Priority 6 n=8 and n=24 development targets, all post-hoc targets, and all prior confirmatory targets.

## Frozen attacker settings

Use the exact `GEN_IDLG_STYLE` settings from the n=24 expansion:

- 4 restarts per group
- 600 optimization steps
- lr = 0.1
- known group labels under the existing PaySim threat-model assumption
- no tabular-TV term
- no hyperparameter changes after target creation

## Decision rule

The confirmatory cell PASSes only if it passes both controls under the frozen gate:

- baseline reconstruction beats Prior with mean difference < 0, median difference < 0, and exact one-sided sign-test p < 0.05;
- baseline reconstruction beats Zero-update with mean difference < 0, median difference < 0, and exact one-sided sign-test p < 0.05.

If the confirmatory run PASSes, report this as a confirmed successful Priority 6 attacker escalation for this cell and discuss how it affects the interpretation of RQ1. Do not rewrite prior frozen reports; only add a new report.

If the confirmatory run FAILs, report this as small-sample optimism in the development pilots and keep the prior frozen RQ1 conclusions unchanged.

## One-shot commitment

Run this confirmatory target set exactly once. Do not repeat, extend, or retune after seeing results. Technical replay is allowed only for implementation/runtime failure, documented by a separate pre-replay amendment.
