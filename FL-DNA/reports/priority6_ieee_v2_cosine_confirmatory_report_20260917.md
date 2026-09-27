# Priority 6 confirmatory report: IEEE-CIS v2 GEN_COSINE_TV

Status: CONFIRMATORY_PASS
Date: 2026-09-17

## Scope

This report covers one Priority 6 confirmatory escalation only:

- Dataset: IEEE-CIS Fraud Detection
- Defense: DNA Transform v2 ratio0.95 eta0.01
- Attacker generation: `GEN_COSINE_TV`
- Amendment: `protocols/amendments/2026-09-17_priority6_ieee_v2_cosine_confirmatory.md`

No other Priority 6 cell is interpreted from this run.

## Pre-run power and target contract

Power analysis used the existing RQ1 exact one-sided binomial sign-test method:

- p0 = 0.50
- p1 = 0.70
- alpha = 0.05
- target power = 0.80
- tie_rate = 0.0
- dropout_rate = 0.05

Power artifact:

- `artifacts/priority6_sota_attackers/confirmatory_power_p1_0p70_20260917/rq1_power_analysis.json`

Frozen planning result:

- required effective non-tied n = 37
- target draw = 39
- planning rejection threshold = 24 wins among 37 non-tied groups
- actual alpha at planning n = 0.04943587479647249
- actual power at p1 = 0.8070956916527874

The actual confirmatory run produced 39 non-tied groups, so the exact one-sided p-values below are computed at n=39. For n=39, the exact alpha<=0.05 rejection threshold is at least 26 wins.

## Data firewall

Target artifact:

- `artifacts/priority6_sota_attackers/confirmatory_ieee_v2_cosine_targets_20260917/ieee_priority6_bundle.pt`

Target firewall:

- `artifacts/priority6_sota_attackers/confirmatory_ieee_v2_cosine_targets_20260917/ieee_target_firewall.json`

Firewall result:

- groups = 39
- records_per_group = 4
- fraud_per_group = 1
- source rows = 156
- checked historical IEEE target files = 4
- max overlap with historical target pools = 0
- warmup-target overlap = 0
- disjointness gate = PASS
- dataset SHA-256 = `3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642`
- bundle SHA-256 = `3332a2d82ad12a5885380e74c937ee4cfed4dcae641edfaf790f2ca71b457def`

## Commands run

Power analysis:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/rq1_power_analysis.py --p0 0.50 --p1 0.70 --alpha 0.05 --power 0.80 --tie-rate 0.0 --dropout-rate 0.05 --maximum-n 150 --output-dir artifacts/priority6_sota_attackers/confirmatory_power_p1_0p70_20260917
```

Target creation:

```bash
.venv-phase1/bin/python experiments/run_priority6_sota_style_attackers.py prepare-ieee --output artifacts/priority6_sota_attackers/confirmatory_ieee_v2_cosine_targets_20260917 --seed 2026091782 --groups 39 --max-rows 50000 --amendment protocols/amendments/2026-09-17_priority6_ieee_v2_cosine_confirmatory.md
```

Confirmatory run:

```bash
.venv-phase1/bin/python experiments/run_priority6_sota_style_attackers.py execute --dataset ieee --bundle artifacts/priority6_sota_attackers/confirmatory_ieee_v2_cosine_targets_20260917/ieee_priority6_bundle.pt --output artifacts/priority6_sota_attackers/confirmatory_ieee_v2_cosine_run_20260917 --seed 2026091784 --workers 8 --groups 39 --defenses v2_ratio0p95_eta0p01 --generations GEN_COSINE_TV --amendment protocols/amendments/2026-09-17_priority6_ieee_v2_cosine_confirmatory.md
```

## Results

Artifact:

- `artifacts/priority6_sota_attackers/confirmatory_ieee_v2_cosine_run_20260917/priority6_gate_report.json`

Completed jobs: 156/156.

| Control | Wins | Non-ties | Mean difference | Median difference | Exact one-sided p | Gate |
|---|---:|---:|---:|---:|---:|---|
| Prior | 36 | 39 | -18.22005511185879 | -1.8663490465690433 | 0.000000018044374883174896 | PASS |
| Zero-update | 36 | 39 | -18.15571436415269 | -2.0107084239376354 | 0.000000018044374883174896 | PASS |

Overall gate: PASS.

## Interpretation

This is a confirmed successful attacker escalation for IEEE-CIS v2 under the Priority 6 protocol. The `GEN_COSINE_TV` attacker beat both Prior and Zero-update controls on a fresh source-disjoint confirmatory target set, in a single pre-registered run.

This result affects the interpretation of RQ1 by showing that DNA Transform v2's earlier "attacker not found" development state was not a stable privacy conclusion once a stronger published-style cosine attacker was validated and escalated. It reinforces the rule that attacker failure is not privacy evidence unless the attacker itself has passed appropriate controls.

This report does not rewrite or invalidate earlier frozen RQ1 reports. It is an additional confirmatory finding that should be cited alongside the earlier v1/v2/DP conclusions.

## Gate status

CONFIRMATORY PASS.
