# Priority 6 confirmatory report: PaySim v1 medium GEN_IDLG_STYLE

Status: CONFIRMATORY_PASS
Date: 2026-09-17

## Scope

This report covers one Priority 6 confirmatory escalation only:

- Dataset: PaySim (`datasets/creditcard.csv`)
- Defense: DNA Transform v1 medium
- Attacker generation: `GEN_IDLG_STYLE`
- Amendment: `protocols/amendments/2026-09-17_priority6_paysim_v1_idlg_confirmatory.md`

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

- `artifacts/priority6_sota_attackers/confirmatory_paysim_v1_idlg_targets_20260917/paysim_priority6_v1_idlg_confirmatory_targets.pt`

Target provenance:

- `artifacts/priority6_sota_attackers/confirmatory_paysim_v1_idlg_targets_20260917/paysim_priority6_v1_idlg_confirmatory_targets.provenance.json`

Firewall result:

- groups = 39
- records_per_group = 4
- fraud_per_group = 1
- source rows = 156
- max overlap with existing target pools = 0
- dataset SHA-256 = `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`

## Commands run

Power analysis:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/rq1_power_analysis.py --p0 0.50 --p1 0.70 --alpha 0.05 --power 0.80 --tie-rate 0.0 --dropout-rate 0.05 --maximum-n 150 --output-dir artifacts/priority6_sota_attackers/confirmatory_power_p1_0p70_20260917
```

Target creation:

```bash
mkdir -p artifacts/priority6_sota_attackers/confirmatory_paysim_v1_idlg_targets_20260917
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py artifacts/priority6_sota_attackers/confirmatory_paysim_v1_idlg_targets_20260917 --output-name paysim_priority6_v1_idlg_confirmatory_targets.pt --groups 39 --records-per-group 4 --fraud-per-group 1 --seed 2026091781 --purpose 'Priority 6 confirmatory PaySim v1_medium GEN_IDLG_STYLE p1_0p70 one-shot'
```

Confirmatory run:

```bash
.venv-phase1/bin/python experiments/run_priority6_sota_style_attackers.py execute --dataset paysim --target artifacts/priority6_sota_attackers/confirmatory_paysim_v1_idlg_targets_20260917/paysim_priority6_v1_idlg_confirmatory_targets.pt --output artifacts/priority6_sota_attackers/confirmatory_paysim_v1_idlg_run_20260917 --seed 2026091783 --workers 8 --groups 39 --defenses v1_medium --generations GEN_IDLG_STYLE --amendment protocols/amendments/2026-09-17_priority6_paysim_v1_idlg_confirmatory.md
```

## Results

Artifact:

- `artifacts/priority6_sota_attackers/confirmatory_paysim_v1_idlg_run_20260917/priority6_gate_report.json`

Completed jobs: 156/156.

| Control | Wins | Non-ties | Mean difference | Median difference | Exact one-sided p | Gate |
|---|---:|---:|---:|---:|---:|---|
| Prior | 29 | 39 | -1636.6977484245251 | -268.8062994808507 | 0.0016889239559532143 | PASS |
| Zero-update | 32 | 39 | -1820.1198764476546 | -506.4237112093824 | 0.000035127392038702965 | PASS |

Overall gate: PASS.

## Interpretation

This is a confirmed successful attacker escalation for PaySim v1 medium under the Priority 6 protocol. The `GEN_IDLG_STYLE` attacker beat both Prior and Zero-update controls on a fresh source-disjoint confirmatory target set, in a single pre-registered run.

This result affects the interpretation of RQ1 by showing that, for PaySim v1 medium, a published-style fixed-label attacker can reconstruct better than controls at the bounded 4-record/1-fraud/1-step scope. It strengthens the claim that RQ1 conclusions depend materially on attacker strength and that weak attacker failures should not be interpreted as privacy.

This report does not rewrite or invalidate earlier frozen RQ1 reports. It is an additional confirmatory finding that should be cited alongside the earlier v1/v2/DP conclusions.

## Gate status

CONFIRMATORY PASS.
