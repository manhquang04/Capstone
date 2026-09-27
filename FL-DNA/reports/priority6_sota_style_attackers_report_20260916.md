# Priority 6 report: SOTA-style attacker variants

Status: COMPLETED_DEVELOPMENT_GATE_WITH_N24_EXPANSION
Date: 2026-09-17

## Summary

Package A added and tested two literature-style gradient-inversion attacker variants:

- `GEN_COSINE_TV`: global cosine-similarity gradient/update matching in the spirit of Geiping et al. The supervisor explicitly removed the proposed tabular-TV feature-graph term before any gate was run, so the frozen objective is cosine matching plus public range/domain constraints only (`lambda_tabular_tv = 0`).
- `GEN_IDLG_STYLE`: fixed-label/label-known simplification in the spirit of iDLG. PaySim uses known group labels under the project threat model. IEEE-CIS uses the public one-fraud-of-four label multiset in canonical order and does not use oracle row membership.

Minimal n=8 development gates were run on new source-disjoint targets for PaySim and IEEE-CIS, for both v1 medium and v2 ratio0.95/eta0.01. Two cells passed both Prior and Zero-update controls and were expanded, under a separate pre-run amendment, to n=24 development pilots. Both n=24 expansions also passed.

These results strengthen the attacker side of the study by showing that at least two published-style objectives can beat controls in selected development settings. They do not change any frozen RQ1/RQ2/RQ3 confirmatory conclusion, and they do not by themselves authorize any new confirmatory run.

## Literature basis

Primary sources checked directly:

- Ligeng Zhu, Zhijian Liu, Song Han, "Deep Leakage from Gradients", NeurIPS 2019 / arXiv:1906.08935. This is the original DLG-style gradient matching baseline. Source: https://arxiv.org/abs/1906.08935
- Bo Zhao, Konda Reddy Mopuri, Hakan Bilen, "iDLG: Improved Deep Leakage from Gradients", arXiv:2001.02610, 2020. This motivates fixing or inferring labels instead of optimizing labels jointly. Source: https://arxiv.org/abs/2001.02610
- Jonas Geiping, Hartmut Bauermeister, Hannah Droge, Michael Moeller, "Inverting Gradients -- How easy is it to break privacy in federated learning?", NeurIPS 2020 / arXiv:2003.14053. This motivates magnitude-invariant/cosine-style gradient matching. Source: https://arxiv.org/abs/2003.14053

## Amendments

- Design amendment: `protocols/amendments/2026-09-16_priority6_sota_style_attackers_design.md`
- PaySim target technical replay: `protocols/amendments/2026-09-16_priority6_paysim_target_technical_replay.md`
- PaySim IDLG technical replay: `protocols/amendments/2026-09-16_priority6_paysim_idlg_technical_replay.md`
- n=24 expansion amendment: `protocols/amendments/2026-09-17_priority6_sota_style_attackers_n24_expansion.md`

## Implementation

Main runner:

- `experiments/run_priority6_sota_style_attackers.py`

The runner implements:

- `prepare-ieee`: creates source-disjoint IEEE-CIS target bundles.
- `execute`: runs the frozen attacker generation(s), selects the best restart per group by objective, and applies the exact one-sided sign-test gate against Prior and Zero-update controls.

The runner was extended after n=8 to accept `--groups`, `--defenses`, and `--generations` so the n=24 expansion could run only the two authorized passing cells. This did not change the loss functions, optimization parameters, target composition, or gate rule.

## Data firewall

PaySim n=8 target:

- Artifact: `artifacts/priority6_sota_attackers/targets_paysim/paysim_priority6_targets_replay1.pt`
- Seed: `2026091665`
- Source rows: 32
- Firewall: max overlap with existing target pools = 0

IEEE-CIS n=8 target:

- Artifact: `artifacts/priority6_sota_attackers/targets_ieee/ieee_priority6_bundle.pt`
- Seed: `2026091662`
- Source rows: 32
- Firewall: max overlap with historical IEEE target pools = 0; warmup-target overlap = 0
- Bundle SHA-256: `bf46d5b8a9c472d9ef8c3b5a2fb2c4c491f01c784e7a2fda285d6472081658af`

PaySim n=24 target:

- Artifact: `artifacts/priority6_sota_attackers/targets_paysim_n24/paysim_priority6_n24_targets.pt`
- Seed: `2026091761`
- Source rows: 96
- Firewall: max overlap with existing target pools = 0

IEEE-CIS n=24 target:

- Artifact: `artifacts/priority6_sota_attackers/targets_ieee_n24/ieee_priority6_bundle.pt`
- Seed: `2026091762`
- Source rows: 96
- Firewall: max overlap with historical IEEE target pools = 0; warmup-target overlap = 0
- Bundle SHA-256: `44a426221c34e01eab6faafbacea9abd80aaccbdae6f9f09053f6360642e579d`

## Commands run

IEEE-CIS n=8 target:

```bash
.venv-phase1/bin/python experiments/run_priority6_sota_style_attackers.py prepare-ieee --output artifacts/priority6_sota_attackers/targets_ieee --seed 2026091662 --groups 8 --max-rows 50000
```

PaySim n=8 target replay:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py artifacts/priority6_sota_attackers/targets_paysim --output-name paysim_priority6_targets_replay1.pt --groups 8 --records-per-group 4 --fraud-per-group 1 --seed 2026091665 --purpose 'Priority 6 SOTA-style attacker development gate PaySim n8 replay'
```

PaySim n=8 gate:

```bash
.venv-phase1/bin/python experiments/run_priority6_sota_style_attackers.py execute --dataset paysim --target artifacts/priority6_sota_attackers/targets_paysim/paysim_priority6_targets_replay1.pt --output artifacts/priority6_sota_attackers/gates_paysim_20260916 --seed 2026091663 --workers 8
```

IEEE-CIS n=8 gate:

```bash
.venv-phase1/bin/python experiments/run_priority6_sota_style_attackers.py execute --dataset ieee --bundle artifacts/priority6_sota_attackers/targets_ieee/ieee_priority6_bundle.pt --output artifacts/priority6_sota_attackers/gates_ieee_20260917 --seed 2026091664 --workers 8
```

PaySim n=24 target and gate:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py artifacts/priority6_sota_attackers/targets_paysim_n24 --output-name paysim_priority6_n24_targets.pt --groups 24 --records-per-group 4 --fraud-per-group 1 --seed 2026091761 --purpose 'Priority 6 n24 expansion PaySim v1_medium GEN_IDLG_STYLE'
.venv-phase1/bin/python experiments/run_priority6_sota_style_attackers.py execute --dataset paysim --target artifacts/priority6_sota_attackers/targets_paysim_n24/paysim_priority6_n24_targets.pt --output artifacts/priority6_sota_attackers/gates_paysim_n24_20260917 --seed 2026091763 --workers 8 --groups 24 --defenses v1_medium --generations GEN_IDLG_STYLE
```

IEEE-CIS n=24 target and gate:

```bash
.venv-phase1/bin/python experiments/run_priority6_sota_style_attackers.py prepare-ieee --output artifacts/priority6_sota_attackers/targets_ieee_n24 --seed 2026091762 --groups 24 --max-rows 50000
.venv-phase1/bin/python experiments/run_priority6_sota_style_attackers.py execute --dataset ieee --bundle artifacts/priority6_sota_attackers/targets_ieee_n24/ieee_priority6_bundle.pt --output artifacts/priority6_sota_attackers/gates_ieee_n24_20260917 --seed 2026091764 --workers 8 --groups 24 --defenses v2_ratio0p95_eta0p01 --generations GEN_COSINE_TV
```

## n=8 development gate results

Gate rule: each cell passes only if it beats both Prior and Zero-update with mean difference < 0, median difference < 0, and exact one-sided sign-test p < 0.05.

| Dataset | Defense | Generation | Prior wins | Prior p | Zero wins | Zero p | Gate |
|---|---|---|---:|---:|---:|---:|---|
| PaySim | v1_medium | GEN_COSINE_TV | 6/8 | 0.14453125 | 6/8 | 0.14453125 | FAIL |
| PaySim | v1_medium | GEN_IDLG_STYLE | 7/8 | 0.03515625 | 8/8 | 0.00390625 | PASS |
| PaySim | v2_ratio0p95_eta0p01 | GEN_COSINE_TV | 5/8 | 0.36328125 | 5/8 | 0.36328125 | FAIL |
| PaySim | v2_ratio0p95_eta0p01 | GEN_IDLG_STYLE | 6/8 | 0.14453125 | 7/8 | 0.03515625 | FAIL |
| IEEE-CIS | v1_medium | GEN_COSINE_TV | 5/8 | 0.36328125 | 5/8 | 0.36328125 | FAIL |
| IEEE-CIS | v1_medium | GEN_IDLG_STYLE | 6/8 | 0.14453125 | 6/8 | 0.14453125 | FAIL |
| IEEE-CIS | v2_ratio0p95_eta0p01 | GEN_COSINE_TV | 7/8 | 0.03515625 | 7/8 | 0.03515625 | PASS |
| IEEE-CIS | v2_ratio0p95_eta0p01 | GEN_IDLG_STYLE | 4/8 | 0.63671875 | 3/8 | 0.85546875 | FAIL |

## n=24 development expansion results

Only the two n=8 PASS cells were expanded.

| Dataset | Defense | Generation | Prior wins | Prior p | Zero wins | Zero p | Gate |
|---|---|---|---:|---:|---:|---:|---|
| PaySim | v1_medium | GEN_IDLG_STYLE | 18/24 | 0.0113279223 | 22/24 | 0.0000179410 | PASS |
| IEEE-CIS | v2_ratio0p95_eta0p01 | GEN_COSINE_TV | 18/24 | 0.0113279223 | 18/24 | 0.0113279223 | PASS |

## Interpretation

Package A answers the strawman-attacker concern partially but usefully:

- `GEN_IDLG_STYLE` is a stronger PaySim v1 attacker than the prior L2-style baseline in this development setting.
- `GEN_COSINE_TV`, with the tabular-TV term removed and range constraints only, is a stronger IEEE-CIS v2 attacker in this development setting.
- The passing cells remained stable when expanded from n=8 to n=24 on fresh source-disjoint development targets.
- The failures in other cells are not evidence that the corresponding defense is private; they only mean these two specific published-style attacker variants did not beat both controls there under the frozen development gate.

No confirmatory experiment was run in Package A. If the supervisor wants to promote either passing cell, the next valid step is a new pre-run confirmatory amendment with power analysis and a fresh source-disjoint target set.

## Artifact IDs

- PaySim n=8 gate: `artifacts/priority6_sota_attackers/gates_paysim_20260916/priority6_gate_report.json`
- IEEE-CIS n=8 gate: `artifacts/priority6_sota_attackers/gates_ieee_20260917/priority6_gate_report.json`
- PaySim n=24 gate: `artifacts/priority6_sota_attackers/gates_paysim_n24_20260917/priority6_gate_report.json`
- IEEE-CIS n=24 gate: `artifacts/priority6_sota_attackers/gates_ieee_n24_20260917/priority6_gate_report.json`

## Gate status

Priority 6 Package A development gate: COMPLETED.

Authorized follow-up: none automatically. Confirmatory escalation requires a separate amendment.
