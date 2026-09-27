# Priority 14 — n=24 clean-vector DNA-vs-DP pilot

Status: DEVELOPMENT PILOT COMPLETE; CONFIRMATORY NOT RUN

Date: 2026-09-21

## Scope

This report covers the supervisor-authorized Priority 14 n=24 development pilot
for two clean-vector DNA-vs-DP head-to-head cells:

1. PaySim DNA Transform v1 stronger vs distortion-matched DP 0.0004.
2. PaySim DNA Transform v2 ratio0.95/eta0.01 vs distortion-matched DP 0.00105.

The pilot keeps the Priority 14 corrected vector definition: trainable
parameters only, all buffers excluded. It uses the original raw-lift/harddiff
attacker and the corrected `_nonnegative_penalty(latent, meta)` helper.

No confirmatory target set was created and no confirmatory run was executed.

## Amendments

- n=8 base probe:
  `protocols/amendments/2026-09-21_priority14_fresh_dna_vs_dp_probe.md`
- technical replay for `_nonnegative_penalty` correction:
  `protocols/amendments/2026-09-21_priority14_technical_replay.md`
- n=24 escalation authorization:
  `protocols/amendments/2026-09-21_priority14_n24_pilot_escalation.md`
- confirmatory escalation draft, not authorized for execution:
  `protocols/amendments/2026-09-21_priority14_confirmatory_escalation_draft.md`

## Target sets and data firewall

Both target sets were newly generated after the n=24 amendment. Provenance
reports show `max_overlap_with_existing_targets = 0`.

| Cell | Target path | Target SHA-256 | Provenance SHA-256 | Max overlap | Dataset SHA-256 |
|---|---|---:|---:|---:|---:|
| v1 stronger vs DP 0.0004 | `artifacts/priority14_fresh_probe/v1_stronger_n24_targets_20260921/paysim_priority14_v1_stronger_n24_targets.pt` | `33136fe103f7427b60446d3079b76917694a14382491f6fcffcf3b9f7b26e3ae` | `cb0b0f21f4f49bad3ba45af02b10c1c60b2f784d0bf51a173465ac383c1f35ba` | 0 | `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b` |
| v2 ratio0.95 vs DP 0.00105 | `artifacts/priority14_fresh_probe/v2_ratio0p95_n24_targets_20260921/paysim_priority14_v2_ratio0p95_n24_targets.pt` | `e4bbf2d84e8e2481cc8d6d5d41d6525adb003b2dca0e9c1d16e6fdc1cd688020` | `9dbc8bcabaedd9607547369145793dd881f08fd33c4da3b80a627f8e20774cd7` | 0 | `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b` |

## Commands run

Target generation:

```bash
mkdir -p FL-DNA/artifacts/priority14_fresh_probe/v1_stronger_n24_targets_20260921 \
  FL-DNA/artifacts/priority14_fresh_probe/v2_ratio0p95_n24_targets_20260921

PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python \
  FL-DNA/experiments/create_phase4_source_disjoint_targets.py \
  FL-DNA/artifacts/priority14_fresh_probe/v1_stronger_n24_targets_20260921 \
  --output-name paysim_priority14_v1_stronger_n24_targets.pt \
  --groups 24 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092122 \
  --purpose 'Priority 14 clean-vector DNA-vs-DP n24 pilot v1_stronger'

PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python \
  FL-DNA/experiments/create_phase4_source_disjoint_targets.py \
  FL-DNA/artifacts/priority14_fresh_probe/v2_ratio0p95_n24_targets_20260921 \
  --output-name paysim_priority14_v2_ratio0p95_n24_targets.pt \
  --groups 24 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092123 \
  --purpose 'Priority 14 clean-vector DNA-vs-DP n24 pilot v2_ratio0p95'
```

Pilot execution:

```bash
PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python \
  FL-DNA/experiments/priority14_fresh_dna_vs_dp_probe.py \
  --cell v1_stronger_vs_dp_0p0004 \
  --target FL-DNA/artifacts/priority14_fresh_probe/v1_stronger_n24_targets_20260921/paysim_priority14_v1_stronger_n24_targets.pt \
  --output FL-DNA/artifacts/priority14_fresh_probe/v1_stronger_n24_pilot_20260921 \
  --seed 2026092124 --groups 24 --restarts 4 --workers 8 \
  --amendment protocols/amendments/2026-09-21_priority14_n24_pilot_escalation.md

PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python \
  FL-DNA/experiments/priority14_fresh_dna_vs_dp_probe.py \
  --cell v2_ratio0p95_eta0p01_vs_dp_0p00105 \
  --target FL-DNA/artifacts/priority14_fresh_probe/v2_ratio0p95_n24_targets_20260921/paysim_priority14_v2_ratio0p95_n24_targets.pt \
  --output FL-DNA/artifacts/priority14_fresh_probe/v2_ratio0p95_n24_pilot_20260921 \
  --seed 2026092125 --groups 24 --restarts 4 --workers 8 \
  --amendment protocols/amendments/2026-09-21_priority14_n24_pilot_escalation.md
```

Power analysis:

```bash
mkdir -p FL-DNA/artifacts/priority14_fresh_probe/confirmatory_power_p1_0p70_20260921

PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python \
  FL-DNA/experiments/rq1_power_analysis.py \
  --p0 0.5 --p1 0.70 --alpha 0.05 --power 0.80 \
  --tie-rate 0 --dropout-rate 0 --maximum-n 150 \
  --output-dir FL-DNA/artifacts/priority14_fresh_probe/confirmatory_power_p1_0p70_20260921
```

Note: before the successful power-analysis command above, one attempted command
used unsupported CLI options `--output-json` and `--output-csv`. It failed before
creating any output and did not affect the results.

## Results

Zero-threshold head-to-head rule:

```text
D_i = MSE_DNA_i - MSE_DP_i
```

DNA win means `D_i > 0`; DP win means `D_i < 0`.

| Cell | DNA wins | DP wins | Exact ties | Non-tied n | DNA one-sided p | DP one-sided p | Mean D | Median D | Artifact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v1 stronger vs DP 0.0004 | 0 | 24 | 0 | 24 | 1.0 | 5.960464477539063e-08 | -0.0016173165 | -0.0016199085 | `artifacts/priority14_fresh_probe/v1_stronger_n24_pilot_20260921/priority14_probe_report.json` |
| v2 ratio0.95 vs DP 0.00105 | 0 | 24 | 0 | 24 | 1.0 | 5.960464477539063e-08 | -0.0111456242 | -0.0111635612 | `artifacts/priority14_fresh_probe/v2_ratio0p95_n24_pilot_20260921/priority14_probe_report.json` |

Artifact hashes:

| Artifact | SHA-256 |
|---|---:|
| `artifacts/priority14_fresh_probe/v1_stronger_n24_pilot_20260921/priority14_probe_report.json` | `00d0d325ae56a5a79a3a973f9443939a489491accb5a43798a729085154c68e1` |
| `artifacts/priority14_fresh_probe/v2_ratio0p95_n24_pilot_20260921/priority14_probe_report.json` | `1b8f264f7c6ef50aba4c6e29407935a39e113c1b0ccf8c10bef2b5c95feb4c43` |
| `artifacts/priority14_fresh_probe/confirmatory_power_p1_0p70_20260921/rq1_power_analysis.json` | `9e013f0e48b01018db5ca4fd14ceabb5fe9bcb2487f9eb0f96313401338cd440` |
| `experiments/priority14_fresh_dna_vs_dp_probe.py` | `b07ef173308d211833185ceed762cb40602a4c5c8a4f89efc6a0359067cd23d1` |

## Confirmatory power analysis

Because both n=24 cells still show DP wins 24/24, the escalation amendment
requires confirmatory power analysis but not confirmatory execution.

Power artifact:

`artifacts/priority14_fresh_probe/confirmatory_power_p1_0p70_20260921/rq1_power_analysis.json`

Primary calculation:

- exact one-sided binomial sign test;
- p0 = 0.50;
- p1 = 0.70, inherited from the existing RQ1/Priority 6 convention and not
  refit from the observed 24/24 pilot result;
- alpha = 0.05;
- target power = 0.80;
- tie rate = 0.0;
- dropout rate = 0.0;
- required effective non-tied n = 37;
- rejection threshold = at least 24 wins among 37 non-tied groups;
- actual alpha = 0.04943587479647249;
- actual power = 0.8070956916527874;
- initial draw count = 37 under zero-tie/zero-dropout assumptions.

A separate draft amendment records this calculation:

`protocols/amendments/2026-09-21_priority14_confirmatory_escalation_draft.md`

It remains `DRAFT — AWAITING SUPERVISOR DECISION`; it does not authorize
confirmatory target generation or execution.

## Gate status and next allowed step

Development n=24 pilot status: both cells show the same DP-win direction as n=8
and pass the "strong pattern persists" criterion for drafting a confirmatory
amendment.

Next protocol-allowed step: supervisor review of
`2026-09-21_priority14_confirmatory_escalation_draft.md`.

Not performed:

- no confirmatory target set was generated;
- no confirmatory execution was run;
- no attacker/DNA/DP parameter was changed after observing n=24 results.

## Final checks

Commands:

```bash
git -C FL-DNA diff --check

FL-DNA/.venv-phase1/bin/python -m py_compile \
  FL-DNA/experiments/priority14_fresh_dna_vs_dp_probe.py \
  FL-DNA/experiments/rq1_power_analysis.py \
  FL-DNA/experiments/create_phase4_source_disjoint_targets.py

ps -axo pid,command | rg 'priority14_fresh_dna_vs_dp_probe|create_phase4_source_disjoint_targets|rq1_power_analysis' | rg -v 'rg ' || true

rg -n "torch\\.set_num_threads\\(1\\)" \
  FL-DNA/experiments/priority14_fresh_dna_vs_dp_probe.py
```

Results:

- `git diff --check`: PASS.
- `py_compile`: PASS.
- Background workload check: no Priority 14, target-generation, or power-analysis
  workload remained running.
- `torch.set_num_threads(1)` remains present in
  `experiments/priority14_fresh_dna_vs_dp_probe.py` at the worker and top-level
  execution points.
