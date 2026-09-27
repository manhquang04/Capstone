# Priority 12 corrected-vector reconfirmation: PaySim v1-stronger / GEN_IDLG_STYLE

Date: 2026-09-21  
Status: STOPPED AT DEVELOPMENT GATE (n=8 FAIL)

## Scope

This replay fixes the Priority 11 buffer-contamination issue by using only trainable parameters for both the DNA transform input vector and attacker loss vector.

- Dataset: PaySim / `creditcard.csv`
- Defense: DNA Transform v1 stronger (`mix_ratio=0.12`, `keep_ratio=0.82`, `shrink_factor=0.35`)
- Attacker: `GEN_IDLG_STYLE`
- Vector scope: trainable parameters only; all buffers excluded (`running_mean`, `running_var`, `num_batches_tracked`, and any other model buffers)
- Amendment: `protocols/amendments/2026-09-21_priority12_corrected_vector_reconfirmation.md`

## Data firewall

Fresh target set:

- Target artifact: `artifacts/priority12_corrected_vector/paysim_v1_stronger_n8_targets_20260921/paysim_priority12_v1_stronger_n8_targets.pt`
- Provenance: `artifacts/priority12_corrected_vector/paysim_v1_stronger_n8_targets_20260921/paysim_priority12_v1_stronger_n8_targets.provenance.json`
- Provenance SHA-256: `a3f3fc7fbcf6e8f463768efdab0418a7d1a5b9bfb1b951f45b2f30f5f63e8153`
- Dataset SHA-256: `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`
- `max_overlap_with_existing_targets`: `0`

## Commands run

Target generation:

```bash
PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python FL-DNA/experiments/create_phase4_source_disjoint_targets.py \
  FL-DNA/artifacts/priority12_corrected_vector/paysim_v1_stronger_n8_targets_20260921 \
  --output-name paysim_priority12_v1_stronger_n8_targets.pt \
  --groups 8 \
  --records-per-group 4 \
  --fraud-per-group 1 \
  --seed 2026092113 \
  --purpose 'Priority 12 corrected-vector PaySim v1_stronger GEN_IDLG_STYLE n8 development gate'
```

Development gate:

```bash
PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python FL-DNA/experiments/run_priority12_corrected_vector_attackers.py execute \
  --dataset paysim \
  --target FL-DNA/artifacts/priority12_corrected_vector/paysim_v1_stronger_n8_targets_20260921/paysim_priority12_v1_stronger_n8_targets.pt \
  --output FL-DNA/artifacts/priority12_corrected_vector/paysim_v1_stronger_n8_gate_20260921 \
  --seed 2026092116 \
  --workers 8 \
  --groups 8 \
  --defenses v1_stronger \
  --generations GEN_IDLG_STYLE \
  --amendment protocols/amendments/2026-09-21_priority12_corrected_vector_reconfirmation.md
```

## Result

Output report:

- `artifacts/priority12_corrected_vector/paysim_v1_stronger_n8_gate_20260921/priority12_gate_report.json`
- SHA-256: `703ac902ce3bee795ebf3c9584567c0dd0a9deefc69a33e6be2a80b79386c7ac`
- Completed jobs: 32

| Control | Wins / non-ties | One-sided exact sign-test p | Mean difference | Median difference | Gate |
|---|---:|---:|---:|---:|---|
| Prior | 7 / 8 | 0.03515625 | -1364.685808 | -375.486175 | PASS |
| Zero-update | 5 / 8 | 0.36328125 | -1534.325024 | -40.329546 | FAIL |

Overall development gate: FAIL because both controls must pass.

## Protocol consequence

Because the n=8 development gate failed against the Zero-update control, this cell was not escalated to n=24 or confirmatory. This follows the Priority 12 escalation rule exactly.

