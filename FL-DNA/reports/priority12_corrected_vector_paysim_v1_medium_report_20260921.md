# Priority 12 corrected-vector reconfirmation: PaySim v1-medium / GEN_IDLG_STYLE

Date: 2026-09-21  
Status: STOPPED AT DEVELOPMENT GATE (n=8 FAIL)

## Scope

This replay fixes the Priority 11 buffer-contamination issue by using only trainable parameters for both the DNA transform input vector and attacker loss vector.

- Dataset: PaySim / `creditcard.csv`
- Defense: DNA Transform v1 medium (`mix_ratio=0.10`, `keep_ratio=0.85`, `shrink_factor=0.40`)
- Attacker: `GEN_IDLG_STYLE`
- Vector scope: trainable parameters only; all buffers excluded (`running_mean`, `running_var`, `num_batches_tracked`, and any other model buffers)
- Amendment: `protocols/amendments/2026-09-21_priority12_corrected_vector_reconfirmation.md`

## Data firewall

Fresh target set:

- Target artifact: `artifacts/priority12_corrected_vector/paysim_v1_medium_n8_targets_20260921/paysim_priority12_v1_medium_n8_targets.pt`
- Provenance: `artifacts/priority12_corrected_vector/paysim_v1_medium_n8_targets_20260921/paysim_priority12_v1_medium_n8_targets.provenance.json`
- Provenance SHA-256: `8f522ab406f5023348b2f588d742035ac172aabf5e1d2e934358db61fc4b73c1`
- Dataset SHA-256: `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`
- `max_overlap_with_existing_targets`: `0`

## Commands run

Target generation:

```bash
PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python FL-DNA/experiments/create_phase4_source_disjoint_targets.py \
  FL-DNA/artifacts/priority12_corrected_vector/paysim_v1_medium_n8_targets_20260921 \
  --output-name paysim_priority12_v1_medium_n8_targets.pt \
  --groups 8 \
  --records-per-group 4 \
  --fraud-per-group 1 \
  --seed 2026092112 \
  --purpose 'Priority 12 corrected-vector PaySim v1_medium GEN_IDLG_STYLE n8 development gate'
```

Development gate:

```bash
PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python FL-DNA/experiments/run_priority12_corrected_vector_attackers.py execute \
  --dataset paysim \
  --target FL-DNA/artifacts/priority12_corrected_vector/paysim_v1_medium_n8_targets_20260921/paysim_priority12_v1_medium_n8_targets.pt \
  --output FL-DNA/artifacts/priority12_corrected_vector/paysim_v1_medium_n8_gate_20260921 \
  --seed 2026092115 \
  --workers 8 \
  --groups 8 \
  --defenses v1_medium \
  --generations GEN_IDLG_STYLE \
  --amendment protocols/amendments/2026-09-21_priority12_corrected_vector_reconfirmation.md
```

## Result

Output report:

- `artifacts/priority12_corrected_vector/paysim_v1_medium_n8_gate_20260921/priority12_gate_report.json`
- SHA-256: `74eec07c50e8e1e0b8371e7779053cc342645a89c194f9c6537fd4934af7617a`
- Completed jobs: 32

| Control | Wins / non-ties | One-sided exact sign-test p | Mean difference | Median difference | Gate |
|---|---:|---:|---:|---:|---|
| Prior | 5 / 8 | 0.36328125 | -1146.197230 | -617.392801 | FAIL |
| Zero-update | 6 / 8 | 0.14453125 | -915.207496 | -319.606309 | FAIL |

Overall development gate: FAIL.

## Protocol consequence

Because the n=8 development gate failed, this cell was not escalated to n=24 or confirmatory. This follows the Priority 12 escalation rule exactly.

