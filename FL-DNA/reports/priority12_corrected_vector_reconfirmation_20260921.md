# Priority 12 corrected-vector reconfirmation summary

Date: 2026-09-21  
Status: COMPLETE FOR THE THREE AUTHORIZED CELLS — ALL STOPPED AT n=8 DEVELOPMENT GATE

## What changed

Priority 12 addressed the Priority 11 audit finding that some tabular RQ1 pipelines had attacked/update-transformed vectors containing BatchNorm buffers. A new amendment and runner were added before running any new gate:

- Amendment: `protocols/amendments/2026-09-21_priority12_corrected_vector_reconfirmation.md`
- Runner: `experiments/run_priority12_corrected_vector_attackers.py`

The corrected vector definition is:

- include only trainable parameters (`model.named_parameters()` / `model.parameters()` with `requires_grad=True`);
- exclude all non-trainable buffers, including `running_mean`, `running_var`, `num_batches_tracked`, and any other `model.named_buffers()` entries;
- apply this same trainable-only vector scope to both the DNA transform input and attacker loss/cosine matching vector.

No DNA mathematical formula or attacker generation name was changed.

## Authorized cells

| Cell | Dataset | Defense | Attacker | Old contaminated status | Corrected-vector n=8 result | Escalated? |
|---|---|---|---|---|---|---|
| 1 | PaySim | v1-medium | GEN_IDLG_STYLE | Confirmatory PASS in Priority 6 | FAIL: Prior 5/8 p=0.36328125; Zero 6/8 p=0.14453125 | No |
| 2 | PaySim | v1-stronger | GEN_IDLG_STYLE | Confirmatory PASS in Priority 9 | FAIL overall: Prior 7/8 p=0.03515625; Zero 5/8 p=0.36328125 | No |
| 3 | IEEE-CIS | v2 ratio0.95/eta0.01 | GEN_COSINE_TV | Confirmatory PASS in Priority 6 | FAIL: Prior 4/8 p=0.63671875; Zero 6/8 p=0.14453125 | No |

Because all three corrected-vector development gates failed at n=8, no n=24 pilot and no confirmatory run was executed for Priority 12.

## Artifact IDs

| Cell | Target / bundle artifact | Gate report | Gate report SHA-256 | Firewall evidence |
|---|---|---|---|---|
| PaySim v1-medium | `artifacts/priority12_corrected_vector/paysim_v1_medium_n8_targets_20260921/paysim_priority12_v1_medium_n8_targets.pt` | `artifacts/priority12_corrected_vector/paysim_v1_medium_n8_gate_20260921/priority12_gate_report.json` | `74eec07c50e8e1e0b8371e7779053cc342645a89c194f9c6537fd4934af7617a` | provenance `max_overlap_with_existing_targets=0` |
| PaySim v1-stronger | `artifacts/priority12_corrected_vector/paysim_v1_stronger_n8_targets_20260921/paysim_priority12_v1_stronger_n8_targets.pt` | `artifacts/priority12_corrected_vector/paysim_v1_stronger_n8_gate_20260921/priority12_gate_report.json` | `703ac902ce3bee795ebf3c9584567c0dd0a9deefc69a33e6be2a80b79386c7ac` | provenance `max_overlap_with_existing_targets=0` |
| IEEE-CIS v2 | `artifacts/priority12_corrected_vector/ieee_v2_ratio0p95_n8_targets_20260921/ieee_priority6_bundle.pt` | `artifacts/priority12_corrected_vector/ieee_v2_ratio0p95_n8_gate_20260921/priority12_gate_report.json` | `c0eb9a36495b8c8e14f11a9540efd849c942156e8566f8d7e367cc29f585a824` | `disjointness_gate=PASS`, `max_overlap=0`, `warmup_target_overlap=0` |

## Commands actually run

PaySim v1-medium target and gate:

```bash
PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python FL-DNA/experiments/create_phase4_source_disjoint_targets.py FL-DNA/artifacts/priority12_corrected_vector/paysim_v1_medium_n8_targets_20260921 --output-name paysim_priority12_v1_medium_n8_targets.pt --groups 8 --records-per-group 4 --fraud-per-group 1 --seed 2026092112 --purpose 'Priority 12 corrected-vector PaySim v1_medium GEN_IDLG_STYLE n8 development gate'
PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python FL-DNA/experiments/run_priority12_corrected_vector_attackers.py execute --dataset paysim --target FL-DNA/artifacts/priority12_corrected_vector/paysim_v1_medium_n8_targets_20260921/paysim_priority12_v1_medium_n8_targets.pt --output FL-DNA/artifacts/priority12_corrected_vector/paysim_v1_medium_n8_gate_20260921 --seed 2026092115 --workers 8 --groups 8 --defenses v1_medium --generations GEN_IDLG_STYLE --amendment protocols/amendments/2026-09-21_priority12_corrected_vector_reconfirmation.md
```

PaySim v1-stronger target and gate:

```bash
PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python FL-DNA/experiments/create_phase4_source_disjoint_targets.py FL-DNA/artifacts/priority12_corrected_vector/paysim_v1_stronger_n8_targets_20260921 --output-name paysim_priority12_v1_stronger_n8_targets.pt --groups 8 --records-per-group 4 --fraud-per-group 1 --seed 2026092113 --purpose 'Priority 12 corrected-vector PaySim v1_stronger GEN_IDLG_STYLE n8 development gate'
PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python FL-DNA/experiments/run_priority12_corrected_vector_attackers.py execute --dataset paysim --target FL-DNA/artifacts/priority12_corrected_vector/paysim_v1_stronger_n8_targets_20260921/paysim_priority12_v1_stronger_n8_targets.pt --output FL-DNA/artifacts/priority12_corrected_vector/paysim_v1_stronger_n8_gate_20260921 --seed 2026092116 --workers 8 --groups 8 --defenses v1_stronger --generations GEN_IDLG_STYLE --amendment protocols/amendments/2026-09-21_priority12_corrected_vector_reconfirmation.md
```

IEEE-CIS v2 target bundle and gate:

```bash
PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python FL-DNA/experiments/run_priority12_corrected_vector_attackers.py prepare-ieee --output FL-DNA/artifacts/priority12_corrected_vector/ieee_v2_ratio0p95_n8_targets_20260921 --seed 2026092114 --groups 8 --max-rows 50000 --amendment protocols/amendments/2026-09-21_priority12_corrected_vector_reconfirmation.md
PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python FL-DNA/experiments/run_priority12_corrected_vector_attackers.py execute --dataset ieee --bundle FL-DNA/artifacts/priority12_corrected_vector/ieee_v2_ratio0p95_n8_targets_20260921/ieee_priority6_bundle.pt --output FL-DNA/artifacts/priority12_corrected_vector/ieee_v2_ratio0p95_n8_gate_20260921 --seed 2026092117 --workers 8 --groups 8 --defenses v2_ratio0p95_eta0p01 --generations GEN_COSINE_TV --amendment protocols/amendments/2026-09-21_priority12_corrected_vector_reconfirmation.md
```

Validation:

```bash
PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python -m py_compile FL-DNA/experiments/run_priority12_corrected_vector_attackers.py
git diff --check
```

Both validation commands completed without reported errors.

## Gate decision

Priority 12 requested the old PASS cells be rerun under the corrected trainable-only vector definition, with full escalation discipline. The corrected-vector evidence does not pass the first n=8 development gate for any of the three high-value cells, so the protocol-mandated next step is to stop here and not spend n=24 or confirmatory budget on these cells.

