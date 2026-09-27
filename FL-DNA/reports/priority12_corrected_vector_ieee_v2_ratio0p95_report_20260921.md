# Priority 12 corrected-vector reconfirmation: IEEE-CIS v2 ratio0.95/eta0.01 / GEN_COSINE_TV

Date: 2026-09-21  
Status: STOPPED AT DEVELOPMENT GATE (n=8 FAIL)

## Scope

This replay fixes the Priority 11 buffer-contamination issue by using only trainable parameters for both the DNA transform input vector and attacker loss vector.

- Dataset: IEEE-CIS Fraud Detection
- Defense: DNA Transform v2 (`compression_ratio=0.95`, `quantization_eta=0.01`)
- Attacker: `GEN_COSINE_TV`
- Vector scope: trainable parameters only; all buffers excluded (`running_mean`, `running_var`, `num_batches_tracked`, and any other model buffers)
- Amendment: `protocols/amendments/2026-09-21_priority12_corrected_vector_reconfirmation.md`

## Data firewall

Fresh target bundle:

- Bundle: `artifacts/priority12_corrected_vector/ieee_v2_ratio0p95_n8_targets_20260921/ieee_priority6_bundle.pt`
- Firewall report: `artifacts/priority12_corrected_vector/ieee_v2_ratio0p95_n8_targets_20260921/ieee_target_firewall.json`
- Firewall SHA-256: `87eb4e025446e71ff97fb57435c634c7532fbd342fba4ca29de24a9605022b30`
- Dataset SHA-256: `3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642`
- Bundle SHA-256: `f9836ecdca283e7a8dba0f6136f74606b598b01d0939c44eedd8ac09b8695acb`
- `disjointness_gate`: `PASS`
- `max_overlap`: `0`
- `warmup_target_overlap`: `0`

## Commands run

Target bundle generation:

```bash
PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python FL-DNA/experiments/run_priority12_corrected_vector_attackers.py prepare-ieee \
  --output FL-DNA/artifacts/priority12_corrected_vector/ieee_v2_ratio0p95_n8_targets_20260921 \
  --seed 2026092114 \
  --groups 8 \
  --max-rows 50000 \
  --amendment protocols/amendments/2026-09-21_priority12_corrected_vector_reconfirmation.md
```

Development gate:

```bash
PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python FL-DNA/experiments/run_priority12_corrected_vector_attackers.py execute \
  --dataset ieee \
  --bundle FL-DNA/artifacts/priority12_corrected_vector/ieee_v2_ratio0p95_n8_targets_20260921/ieee_priority6_bundle.pt \
  --output FL-DNA/artifacts/priority12_corrected_vector/ieee_v2_ratio0p95_n8_gate_20260921 \
  --seed 2026092117 \
  --workers 8 \
  --groups 8 \
  --defenses v2_ratio0p95_eta0p01 \
  --generations GEN_COSINE_TV \
  --amendment protocols/amendments/2026-09-21_priority12_corrected_vector_reconfirmation.md
```

## Result

Output report:

- `artifacts/priority12_corrected_vector/ieee_v2_ratio0p95_n8_gate_20260921/priority12_gate_report.json`
- SHA-256: `c0eb9a36495b8c8e14f11a9540efd849c942156e8566f8d7e367cc29f585a824`
- Completed jobs: 32

| Control | Wins / non-ties | One-sided exact sign-test p | Mean difference | Median difference | Gate |
|---|---:|---:|---:|---:|---|
| Prior | 4 / 8 | 0.63671875 | -2.082899 | -0.196406 | FAIL |
| Zero-update | 6 / 8 | 0.14453125 | -1.858080 | -0.186212 | FAIL |

Overall development gate: FAIL.

## Protocol consequence

Because the n=8 development gate failed, this cell was not escalated to n=24 or confirmatory. This follows the Priority 12 escalation rule exactly.

