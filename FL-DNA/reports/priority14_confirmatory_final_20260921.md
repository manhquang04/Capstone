# Priority 13-14 final report — clean-vector DNA-vs-DP head-to-head

Status: PRIORITY 14 CONFIRMATORY COMPLETE

Date: 2026-09-21

This report summarizes the full Priority 13-14 chain from the BatchNorm-buffer
contamination finding through the final fresh clean-vector confirmatory run.
It reports the evidence only. It does not reinterpret the overall RQ1 conclusion.

## Initial finding

Priority 11 found that several tabular RQ1 attack pipelines had constructed the
attacked vector from `state_dict()`-style content that included non-trainable
BatchNorm buffers (`running_mean`, `running_var`, `num_batches_tracked`), rather
than only trainable parameters.

Priority 13 then replayed two old null DNA-vs-DP comparisons by excluding those
buffers at scoring time. That replay was descriptive only because the stored
candidates had originally been optimized under the contaminated objective.

## Five checks before final confirmatory execution

| Layer | Purpose | Result |
|---|---|---|
| 1. Buffer-exclusion replay with old candidates | Check whether old null DNA-vs-DP conclusions visibly change when scoring only trainable parameters. | At zero threshold: v1 stronger old n=44 replay gave 0 DNA wins / 44 DP wins / 0 ties; v2 old n=176 replay gave 0 DNA wins / 176 DP wins / 0 ties. |
| 2. Threshold sanity check | Check whether the old feature-MSE tie threshold was hiding all trainable-update differences. | Old frozen thresholds produced all ties, but zero-threshold decomposition showed nonzero differences in every pair. |
| 3. Permutation-mismatch check | Test whether v1 loses only because position-wise MSE is unfair to a permutation-style transform. | v1 sorted-value MSE still gave 0 DNA wins / 44 DP wins / 0 ties; v2, which does not use the same v1 permutation mechanism, also gave 0 DNA wins / 176 DP wins / 0 ties. |
| 4. Fresh n=8 clean-vector probe | Optimize the attacker directly on trainable parameters only, instead of rescoring contaminated old candidates. | Corrected `_nonnegative_penalty(latent, meta)` run gave 0 DNA wins / 8 DP wins / 0 ties for both v1 stronger and v2 ratio0.95. |
| 5. Fresh n=24 pilot | Check whether the n=8 pattern persists at the intermediate development scale before confirmatory execution. | Both cells gave 0 DNA wins / 24 DP wins / 0 ties, DP one-sided p = 5.960464477539063e-08. |

The n=24 result triggered the pre-authorized confirmatory power analysis and the
supervisor then authorized the final confirmatory run.

## Confirmatory authorization

Amendment:

`protocols/amendments/2026-09-21_priority14_confirmatory_escalation_draft.md`

The amendment was changed from draft to `FROZEN/AUTHORIZED` before target
generation or confirmatory execution.

Frozen conditions checked before execution:

- p1 = 0.70, inherited from the existing RQ1/Priority 6 convention and not
  refit from the n=24 result.
- Required effective non-tied n = 37.
- Draw n = 39 for each cell, using the small project-standard buffer.
- Rejection threshold = at least 24 wins among 37 non-tied groups.
- Actual alpha = 0.04943587479647249.
- Actual power at p1 = 0.8070956916527874.
- Vector scope = trainable parameters only; all buffers excluded.
- Attacker = original raw-lift/harddiff attacker with balanced-tensor objective.
- Penalty = corrected `0.001 * _nonnegative_penalty(latent, meta)`.
- Exactly one confirmatory execution per cell.

No condition above failed before execution.

## Target sets and data firewall

Both confirmatory target sets were newly generated after the amendment was
authorized. Both provenance files report `max_overlap_with_existing_targets = 0`.

| Cell | Target path | Target SHA-256 | Provenance SHA-256 | Groups | Max overlap | Dataset SHA-256 |
|---|---|---:|---:|---:|---:|---:|
| v1 stronger vs DP 0.0004 | `artifacts/priority14_fresh_probe/v1_stronger_confirmatory_targets_20260921/paysim_priority14_v1_stronger_confirmatory_targets.pt` | `2b0584918b362526ad28ce1e473b91dc56e2e9d42e257e5a4ddca643e8450fcd` | `7417cb569026ec010150f4e87787653bd933b175d81eb51602b0b6f914135487` | 39 | 0 | `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b` |
| v2 ratio0.95 vs DP 0.00105 | `artifacts/priority14_fresh_probe/v2_ratio0p95_confirmatory_targets_20260921/paysim_priority14_v2_ratio0p95_confirmatory_targets.pt` | `d57097a27fdb32dc41a0fc2a1d21a6a0f1ddc2af207a767280839c8294397c32` | `72b5e07b9f48e8191e497fa6a45c4f4c1882719d37d32c94e6f8b9c4bca7e624` | 39 | 0 | `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b` |

## Commands run

Amendment status update was made before the following target generation commands.

Target generation:

```bash
mkdir -p FL-DNA/artifacts/priority14_fresh_probe/v1_stronger_confirmatory_targets_20260921 \
  FL-DNA/artifacts/priority14_fresh_probe/v2_ratio0p95_confirmatory_targets_20260921

PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python \
  FL-DNA/experiments/create_phase4_source_disjoint_targets.py \
  FL-DNA/artifacts/priority14_fresh_probe/v1_stronger_confirmatory_targets_20260921 \
  --output-name paysim_priority14_v1_stronger_confirmatory_targets.pt \
  --groups 39 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092126 \
  --purpose 'Priority 14 clean-vector DNA-vs-DP confirmatory v1_stronger'

PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python \
  FL-DNA/experiments/create_phase4_source_disjoint_targets.py \
  FL-DNA/artifacts/priority14_fresh_probe/v2_ratio0p95_confirmatory_targets_20260921 \
  --output-name paysim_priority14_v2_ratio0p95_confirmatory_targets.pt \
  --groups 39 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092127 \
  --purpose 'Priority 14 clean-vector DNA-vs-DP confirmatory v2_ratio0p95'
```

Confirmatory execution:

```bash
PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python \
  FL-DNA/experiments/priority14_fresh_dna_vs_dp_probe.py \
  --cell v1_stronger_vs_dp_0p0004 \
  --target FL-DNA/artifacts/priority14_fresh_probe/v1_stronger_confirmatory_targets_20260921/paysim_priority14_v1_stronger_confirmatory_targets.pt \
  --output FL-DNA/artifacts/priority14_fresh_probe/v1_stronger_confirmatory_20260921 \
  --seed 2026092128 --groups 39 --restarts 4 --workers 8 \
  --amendment protocols/amendments/2026-09-21_priority14_confirmatory_escalation_draft.md

PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python \
  FL-DNA/experiments/priority14_fresh_dna_vs_dp_probe.py \
  --cell v2_ratio0p95_eta0p01_vs_dp_0p00105 \
  --target FL-DNA/artifacts/priority14_fresh_probe/v2_ratio0p95_confirmatory_targets_20260921/paysim_priority14_v2_ratio0p95_confirmatory_targets.pt \
  --output FL-DNA/artifacts/priority14_fresh_probe/v2_ratio0p95_confirmatory_20260921 \
  --seed 2026092129 --groups 39 --restarts 4 --workers 8 \
  --amendment protocols/amendments/2026-09-21_priority14_confirmatory_escalation_draft.md
```

## Confirmatory results

Zero-threshold head-to-head rule:

```text
D_i = MSE_DNA_i - MSE_DP_i
```

DNA win means `D_i > 0`; DP win means `D_i < 0`.

| Cell | DNA wins | DP wins | Exact ties | Non-tied n | DNA one-sided p | DP one-sided p | Mean D | Median D | Artifact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v1 stronger vs DP 0.0004 | 0 | 39 | 0 | 39 | 1.0 | 1.8189894035458565e-12 | -0.0016166627 | -0.0016185609 | `artifacts/priority14_fresh_probe/v1_stronger_confirmatory_20260921/priority14_probe_report.json` |
| v2 ratio0.95 vs DP 0.00105 | 0 | 39 | 0 | 39 | 1.0 | 1.8189894035458565e-12 | -0.0111420003 | -0.0111530246 | `artifacts/priority14_fresh_probe/v2_ratio0p95_confirmatory_20260921/priority14_probe_report.json` |

Artifact hashes:

| Artifact | SHA-256 |
|---|---:|
| `protocols/amendments/2026-09-21_priority14_confirmatory_escalation_draft.md` | `2be25ef2d14ae3191bb44b2931d9aafe664c015ced7d8d436a52e15d49298156` |
| `artifacts/priority14_fresh_probe/v1_stronger_confirmatory_20260921/priority14_probe_report.json` | `29469ee1170f03d24e3e82f3a8f5eed72c9644e247f50cce26b299299bd65ada` |
| `artifacts/priority14_fresh_probe/v2_ratio0p95_confirmatory_20260921/priority14_probe_report.json` | `7d833fb746f82fefe61838b836ef11f54d1113dbfe9249b9818d7bf023c79a16` |
| `artifacts/priority14_fresh_probe/confirmatory_power_p1_0p70_20260921/rq1_power_analysis.json` | `9e013f0e48b01018db5ca4fd14ceabb5fe9bcb2487f9eb0f96313401338cd440` |
| `experiments/priority14_fresh_dna_vs_dp_probe.py` | `b07ef173308d211833185ceed762cb40602a4c5c8a4f89efc6a0359067cd23d1` |

## Execution boundary

Performed:

- converted the confirmatory amendment to `FROZEN/AUTHORIZED` before creating
  confirmatory targets;
- generated fresh source-disjoint n=39 target sets for both cells;
- ran exactly one confirmatory execution for each cell;
- used the corrected `_nonnegative_penalty(latent, meta)` implementation.

Not performed:

- no second confirmatory run;
- no target reuse from n=8, n=24, Priority 6/8/9/12/13, or any earlier pool;
- no attacker, DNA, DP, p1, alpha, or vector-scope retuning after seeing
  results;
- no RQ1-wide interpretation or report rewrite.

## Final checks

Final check commands:

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
- `torch.set_num_threads(1)` remains present in the Priority 14 runner.
