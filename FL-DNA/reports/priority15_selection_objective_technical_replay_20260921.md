# Priority 15 — selection-objective trainable-key technical replay

Status: STAGE 1 COMPLETE; BOTH FAMILIES STOPPED AFTER DECISIVE DEVELOPMENT-GATE FAILURE

Date: 2026-09-21

This report covers the technical replay requested after finding that three
Phase-4 tabular attack backends used floating `state_dict()` entries, including
BatchNorm buffers, in the optimizer's candidate-selection objective. It reports
the replay evidence only and does not reinterpret the overall RQ1 conclusion.

## What changed

Affected backends:

- `experiments/run_phase4_harddiff_reparam_for_misselected.py`
- `experiments/run_phase4_dna_v2_iht_attack.py`
- `experiments/run_phase4_simple_defense_attack.py`

Previous selection-objective key rule:

```python
keys = [key for key, value in observed.items() if value.is_floating_point()]
```

This included floating BatchNorm buffers such as `running_mean` and
`running_var`.

Corrected key rule:

```python
def _trainable_keys(model):
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]

keys = _trainable_keys(model)
```

For `run_phase4_simple_defense_attack.py`, the observed/candidate defense
application now applies the defense only to those trainable keys and clones
non-key entries unchanged. This prevents buffer tensors from entering either the
defense plan or the matching loss.

Technical replay amendment:

`protocols/amendments/2026-09-21_selection_objective_trainable_keys_technical_replay.md`

This is a technical replay because it restores the pre-specified trainable-only
vector definition. It does not change target scope, seeds, attacker generation,
optimizer settings, restarts, iterations, DNA parameters, DP parameters, or
statistical thresholds.

## Implementation hashes after fix

| File | SHA-256 |
|---|---:|
| `experiments/run_phase4_harddiff_reparam_for_misselected.py` | `a1033494f55b23baa2c27f2685d8e8e81081d5bdffe715cf51eb2848c9c576fc` |
| `experiments/run_phase4_dna_v2_iht_attack.py` | `c7697467c5884e0bf4d759e27ab0c3ebcd65c30f38e029042e70cbd99cc6204d` |
| `experiments/run_phase4_simple_defense_attack.py` | `8e96e0164f79f71f4a7505adf50d63d2b72d70d865250b59597c104529036eee` |
| `experiments/run_strong_update_dp_attack.py` | `b6978642c68b53ace139dac8b9f36e3c52d8142e6eba1a4fbf01ea475c79fa53` |
| `experiments/analyze_strong_update_dp_attack.py` | `782f06f6f109bfd5b16e72b649f51229e742858a3cd05804117963947838357b` |

`run_strong_update_dp_attack.py` was patched only to accept a dynamic `--groups`
argument for n=24/n=39 stages. `analyze_strong_update_dp_attack.py` was patched
only to infer group count and report wins/losses/ties explicitly. No scientific
setting was changed.

## Escalation rule applied

For both affected result families:

- Stage 1: n=8 development probe on fresh source-disjoint target pool.
- Stage 2: n=24 pilot only if Stage 1 does not fail decisively.
- Stage 3: confirmatory only after Stage 2.

RQ1-v2 has a pre-registered confirmatory draw of 176 targets in
`protocols/config/rq1_v2_confirmatory.json`; if family A had reached Stage 3,
that n=176 design would have been followed instead of the generic n=39 design.

Both families failed decisively at Stage 1. Therefore Stage 2 and Stage 3 were
not run for either family.

## Family A — RQ1-v2 DNA-v2 vs DP-v2

### Stage 1 target pool

Target:

`artifacts/priority15_selection_objective_replay/rq1_v2_stage1_n8_targets_20260921/paysim_priority15_rq1_v2_stage1_n8_targets.pt`

Config:

`protocols/config/priority15_rq1_v2_stage1_n8.json`

Analysis:

`artifacts/priority15_selection_objective_replay/rq1_v2_stage1_n8_analysis_20260921/summary.json`

Data firewall:

- groups: 8;
- source rows: `[25198, 458632, 566149, 897950, 900905, 1229108, 1333005, 1508296, 1975259, 2203502, 2715320, 2787434, 2810406, 2986113, 3088296, 3118763, 3126470, 3340742, 4048578, 4048806, 4100870, 4823479, 5042687, 5169393, 5178428, 5201564, 5807181, 5839066, 6098599, 6290960, 6296660, 6308479]`;
- max overlap with existing targets: 0;
- provenance SHA-256: `c1b5fcd86928a5049e91fa3fa0105f6f05f115ac19a3288b7d29ef71bece2901`;
- config SHA-256: `281ede087480c44eb200eeae3d8dcbae29ec8b32e54f9847faa11fd8e0b27389`;
- analysis SHA-256: `d7c7157ad7e2fc236b06f66b911987673787943bd568a22265546167f0360fcc`.

### Stage 1 commands

```bash
PYTHONPATH=. .venv-phase1/bin/python \
  experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority15_selection_objective_replay/rq1_v2_stage1_n8_targets_20260921 \
  --output-name paysim_priority15_rq1_v2_stage1_n8_targets.pt \
  --groups 8 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092130 \
  --purpose 'Priority 15 selection-objective technical replay RQ1-v2 Stage 1 n8'

PYTHONPATH=. .venv-phase1/bin/python \
  experiments/run_rq1_v2_confirmatory.py \
  --config protocols/config/priority15_rq1_v2_stage1_n8.json \
  --output-dir artifacts/priority15_selection_objective_replay/rq1_v2_stage1_n8_run_20260921 \
  --workers 8

PYTHONPATH=. .venv-phase1/bin/python \
  experiments/analyze_rq1_v2_confirmatory.py \
  --config protocols/config/priority15_rq1_v2_stage1_n8.json \
  --run-dir artifacts/priority15_selection_objective_replay/rq1_v2_stage1_n8_run_20260921 \
  --output-dir artifacts/priority15_selection_objective_replay/rq1_v2_stage1_n8_analysis_20260921
```

Execution completed with 24/24 jobs successful.

### Stage 1 control gates

Wins mean the branch's selected reconstruction beat the relevant control under
the original branch-gate sign convention (`difference < 0`). Exact p-values are
one-sided exact sign-test p-values.

| Branch | Control | wins | losses | ties | non-tied n | exact p | mean diff | median diff | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| raw | Prior | 4 | 4 | 0 | 8 | 0.63671875 | -260.255852 | 0.873496 | FAIL |
| raw | Zero-update | 6 | 2 | 0 | 8 | 0.14453125 | -878.560415 | -140.590968 | FAIL |
| dna_v2 | Prior | 5 | 3 | 0 | 8 | 0.36328125 | -97.037487 | -38.522704 | FAIL |
| dna_v2 | Zero-update | 4 | 4 | 0 | 8 | 0.63671875 | 87.343292 | -4.607654 | FAIL |
| dp_v2 | Prior | 3 | 5 | 0 | 8 | 0.85546875 | 246.015243 | 40.814628 | FAIL |
| dp_v2 | Zero-update | 4 | 4 | 0 | 8 | 0.63671875 | 123.818783 | -14.588006 | FAIL |

Primary DNA-v2 vs DP-v2 descriptive result at Stage 1:

| DNA wins | DP wins | ties | non-tied n | exact one-sided p for DNA advantage |
|---:|---:|---:|---:|---:|
| 4 | 4 | 0 | 8 | 0.63671875 |

Stage-1 decision for Family A: STOP. The raw, DNA-v2, and DP-v2 branches all
failed their controls; the primary contrast is invalid and no n=24 pilot was
run.

## Family B — strong update-DP utility-ceiling attacker gate

### Stage 1 target pool

Target:

`artifacts/priority15_selection_objective_replay/strong_dp_stage1_n8_targets_20260921/paysim_priority15_strong_dp_stage1_n8_targets.pt`

Analysis:

`artifacts/priority15_selection_objective_replay/strong_dp_stage1_n8_analysis_20260921/attack_summary.json`

Data firewall:

- groups: 8;
- source rows: `[576341, 1059574, 1427931, 1724597, 1920363, 2108209, 2141615, 2440758, 2748916, 2823164, 3048663, 3083561, 3425288, 3629948, 3675349, 3960335, 4128791, 4239379, 4537934, 4542925, 4584630, 5180709, 5307357, 5393914, 5426793, 5536639, 5563709, 5575916, 5747125, 5967471, 6192136, 6317596]`;
- max overlap with existing targets: 0;
- provenance SHA-256: `83e23cb0e9fea75eb25e9eb6e74889b9738f8b8fe2f8b93c75bc9fd68f5de591`;
- analysis SHA-256: `213da70c603259c68f73b77513e98a66c0e07fa503fb4ebfc1a17e37bed8768f`.

### Stage 1 commands

```bash
PYTHONPATH=. .venv-phase1/bin/python \
  experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority15_selection_objective_replay/strong_dp_stage1_n8_targets_20260921 \
  --output-name paysim_priority15_strong_dp_stage1_n8_targets.pt \
  --groups 8 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092131 \
  --purpose 'Priority 15 selection-objective technical replay strong-DP Stage 1 n8'

PYTHONPATH=. .venv-phase1/bin/python \
  experiments/run_strong_update_dp_attack.py \
  --target artifacts/priority15_selection_objective_replay/strong_dp_stage1_n8_targets_20260921/paysim_priority15_strong_dp_stage1_n8_targets.pt \
  --output-dir artifacts/priority15_selection_objective_replay/strong_dp_stage1_n8_run_20260921 \
  --groups 8 --workers 8

PYTHONPATH=. .venv-phase1/bin/python \
  experiments/analyze_strong_update_dp_attack.py \
  --run-dir artifacts/priority15_selection_objective_replay/strong_dp_stage1_n8_run_20260921 \
  --output-dir artifacts/priority15_selection_objective_replay/strong_dp_stage1_n8_analysis_20260921
```

Execution completed with 40/40 jobs successful.

### Stage 1 control gates

| Branch | Control | wins | losses | ties | non-tied n | exact p | mean diff | median diff | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| raw | Prior | 6 | 2 | 0 | 8 | 0.14453125 | -1446.151732 | -40.963586 | FAIL |
| raw | Zero-update | 3 | 5 | 0 | 8 | 0.85546875 | -279.448839 | 83.274718 | FAIL |
| epsilon_100 | Prior | 5 | 3 | 0 | 8 | 0.36328125 | -29.910257 | -23.355700 | FAIL |
| epsilon_100 | Zero-update | 6 | 2 | 0 | 8 | 0.14453125 | -41.668684 | -28.079870 | FAIL |
| epsilon_50 | Prior | 5 | 3 | 0 | 8 | 0.36328125 | -955.381229 | -26.533084 | FAIL |
| epsilon_50 | Zero-update | 5 | 3 | 0 | 8 | 0.36328125 | -876.459018 | -30.802852 | FAIL |
| epsilon_10 | Prior | 5 | 3 | 0 | 8 | 0.36328125 | 178.581320 | -46.710510 | FAIL |
| epsilon_10 | Zero-update | 3 | 5 | 0 | 8 | 0.85546875 | 280.131177 | 6.754321 | FAIL |
| epsilon_1 | Prior | 6 | 2 | 0 | 8 | 0.14453125 | 829.764379 | -28.928256 | FAIL |
| epsilon_1 | Zero-update | 6 | 2 | 0 | 8 | 0.14453125 | 1182.586504 | -14.616550 | FAIL |

Stage-1 decision for Family B: STOP. Every branch failed at least one control;
therefore no n=24 pilot or n=39 confirmatory run was run.

## What was not run

- No Stage 2 n=24 pilot was run for RQ1-v2 because Stage 1 failed decisively.
- No RQ1-v2 n=176 confirmatory replay was run.
- No Stage 2 n=24 pilot was run for strong-DP because Stage 1 failed decisively.
- No strong-DP n=39 confirmatory replay was run.
- No target set from the old RQ1-v2 n=176 run or old strong-DP run was reused.

## Final status

The technical code fix is in place and Stage 1 replay was completed for both
affected result families on fresh source-disjoint targets. Both families failed
the corrected-objective development gates at n=8, so the escalation chain ended
there under the predeclared rule.

No overall RQ1/RQ2/RQ3 interpretation is made in this report.
