# Priority 18 — DP-vs-transform head-to-head independence-assumption robustness check

**Date:** 2026-09-27
**Status:** COMPLETE
**Scope:** statistical re-analysis of existing DP-vs-transform head-to-head confirmatory data. No new targets, model training, or attacker runs were created. No files under `Latex/` were edited.

## Purpose

The DP-vs-transform head-to-head sign tests in Table `tab:dpvsdna` report
unanimous DP wins (`39/39`, one-sided exact sign-test p-value
`1.8189894035458565e-12`) for:

1. Transform v1 stronger vs DP 0.0004;
2. Transform v1 medium vs DP 0.000315;
3. Transform v2 ratio0.95/eta0.01 vs DP 0.00105.

This analysis checks how much that p-value depends on treating the 39 paired
targets as independent Bernoulli trials, and documents the actual shared vs.
per-target components in the code.

## Step 1 — Actual independence structure from code

### Target data

Targets within a run are source-disjoint rows, sampled without replacement:

- `experiments/create_phase4_source_disjoint_targets.py` reads the PaySim data and computes the available rows after exclusions at lines 64-68.
- It calls `make_groups(...)` with the requested number of groups and a derived seed at lines 69-76.
- `experiments/run_phase3_adam_ladder.py` implements `make_groups(...)`; fraud and non-fraud row pools are sampled with `replace=False` at lines 52-56, then split into groups at lines 58-64.
- The target generator records overlaps against existing target artifacts and `max_overlap_with_existing_targets` at lines 81-100 of `create_phase4_source_disjoint_targets.py`.

Finding: target source rows are disjoint within each confirmatory target set and disjoint from earlier target pools by construction/provenance.

### Model checkpoint and local update generation

All targets in a given tabular head-to-head run share the same pre-local model checkpoint:

- `experiments/run_phase3_adam_ladder.py::capture` constructs `FraudMLP`, then loads `REFERENCE/'pre_local.pt'` at lines 68-72.

However, each target group uses its own local update seed:

- Priority 14 runner derives `local_seed = derive_seed(job["seed"], "priority14-local", job["cell"], job["group"])` at `experiments/priority14_fresh_dna_vs_dp_probe.py` lines 209-214.
- Priority 16 runner does the same at `experiments/priority16_v1_medium_dna_vs_dp_probe.py` lines 204-209.
- `privacy/seed_manager.py::derive_seed` hashes the run seed plus all provided parts into a deterministic child seed at lines 16-20.

Finding: there is one shared starting checkpoint per run, but the local target data and local-update RNG state are group-specific. The experiment does not train an independently initialized global checkpoint per target.

### Attacker restarts

Each target/restart candidate initialization is group- and restart-specific:

- Priority 14 `_optimize_branch` calls `_initial(... derive_seed(job["seed"], "priority14-init", job["cell"], branch, job["group"], job["restart"]) ...)` at lines 142-149.
- Priority 16 uses the same pattern at lines 137-144.

Finding: restarts are not one shared candidate initialization reused across all targets.

### DP noise

DP hyperparameters are shared across targets within a cell, but the noise plan is group-specific:

- Priority 14 `_dp_payload` constructs the DP arguments with shared `clip_norm`, `noise_multiplier`, `defense_seed`, and `mc_noise_samples` at lines 121-132.
- The actual clipping/noise plan in `experiments/run_phase4_simple_defense_attack.py::_plan` derives the direct clipping-noise seed with `derive_seed(args.defense_seed, "clipping-noise", group_id)` at lines 285-295.
- For the Monte-Carlo mean objective, it derives the MC seed with `derive_seed(args.defense_seed, "clipping-noise-mc", group_id)` at lines 306-314.
- Candidate-side MC defense uses `plan["mc_noise_mean"]` at lines 217-239, specifically line 236 for `clipping_noise_mc`.

Finding: despite a fixed run-level `defense_seed`, the DP noise is not identical across all 39 targets; it is deterministically group-specific. The DP clip norm/noise multiplier are shared by design.

### DNA Transform v1 randomness

Transform v1 has shared configuration constants but not one single run-level permutation matrix applied identically to every target:

- Priority 14 freezes v1-stronger config, including seed `681958327`, at `experiments/priority14_fresh_dna_vs_dp_probe.py` lines 60-65.
- Priority 16 freezes v1-medium config, including seed `681958327`, at `experiments/priority16_v1_medium_dna_vs_dp_probe.py` lines 60-65.
- Observed v1 updates are transformed by `_transmit_observed(...)`, which calls `transform_update_array(..., tensor_index=tensor_index)` at `experiments/run_phase4_dna_level1_forward_attack.py` lines 91-99.
- `dna_encoder/transform_defense.py::transform_update_array` computes a block seed from the block's float32 bytes, base seed, tensor index, and block index at lines 57-76, with `_dna_block_seed_from_float32_block(...)` defined at lines 110-142.
- Candidate surrogate realizations are group/restart-specific: `_surrogate_plan_from_state(...)` uses `_surrogate_seed(config, group_id, realization_id, tensor_index, block_index)` at `experiments/run_phase4_dna_level1_forward_attack.py` lines 57-74, and `_surrogate_seed(...)` includes `group_id` and `candidate_id` at `experiments/analyze_phase4_dna_level1_surrogate_inversion.py` lines 30-38.

Finding: v1 shares its high-level config and base seed across targets, but the observed transform is data-dependent per update block; the surrogate candidate transform is explicitly group/restart-specific. There is no single identical v1 permutation/mask applied to every target.

### DNA Transform v2 randomness

Transform v2 has a stronger shared run-level component than v1:

- Priority 14 freezes v2 config with `compression_ratio=0.95`, `quantization_eta=0.01`, and seed `20260916` at `experiments/priority14_fresh_dna_vs_dp_probe.py` lines 66-70.
- `_observed_sketches(...)` calls `transform_update_array_v2(..., tensor_index=tensor_index, quantization_seed=group_id)` at `experiments/run_phase4_dna_v2_sketch_space_attack.py` lines 107-127.
- `dna_encoder/transform_defense_v2.py::transform_update_array_v2` derives the projection seed only from config seed and tensor index at lines 61-83, then creates signs and sampled indices at lines 87-91.
- Quantization is group-specific via `quantization_seed=group_id` passed above and then `q_seed = _derive_seed(seed, quantization_seed)` at `transform_defense_v2.py` lines 93-95.

Finding: v2 uses the same projection/sign/sample pattern per tensor across all 39 targets, while quantization randomness is group-specific. This shared projection is a plausible within-run correlation source.

## Step 2 — Robustness checks

Analysis script:

- `experiments/priority18_head_to_head_independence_check.py`
- SHA-256: `14a4ba3a8e355a61e57de8588fda63994d51faa71d8bb5a5c4e5c4d335572ef2`

Output:

- `artifacts/priority18_head_to_head_independence/priority18_independence_check_20260927.json`
- SHA-256: `4fefdb0a81f51c94440597ff87821a86ee71397e692bb5a0e5144b36ee05bdc8`

Command:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority18_head_to_head_independence_check.py \
  --output artifacts/priority18_head_to_head_independence/priority18_independence_check_20260927.json \
  --permutations 100000 \
  --seed 20260927
```

### Per-target D summaries

| Cell | n | DP wins | DNA wins | Ties | Mean D | SD(D) | Min D | Max D | Exact targetwise p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 stronger vs DP 0.0004 | 39 | 39 | 0 | 0 | -0.0016166626669785634 | 0.00002239451446454247 | -0.0016648222065312667 | -0.0015611634361454207 | 1.8189894035458565e-12 |
| v1 medium vs DP 0.000315 | 39 | 39 | 0 | 0 | -0.0010026436558672064 | 0.000014117016824531628 | -0.001032592196022129 | -0.0009677024415664272 | 1.8189894035458565e-12 |
| v2 ratio0.95/eta0.01 vs DP 0.00105 | 39 | 39 | 0 | 0 | -0.011142000339682493 | 0.00015395491425476958 | -0.011472191521304603 | -0.010762953239998747 | 1.8189894035458565e-12 |

### Permutation sanity check

The permutation check randomly flips each paired target's DP/DNA label with
probability 0.5 under the paired null. It ran 100,000 simulations per cell.
This sanity check assumes targetwise flips and therefore does not model
run-level clustering; it is reported only as a check that the targetwise sign
test matches a paired-label permutation null.

| Cell | Extreme permutations / 100,000 | Empirical p | Resolution note |
| --- | ---: | ---: | --- |
| v1 stronger vs DP 0.0004 | 0 / 100000 | 0.0 | Monte Carlo resolution is 1/100000; exact p is 2^-39 |
| v1 medium vs DP 0.000315 | 0 / 100000 | 0.0 | Monte Carlo resolution is 1/100000; exact p is 2^-39 |
| v2 ratio0.95/eta0.01 vs DP 0.00105 | 0 / 100000 | 0.0 | Monte Carlo resolution is 1/100000; exact p is 2^-39 |

### Conservative effective-n sensitivity

If positive within-run correlation reduces the effective number of independent
Bernoulli trials, then the exact p-value for unanimous DP wins becomes `2^-n_eff`
rather than `2^-39`.

| Effective independent units | One-sided p for unanimous DP wins |
| ---: | ---: |
| 1 | 0.5 |
| 2 | 0.25 |
| 3 | 0.125 |
| 4 | 0.0625 |
| 5 | 0.03125 |
| 6 | 0.015625 |
| 7 | 0.0078125 |
| 8 | 0.00390625 |
| 9 | 0.001953125 |
| 10 | 0.0009765625 |
| 12 | 0.000244140625 |
| 15 | 0.000030517578125 |
| 20 | 0.00000095367431640625 |
| 25 | 0.000000029802322387695312 |
| 30 | 0.0000000009313225746154785 |
| 35 | 0.000000000029103830456733704 |
| 37 | 0.000000000007275957614183426 |
| 39 | 0.0000000000018189894035458565 |

Plain implication: if the most conservative interpretation treats each entire
confirmatory run as only one independent cluster, the formal p-value becomes
`0.5`, not significant. If the effective sample size is at least 5 independent
units, unanimous wins are already below 0.05; at 8 independent units the
p-value is `0.00390625`; at the original targetwise n=39 it is
`1.8189894035458565e-12`.

## Honest verdict

The qualitative sign-consistency result survives: every target in every cell
has `D_i < 0`, so DP wins 39/39 with no exceptions for v1-stronger, v1-medium,
and v2. The target rows are source-disjoint and each target has group-specific
local-update RNG and group-specific DP noise, so the 39 signs are not literal
copies of one identical noise draw or one identical target.

However, the extremity of the nominal p-value (`1.82e-12`) does depend on the
targetwise independence assumption. The code shows shared run-level components:

- all targets share the same pre-local checkpoint;
- all targets in a cell share the same DP hyperparameters;
- v2 additionally shares the same projection/sign/sample pattern per tensor;
- v1 shares the same high-level DNA config/base seed, though observed transforms
  are data-dependent and candidate surrogates are group/restart-specific.

Therefore, the paper should not overstate `p=1.82e-12` as if correlation were
impossible. Under a maximally conservative one-cluster-per-run treatment, the
formal p-value would be only `0.5`. Under any less extreme effective-n treatment
with at least 5 independent units, unanimous wins remain nominally significant.

Recommended wording for the evidence level: the head-to-head result is very
strong as a sign-consistency/descriptive pattern (`39/39`, zero exceptions in
all three cells), and the targetwise exact sign-test p-value is `1.82e-12`, but
the latter should be presented with the limitation that within-run correlation
could reduce the effective sample size.

## Final check

No target generation, training, or attacker run was performed for Priority 18.
No amendment was created. No file under `Latex/` was edited.
