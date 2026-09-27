# Priority 19 — RQ1 attacker-vs-control independence-assumption audit

**Date:** 2026-09-27  
**Status:** COMPLETE  
**Scope:** code audit plus statistical re-analysis of existing RQ1 attacker-vs-control sign-test data. No new targets, no model training, no attacker optimization, and no amendment were created. No files under `Latex/` were edited.

## Purpose

Priority 18 audited the DP-vs-transform head-to-head sign tests and found that
the nominal targetwise p-value (`1.8189894035458565e-12` for 39/39 unanimous
wins) is fragile under a maximally conservative single-cluster reading because
targets in that pipeline share run-level components. This Priority 19 audit
extends the same question to the CIFAR-10 image-domain RQ1 confirmatory break:

- CIFAR-10 / minimal LeNet / DNA Transform v2 `ratio0.95 eta0.01` /
  `GEN_COSINE_TV`;
- confirmatory `n=39`, reported as `39/39` wins vs Prior and Zero-update,
  p = `1.8189894035458565e-12`;
- development n=24 pilot for the same cell, reported as `24/24` wins vs both
  controls, p = `5.960464477539063e-08`.

For context, I also inspected one tabular attacker-vs-control confirmatory cell
from Priority 6: PaySim / v1-medium / `GEN_IDLG_STYLE`.

## Step 1 — Actual independence structure from code

### CIFAR-10 target data

The CIFAR target generator samples fresh CIFAR training indices without
replacement and records overlap evidence:

- `experiments/run_priority7_image_domain_gate.py` loads CIFAR-10 in
  `prepare_targets(...)` at lines 241-246.
- It collects historical CIFAR target indices at lines 229-238, removes them
  from the available pool at lines 247-250, and samples the requested images
  with `replace=False` at line 254.
- It writes `target_indices`, per-manifest overlaps, `max_overlap`, and a
  `disjointness_gate` at lines 281-299.

Finding: CIFAR source images are sampled without replacement within a target
set and source-disjoint from prior CIFAR image-domain pools by manifest.

### CIFAR-10 model checkpoint / model initialization

Unlike the tabular head-to-head pipeline audited in Priority 18, the CIFAR
runner does **not** load one shared `pre_local.pt` or one shared trained
checkpoint for all targets:

- `_make_model(seed)` reseeds and constructs a fresh `LeNetCIFAR()` at
  `experiments/run_priority7_image_domain_gate.py` lines 99-103.
- `_image_job(...)` calls
  `_make_model(derive_seed(job["seed"], "model", job["group"]))` at line 310,
  so the model initialization seed includes the target group.
- `derive_seed(...)` hashes the run seed and all provided parts into a
  deterministic child seed at `privacy/seed_manager.py` lines 16-20.

Finding: each CIFAR target group uses an independently derived model
initialization. There is no single shared pre-local model checkpoint across all
39 CIFAR confirmatory targets.

### CIFAR-10 local update and attacker restarts

The local update and attacker initialization are also derived per target group:

- `_named_update(...)` computes the update directly from the target group's
  images and labels at `run_priority7_image_domain_gate.py` lines 106-111.
- `_image_job(...)` loads one target group at lines 306-309 and computes the
  observed update at line 311.
- The job-level RNG seed includes defense, generation, group, and restart at
  line 305.
- The attacker's initial latent image seed includes defense, generation, group,
  and restart at lines 316-318.
- Jobs are created for each `(defense, generation, group, restart)` at lines
  450-468.

Finding: local target data, local update computation, and restart
initializations are group/restart-specific. Shared attacker hyperparameters
(`iterations`, `attack_lr`, `local_lr`, and `restarts`) are fixed by protocol
but are not themselves a common random artifact.

### CIFAR-10 Transform v2 randomness

CIFAR's Transform v2 path shares the same projection/sign/sample pattern per
tensor across targets, while quantization randomness is target-group-specific:

- The CIFAR runner freezes `V2 = {compression_ratio=0.95, quantization_eta=0.01,
  seed=20260916}` at `run_priority7_image_domain_gate.py` line 49.
- `_defense_payload(...)` passes the observed update and `group_id` into
  `_observed_sketches(...)` at lines 135-139.
- `_observed_sketches(...)` calls `transform_update_array_v2(...)` with
  `tensor_index=tensor_index` and `quantization_seed=group_id` at
  `experiments/run_phase4_dna_v2_sketch_space_attack.py` lines 107-127.
- `transform_update_array_v2(...)` derives the projection seed only from the
  v2 base seed and tensor index at `dna_encoder/transform_defense_v2.py` lines
  61-83, then creates signs and sampled indices at lines 87-91.
- The same function derives the quantization seed from that projection seed and
  `quantization_seed` at lines 93-95.

Finding: the CIFAR v2 cell has no shared model checkpoint, but it does share a
run-level v2 projection/sign/sample pattern per tensor across targets. This is
a plausible within-run correlation source. Quantization is group-specific via
`quantization_seed=group_id`.

### CIFAR-10 scoring and sign test

The reported CIFAR attack-success sign test is based on image reconstruction
MSE:

- `_score(...)` computes per-image MSE/PSNR/SSIM and averages image MSE over
  the group at `run_priority7_image_domain_gate.py` lines 370-380.
- `_evaluate(...)` selects the best baseline and zero-update restart per group
  by objective at lines 387-394, then records `baseline_minus_prior` and
  `baseline_minus_zero` at lines 399-410.
- The sign gate removes exact zero ties, counts wins where the attacker's
  `baseline_minus_control < 0`, computes the one-sided sign-test p-value, and
  requires negative mean, negative median, and p < 0.05 at lines 413-432.

### Optional tabular comparison: PaySim v1-medium / GEN_IDLG_STYLE

The Priority 6 PaySim attacker-vs-control confirmatory runner has the same
shared-pre-local-checkpoint structure as the tabular family:

- PaySim targets are source-disjoint: `create_phase4_source_disjoint_targets.py`
  builds available rows after exclusions at lines 64-68, calls `make_groups`
  at lines 69-76, and records target overlaps at lines 81-103.
- `run_phase3_adam_ladder.py::capture(...)` constructs `FraudMLP`, then loads
  `REFERENCE/'pre_local.pt'` at lines 68-72.
- `run_priority6_sota_style_attackers.py::_paysim_job(...)` uses a
  group-derived local seed at lines 356-365 and a generation/defense/group/
  restart-derived initialization seed at lines 366-371.
- The v1 defense path uses `_transmit_observed(...)` and
  `_surrogate_plan_from_state(...)` at `run_priority6_sota_style_attackers.py`
  lines 243-248. `_transmit_observed(...)` calls `transform_update_array(...)`
  per tensor at `run_phase4_dna_level1_forward_attack.py` lines 91-99.
- `transform_update_array(...)` derives observed v1 block seeds from the
  block's float32 bytes plus base seed/tensor/block indices at
  `dna_encoder/transform_defense.py` lines 57-76 and 110-142.
- Candidate surrogate seeds include group and candidate/restart identifiers at
  `run_phase4_dna_level1_forward_attack.py` lines 57-74 and
  `analyze_phase4_dna_level1_surrogate_inversion.py` lines 30-38.
- Priority 6 evaluation uses the same restart selection and sign-test pattern:
  `run_priority6_sota_style_attackers.py` lines 522-558.

Finding: the PaySim Priority 6 attacker-vs-control cell shares one tabular
pre-local checkpoint across targets, but local seeds, target data, attacker
initializations, and v1 surrogate plans are group/restart-specific. This
checkpoint-sharing pattern generalizes across tabular RQ1 attacker-vs-control
pipelines, but **not** to the CIFAR-10 image-domain runner.

## Step 2 — CIFAR-10 robustness checks

Analysis script:

- `experiments/priority19_rq1_confirmatory_independence_audit.py`
- SHA-256: `e4e54f52efacf98a5af1d3d731cf1ca7bbfb6d53dc92e1ab4bdfec83842b7b26`

Output:

- `artifacts/priority19_rq1_confirmatory_independence/priority19_cifar_independence_20260927.json`
- SHA-256: `4525972c32a5837e768ed3b90fdd5f1332ed3cf231c6f4a9fc90cdd363237531`

Command:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority19_rq1_confirmatory_independence_audit.py \
  --output artifacts/priority19_rq1_confirmatory_independence/priority19_cifar_independence_20260927.json \
  --permutations 100000 \
  --seed 20260927
```

### CIFAR confirmatory n=39

Source report/artifact:

- Report: `reports/priority7_cifar_v2_cosine_confirmatory_report_20260917.md`
- Gate JSON:
  `artifacts/priority7_image_domain/confirmatory_v2_cosine_gate_20260917/priority7_image_gate_report.json`
- Gate JSON SHA-256:
  `fd7befa0647ad0589e11d0b38c0214db02b1983b321d595504bd6366eacd634e`
- Target SHA-256:
  `9c12af4cd51cece9bb15d9c0522719cf1b7b0ecbf610dcf4b5e78db7b7c74b22`

| Control | Wins | Losses | Ties | Non-ties | Mean ΔMSE | SD(ΔMSE) | Median ΔMSE | Exact p | Permutation extreme / 100,000 | Empirical p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prior | 39 | 0 | 0 | 39 | -0.016728334032770006 | 0.005128196986597894 | -0.01743950903657228 | 1.8189894035458565e-12 | 0 | 0.0 |
| Zero-update | 39 | 0 | 0 | 39 | -0.01637854046236111 | 0.005299054594921829 | -0.01695496904062306 | 1.8189894035458565e-12 | 0 | 0.0 |

The permutation sanity check flips each paired target label independently under
the null. As in Priority 18, this checks agreement with the targetwise sign
test but does **not** model run-level clustering; with 100,000 repetitions the
Monte Carlo resolution is only `1e-5`, so unanimous 39/39 wins naturally produce
zero observed extreme permutations while the exact targetwise p-value is `2^-39`.

### CIFAR n=24 development pilot

Source report/artifact:

- Report: `reports/priority7_cifar_v2_cosine_n24_development_report_20260917.md`
- Gate JSON:
  `artifacts/priority7_image_domain/n24_scope4_v2_cosine_gate_20260917/priority7_image_gate_report.json`
- Gate JSON SHA-256:
  `1828c7602a00fb53fddf9e9ceff54f8d2d0dd6b46787518b5eec3c5f67c4fc56`
- Target SHA-256:
  `7e8790d303e21ef3601561af2b6ef2cddee2925a9e3e285d2a0f45caa95a7e7e`

| Control | Wins | Losses | Ties | Non-ties | Mean ΔMSE | SD(ΔMSE) | Median ΔMSE | Exact p | Permutation extreme / 100,000 | Empirical p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prior | 24 | 0 | 0 | 24 | -0.01904482279528705 | 0.004946507898506446 | -0.018433599554450134 | 5.960464477539063e-08 | 0 | 0.0 |
| Zero-update | 24 | 0 | 0 | 24 | -0.019151096625682266 | 0.0053238387827858665 | -0.018847762370137716 | 5.960464477539063e-08 | 0 | 0.0 |

### Conservative effective-n sensitivity

If positive within-run correlation reduces the effective number of independent
Bernoulli trials, then a unanimous-win exact p-value is `2^-n_eff`, not
necessarily `2^-39`.

| Effective independent units | One-sided p for unanimous wins |
|---:|---:|
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
| 24 | 0.000000059604644775390625 |
| 30 | 0.0000000009313225746154785 |
| 35 | 0.000000000029103830456733704 |
| 37 | 0.000000000007275957614183426 |
| 39 | 0.0000000000018189894035458565 |

Plain implication: under the most conservative single-cluster reading, the
formal p-value for the CIFAR confirmatory cell collapses to `0.5`, exactly as
in Priority 18. If the effective number of independent units is at least 5,
unanimous wins fall below 0.05; at 8 units the p-value is `0.00390625`; at the
targetwise `n=39` it is `1.8189894035458565e-12`.

## Verdict

The CIFAR-10 confirmatory break is **meaningfully more independent than the
tabular DP-vs-transform head-to-head cells with respect to model state**:
there is no shared pre-local checkpoint; every target group gets a fresh
minimal LeNet initialization from a group-derived seed.

However, the CIFAR v2 cell is **not fully free of run-level shared structure**:
all targets share the same Transform v2 projection/sign/sample pattern per
tensor, because the projection seed is derived from the fixed v2 base seed and
tensor index. Quantization randomness, target images, model initialization,
and attacker restarts are group-specific.

Therefore, the qualitative sign pattern survives the audit: the attacker beats
both controls on `39/39` CIFAR confirmatory targets, with zero exceptions, and
the earlier n=24 pilot also shows `24/24` vs both controls. But the exact
targetwise p-value `1.8189894035458565e-12` remains subject to the same class of
effective-sample-size caveat as Priority 18, though for a narrower reason
(shared v2 projection rather than shared tabular checkpoint plus shared v2
projection). Under a maximally conservative one-run/one-cluster reading, the
formal p-value is only `0.5`; under any effective-n assumption of at least five
independent units, the unanimous sign pattern remains nominally significant.

For the optional PaySim Priority 6 comparison, the code-only audit shows the
tabular attacker-vs-control pipeline does share one `pre_local.pt` checkpoint
across targets, just like the broader tabular RQ1 family. Thus the checkpoint
sharing issue generalizes across tabular RQ1 attacker-vs-control pipelines but
does not generalize to the CIFAR image-domain runner.

## Artifact and code hashes

| File | SHA-256 |
|---|---|
| `experiments/priority19_rq1_confirmatory_independence_audit.py` | `e4e54f52efacf98a5af1d3d731cf1ca7bbfb6d53dc92e1ab4bdfec83842b7b26` |
| `artifacts/priority19_rq1_confirmatory_independence/priority19_cifar_independence_20260927.json` | `4525972c32a5837e768ed3b90fdd5f1332ed3cf231c6f4a9fc90cdd363237531` |
| `experiments/run_priority7_image_domain_gate.py` | `904890af8df1c82032fa0cbae358c9f4664b9bb5468a45d8e9a1c686f4894786` |
| `dna_encoder/transform_defense_v2.py` | `884d6b52dc149fcd9f8fe3d45e1a8e8b75a5d0ecf3b81d2ff1c1628584ec7c76` |
| `experiments/run_phase4_dna_v2_sketch_space_attack.py` | `c4b3cac16a2726f1c59b289eed9becf78419c8c781a72110753c75bc24fd6cc7` |
| `experiments/run_priority6_sota_style_attackers.py` | `0780fa2d4aa9a8c56a49b58910b55ab8de22e32d553ee8ffc8d83efa3622e9f0` |
| `reports/priority7_cifar_v2_cosine_confirmatory_report_20260917.md` | `a7154fd396e7ddf0b778088ba8ec1fd6585860e0fceae4461e88b1db0f5e88fe` |
| `reports/priority7_cifar_v2_cosine_n24_development_report_20260917.md` | `0a6db32b82c163fac3a2a1d23e6b3ec776c9ebb8798f5ea689bd6b188201f04f` |
| `reports/priority6_paysim_v1_idlg_confirmatory_report_20260917.md` | `4de8f9d06def92c004a7031842079caf2ae2602ed4785dfcc441069e43338729` |

## Final checks

- No target generation: PASS.
- No model training: PASS.
- No attacker optimization: PASS.
- No amendment created: PASS.
- No file under `Latex/` edited: PASS.
- `torch.set_num_threads(1)` unchanged by this task.
