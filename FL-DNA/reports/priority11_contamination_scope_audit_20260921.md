# Priority 11 contamination-scope audit for tabular RQ1 tests

**Date:** 2026-09-21  
**Scope:** code audit only. No experiment was rerun, no target set was created, no
artifact was modified, and no frozen conclusion was reinterpreted here.

## Executive map

Code-level audit shows the following:

- All post-Phase-3 bounded-update tabular RQ1 tests that use
  `attacks.local_update.simulate()` / `experiments.run_phase3_adam_ladder.capture()`
  are **contaminated** by floating BatchNorm buffers. This includes the RQ1 primary
  confirmatory family, Group 2/3 replications, Priority 2 clean confirmatory,
  RQ1-v2 confirmatory/development gates, Priority 5.3 IEEE-CIS development gate,
  Priority 6/8/9/10 SOTA-style tabular gates, strong-DP attack gates, and scope
  boundary raw screens.
- The early single-sample DLG-style helpers based on
  `attacks.gradient_inversion.parameter_gradients()` are **clean** because they use
  trainable `model.parameters()` only.

This report only maps scope of contamination. It does not propose fixes or alter
any prior RQ1 conclusion.

## Code evidence used for classification

### Evidence A — `simulate()` returns parameter deltas plus model buffers

Source: `attacks/local_update.py:35-61`

```python
35 def simulate(model, criterion, x, y, batches, rng_state, lr=1e-3, create_graph=True):
36     """Fresh Adam state, train mode, fixed dropout realization; return full state delta."""
37     params = dict(model.named_parameters())
38     initial = {k: v.detach().clone() for k, v in params.items()}
39     buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
40     initial_buffers = {k: v.clone() for k, v in buffers.items()}
...
59     delta = {k: v - initial[k] for k, v in params.items()}
60     delta.update({k: v - initial_buffers[k] for k, v in tracked.items()})
61     return delta
```

Classification rule: any pipeline whose attacked/defended update dictionary is
created from this `simulate()` output is **contaminated** unless it explicitly
filters to trainable parameter names before defense/loss. The audited RQ1 tabular
pipelines below do not apply such a trainable-only filter.

### Evidence B — Phase-3/Phase-4 PaySim capture uses `simulate()`

Source: `experiments/run_phase3_adam_ladder.py:68-81`

```python
68 def capture(group, seed, batch_size):
...
71     model=FraudMLP(x.shape[1]).train();model.load_state_dict(torch.load(REFERENCE/'pre_local.pt',weights_only=False))
72     initial=copy.deepcopy(model.state_dict()); criterion=common.BinaryFocalLoss(); rng=torch.Generator().manual_seed(seed).get_state()
73     observed=simulate(model,criterion,x,y,batches,rng)
...
79     for key,value in native.state_dict().items():
80         torch.testing.assert_close(observed[key],value-initial[key],atol=2e-7,rtol=2e-4)
81     return model,criterion,x,y,batches,rng,observed
```

Because `observed` comes from `simulate()`, it includes the floating BatchNorm
running-stat buffers returned by Evidence A.

### Evidence C — raw/DNA/DP Phase-4 branches use all floating keys from `observed`

Raw branch:

Source: `experiments/run_phase4_harddiff_reparam_for_misselected.py:168-192`

```python
168     model, criterion, x, y, batches, rng, observed = capture(
...
171     keys = [key for key, value in observed.items() if value.is_floating_point()]
...
182         signal = observed if method == "baseline" else {key: torch.zeros_like(value) for key, value in observed.items()}
...
191             delta = simulate(model, criterion, reconstruction, y, batches, rng)
192             gradient_loss = update_objective(delta, signal, keys, reference=observed, mode="balanced_tensor")
```

DNA v1 branch:

Source: `experiments/run_phase4_dna_level1_forward_attack.py:106-141`

```python
106     model, criterion, x, y, batches, rng, observed = capture(group, local_seed, protocol["batch_size"])
107     keys = [key for key, value in observed.items() if value.is_floating_point()]
...
115     transmitted = _transmit_observed(observed, transform_config)
116     surrogate_plan = _surrogate_plan_from_state(observed, transform_config, group_id, realization_id)
...
139         delta = simulate(model, criterion, reconstruction, y, batches, rng)
140         candidate_transmitted = _apply_surrogate_realization_torch(delta, surrogate_plan)
141         gradient_loss = update_objective(candidate_transmitted, signal, keys, reference=transmitted, mode="balanced_tensor")
```

DP/clipping-noise branch:

Source: `experiments/run_phase4_simple_defense_attack.py:332-359`

```python
332     model, criterion, x, y, batches, rng, observed = capture(group, local_seed, protocol["batch_size"])
333     keys = _floating_keys(observed)
334     plan = _plan(args.defense, observed, keys, group_id, args)
335     defended_observed = _apply_observed_defense(observed, keys, args.defense, plan)
336     signal = defended_observed if method == "baseline" else {key: torch.zeros_like(value) for key, value in defended_observed.items()}
...
357         delta = simulate(model, criterion, reconstruction, y, batches, rng)
358         candidate_defended = _apply_candidate_defense(delta, keys, args.defense, plan)
359         gradient_loss = update_objective(candidate_defended, signal, keys, reference=defended_observed, mode="balanced_tensor")
```

Classification: all three branches are **contaminated**.

### Evidence D — RQ1 confirmatory wrappers call the contaminated Phase-4 branches

Primary RQ1 wrapper:

Source: `experiments/run_rq1_confirmatory.py:35-53`

```python
35 def command(branch: str, run: Path, group: int, python: str) -> list[str]:
36     if branch == "raw":
37         return [python, "experiments/run_phase4_harddiff_reparam_for_misselected.py", str(run),
...
40     if branch == "dna":
41         return [python, "experiments/run_phase4_dna_level1_forward_attack.py", str(run),
...
47     if branch == "dp":
48         return [python, "experiments/run_phase4_simple_defense_attack.py", str(run),
...
52                 "--clip-norm", "100", "--noise-multiplier", "0.00025",
53                 "--mc-noise-samples", "100", "--defense-seed", "314159265"]
```

Variant RQ1 wrapper:

Source: `experiments/run_rq1_variant_confirmatory.py:25-34`

```python
25 def _command(branch: str, run: Path, group: int, python: str, config: dict, target: Path) -> list[str]:
26     attack=config["attack"]
27     if branch == "raw":
28         return [python,"experiments/run_phase4_harddiff_reparam_for_misselected.py",str(run),...
29     if branch == "dna":
30         dna=config["dna_transform"]
31         return [python,"experiments/run_phase4_dna_level1_forward_attack.py",str(run),...
32     if branch == "dp":
33         dp=config["dp_distortion_matched"]
34         return [python,"experiments/run_phase4_simple_defense_attack.py",str(run),...
```

Classification: any RQ1 test run via either wrapper is **contaminated**.

### Evidence E — IEEE-CIS original development gate uses `simulate()` and all floating keys

Source: `experiments/run_ieee_cis_rq1_development_gate.py:276-329`

```python
276 def capture(group: dict, state_dict: dict, pos_weight: torch.Tensor, seed: int, lr: float) -> tuple[FraudMLP, nn.Module, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
...
284     observed = simulate(model, criterion, x, y, batches, rng_state, lr=lr, create_graph=False)
285     return model, criterion, x, y, rng_state, observed
...
302     model, criterion, original, true_labels, rng_state, observed = capture(
...
305     keys = [key for key, value in observed.items() if value.is_floating_point()]
...
314         signal = observed if method == "baseline" else {key: torch.zeros_like(value) for key, value in observed.items()}
...
328             simulated = simulate(model, criterion, candidate_x, candidate_y, batches, rng_state, lr=local_lr)
329             loss = update_objective(simulated, signal, keys, reference=observed, mode="balanced_tensor")
```

Classification: **contaminated**.

### Evidence F — Priority 6/8/9/10 SOTA-style tabular runner uses contaminated capture/simulate

PaySim path:

Source: `experiments/run_priority6_sota_style_attackers.py:364-390`

```python
364     local_seed = derive_seed(job["seed"], "priority6-paysim-local", job["group"])
365     model, criterion, original, true_labels, batches, rng, observed = capture_paysim(group, local_seed, 4)
...
373     observed_defended, payload = _defense_payload(observed, job["defense"], job["group"], job["restart"])
...
387             delta = simulate(model, criterion, reconstruction, true_labels, batches, rng)
388             defended = _candidate_defended(delta, payload, job["defense"])
389             penalty = 0.001 * _nonnegative_penalty(latent, meta)
390             loss = _loss(defended, signal, observed_defended, job["generation"], penalty)
```

IEEE-CIS path:

Source: `experiments/run_priority6_sota_style_attackers.py:432-474`

```python
432     model, criterion, original, true_labels, rng, observed = capture_ieee(
...
447     observed_defended, payload = _defense_payload(observed, job["defense"], job["group"], job["restart"])
...
472             delta = simulate(model, criterion, candidate_x, candidate_y, batches, rng, lr=bundle["local_lr"])
473             defended = _candidate_defended(delta, payload, job["defense"])
474             loss = _loss(defended, signal, observed_defended, job["generation"], torch.zeros((), dtype=numeric.dtype))
```

Classification: **contaminated**.

### Evidence G — RQ1-v2 IHT/sketch/lifted development and confirmatory code uses contaminated observed/simulate

RQ1-v2 confirmatory wrapper:

Source: `experiments/run_rq1_v2_confirmatory.py:25-82`

```python
25 def _command(branch: str, run: Path, group: int, python: str, config: dict, target: Path) -> list[str]:
...
27     if branch == "raw":
30             "experiments/run_phase4_harddiff_reparam_for_misselected.py",
...
43     if branch == "dna_v2":
47             "experiments/run_phase4_dna_v2_iht_attack.py",
...
76     if branch == "dp_v2":
80             "experiments/run_phase4_simple_defense_attack.py",
```

IHT attacker:

Source: `experiments/run_phase4_dna_v2_iht_attack.py:208-240`

```python
208     model, criterion, x, y, batches, rng, observed = _capture_for_iht(group, local_seed, protocol["batch_size"])
209     keys = _floating_keys(observed)
...
215     recovered_update, iht_diagnostics = _iht_recovered_update(observed, v2_config, group_id, args)
216     signal = recovered_update if method == "baseline" else {key: torch.zeros_like(value) for key, value in recovered_update.items()}
...
238         delta = simulate(model, criterion, reconstruction, y, batches, rng)
239         gradient_loss = update_objective(delta, signal, keys, reference=recovered_update, mode="balanced_tensor")
```

Sketch-space STE attacker:

Source: `experiments/run_phase4_dna_v2_sketch_space_attack.py:158-192`

```python
158     model, criterion, x, y, batches, rng, observed = capture(group, local_seed, protocol["batch_size"])
159     keys = _floating_keys(observed)
...
165     observed_sketch_payload = _observed_sketches(observed, v2_config, group_id)
166     observed_sketch = {key: value["sketch"].to(dtype=observed[key].dtype) for key, value in observed_sketch_payload.items()}
167     signal = observed_sketch if method == "baseline" else _zero_like_signal(observed_sketch)
...
190         delta = simulate(model, criterion, reconstruction, y, batches, rng)
191         candidate_sketch = _candidate_sketches(delta, observed_sketch_payload, args)
192         sketch_loss = update_objective(candidate_sketch, signal, list(signal), reference=observed_sketch, mode="balanced_tensor")
```

Lifted v2/raw-lift development attacker:

Source: `experiments/run_phase4_dna_v2_development_gate.py:94-125`

```python
94     model, criterion, x, y, batches, rng, observed = capture(group, local_seed, protocol["batch_size"])
95     keys = _floating_keys(observed)
...
101     defended_observed = _v2_lifted_update(observed, v2_config, group_id)
102     signal = defended_observed if method == "baseline" else {key: torch.zeros_like(value) for key, value in defended_observed.items()}
...
123         delta = simulate(model, criterion, reconstruction, y, batches, rng)
124         gradient_loss = update_objective(delta, signal, keys, reference=defended_observed, mode="balanced_tensor")
125         loss = gradient_loss
```

Classification: **contaminated**.

### Evidence H — Level-2 realization-known direct inversion uses contaminated capture/observed

Source: `experiments/run_rq1_conservative_level2.py:38-42`

```python
38         model,criterion,_,y,batches,rng,observed=capture(group,local_seed,baseline["protocol"]["batch_size"])
39         config=DNATransformConfig(256,0.08,0.88,0.45,derive_seed(DNA_RUN_SEED,"phase4-dna-transform",TARGET.name,group_id))
40         recovered,tensor_rows,matrix_rows=_recover_state(observed,config)
41         transform_rows.extend({"group_id":group_id,**row} for row in tensor_rows); block_rows.extend({"group_id":group_id,**row} for row in matrix_rows)
42         selected.append(_evaluate_group_with_recovered_signal(report_dir,group,group_id,restarts,model,criterion,y,batches,rng,observed,recovered))
```

Source: `experiments/run_phase4_dna_level2_direct_inversion.py:122-133`

```python
122 def _evaluate_group_with_recovered_signal(report_dir, group, group_id, restarts, model, criterion, y, batches, rng, observed, recovered):
123     keys = [key for key, value in observed.items() if value.is_floating_point()]
...
130             ("zero_update", {key: torch.zeros_like(value) for key, value in observed.items()}),
...
133             delta = simulate(model, criterion, loaded["reconstruction"], y, batches, rng)
```

Classification: **contaminated**.

### Evidence I — strong-DP exploratory gate reuses contaminated raw/simple-defense runners

Source: `experiments/run_strong_update_dp_attack.py:12-19`

```python
12 def job_run(job, python, out, target):
...
16     if job["branch"]=="raw":
17         cmd=[python,"experiments/run_phase4_harddiff_reparam_for_misselected.py",str(folder),...
18     else:
19         cmd=[python,"experiments/run_phase4_simple_defense_attack.py",str(folder),...
```

Classification: **contaminated**.

### Evidence J — early DLG-style single-sample helpers are parameter-only

Source: `attacks/gradient_inversion.py:37-52`

```python
37 def parameter_gradients(
...
44     """Compute per-parameter gradients for one known-label sample."""
45     model.eval()
46     logits = model(features)
47     loss = criterion(logits, label)
48     params = [parameter for parameter in model.parameters() if parameter.requires_grad]
49     gradients = torch.autograd.grad(
50         loss,
51         params,
52         create_graph=create_graph,
```

Source: `experiments/run_phase1_attack_validation.py:133-134`

```python
133 def _gradients(model, criterion, features: torch.Tensor, labels: torch.Tensor, create_graph: bool) -> list[torch.Tensor]:
134     return parameter_gradients(model, criterion, features.reshape(1, -1), labels.reshape(1, 1), create_graph=create_graph)
```

Source: `experiments/run_phase2_diagnostic.py:80-82`

```python
80     for sample_id, (original, label, validation_index) in enumerate(zip(features, labels, indices)):
81         observed = parameter_gradients(model, criterion, original.reshape(1, -1), label.reshape(1, 1))
82         for restart in range(RESTARTS):
```

Classification: **clean**.

## Audit table

| Test / result family | Dataset | File / experiment | Clean or contaminated? | Code evidence |
|---|---|---|---|---|
| Early Phase 1 single-sample attack validation | PaySim/creditcard tabular | `experiments/run_phase1_attack_validation.py`; `attacks/gradient_inversion.py` | **Clean** | Evidence J: `parameter_gradients()` uses `model.parameters()` with `requires_grad`; no buffers. |
| Early Phase 2 diagnostic gradient inversion | PaySim/creditcard tabular | `experiments/run_phase2_diagnostic.py`; `attacks/gradient_inversion.py` | **Clean** | Evidence J. |
| Phase 3 bounded Adam validation / pre-Phase-4 raw-style gates | PaySim | `experiments/run_phase3_adam_ladder.py`; `experiments/run_phase4_harddiff_reparam_for_misselected.py` | **Contaminated** | Evidence A+B+C raw branch. |
| RQ1 Stage-1 audit/smoke/calibration raw/DNA/DP development checks | PaySim | `protocols/stage1_execution_report_2026-09-12.md` commands using `run_phase4_harddiff_reparam_for_misselected.py`, `run_phase4_dna_level1_forward_attack.py`, `run_phase4_simple_defense_attack.py` | **Contaminated** | Evidence C. |
| RQ1 primary confirmatory: `dna_conservative` vs DP distortion-matched, original frozen execution | PaySim | `experiments/run_rq1_confirmatory.py`; report `protocols/confirmatory_execution_report_2026-09-13.md` | **Contaminated** | Evidence D calls Evidence C branch scripts. |
| RQ1 primary analysis output (`19/39`, p≈0.625 in execution report) | PaySim | `experiments/analyze_rq1_confirmatory.py` | **Contaminated inputs** | Analysis consumes artifacts generated by `run_rq1_confirmatory.py`; Evidence D. |
| RQ1 Group 2 medium confirmatory | PaySim | `experiments/run_rq1_variant_confirmatory.py`; report `reports/rq1_group2_report.md`; config `protocols/config/rq1_medium_confirmatory.json` | **Contaminated** | Evidence D variant wrapper calls Evidence C branch scripts. |
| RQ1 Group 2 stronger confirmatory | PaySim | `experiments/run_rq1_variant_confirmatory.py`; report `reports/rq1_group2_report.md`; config `protocols/config/rq1_stronger_confirmatory.json` | **Contaminated** | Evidence D variant wrapper calls Evidence C branch scripts. |
| RQ1 conservative Level-2 realization-known direct inversion | PaySim | `experiments/run_rq1_conservative_level2.py`; `experiments/run_phase4_dna_level2_direct_inversion.py`; report `reports/rq1_group2_report.md` | **Contaminated** | Evidence H. |
| RQ1 Group 3 Replication 1 | PaySim | `experiments/run_rq1_variant_confirmatory.py`; config `protocols/config/rq1_group3_replication_1.json`; report `reports/group3_replication_report.md` | **Contaminated** | Evidence D variant wrapper calls Evidence C branch scripts. |
| RQ1 Group 3 Replication 2 | PaySim | `experiments/run_rq1_variant_confirmatory.py`; config `protocols/config/rq1_group3_replication_2.json`; report `reports/group3_replication_report.md` | **Contaminated** | Evidence D variant wrapper calls Evidence C branch scripts. |
| RQ1 Group 3 Replication 3 / pooled exploratory analysis | PaySim | `experiments/run_rq1_variant_confirmatory.py`; config `protocols/config/rq1_group3_replication_3.json`; report `reports/group3_replication_report.md` | **Contaminated** | Evidence D variant wrapper calls Evidence C branch scripts. |
| RQ1 scope-boundary raw screen at records-per-group 8 | PaySim | `experiments/run_rq1_raw_scope_screening.py`; report `reports/rq1_scope_boundary_report.md` | **Contaminated** | `run_rq1_raw_scope_screening.py` dispatches to `run_phase4_harddiff_reparam_for_misselected.py`; Evidence C raw branch. |
| Priority 2 clean RQ1 confirmatory (`dna_conservative` vs `dp_0.00025`; report says observed 44/81, p=0.2526) | PaySim | `experiments/run_rq1_variant_confirmatory.py`; report `reports/priority2_rq1_clean_confirmatory_report.md`; config `protocols/config/rq1_priority2_clean_confirmatory.json` | **Contaminated** | Evidence D variant wrapper calls Evidence C branch scripts. |
| Strong-DP exploratory attack gates (`epsilon_100`, `epsilon_50`, `epsilon_10`, `epsilon_1`) | PaySim | `experiments/run_strong_update_dp_attack.py`; report `reports/strong_update_dp_work3_attack_gate_report.md` | **Contaminated** | Evidence I dispatches to contaminated raw/simple-defense runners. |
| DNA Transform v2 Step-5 raw-lift development gate (`4/8`, p=0.6367 both controls in report) | PaySim | `experiments/run_phase4_dna_v2_development_gate.py`; report `reports/dna_transform_v2_step5_development_calibration_report.md` / diagnosis report | **Contaminated** | Evidence G lifted v2 development attacker. |
| DNA Transform v2 structured sketch-space STE development variants | PaySim | `experiments/run_phase4_dna_v2_sketch_space_attack.py`; report `reports/dna_transform_v2_structured_attacker_development_report.md` | **Contaminated** | Evidence G sketch-space attacker. |
| DNA Transform v2 final compressed-sensing / IHT development gate (`IHT-0.20` n=8/n=24 validation) | PaySim | `experiments/run_phase4_dna_v2_iht_attack.py`; report `reports/dna_transform_v2_final_compressed_sensing_attacker_report.md` and expanded pilot report | **Contaminated** | Evidence G IHT attacker. |
| RQ1-v2 confirmatory (`n=176`, reported p≈0.701 for v2 contrast) | PaySim | `experiments/run_rq1_v2_confirmatory.py`; `experiments/run_phase4_dna_v2_iht_attack.py`; report `reports/rq1_v2_confirmatory_execution_report.md` | **Contaminated** | Evidence G confirmatory wrapper + IHT/raw/DP branch code. |
| Priority 5.3 IEEE-CIS original RQ1 development gate (`4/8`, `6/8` style raw gate outcomes) | IEEE-CIS | `experiments/run_ieee_cis_rq1_development_gate.py`; report `reports/priority5_ieee_cis_rq1_development_gate_report_20260916.md` | **Contaminated** | Evidence E. |
| Priority 6 SOTA-style attacker development gates, n=8 and n=24 | PaySim + IEEE-CIS | `experiments/run_priority6_sota_style_attackers.py`; report `reports/priority6_sota_style_attackers_report_20260916.md` | **Contaminated** | Evidence F. |
| Priority 6 PaySim v1-medium `GEN_IDLG_STYLE` confirmatory, n=39 | PaySim | `experiments/run_priority6_sota_style_attackers.py`; report `reports/priority6_paysim_v1_idlg_confirmatory_report_20260917.md` | **Contaminated** | Evidence F PaySim path. |
| Priority 6 IEEE-CIS v2 `GEN_COSINE_TV` confirmatory, n=39 | IEEE-CIS | `experiments/run_priority6_sota_style_attackers.py`; report `reports/priority6_ieee_v2_cosine_confirmatory_report_20260917.md` | **Contaminated** | Evidence F IEEE-CIS path. |
| Priority 8 v2 ratio 0.90 sensitivity gate, n=8 | IEEE-CIS | `experiments/run_priority6_sota_style_attackers.py`; report `reports/priority8_v2_ratio0p90_sensitivity_report_20260917.md` | **Contaminated** | Evidence F IEEE-CIS path. |
| Priority 9 v1 family sensitivity: conservative n=8, stronger n=8/n=24, stronger n=39 confirmatory | PaySim | `experiments/run_priority6_sota_style_attackers.py`; report `reports/priority9_v1_config_family_sensitivity_report_20260917.md` | **Contaminated** | Evidence F PaySim path. |
| Priority 10 v3 sensitivity-guided sketching probe, n=8 | IEEE-CIS | `experiments/run_priority6_sota_style_attackers.py`; report `reports/priority10_v3_sensitivity_guided_sketching_probe_20260921.md` | **Contaminated** | Evidence F IEEE-CIS path; v3 was added to same runner. |

## Notes on the user's specifically named items

| Named item | File / experiment found | Classification |
|---|---|---|
| RQ1 primary confirmatory v1 vs DP distortion-matched | `experiments/run_rq1_confirmatory.py` dispatching to `run_phase4_harddiff_reparam_for_misselected.py`, `run_phase4_dna_level1_forward_attack.py`, `run_phase4_simple_defense_attack.py`; report `protocols/confirmatory_execution_report_2026-09-13.md` | **Contaminated** |
| RQ1-v2 confirmatory, n=176, p≈0.701 | `experiments/run_rq1_v2_confirmatory.py` + `experiments/run_phase4_dna_v2_iht_attack.py`; report `reports/rq1_v2_confirmatory_execution_report.md` | **Contaminated** |
| Priority 5.3 IEEE-CIS original attacker development gate | `experiments/run_ieee_cis_rq1_development_gate.py`; report `reports/priority5_ieee_cis_rq1_development_gate_report_20260916.md` | **Contaminated** |
| Other tabular RQ1 tests in protocols/reports | Group2/Group3 variants, Priority2, Priority6/8/9/10, strong-DP, v2 development gates, scope boundary | **Contaminated**, except early Phase1/Phase2 DLG-style helpers using `parameter_gradients()` |

## Bottom-line scope statement

Within the audited tabular RQ1 pipeline, every post-Phase-3 bounded-update
development/confirmatory test I found in `protocols/` or `reports/` is affected by
BatchNorm-buffer contamination because it uses full-state update dictionaries from
`simulate()` and then matches/transforms all floating keys.

The only clean tabular gradient-inversion code path found is the early
single-sample `parameter_gradients()`/DLG-style path, which is not the pipeline used
for the later RQ1 confirmatory/development results.

