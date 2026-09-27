# Priority 10 urgent verification: update-vector definition in frozen RQ1 pipelines

**Date:** 2026-09-21  
**Scope:** code-level verification only. No experiment, target generation, attacker run,
protocol change, or result reinterpretation was performed.

## Direct answer

For the tabular RQ1 pipelines used by **Priority 6, Priority 8, and Priority 9**:

**YES — the real attack pipeline includes floating BatchNorm buffers
(`running_mean`, `running_var`) in the update dictionaries transformed by DNA v1/v2
and in the attacker matching loss.**

`num_batches_tracked` is also returned in the full state-delta dictionary by
`simulate()`, but the defense/loss code filters on `is_floating_point()` for the
transformed/sketched/loss-matched tensors. Therefore the material inclusion is the
floating BatchNorm running-stat buffers, not the integer counter.

For the image-domain **Priority 7** pipeline:

**NO — that pipeline constructs updates from trainable `named_parameters()` gradients
only, so BatchNorm buffers are not included there.**

For the generic `attacks/gradient_inversion.py` DLG-style helper:

**NO — it uses `model.parameters()` with `requires_grad`, so it does not include
buffers.** This helper is not the decisive path for the tabular Priority 6/8/9
SOTA-style attacker runs.

## Evidence: tabular Priority 6/8/9 path

Priority 8 and Priority 9 reports show they invoke
`experiments/run_priority6_sota_style_attackers.py`; therefore the Priority 6 runner
is the relevant real pipeline for those tabular SOTA-style gates/confirmatory runs.

### 1. `simulate()` returns parameters plus BatchNorm buffers

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

The project model contains BatchNorm layers:

Source: `models/fraud_mlp.py:16-32`

```python
16 self.network = nn.Sequential(
17     nn.Linear(input_dim, 128),
18     nn.BatchNorm1d(128),
...
22     nn.Linear(128, 64),
23     nn.BatchNorm1d(64),
...
27     nn.Linear(64, 32),
28     nn.BatchNorm1d(32),
...
32     nn.Linear(32, 1),
```

So `model.named_buffers()` contains BatchNorm buffers such as
`network.1.running_mean`, `network.1.running_var`, etc.

### 2. Priority 6 runner obtains `observed` from capture functions, then passes it to DNA

PaySim path:

Source: `experiments/run_priority6_sota_style_attackers.py:364-390`

```python
364 local_seed = derive_seed(job["seed"], "priority6-paysim-local", job["group"])
365 model, criterion, original, true_labels, batches, rng, observed = capture_paysim(group, local_seed, 4)
...
373 observed_defended, payload = _defense_payload(observed, job["defense"], job["group"], job["restart"])
...
387 delta = simulate(model, criterion, reconstruction, true_labels, batches, rng)
388 defended = _candidate_defended(delta, payload, job["defense"])
389 penalty = 0.001 * _nonnegative_penalty(latent, meta)
390 loss = _loss(defended, signal, observed_defended, job["generation"], penalty)
```

IEEE-CIS path:

Source: `experiments/run_priority6_sota_style_attackers.py:432-474`

```python
432 model, criterion, original, true_labels, rng, observed = capture_ieee(
...
447 observed_defended, payload = _defense_payload(observed, job["defense"], job["group"], job["restart"])
...
472 delta = simulate(model, criterion, candidate_x, candidate_y, batches, rng, lr=bundle["local_lr"])
473 defended = _candidate_defended(delta, payload, job["defense"])
474 loss = _loss(defended, signal, observed_defended, job["generation"], torch.zeros((), dtype=numeric.dtype))
```

Because candidate `delta` is explicitly recomputed by `simulate()`, it has the same
parameter-plus-buffer structure shown above.

### 3. DNA v1/v2 payload construction transforms/sketches all floating keys in `observed`

Priority 6 defense dispatcher:

Source: `experiments/run_priority6_sota_style_attackers.py:243-253`

```python
243 def _defense_payload(observed: dict[str, torch.Tensor], defense: str, group_id: int, realization: int):
244     if defense in V1_CONFIGS:
245         config = DNATransformConfig(**V1_CONFIGS[defense])
246         transmitted = _transmit_observed(observed, config)
247         plan = _surrogate_plan_from_state(observed, config, group_id, realization)
248         return transmitted, ("v1", plan)
249     if defense in V2_CONFIGS:
250         config = DNATransformV2Config(**V2_CONFIGS[defense])
251         payload = _observed_sketches(observed, config, group_id)
252         sketch = {key: item["sketch"].to(dtype=observed[key].dtype) for key, item in payload.items()}
253         return sketch, ("v2", payload)
```

DNA v1 helper:

Source: `experiments/run_phase4_dna_level1_forward_attack.py:57-99`

```python
57 def _surrogate_plan_from_state(state_delta, config, group_id, realization_id):
58     plan = {}
59     for tensor_index, (name, value) in enumerate(state_delta.items()):
60         if not value.is_floating_point():
61             continue
...
73         plan[name] = rows
...
91 def _transmit_observed(observed, config):
92     transmitted = {}
93     for tensor_index, (name, delta) in enumerate(observed.items()):
94         if delta.is_floating_point():
95             array, _ = transform_update_array(delta.detach().cpu().numpy(), config, tensor_index=tensor_index)
96             transmitted[name] = torch.from_numpy(array).to(dtype=delta.dtype)
97         else:
98             transmitted[name] = delta.detach().cpu()
99     return transmitted
```

DNA v2 helper:

Source: `experiments/run_phase4_dna_v2_sketch_space_attack.py:107-126`

```python
107 def _observed_sketches(
...
113     for tensor_index, (name, value) in enumerate(observed.items()):
114         if not value.is_floating_point():
115             continue
116         sketch, metadata = transform_update_array_v2(
117             value.detach().cpu().numpy().astype(np.float32, copy=False),
118             config,
119             tensor_index=tensor_index,
120             quantization_seed=group_id,
121         )
122         sketches[name] = {
123             "sketch": torch.from_numpy(sketch.copy()).to(dtype=value.dtype),
124             "quantization_delta": float(metadata.quantization_delta),
125             "tensor_index": tensor_index,
126         }
```

The transform functions themselves operate on whichever array they are handed; they
do not distinguish parameters from buffers:

Source: `dna_encoder/transform_defense.py:36-53`

```python
36 def transform_update_array(
...
49     original = np.asarray(array, dtype=np.float32)
...
52     flat = original.reshape(-1).astype(np.float32, copy=True)
53     transformed = flat.copy()
```

Source: `dna_encoder/transform_defense_v2.py:61-82`

```python
61 def transform_update_array_v2(
...
77     original = np.asarray(array, dtype=np.float32)
78     flat = original.reshape(-1).astype(np.float64, copy=False)
79     original_size = int(flat.size)
80     padded_size = _next_power_of_two(max(1, original_size))
81     sketch_size = _sketch_size(padded_size, config.compression_ratio)
82     seed = _derive_seed(config.seed, tensor_index)
```

### 4. The attacker loss also uses all floating keys in the defended signal

`GEN_COSINE_TV`:

Source: `experiments/run_priority6_sota_style_attackers.py:234-240`

```python
234 def _global_cosine(candidate: dict[str, torch.Tensor], signal: dict[str, torch.Tensor]) -> torch.Tensor:
235     keys = [key for key in signal if key in candidate and signal[key].is_floating_point()]
...
238     left = torch.cat([candidate[key].reshape(-1) for key in keys])
239     right = torch.cat([signal[key].reshape(-1).to(dtype=left.dtype) for key in keys])
240     return 1.0 - F.cosine_similarity(left, right, dim=0, eps=1e-12)
```

`GEN_IDLG_STYLE`:

Source: `experiments/run_priority6_sota_style_attackers.py:334-340`

```python
334 def _loss(candidate, signal, reference, generation, range_penalty):
335     if generation == "GEN_COSINE_TV":
336         return _global_cosine(candidate, signal) + range_penalty
337     if generation == "GEN_IDLG_STYLE":
338         keys = [key for key, value in signal.items() if value.is_floating_point()]
339         return update_objective(candidate, signal, keys, reference=reference, mode="balanced_tensor") + range_penalty
340     raise ValueError(generation)
```

The objective comment itself says it compares floating parameters and BN buffers:

Source: `experiments/phase3_bounded_validation.py:35-57`

```python
35 def update_objective(candidate, observed, keys, reference=None, mode="equal_mse"):
36     """Compare transmitted floating parameters and BN buffers."""
...
41         for key in keys:
42             candidate_value = candidate[key].reshape(-1)
43             observed_value = observed[key].reshape(-1)
...
57         return torch.stack(terms).mean()
```

## Evidence: Priority 7 image-domain path

The CIFAR-10 image-domain runner constructs the update from trainable parameters only:

Source: `experiments/run_priority7_image_domain_gate.py:106-111`

```python
106 def _named_update(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, lr: float) -> dict[str, torch.Tensor]:
107     logits = model(images)
108     loss = F.cross_entropy(logits, labels)
109     params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
110     grads = torch.autograd.grad(loss, [param for _, param in params], create_graph=images.requires_grad)
111     return {name: -lr * grad for (name, _), grad in zip(params, grads)}
```

Then its loss only matches keys from that parameter-only signal:

Source: `experiments/run_priority7_image_domain_gate.py:114-126`

```python
114 def _global_cosine(candidate: dict[str, torch.Tensor], signal: dict[str, torch.Tensor]) -> torch.Tensor:
115     keys = [key for key in signal if key in candidate and signal[key].is_floating_point()]
...
121 def _dict_l2(candidate: dict[str, torch.Tensor], signal: dict[str, torch.Tensor]) -> torch.Tensor:
122     losses = []
123     for key, target in signal.items():
124         if key in candidate and target.is_floating_point():
125             losses.append(F.mse_loss(candidate[key], target.to(dtype=candidate[key].dtype, device=candidate[key].device)))
126     return torch.stack(losses).mean()
```

So the Priority 7 image-domain pipeline does not include BatchNorm buffers in its
update/loss vector.

## Evidence: generic `attacks/gradient_inversion.py`

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

This helper is parameter-gradient-only and does not include buffers.

## Status for supervisor decision

The code evidence answers the urgent question as follows:

- **Priority 6 / Priority 8 / Priority 9 tabular RQ1 attacker pipelines:** YES,
  floating BatchNorm buffers are included in the real defended update vector and
  attacker matching loss.
- **Priority 7 image-domain RQ1 attacker pipeline:** NO, only trainable parameter
  gradients are included.
- **Generic `attacks/gradient_inversion.py`:** NO, only trainable parameter
  gradients are included.

Per instruction, I made no code fix and did not reinterpret or modify any frozen RQ1
conclusion.

