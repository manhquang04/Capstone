# Priority 27 — gaps and reusable evaluation harness report

Date: 2026-09-30  
Amendment: `protocols/amendments/2026-09-29_priority27_gaps_and_harness.md`  
Amendment SHA-256: `042b84b2339c3525bc7c14146349277f8c2eba4dd1e07a6c640a1531714366c5`

## Executive summary

Priority 27 was run under the pre-registered reduced plan in the amendment. The full requested scope was estimated to exceed the approximately four-day compute budget, so the amendment froze the allowed reduction before execution: C4 and C5 were dropped, and C2 grids were shrunk to five initial multipliers plus one pre-declared extension.

Main outcomes:

- Part H harness was implemented and unit-tested.
- C1 found no qualified gradient-only instrument. All Adam and SGD cells failed the n=8 positive-control gate, so no C1 n=24 pilot was run.
- C2 found bracketed utility-matched comparators for `full_state_single_clip` and `per_tensor_clip`, both selecting `sigma=3e-05` for v1-conservative and v2. `fedbn_trainable_only` was not available: no tested sigma met the utility threshold, even after the pre-declared extension.
- C3 therefore ran only the BN-statistics instrument (T1), for four available transform × DP-variant cells. In all four cells, the transform branch had significantly larger standardized batch-mean reconstruction MSE than the matched DP branch after Holm correction. This is an RQ1 extension analysis only; it does not replace the paper's registered RQ1 answer.

## Files created or modified

### Protocol and harness

| File | SHA-256 |
|---|---:|
| `protocols/amendments/2026-09-29_priority27_gaps_and_harness.md` | `042b84b2339c3525bc7c14146349277f8c2eba4dd1e07a6c640a1531714366c5` |
| `experiments/harness/__init__.py` | `227033c254200b2410427fa22940ef98917535ff4bf0333373e8d49e90d1c8dc` |
| `experiments/harness/interfaces.py` | `7c6f4f4eacc42607d59b8990774deaceaaf26352d16b3c37a61eaa67dbc949b8` |
| `experiments/harness/metrics.py` | `175084d771d0ae3b6a86ad12802166c0b456f71df2b82e82216bc8c177270670` |
| `experiments/harness/attacks.py` | `f65fa898763d5cf58f93f4c49f3b14ee850b925ae4da9951a80f30e0d8fdf29f` |
| `experiments/harness/validators.py` | `2f8dc3a8bb9e6ce2f2a03e9e4d508915f7f51ecc071e7816be9f0265178beeed` |
| `experiments/harness/reporting.py` | `2f7d7dca8035509091b7c1e0ae43445d800e41850776c200245dd8cbccb10749` |
| `tests/test_priority27_harness.py` | `16623ccecf100297792d75883875ea670c07fd8677e7085bf12fffdcbfad415c` |

### Priority 27 scripts and config

| File | SHA-256 |
|---|---:|
| `experiments/priority27_c1_gradient_gate.py` | `48809da3abdec20166afbc8d67bd3ac574e73a40cf85452dad71c46922c36101` |
| `experiments/priority27_dp_variants.py` | `a7495100b0bdebf1cec6a882a1f120eb6c6112e5069e668efb3e12e125713bd4` |
| `experiments/run_fraud_fl_dp_priority27.py` | `594c95daa7cf17b4e82973e1b984b7e4c720afd2a329c1e62e22b76ee99d775d` |
| `experiments/priority27_clip_probe.py` | `e6aef4867e2c7d0701c933c0cc426bbc5570b109c4831ec652293f16c0070f5c` |
| `experiments/run_priority27_utility_grid.py` | `4050b7ddb8272c86010a8dc5efa9e3c7dc1a1b80e466a7459897c88015da68f0` |
| `experiments/analyze_priority27_utility_grid.py` | `8e59fe6f23231779d3b8f3a30261e9dea5697692f3afbd17c268b784ba4e9f5b` |
| `experiments/priority27_c3_t1_confirm.py` | `53acfacd57fe54c938d0e35f6d2b2d7505c11b873844bc7e8e13f2f301eb2cda` |
| `protocols/config/priority27_c2_utility_grid.json` | `b1b1e41a3eb89d43992908bf7ae9d69fdde864bffd5b01ac6a0d80b71dcd3f5e` |

## Commands run

Harness checks:

```bash
PYTHONPATH=. .venv-phase1/bin/python -m pytest tests/test_priority27_harness.py -q
```

C1 target generation used `experiments/create_phase4_source_disjoint_targets.py` with 8 groups, 1 fraud per group, and batch sizes 4, 8, and 16 for each of Adam and SGD. Each generated target bundle reported `max_overlap_with_existing_targets=0`.

C1 gate commands followed this pattern:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority27_c1_gradient_gate.py \
  --target artifacts/priority27_gaps_harness/c1_dev_<setting>_bs<batch>_targets_20260929/paysim_priority27_c1_dev_<setting>_bs<batch>_targets.pt \
  --output-dir artifacts/priority27_gaps_harness/c1_dev_<setting>_bs<batch>_gate_20260929 \
  --seed <seed> --setting <adam|sgd> --batch-size <batch> \
  --steps 2000 --restarts 8 --attacker-lr 0.05 --workers 4
```

C2 clip probe:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority27_clip_probe.py \
  --seeds 270101 270102 \
  --output-dir artifacts/priority27_gaps_harness/c2_clip_probe_20260929 \
  --max-rows 500000 --num-rounds 50 --num-clients 3
```

C2 initial grid:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority27_utility_grid.py \
  --config protocols/config/priority27_c2_utility_grid.json \
  --output-dir artifacts/priority27_gaps_harness/c2_utility_grid_initial_20260929 \
  --workers 9
```

C2 pre-declared extension:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority27_utility_grid.py \
  --config protocols/config/priority27_c2_utility_grid.json \
  --output-dir artifacts/priority27_gaps_harness/c2_utility_grid_extension_20260929 \
  --workers 9 --extension-only
```

C2 final analysis:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/analyze_priority27_utility_grid.py \
  --config protocols/config/priority27_c2_utility_grid.json \
  --run-dir artifacts/priority27_gaps_harness/c2_utility_grid_initial_20260929 \
  --extension-run-dir artifacts/priority27_gaps_harness/c2_utility_grid_extension_20260929 \
  --output-dir results/priority27_gaps_harness/c2_utility_grid_with_extension_20260930
```

C3 target generation used `experiments/create_phase4_source_disjoint_targets.py` for an n=8 development bundle and an n=39 confirmatory bundle; both reported zero overlap with existing target sources.

C3 confirmatory:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority27_c3_t1_confirm.py \
  --dev-target artifacts/priority27_gaps_harness/c3_t1_dev_targets_20260930/paysim_priority27_c3_t1_dev_targets.pt \
  --confirm-target artifacts/priority27_gaps_harness/c3_t1_confirm_targets_20260930/paysim_priority27_c3_t1_confirm_targets.pt \
  --utility-summary results/priority27_gaps_harness/c2_utility_grid_with_extension_20260930/utility_grid_summary.json \
  --output-dir artifacts/priority27_gaps_harness/c3_t1_confirm_20260930 \
  --seed 2026093003
```

## Part H — harness unit tests

Result:

```text
6 passed
```

Covered checks include interface behavior, tabular/image scoring, positive-control gate refusal, server-secret declaration enforcement, DP spec recording, and decoy scoring against the current target.

## Part C1 — gradient-only instrument qualification

Gate rule: exact one-sided sign test against both Prior and decoy at n=8. A cell qualifies only if it beats both controls. No cell qualified; therefore no C1 n=24 pilot or C3 gradient-only confirmatory arm was run.

| Setting | Batch | Wins vs Prior | p vs Prior | Wins vs decoy | p vs decoy | Qualified | Target SHA-256 |
|---|---:|---:|---:|---:|---:|---|---|
| Adam | 4 | 3/8 | 0.855469 | 7/8 | 0.035156 | No | `585bf458acfb4ff30a0e6e7d4b64b0094853d1595b60e51f42cbb5ae206706a9` |
| Adam | 8 | 1/8 | 0.996094 | 6/8 | 0.144531 | No | `b458be196d337e2818606fad4d0d75d01de1f684d8bd8569568fb7699684d52b` |
| Adam | 16 | 0/8 | 1.000000 | 6/8 | 0.144531 | No | `d94b1f7c8cded8e94fb2bb829e7d901bfdfde6faf0140667ff4e0101166cb686` |
| SGD | 4 | 2/8 | 0.964844 | 5/8 | 0.363281 | No | `b9177d7d1d03c231af1d507a7e0b9355da8e53cdf0cec15b75ed187db853fccd` |
| SGD | 8 | 4/8 | 0.636719 | 4/8 | 0.636719 | No | `d182b1dae0b10705c84875fd4fe5db1eaf8f25b7fdbf11ab1e8cbe32a515ed1f` |
| SGD | 16 | 3/8 | 0.855469 | 5/8 | 0.363281 | No | `73e66d06c35c2dbeb2f6d72059071faaf5ba25653781d6c0bc8c878dd5e1726d` |

Adam-sign diagnostic:

| Adam batch | cosine(step, -lr sign(g)) | sign agreement | corr(|step|, |g|) |
|---:|---:|---:|---:|
| 4 | 0.992070 | 0.999901 | 0.143178 |
| 8 | 0.991852 | 0.999941 | 0.113472 |
| 16 | 0.991517 | 0.999901 | 0.080330 |

This supports the pre-registered diagnostic that the first fresh-state Adam step is nearly a sign update and preserves little coordinate magnitude information.

## Part C2 — DP comparator calibration

Clip probe summary:

| Quantity | Value |
|---|---:|
| Full-state C95 | 283.364730834961 |
| Trainable-only C95 | 1.3000768184661866 |
| Dominant per-tensor C95 | `network.1.running_var = 283.20083770751955` |
| Clip probe artifact | `artifacts/priority27_gaps_harness/c2_clip_probe_20260929/clip_probe_summary.json` |
| Clip probe SHA-256 | `8a8ade50170aab2f452521ac20a31dd1372fdaed7d6ae83a2c18257f0ab1de95` |

Grid execution:

| Artifact | SHA-256 |
|---|---:|
| `artifacts/priority27_gaps_harness/c2_utility_grid_initial_20260929/execution_summary.json` | `379e9fe7eb73633b3834fa51857b8b529348a4675e2262078162ef1e9a0835bd` |
| `artifacts/priority27_gaps_harness/c2_utility_grid_extension_20260929/execution_summary.json` | `efc7bdac39c1625a04c7dd865e3861a39b92542c1853309419dde464dd42f8d9` |
| `results/priority27_gaps_harness/c2_utility_grid_with_extension_20260930/utility_grid_summary.json` | `3b97e968c620f32bdd808c88b3dab19790a1b452b5b72202acb6c4d91ecf26be` |

Final selected utility-matched comparators:

| DP variant | Transform | Threshold ΔF1 | Selected sigma | Mean ΔF1 | SD ΔF1 | Mean ΔAUC | Mean ΔPR-AUC | Bracketed | ε one release | ε 50 releases |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| full_state_single_clip | v1 conservative | -0.0036 | 3e-05 | -0.003056 | 0.017002 | -0.000026 | -0.002970 | Yes | 5.557e8 | 2.778e10 |
| full_state_single_clip | v2 0.95/0.01 | -0.0167 | 3e-05 | -0.003056 | 0.017002 | -0.000026 | -0.002970 | Yes | 5.557e8 | 2.778e10 |
| per_tensor_clip | v1 conservative | -0.0036 | 3e-05 | -0.001026 | 0.021670 | -0.000355 | -0.005090 | Yes | 5.557e8 | 2.778e10 |
| per_tensor_clip | v2 0.95/0.01 | -0.0167 | 3e-05 | -0.001026 | 0.021670 | -0.000355 | -0.005090 | Yes | 5.557e8 | 2.778e10 |
| fedbn_trainable_only | v1 conservative | -0.0036 | not available | - | - | - | - | No | - | - |
| fedbn_trainable_only | v2 0.95/0.01 | -0.0167 | not available | - | - | - | - | No | - | - |

For `fedbn_trainable_only`, even the smallest tested sigma in the initial grid/extension did not meet either transform's threshold; the closest recorded value at sigma `1e-06` had mean ΔF1 `-0.247824`, far below the thresholds. It was therefore excluded from C3.

## Part C3 — RQ1-style extension with T1 BN instrument

Targets:

| Bundle | Path | SHA-256 | Overlap |
|---|---|---:|---:|
| Development n=8 | `artifacts/priority27_gaps_harness/c3_t1_dev_targets_20260930/paysim_priority27_c3_t1_dev_targets.pt` | `50c25dad67beae8687c0deafe003130b8e81c9499a6637eaa275af4055d034f7` | 0 |
| Confirmatory n=39 | `artifacts/priority27_gaps_harness/c3_t1_confirm_targets_20260930/paysim_priority27_c3_t1_confirm_targets.pt` | `71eb36f55fd29fe0d04c3e2cdd2f7c626453815c3975187c07dea2b9159e7b04` | 0 |

Output artifacts:

| Artifact | SHA-256 |
|---|---:|
| `artifacts/priority27_gaps_harness/c3_t1_confirm_20260930/c3_t1_summary.json` | `fd0a6573ceeb15014b98c79270152f6024834555d932aa132ee35b912df62b92` |
| `artifacts/priority27_gaps_harness/c3_t1_confirm_20260930/confirm_t1_comparisons.csv` | `1dc43555acca4b6f06016b9b025c3fdbdf416dcf0bc3ce7bcd347ef4bcc3f159` |

Branch MSE summary, standardized input-space batch mean:

| Branch | Mean MSE | Median MSE |
|---|---:|---:|
| none | 6.872e-12 | 9.873e-13 |
| population_mean | 1.650360 | 0.376834 |
| v1_conservative | 3.312579 | 0.131084 |
| v2_0p95_eta0p01 | 9.181747 | 0.902777 |
| DP full_state for v1 | 1.210833 | 0.037985 |
| DP full_state for v2 | 1.206129 | 0.026529 |
| DP per_tensor for v1 | 0.796705 | 0.033901 |
| DP per_tensor for v2 | 0.800048 | 0.023474 |

Pre-registered DNA-vs-DP exact sign tests:

| Transform | DP variant | DNA wins | DP wins | Ties | n | p(DNA>DP) | p(DP>DNA) | Holm p(DNA>DP) | Holm p(DP>DNA) | Descriptive label |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v1 conservative | full_state_single_clip | 30 | 9 | 0 | 39 | 5.325e-4 | 0.999853 | 5.325e-4 | 1.000000 | transform better |
| v1 conservative | per_tensor_clip | 33 | 6 | 0 | 39 | 7.150e-6 | 0.999999 | 1.430e-5 | 1.000000 | transform better |
| v2 0.95/0.01 | full_state_single_clip | 37 | 2 | 0 | 39 | 1.421e-9 | 1.000000 | 5.683e-9 | 1.000000 | transform better |
| v2 0.95/0.01 | per_tensor_clip | 37 | 2 | 0 | 39 | 1.421e-9 | 1.000000 | 5.683e-9 | 1.000000 | transform better |

Note on v1 adaptive BN evaluation: the C3 script evaluates the pre-registered v1 BN estimators that are meaningful for the 128-dimensional `running_mean` statistic in this run: direct transformed BN and debiased estimate `(T - m * mean(T))/(1-m)`. The per-target comparison uses the best of these estimators. This is favorable to the attacker / unfavorable to the transform. The block-sum and surrogate-permutation ideas from the amendment were not separately instantiated for this BN vector in this run.

## C4 and C5

C4 and C5 were not run. This was frozen before execution in the amendment's compute-reduction plan:

1. Drop C4.
2. Drop C5.
3. Shrink C2 grids.

No C4/C5 artifacts or target sets were created.

## Final checks

The following checks were run after implementation/reporting:

```bash
PYTHONPATH=. .venv-phase1/bin/python -m pytest tests/test_priority27_harness.py -q
PYTHONPATH=. .venv-phase1/bin/python -m py_compile \
  experiments/priority27_c1_gradient_gate.py \
  experiments/priority27_clip_probe.py \
  experiments/run_priority27_utility_grid.py \
  experiments/analyze_priority27_utility_grid.py \
  experiments/priority27_c3_t1_confirm.py \
  experiments/run_fraud_fl_dp_priority27.py \
  experiments/priority27_dp_variants.py
git diff --check
pgrep -fl 'priority27|run_fraud_fl_dp_priority27|run_priority27'
```

Final check outputs are summarized in the handoff message. No workload was intentionally left running.

## Plain summary

Under the reduced Priority 27 plan, the only qualified instrument remained the BN-statistics channel. No gradient-only instrument qualified at n=8, including FedSGD-style secondary cells. Utility-matched DP was bracketed for two DP variants (`full_state_single_clip` and `per_tensor_clip`) but not for `fedbn_trainable_only`. In the T1 extension tests that were actually available, both v1-conservative and v2 showed significantly larger standardized batch-mean reconstruction MSE than the utility-matched DP comparators after Holm correction. These findings are extension evidence and do not replace the registered RQ1 answer already reported for the paper.
