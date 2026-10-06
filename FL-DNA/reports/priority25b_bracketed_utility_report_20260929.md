# Priority 25b — Bracketed utility-matched DP grid and T2 gate attempt

Date: 2026-09-29  
Status: COMPLETED

## Scope and protocol status

This priority extends Priority 25 without modifying any earlier artifact or report.  The protocol amendment was frozen before any Priority 25b data were generated:

- Amendment: `protocols/amendments/2026-09-29_priority25b_bracketed_utility.md`
- SHA-256: `3d686a135ab64e5c3c8c38c542beed40e06cccf28adee4e3e9114df8d7180dee`

Pre-declared handling was applied exactly:

- Priority 25 T1 × utility-matched results remain reported, but are labeled as using a comparator that failed the bracketing requirement.
- In the combined primary RQ1 family, Priority 25 T1 × utility-matched tests are replaced by the Priority 25b T1 × utility-matched tests below.
- Priority 24 T1 × distortion-matched results are unchanged.

No files under `Latex/` were edited.

## Commands run

Utility-grid scripts:

```bash
PYTHONPATH=. .venv-phase1/bin/python -m py_compile \
  experiments/run_priority25b_utility_grid.py \
  experiments/analyze_priority25b_utility_grid.py
```

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority25b_utility_grid.py \
  --config protocols/config/priority25b_utility_grid.json \
  --output-dir artifacts/priority25b_bracketed_utility/utility_grid_initial_20260929 \
  --workers 9
```

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/analyze_priority25b_utility_grid.py \
  --config protocols/config/priority25b_utility_grid.json \
  --run-dir artifacts/priority25b_bracketed_utility/utility_grid_initial_20260929 \
  --output-dir results/priority25b_bracketed_utility/utility_grid_initial_20260929
```

T1 target generation and utility-matched confirmatory:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority25b_bracketed_utility/t1_dev_targets_20260929 \
  --output-name paysim_priority25b_t1_dev_targets.pt \
  --groups 8 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092971 \
  --purpose 'Priority 25b T1 utility-matched development n8'
```

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority25b_bracketed_utility/t1_confirm_targets_20260929 \
  --output-name paysim_priority25b_t1_confirm_targets.pt \
  --groups 39 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092972 \
  --purpose 'Priority 25b T1 utility-matched confirmatory n39'
```

Initial T1 run used the correct scientific inputs but inherited the old Priority 25 amendment metadata field from the reused script; it is retained as a disclosed technical-discard artifact:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority25_t1_utility_matched_rq1.py \
  --dev-target artifacts/priority25b_bracketed_utility/t1_dev_targets_20260929/paysim_priority25b_t1_dev_targets.pt \
  --confirm-target artifacts/priority25b_bracketed_utility/t1_confirm_targets_20260929/paysim_priority25b_t1_confirm_targets.pt \
  --utility-summary results/priority25b_bracketed_utility/utility_grid_initial_20260929/utility_grid_summary.json \
  --output-dir artifacts/priority25b_bracketed_utility/t1_utility_run_20260929 \
  --seed 2026092971
```

The script was then patched to accept an explicit `--amendment` argument and rerun with the same scientific inputs:

```bash
PYTHONPATH=. .venv-phase1/bin/python -m py_compile experiments/priority25_t1_utility_matched_rq1.py
```

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority25_t1_utility_matched_rq1.py \
  --dev-target artifacts/priority25b_bracketed_utility/t1_dev_targets_20260929/paysim_priority25b_t1_dev_targets.pt \
  --confirm-target artifacts/priority25b_bracketed_utility/t1_confirm_targets_20260929/paysim_priority25b_t1_confirm_targets.pt \
  --utility-summary results/priority25b_bracketed_utility/utility_grid_initial_20260929/utility_grid_summary.json \
  --output-dir artifacts/priority25b_bracketed_utility/t1_utility_run_20260929_replay1 \
  --amendment protocols/amendments/2026-09-29_priority25b_bracketed_utility.md \
  --seed 2026092971
```

T2 gate and retry:

```bash
PYTHONPATH=. .venv-phase1/bin/python -m py_compile experiments/priority25b_t2_gate.py
```

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority25b_bracketed_utility/t2_gate_targets_20260929 \
  --output-name paysim_priority25b_t2_gate_targets.pt \
  --groups 8 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092981 \
  --purpose 'Priority 25b T2 gate n8'
```

Two T2 technical attempts failed before producing scientific summaries:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority25b_t2_gate.py \
  --target artifacts/priority25b_bracketed_utility/t2_gate_targets_20260929/paysim_priority25b_t2_gate_targets.pt \
  --output-dir artifacts/priority25b_bracketed_utility/t2_gate_run_20260929 \
  --seed 2026092981 --steps 2000 --restarts 8 --attacker-lr 0.05 --workers 4
```

This failed because the observed update retained an autograd graph.  The script was patched to detach the observed trainable vector.

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority25b_t2_gate.py \
  --target artifacts/priority25b_bracketed_utility/t2_gate_targets_20260929/paysim_priority25b_t2_gate_targets.pt \
  --output-dir artifacts/priority25b_bracketed_utility/t2_gate_run_20260929_replay1 \
  --seed 2026092981 --steps 2000 --restarts 8 --attacker-lr 0.05 --workers 4
```

This failed because final scoring called `simulate(... create_graph=False)` inside `torch.no_grad()`.  The no-grad scope was restricted to input-space scoring only.

Successful T2 gate run:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority25b_t2_gate.py \
  --target artifacts/priority25b_bracketed_utility/t2_gate_targets_20260929/paysim_priority25b_t2_gate_targets.pt \
  --output-dir artifacts/priority25b_bracketed_utility/t2_gate_run_20260929_replay2 \
  --seed 2026092981 --steps 2000 --restarts 8 --attacker-lr 0.05 --workers 4
```

The first retry target-generation command failed before writing output because the output directory did not exist.  After creating the directory, target generation succeeded:

```bash
mkdir -p artifacts/priority25b_bracketed_utility/t2_retry_targets_20260929
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority25b_bracketed_utility/t2_retry_targets_20260929 \
  --output-name paysim_priority25b_t2_retry_targets.pt \
  --groups 8 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092982 \
  --purpose 'Priority 25b T2 predeclared retry gate n8'
```

Pre-declared T2 retry:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority25b_t2_gate.py \
  --target artifacts/priority25b_bracketed_utility/t2_retry_targets_20260929/paysim_priority25b_t2_retry_targets.pt \
  --output-dir artifacts/priority25b_bracketed_utility/t2_retry_run_20260929 \
  --seed 2026092982 --steps 8000 --restarts 16 --attacker-lr 0.05 --workers 4
```

## Part 1 — Bracketed utility grid

Clip norm was frozen at `C = 259.0841131896973`.  The initial 8-point grid was sufficient; the pre-declared `[1e-2, 3e-2]` extension was not run.

| sigma | n | mean ΔF1 | SD ΔF1 | mean ΔAUC | mean ΔPR-AUC | ε one release | ε 50 releases |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-06 | 16 | 0.000384 | 0.020271 | 0.000888 | 0.003290 | 5.0001e+11 | 2.5000e+13 |
| 3e-06 | 16 | -0.000582 | 0.024029 | 0.000400 | 0.005657 | 5.5557e+10 | 2.7778e+12 |
| 1e-05 | 16 | 0.004805 | 0.023041 | 0.000463 | 0.004610 | 5.0005e+09 | 2.5000e+11 |
| 3e-05 | 16 | -0.005682 | 0.016052 | 0.000428 | -0.002713 | 5.5572e+08 | 2.7779e+10 |
| 1e-04 | 16 | -0.039918 | 0.025950 | -0.000265 | -0.037915 | 5.0048e+07 | 2.5003e+09 |
| 3e-04 | 16 | -0.159464 | 0.050377 | -0.008545 | -0.180383 | 5.5716e+06 | 2.7789e+08 |
| 1e-03 | 16 | -0.701673 | 0.088494 | -0.436922 | -0.741901 | 5.0480e+05 | 2.5034e+07 |
| 3e-03 | 16 | -0.726799 | 0.042028 | -0.545189 | -0.753371 | 5.7242e+04 | 2.7891e+06 |

Matching rule results:

| Transform | threshold ΔF1 | selected sigma | selected mean ΔF1 | next sigma | next mean ΔF1 | verdict |
|---|---:|---:|---:|---:|---:|---|
| v1 conservative | -0.0036 | 1e-05 | 0.004805 | 3e-05 | -0.005682 | BRACKETED |
| v2 0.95/0.01 | -0.0167 | 3e-05 | -0.005682 | 1e-04 | -0.039918 | BRACKETED |

## Part 2 — T1 utility-matched confirmatory

Development qualification on fresh targets:

| Instrument | n | wins vs Prior | p vs Prior | wins vs decoy | p vs decoy | qualified |
|---|---:|---:|---:|---:|---:|---|
| T1 BN running-mean recovery | 8 | 8 | 0.00390625 | 8 | 0.00390625 | yes |

Confirmatory DNA-vs-utility-matched-DP:

| Transform | DNA wins | DP wins | ties | non-tied n | p(DNA>DP) | p(DP>DNA) | median MSE ratio DNA/DP | mean MSE ratio DNA/DP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 conservative | 38 | 1 | 0 | 39 | 7.275957614183426e-11 | 0.999999999998181 | 33.05883630632897 | 233.98791548206495 |
| v2 0.95/0.01 | 39 | 0 | 0 | 39 | 1.8189894035458565e-12 | 1.0 | 36.35513451773111 | 82.35164958869629 |

Branch-vs-control descriptive table on the same confirmatory targets:

| Branch | mean MSE | median MSE | wins vs none | p vs none | wins vs Prior | p vs Prior | wins vs decoy | p vs decoy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 1.588e-12 | 9.116e-13 | — | — | 39 | 1.8189894035458565e-12 | 39 | 1.8189894035458565e-12 |
| v1 conservative | 2.466396 | 0.163946 | 39 | 1.8189894035458565e-12 | 25 | 0.05406451070302866 | 30 | 0.00053250981727615 |
| v2 0.95/0.01 | 4.016173 | 0.653771 | 39 | 1.8189894035458565e-12 | 4 | 0.9999999819556251 | 24 | 0.09979543345980349 |
| DP utility for v1 | 0.374605 | 0.003723 | 39 | 1.8189894035458565e-12 | 39 | 1.8189894035458565e-12 | 38 | 7.275957614183426e-11 |
| DP utility for v2 | 0.412939 | 0.020045 | 39 | 1.8189894035458565e-12 | 38 | 7.275957614183426e-11 | 37 | 1.420630724169314e-09 |

Interpretation note requested from Priority 26: T1 cannot read the v2 path even with the key in the earlier seed-leak control; therefore the v2 comparison here measures v2's lossy compression of the BN statistic, not key secrecy.  The v2-with-key vs Prior result remains reported in Priority 26; this report does not alter that result.

## Part 3 — T2 gate attempt

The concrete gradient-only T2 attacker was attempted with the frozen TabLeak-style specification: cosine objective, softmax-style harddiff decoding path, known labels, 8 restarts and 2,000 Adam steps for the first gate, then one pre-declared retry with 16 restarts and 8,000 steps.

| Run | n | steps | restarts | wins vs Prior | p vs Prior | wins vs decoy | p vs decoy | qualified |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| first gate | 8 | 2000 | 8 | 2 | 0.96484375 | 7 | 0.03515625 | no |
| pre-declared retry | 8 | 8000 | 16 | 4 | 0.63671875 | 6 | 0.14453125 | no |

Gradient-only statement: T2 attempted; gate failed at n=8, including the one pre-declared retry.  No T2 confirmatory test was run in Priority 25b.

## Combined primary-family Holm results

Primary family includes:

- Priority 24 T1 × distortion-matched tests: v1 conservative and v2.
- Priority 25b T1 × utility-matched tests: v1 conservative and v2.

Holm correction was applied separately to DNA>DP and DP>DNA p-values.

| Test | raw p DNA>DP | Holm p DNA>DP | raw p DP>DNA | Holm p DP>DNA |
|---|---:|---:|---:|---:|
| P24 v1 conservative distortion | 0.9002045665401965 | 0.9002045665401965 | 0.16839181759496574 | 0.673567270379863 |
| P24 v2 distortion | 0.09979543345980349 | 0.19959086691960698 | 0.9459354892969714 | 1.0 |
| P25b v1 conservative utility | 7.275957614183426e-11 | 2.1827872842550278e-10 | 0.999999999998181 | 1.0 |
| P25b v2 utility | 1.8189894035458565e-12 | 7.275957614183426e-12 | 1.0 | 1.0 |

Combined RQ1 answer under the pre-registered Priority 25b rule: **YES**.  At least one DNA>DP test is significant after Holm correction, and no DP>DNA test is significant after Holm correction.  This “YES” rests on the qualified T1 BN-statistics channel; the gradient-only T2 instrument did not qualify.

## Artifact paths and SHA-256 hashes

| Artifact | SHA-256 |
|---|---|
| `protocols/amendments/2026-09-29_priority25b_bracketed_utility.md` | `3d686a135ab64e5c3c8c38c542beed40e06cccf28adee4e3e9114df8d7180dee` |
| `protocols/config/priority25b_utility_grid.json` | `9bdd81a58e4cb1cd3fc65a5431fcfb3447590d7e5336db5c1e82e707591f1eb1` |
| `artifacts/priority25b_bracketed_utility/utility_grid_initial_20260929/execution_summary.json` | `f5386680534282b1626600b22775fc377ccb8de3b855a425b5900348ad3f1591` |
| `results/priority25b_bracketed_utility/utility_grid_initial_20260929/utility_grid_summary.json` | `6009a64f4d95372c9920b0a7e48cbd654be2a7ad2d3eab51a570dfcbcd23cdbf` |
| `artifacts/priority25b_bracketed_utility/t1_dev_targets_20260929/paysim_priority25b_t1_dev_targets.pt` | `ec33b121a7cd462bebe9ab282ac7eeac34748b3b1f45c9e186f06bea5030dcb0` |
| `artifacts/priority25b_bracketed_utility/t1_confirm_targets_20260929/paysim_priority25b_t1_confirm_targets.pt` | `fcd5b0d8e5af2cfe4b4853d7e595b7cdd9f0a7056253fb4be3e8a4d51119195f` |
| `artifacts/priority25b_bracketed_utility/t1_utility_run_20260929_replay1/priority25_t1_utility_summary.json` | `5823404bb4f43daaa4a80d1155b4575da74d07975d00ec74d7ebe222a116b233` |
| `artifacts/priority25b_bracketed_utility/t2_gate_targets_20260929/paysim_priority25b_t2_gate_targets.pt` | `ce796ef9db6c4ef747a763750a55d6122df1551f78b12ce22571859d907b4526` |
| `artifacts/priority25b_bracketed_utility/t2_gate_run_20260929_replay2/t2_gate_summary.json` | `96c68067ef497233eabd3890518c099a5f68c7b64c9bf03e3cf9c1fa51a17449` |
| `artifacts/priority25b_bracketed_utility/t2_retry_targets_20260929/paysim_priority25b_t2_retry_targets.pt` | `88327bb63a6a36bb55f7ba3effdc6d84b1028392ee371078d09418d237992a07` |
| `artifacts/priority25b_bracketed_utility/t2_retry_run_20260929/t2_gate_summary.json` | `c764c7a3abc8273dd5baec02b49cb6f066dff96c4972451ba64c781e737133b3` |
| `experiments/run_priority25b_utility_grid.py` | `da60fcddc78524e7da0d0811f40d070a0281d622fdb8b330d5cf5689ded690e5` |
| `experiments/analyze_priority25b_utility_grid.py` | `7a4c13c562d6dd5bb1aa8ad45f4fedbef4b927268a32bccc3f3e3e6b288aa09a` |
| `experiments/priority25b_t2_gate.py` | `053f50d2a07d628304ac3f1492bc82eeb070a92c786a7f60e0d7ba35f9a1b344` |
| `experiments/priority25_t1_utility_matched_rq1.py` | `34e6d02a031a75082aaecfca529fc7b24d6f2120e20476d1174253dd2df9839a` |

Source-disjoint checks reported `max_overlap_with_existing_targets = 0` for all newly generated target sets used in this priority.
