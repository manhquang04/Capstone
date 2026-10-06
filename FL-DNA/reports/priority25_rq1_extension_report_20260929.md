# Priority 25 — RQ1 extension: utility-matched DP and gradient-only diagnostic

Date: 2026-09-29  
Status: COMPLETED for the pre-registered reduced Priority 25 scope  
No files under `Latex/` were edited.

## 1. Scope and pre-registration

Amendment:

- `protocols/amendments/2026-09-29_priority25_rq1_extension.md`
- SHA-256: `858059a43818b8053cb030426407cf95068c7f6280d09cf76c43adf98806c11c`

Priority 24 remains unchanged and is not rerun.  Priority 25 adds:

1. utility-matched DP for the already-qualified fraud-domain T1 BN-statistics instrument;
2. a descriptive Adam-step diagnostic for the gradient-only gap.

The full requested scope exceeded the approximately 3-day compute budget.  The amendment therefore applied the user-specified fallback order:

- Part 3 CIFAR DP comparator: dropped;
- T3 FedSGD secondary attacker: dropped;
- Part 1 grid: shrunk to four log-spaced multipliers;
- T2 TabLeak-style gradient-only confirmatory chain: not executed in this amendment; only the pre-registered Adam-sign diagnostic was run.

This reduced scope directly closes the utility-matched-DP gap for a valid fraud-domain instrument.  It does not establish a qualified gradient-only fraud-domain instrument.

## 2. Commands run

Norm probe:

```bash
PYTHONPATH=. MAX_ROWS=500000 NUM_ROUNDS=50 LOCAL_EPOCHS=1 FL_NUM_CLIENTS=3 \
  LOSS_TYPE=focal FOCAL_ALPHA=0.95 FOCAL_GAMMA=2.0 \
  .venv-phase1/bin/python experiments/priority25_full_update_norm_probe.py \
  --output-dir artifacts/priority25_rq1_extension/norm_probe_20260929 \
  --seeds 2501001 2501002
```

Utility grid config frozen after norm probe:

```text
protocols/config/priority25_utility_grid.json
```

Utility grid:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority25_utility_grid.py \
  --config protocols/config/priority25_utility_grid.json \
  --output-dir artifacts/priority25_rq1_extension/utility_grid_20260929 \
  --workers 9
```

Utility grid analysis:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/analyze_priority25_utility_grid.py \
  --config protocols/config/priority25_utility_grid.json \
  --run-dir artifacts/priority25_rq1_extension/utility_grid_20260929 \
  --output-dir results/priority25_rq1_extension/utility_grid_20260929
```

T1 utility target generation:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority25_rq1_extension/t1_utility_dev_targets_20260929 \
  --output-name paysim_priority25_t1_utility_dev_targets.pt \
  --groups 8 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092951 \
  --purpose 'Priority 25 T1 utility-matched development n8'
```

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority25_rq1_extension/t1_utility_confirm_targets_20260929 \
  --output-name paysim_priority25_t1_utility_confirm_targets.pt \
  --groups 39 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092952 \
  --purpose 'Priority 25 T1 utility-matched confirmatory n39'
```

T1 utility confirmatory first attempt:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority25_t1_utility_matched_rq1.py \
  --dev-target artifacts/priority25_rq1_extension/t1_utility_dev_targets_20260929/paysim_priority25_t1_utility_dev_targets.pt \
  --confirm-target artifacts/priority25_rq1_extension/t1_utility_confirm_targets_20260929/paysim_priority25_t1_utility_confirm_targets.pt \
  --utility-summary results/priority25_rq1_extension/utility_grid_20260929/utility_grid_summary.json \
  --output-dir artifacts/priority25_rq1_extension/t1_utility_run_20260929 \
  --seed 2026092951
```

This first attempt failed after computing the scientific rows, at CSV serialization of the derived branch-vs-control table, because the `none` row lacks `higher_mse_than_none_*` fields.  No scientific parameter changed.  The script was patched to use union CSV fieldnames for that derived table and rerun into a new output directory, leaving the failed partial output untouched.

T1 utility technical replay:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority25_t1_utility_matched_rq1.py \
  --dev-target artifacts/priority25_rq1_extension/t1_utility_dev_targets_20260929/paysim_priority25_t1_utility_dev_targets.pt \
  --confirm-target artifacts/priority25_rq1_extension/t1_utility_confirm_targets_20260929/paysim_priority25_t1_utility_confirm_targets.pt \
  --utility-summary results/priority25_rq1_extension/utility_grid_20260929/utility_grid_summary.json \
  --output-dir artifacts/priority25_rq1_extension/t1_utility_run_20260929_replay1 \
  --seed 2026092951
```

Adam-sign diagnostic:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority25_rq1_extension/adam_sign_targets_20260929 \
  --output-name paysim_priority25_adam_sign_targets.pt \
  --groups 8 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092961 \
  --purpose 'Priority 25 Adam sign diagnostic n8'
```

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority25_adam_sign_diagnostic.py \
  --target artifacts/priority25_rq1_extension/adam_sign_targets_20260929/paysim_priority25_adam_sign_targets.pt \
  --output-dir artifacts/priority25_rq1_extension/adam_sign_diagnostic_20260929 \
  --seed 2026092961
```

Combined Holm summary:

```bash
PYTHONPATH=. .venv-phase1/bin/python - <<'PY'
# combines fixed Priority 24 p-values with Priority 25 T1 utility p-values
# output: results/priority25_rq1_extension/final_20260929/
PY
```

## 3. Norm probe and utility grid

Norm probe:

| Quantity | Value |
|---|---:|
| Seeds | 2501001, 2501002 |
| Norm count | 300 |
| Min norm | 27.269006729125977 |
| Median norm | 130.4156723022461 |
| p95 norm | 256.51892395019536 |
| Max norm | 653.20361328125 |
| Frozen clip norm `C = 1.01 * p95` | 259.0841131896973 |

Utility grid results:

| Method | σ | Mean ΔF1 | SD ΔF1 | Mean ΔAUC | Mean ΔPR-AUC | ε one release | ε 50 releases |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dp_sigma_1em06` | 0.000001 | 0.0049527633 | 0.0210673493 | 0.0003708526 | 0.0033725139 | 5.000061512925466e11 | 2.5000251151292547e13 |
| `dp_sigma_3em06` | 0.000003 | -0.0032739191 | 0.0284504623 | -0.0000434471 | 0.0024362698 | 5.555724231293991e10 | 2.7778067068481016e12 |
| `dp_sigma_1em05` | 0.000010 | 0.0136924775 | 0.0164744071 | 0.0003278324 | 0.0077528749 | 5.000480258509299e9 | 2.5000365129254645e11 |
| `dp_sigma_3em05` | 0.000030 | -0.0039327054 | 0.0121106043 | 0.0000445532 | -0.0045144269 | 5.55715568515226e8 | 2.7778908979606583e10 |

Frozen matching rule: choose the largest grid σ whose mean ΔF1 is at least `(RQ2 transform mean ΔF1 − 0.005)`.

| Transform | Threshold | Selected σ | Selected mean ΔF1 | Selected mean ΔAUC | Selected mean ΔPR-AUC | ε one release | ε 50 releases |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 conservative | -0.0036 | 0.000010 | 0.0136924775 | 0.0003278324 | 0.0077528749 | 5.000480258509299e9 | 2.5000365129254645e11 |
| v2 0.95/0.01 | -0.0167 | 0.000030 | -0.0039327054 | 0.0000445532 | -0.0045144269 | 5.55715568515226e8 | 2.7778908979606583e10 |

All 40 grid jobs succeeded: 8 baselines + 32 DP jobs.

## 4. T1 qualification and utility-matched confirmatory tests

T1 re-qualification on fresh Priority 25 development targets:

| Gate | Wins | p-value | Result |
|---|---:|---:|---|
| T1 vs Prior | 8/8 | 0.00390625 | PASS |
| T1 vs decoy | 8/8 | 0.00390625 | PASS |

Tie band from deterministic development replay: `0.0`.

Utility-matched DNA-vs-DP confirmatory:

| Cell | DNA wins | DP wins | Ties | n | p(DNA > DP) | p(DP > DNA) | Median MSE ratio DNA/DP | Mean MSE ratio DNA/DP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 conservative vs utility-matched DP | 36 | 3 | 0 | 39 | 1.8044374883174896e-08 | 0.9999999985793693 | 17.35166683708447 | 130.64180896191363 |
| v2 0.95/0.01 vs utility-matched DP | 39 | 0 | 0 | 39 | 1.8189894035458565e-12 | 1.0 | 23.296163478227683 | 45.97072959989458 |

Within-Priority-25 two-test Holm:

- v1 conservative DNA>DP: `1.8044374883174896e-08`;
- v2 DNA>DP: `3.637978807091713e-12`;
- both DP>DNA adjusted p-values: `1.0`.

Branch-vs-control summary:

| Branch | Mean MSE | Median MSE | Beats Prior | p | Beats decoy | p | Higher MSE than none | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 9.031014805197915e-12 | 1.1374427751466505e-12 | 39/39 | 1.8189894035458565e-12 | 39/39 | 1.8189894035458565e-12 | n/a | n/a |
| v1 conservative | 7.176121544046252 | 0.3050710025307695 | 16/39 | 0.9002045665401965 | 28/39 | 0.004737652139738202 | 39/39 | 1.8189894035458565e-12 |
| v2 0.95/0.01 | 14.240658997845662 | 0.9646040023905407 | 0/39 | 1.0 | 21/39 | 0.3746293123804209 | 39/39 | 1.8189894035458565e-12 |
| utility DP for v1 | 2.0147400128827395 | 0.005131439290306418 | 39/39 | 1.8189894035458565e-12 | 36/39 | 1.8044374883174896e-08 | 39/39 | 1.8189894035458565e-12 |
| utility DP for v2 | 2.043663688331978 | 0.04883805909441932 | 37/39 | 1.420630724169314e-09 | 35/39 | 1.6765807231422514e-07 | 39/39 | 1.8189894035458565e-12 |

## 5. Adam-sign diagnostic

Target set: `artifacts/priority25_rq1_extension/adam_sign_targets_20260929/paysim_priority25_adam_sign_targets.pt`  
Target SHA-256: `d4c10248a1b366b55510614bae300af0e8cf36c6ff0950cccfa6200f60c22ba1`

| Statistic | Value |
|---|---:|
| n | 8 |
| Adam lr | 0.001 |
| Mean cosine(step, -lr·sign(g)) | 0.992050269665054 |
| Median cosine(step, -lr·sign(g)) | 0.9922208670980199 |
| Mean sign agreement on nonzero-gradient coordinates | 1.0 |
| Median sign agreement | 1.0 |
| Mean corr(|step|, |gradient|) | 0.04726431480920708 |
| Median corr(|step|, |gradient|) | 0.051714508964191344 |

Plain diagnostic result: one fresh-state Adam step is almost exactly a sign update in direction, while preserving very little per-coordinate gradient magnitude information.  This supports the amendment's decision not to claim a valid gradient-only fraud-domain instrument without a separately designed and qualified TabLeak-style attacker.

Gradient-only statement under the frozen Priority 25 rule:

```text
not answerable with the gradient-only attackers evaluated
```

## 6. Combined primary-family Holm analysis

Primary family combines fixed Priority 24 T1 distortion-matched p-values and Priority 25 T1 utility-matched p-values.

| Test | DNA wins | DP wins | Ties | Raw p DNA>DP | Holm p DNA>DP | Raw p DP>DNA | Holm p DP>DNA |
|---|---:|---:|---:|---:|---:|---:|---:|
| P24 T1 distortion v1 conservative | 16 | 23 | 0 | 0.9002045665401965 | 0.9002045665401965 | 0.16839181759496574 | 0.673567270379863 |
| P24 T1 distortion v2 | 24 | 15 | 0 | 0.09979543345980349 | 0.19959086691960698 | 0.9459354892969714 | 1.0 |
| P25 T1 utility v1 conservative | 36 | 3 | 0 | 1.8044374883174896e-08 | 5.413312464952469e-08 | 0.9999999985793693 | 1.0 |
| P25 T1 utility v2 | 39 | 0 | 0 | 1.8189894035458565e-12 | 7.275957614183426e-12 | 1.0 | 1.0 |

Combined primary-family result:

- any DNA>DP significant after Holm: `true`;
- any DP>DNA significant after Holm: `false`;
- combined RQ1 answer under the frozen Priority 25 rule: `YES`.

Plain wording: under the reduced Priority 25 primary family, DNA significantly beats utility-matched DP on the qualified fraud-domain T1 instrument, and no primary-family DP>DNA test is significant after Holm.  This "YES" is driven by utility-matched DP; the earlier distortion-matched Priority 24 tests remain `NO_NOT_SHOWN`.

## 7. Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `protocols/amendments/2026-09-29_priority25_rq1_extension.md` | `858059a43818b8053cb030426407cf95068c7f6280d09cf76c43adf98806c11c` |
| `protocols/config/priority25_utility_grid.json` | `5ceda668e4567eee3c0cf2205b4e8f4603e84a7772a46f3566dbb0d38f5d8b5a` |
| `experiments/priority25_full_update_norm_probe.py` | `97d22b6e8ab3930eadcda2959e9a6529ab07d336a6fbe6b41852f890da0708e5` |
| `experiments/run_priority25_utility_grid.py` | `14bc7c0b0ee4836e3bc5b50bf5adf9f6a555b74011934374eb76127da4108360` |
| `experiments/analyze_priority25_utility_grid.py` | `ddd831e9eb1eea54c32c5fc8d8edb39eed381672ee5881a8f2b082db9052d447` |
| `experiments/priority25_t1_utility_matched_rq1.py` | `41fb61e77853f665faa0d9653563aec5d2b578c4f596eb01fffff5f43ba51330` |
| `experiments/priority25_adam_sign_diagnostic.py` | `153bd40a87ce39a66ed09fbda7f9fbe15abe5280566cf39ede07903bf52396a3` |
| `artifacts/priority25_rq1_extension/norm_probe_20260929/norm_probe_summary.json` | `d6831a0469c6c01e132d90356b52dab34acbd85f60d623ee35dc34048f7d567e` |
| `artifacts/priority25_rq1_extension/utility_grid_20260929/execution_summary.json` | `a7f53f384efe9afcae2c96452e1ea3885ffa9aebb1dbba3abc023f9af451227c` |
| `results/priority25_rq1_extension/utility_grid_20260929/utility_grid_summary.json` | `840ea8219791307fee533cb327d5d1e8a2464fa932a7e8b7a445e9865770a4f3` |
| `artifacts/priority25_rq1_extension/t1_utility_dev_targets_20260929/paysim_priority25_t1_utility_dev_targets.pt` | `5912797d6872414cd53639d005ea5c785af30a1851a2bc00901a6c804bf33fde` |
| `artifacts/priority25_rq1_extension/t1_utility_confirm_targets_20260929/paysim_priority25_t1_utility_confirm_targets.pt` | `01d04fb8ad7bed7339cb5f9b58bf6f8db0b6da971a2a74e64e5faadf502c2c16` |
| `artifacts/priority25_rq1_extension/t1_utility_run_20260929_replay1/priority25_t1_utility_summary.json` | `ac640fdb50036d8dd0bd9ea9697680e0274946ff3a2fe0dbc3253e973fc5780d` |
| `artifacts/priority25_rq1_extension/t1_utility_run_20260929_replay1/confirm_t1_per_target.csv` | `dab5c597d2915aaf96aaff1d5cdbf0966409f97ae0d9014670d7650a98f9effe` |
| `artifacts/priority25_rq1_extension/t1_utility_run_20260929_replay1/confirm_t1_examples.csv` | `7a2bcb66f62b42154ac0b6023157f57fcf81c30b72778ef628ce523b16cea735` |
| `artifacts/priority25_rq1_extension/adam_sign_targets_20260929/paysim_priority25_adam_sign_targets.pt` | `d4c10248a1b366b55510614bae300af0e8cf36c6ff0950cccfa6200f60c22ba1` |
| `artifacts/priority25_rq1_extension/adam_sign_diagnostic_20260929/adam_sign_diagnostic_summary.json` | `4e62f45b225b25a48e44f5fe4a2821f1b66e48e050cfdd0f206d7d04f38806e1` |
| `results/priority25_rq1_extension/final_20260929/combined_primary_family_holm.json` | `7483b46e4b751384cb58db73ec68731ba94c10f613d6e0d50f458e5020fce3ed` |

