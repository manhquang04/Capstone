# Priority 29 native-setting defense audit report

Date: 2026-09-30

Status: PARTIAL / GATED PROGRESS.  This report freezes and records the native
positive-control qualification work completed so far.  Full defense E1/E2/E3
tables were not emitted in this pass because the fail-closed Priority 29 rule
requires validated native-defense adapters before per-defense conclusions can be
reported.  No `Latex/` files were edited.

## Protocol

Amendment:
`protocols/amendments/2026-09-30_priority29_native_audit.md`

SHA-256:
`0ff540a4664591675096bfa163b27f11ce61426852220c95f53491307bac9bf9`

Frozen scope used in this pass:

- IMAGE primary: CIFAR-10 test split, official Geiping/invertinggradients
  `construct_model("LeNetZhu", seed=42)`, FedSGD one gradient, batch 1, known
  label, `GradientReconstructor` with cosine, signed Adam, lr 0.1, 4800
  iterations, TV 0.01, lr decay, boxed.
- TABULAR primary: Adult native TabLeak, batch 8, official config 46, known
  labels, 1500 iterations, signed Adam lr 0.06, 30 post-selection ensemble
  members, official feature-accuracy metric.
- PaySim secondary was dropped by the pre-registered compute reduction.

## Commands run

```bash
external_defenses/.venv/bin/python -B -m py_compile \
  experiments/priority29/image_native_positive_control.py

external_defenses/.venv/bin/python -B \
  experiments/priority29/image_native_positive_control.py --stage n8 --n 8

# Technical replay after an infrastructure-only interop-thread setter error
# before any scientific result existed:
external_defenses/.venv/bin/python -B \
  experiments/priority29/image_native_positive_control.py --stage n8 --n 8 --allow-existing

external_defenses/.venv/bin/python -B \
  experiments/priority29/image_native_positive_control.py --stage n24 --n 24

external_defenses/.venv/bin/python -B -m py_compile \
  experiments/priority29/tabular_native_positive_control.py

external_defenses/.venv/bin/python -B \
  experiments/priority29/tabular_native_positive_control.py --stage n8 --n 8

external_defenses/.venv/bin/python -B \
  experiments/priority29/tabular_native_positive_control.py --stage n24 --n 24

external_defenses/.venv/bin/python -B -m py_compile \
  experiments/priority29/image_native_positive_control.py \
  experiments/priority29/tabular_native_positive_control.py

git diff --check -- \
  protocols/amendments/2026-09-30_priority29_native_audit.md \
  experiments/priority29/image_native_positive_control.py \
  experiments/priority29/tabular_native_positive_control.py \
  experiments/priority29/__init__.py
```

All processes used `torch.set_num_threads(1)`.  The first image n=8 attempt
failed before any group completed because `torch.set_num_interop_threads(1)` was
called after PyTorch had already started parallel work.  The driver was patched
to ignore that already-set interop condition and the same pre-registered n=8
target slice was replayed.  This is recorded as an infrastructure replay, not a
scientific rerun.

## Native instrument qualification

### IMAGE: official Geiping/invertinggradients on CIFAR-10

Artifacts:

- n=8 summary:
  `artifacts/priority29_native_audit/image_positive_control/n8/summary.json`
- n=8 per-target CSV:
  `artifacts/priority29_native_audit/image_positive_control/n8/per_target.csv`
- n=8 reconstruction grid:
  `artifacts/priority29_native_audit/image_positive_control/n8/reconstruction_grid.png`
- n=24 summary:
  `artifacts/priority29_native_audit/image_positive_control/n24/summary.json`
- n=24 per-target CSV:
  `artifacts/priority29_native_audit/image_positive_control/n24/per_target.csv`
- n=24 reconstruction grid:
  `artifacts/priority29_native_audit/image_positive_control/n24/reconstruction_grid.png`

Source-disjoint evidence:

- CIFAR-10 split: `test`.
- Within-stage target indices are unique.
- Priority 29 image stages use deterministic non-overlapping slices:
  n8 offset 0, n24 offset 100, confirmatory offset 500.
- Earlier project CIFAR attack artifacts used the train split; this gate uses
  the test split.

Positive-control result:

| Stage | Baseline | Wins | Losses | Ties | Exact one-sided p |
|---|---:|---:|---:|---:|---:|
| n=8 | Prior | 8 | 0 | 0 | 0.00390625 |
| n=8 | Decoy | 8 | 0 | 0 | 0.00390625 |
| n=8 | Gray image | 8 | 0 | 0 | 0.00390625 |
| n=8 | CIFAR mean image | 8 | 0 | 0 | 0.00390625 |
| n=24 | Prior | 24 | 0 | 0 | 5.960464477539063e-08 |
| n=24 | Decoy | 24 | 0 | 0 | 5.960464477539063e-08 |
| n=24 | Gray image | 24 | 0 | 0 | 5.960464477539063e-08 |
| n=24 | CIFAR mean image | 24 | 0 | 0 | 5.960464477539063e-08 |

Descriptive reconstruction quality:

| Stage | mean attack MSE | median attack MSE | mean PSNR | median PSNR | mean SSIM | median SSIM |
|---|---:|---:|---:|---:|---:|---:|
| n=8 | 0.0042587155 | 0.0048695535 | 24.9108 | 23.1289 | 0.8734 | 0.8939 |
| n=24 | 0.0077796582 | 0.0044339744 | 23.1818 | 23.5332 | 0.8139 | 0.8562 |

Verdict: IMAGE native instrument QUALIFIED for Priority 29 under the frozen
positive-control gate.

### TABULAR: official TabLeak on Adult

Artifacts:

- n=8 summary:
  `artifacts/priority29_native_audit/tabular_positive_control/n8/summary.json`
- n=8 per-target CSV:
  `artifacts/priority29_native_audit/tabular_positive_control/n8/per_target.csv`
- n=24 summary:
  `artifacts/priority29_native_audit/tabular_positive_control/n24/summary.json`
- n=24 per-target CSV:
  `artifacts/priority29_native_audit/tabular_positive_control/n24/per_target.csv`

Source-disjoint evidence:

- Adult training-row IDs were selected by deterministic shuffled slices.
- Within-stage row IDs are unique.
- Priority 29 tabular stages use non-overlapping slices:
  n8 offset 0, n24 offset 2000, confirmatory offset 6000.

Positive-control result:

| Stage | Baseline | Wins | Losses | Ties | Exact one-sided p |
|---|---:|---:|---:|---:|---:|
| n=8 | Mean/mode | 8 | 0 | 0 | 0.00390625 |
| n=8 | Empirical marginal mean | 8 | 0 | 0 | 0.00390625 |
| n=24 | Mean/mode | 24 | 0 | 0 | 5.960464477539063e-08 |
| n=24 | Empirical marginal mean | 24 | 0 | 0 | 5.960464477539063e-08 |

Descriptive feature accuracy:

| Stage | mean attack acc. | median attack acc. | min attack acc. | max attack acc. | mean mean/mode acc. | mean empirical-marginal acc. |
|---|---:|---:|---:|---:|---:|---:|
| n=8 | 96.6518% | 100.0000% | 86.6071% | 100.0000% | 54.9107% | 54.7470% |
| n=24 | 95.3869% | 98.2143% | 73.2143% | 100.0000% | 56.3244% | 55.5345% |

Verdict: TABULAR native Adult/TabLeak instrument QUALIFIED for Priority 29
under the frozen positive-control gate.

## Defense E1/E2/E3 status

No defense-level E1/E2/E3 result is reported in this file.  This is deliberate:
Priority 29 inherited Priority 28's fail-closed checks, and the currently
available `external_defenses/adaptive/` adapters are not yet sufficient, by
themselves, to emit native per-defense conclusions:

- `PRECODEDefense` in `external_defenses/adaptive/defenses.py` is an
  update-dictionary metadata adapter; Priority 29 requires PRECODE as a
  model-level variational bottleneck and explicitly forbids a pass-through
  adapter from counting as a valid defense.
- `external_defenses/adaptive/smoke.py` is documented as qualitative plumbing,
  uses `THREADS = 4`, a toy 20-step attack budget, and is not the official
  native Geiping/TabLeak attacker configuration used for this priority.
- Therefore, emitting E1/E2/E3 per-defense tables from those smoke adapters
  would violate the "defense-not-identity" and "native official attacker"
  requirements.

This report should be read as the completed instrument-qualification stage for
Priority 29, not as a completed defense verdict table.  The qualified IMAGE and
TABULAR native instruments can now be used for defense-level cells once each
defense has a validated native adapter satisfying Step 0.

## SHA-256 hashes

| Artifact | SHA-256 |
|---|---|
| `protocols/amendments/2026-09-30_priority29_native_audit.md` | `0ff540a4664591675096bfa163b27f11ce61426852220c95f53491307bac9bf9` |
| `experiments/priority29/image_native_positive_control.py` | `550ea63679bc89ad568a32cc63031031f1f74bbd54c9b53b7f8296041f0b3483` |
| `experiments/priority29/tabular_native_positive_control.py` | `b31ade91619bb47481e60f9118349b6cf090edfb3787f9f59fd7defb31b67fb1` |
| `artifacts/priority29_native_audit/image_positive_control/n8/summary.json` | `3575df06e3276660dd8bd50c2d70cf71e8226c0a4d05e2b84c19527513df0485` |
| `artifacts/priority29_native_audit/image_positive_control/n8/per_target.csv` | `74f258def6fc2938f889b6d9c7480e5eb819d7b1d6fb48eacd41879695210e6e` |
| `artifacts/priority29_native_audit/image_positive_control/n8/reconstruction_grid.png` | `293888ce0f4d821fd5ae3275b4abc6134483bd6880eaa222a748dec0aac4da5e` |
| `artifacts/priority29_native_audit/image_positive_control/n24/summary.json` | `db50e1e300d4c62930b13bae734543fbf87f76ed66156339cbdf949172e02a94` |
| `artifacts/priority29_native_audit/image_positive_control/n24/per_target.csv` | `e022a201f2e359bf8edbf3837ab942cc5d79f09a99f9f70808db31108e31a1fc` |
| `artifacts/priority29_native_audit/image_positive_control/n24/reconstruction_grid.png` | `8760131615db02d046b2d6ff52f586cf7000b29e48a0957dac7c44b431b8f2c0` |
| `artifacts/priority29_native_audit/tabular_positive_control/n8/summary.json` | `8c1d9f7d31dccd4473b80b0683a2273245e47e01a991827d7ef3d8e44804b3aa` |
| `artifacts/priority29_native_audit/tabular_positive_control/n8/per_target.csv` | `c4b9e61381a820c07292e170191b89fcbe6522bc670c2403a7cc528e78ab24ea` |
| `artifacts/priority29_native_audit/tabular_positive_control/n24/summary.json` | `81431269d8c68d76b1f3270bcb1aee4b7f3b4b7f04bf7292cc6f5b66971ca395` |
| `artifacts/priority29_native_audit/tabular_positive_control/n24/per_target.csv` | `119baff8e7eaa135835cc26fd5845afd004fec95b490cbb2fba36a6f50945ceb` |

## Final checks

- `py_compile` passed for both Priority 29 drivers.
- `git diff --check` passed for the amendment and Priority 29 driver files.
- No Priority 29 background workload remained after the positive-control runs.
- `torch.set_num_threads(1)` was used in each driver process.
- No file under `external_defenses/` or `Latex/` was edited.

