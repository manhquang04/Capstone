# Priority 30 native defense adapters and no-balance ablation report

Date: 2026-09-30

Status: EXECUTION IN PROGRESS.  This report first recorded the completed
adapter checks and the PaySim no-balance projection check.  It now additionally
records Stage S0 of the requested long audit execution.  Later stages are
appended as they complete so partial progress is not lost.

No `Latex/` files were edited.  No file under `external_defenses/` was edited.

## Protocol

Amendment:
`protocols/amendments/2026-09-30_priority30_native_defenses.md`

SHA-256:
`13e201883219021a51d08d9326b582814b05b654ff0781a07a592a9199314b00`

The amendment states that Priority 30 continues the frozen Priority 29 design.
The E1/E2/E3 definitions, RQ4 answer rule, Holm family, and compute reduction
remain unchanged.

## Commands run

```bash
external_defenses/.venv/bin/python -B -m py_compile \
  experiments/priority30_native_defenses/native_adapters.py \
  experiments/priority30_native_defenses/run_adapter_checks.py

external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/run_adapter_checks.py

external_defenses/.venv/bin/python -B -m py_compile \
  experiments/priority30_native_defenses/paysim_no_balance_ablation.py

DATALOADER_NUM_WORKERS=0 NUM_ROUNDS=1 external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/paysim_no_balance_ablation.py --check-only

shasum -a 256 \
  protocols/amendments/2026-09-30_priority30_native_defenses.md \
  experiments/priority30_native_defenses/native_adapters.py \
  experiments/priority30_native_defenses/run_adapter_checks.py \
  experiments/priority30_native_defenses/paysim_no_balance_ablation.py \
  artifacts/priority30_native_defenses/adapter_checks/adapter_checks.json

git diff --check -- \
  protocols/amendments/2026-09-30_priority30_native_defenses.md \
  experiments/priority30_native_defenses \
  reports/priority30_native_audit_report_20260930.md

external_defenses/.venv/bin/python -B -m py_compile \
  experiments/priority30_native_defenses/run_audit.py

external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/run_audit.py --stage S0 --n 1 --workers 1

nohup external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/run_audit.py --stage S0 --n 2 --workers 2 \
  >> artifacts/priority30_native_defenses/S0_nohup.log 2>&1 &

external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/run_audit.py --stage S0 --n 2 --workers 2
```

Disclosed infrastructure runs:

1. Initial adapter-check import failed before producing adapter results because
   Python resolved `attacks.*` to FL-DNA's package rather than official Tableak.
   Fixed by placing `external_defenses/tableak` first in `sys.path`.
2. First extended adapter-check run failed before results because v1 transform
   config missed a required seed.  Fixed by freezing a local check seed
   `30_001` for adapter-check purposes and reran.
3. The first `run_audit.py --stage S0 --n 2` background launch did not create
   additional outputs; foreground execution was used to capture the live warning
   stream and complete the smoke run.  The warning was a PyTorch tensor-copy
   warning in image rescaling and did not abort execution.

## Part A — native adapter checks

Artifact:
`artifacts/priority30_native_defenses/adapter_checks/adapter_checks.json`

SHA-256:
`a8f97c93aac392af33b13f5679ec0100cd28ff45d1263b3ee9febfd96e8e4cad`

External commits recorded:

| Source | Commit |
|---|---|
| PRECODE | `c66adc4cdd62993139eafebc1b57fc25b0694874` |
| Soteria | `23cf90e9e5cb41d5dc45e7540ef63a4a0ca0a8ca` |
| DLG / gradient pruning | `d21007fa1540ba2303ebc034976aa331814727c7` |
| FetchSGD / Count-Sketch | `833ca44cc43a9b034515f55485524e4f1d0fad21` |

### Adult PRECODE investigation

The earlier Adult PRECODE `L2=0.0` was **not** evidence that the defense was an
identity transform.  It was a measurement artifact: the PRECODE wrapper changes
parameter names (`prefix.*`, `bottleneck.*`, `classifier.*`), so comparing only
common keys against the original `FullyConnected` model compared no shared
parameter names.

The corrected check maps original keys to their PRECODE-wrapper counterparts:

| Original key | PRECODE key |
|---|---|
| `layers.1.layers.0.weight` | `prefix.1.layers.0.weight` |
| `layers.1.layers.0.bias` | `prefix.1.layers.0.bias` |
| `layers.2.layers.0.weight` | `prefix.2.layers.0.weight` |
| `layers.2.layers.0.bias` | `prefix.2.layers.0.bias` |
| `layers.3.weight` | `classifier.weight` |
| `layers.3.bias` | `classifier.bias` |

Corrected result: Adult PRECODE passes defense-not-identity with mapped L2
`1.654908701768933`, and its stochastic pass-to-pass gradient L2 is
`2.7832735270787436`.  The bottleneck is inserted on the representation path
before the final classifier, consistent with the official PRECODE placement.

### IMAGE checks: CIFAR-10 test index 0 / LeNet-Zhu

| Adapter | Check | Result | Effect / note |
|---|---|---:|---:|
| PRECODE model-level bottleneck | defense-not-identity | PASS | L2 vs undef common names = 33.77046694360729 |
| PRECODE stochasticity | stochastic forward/gradient | PASS | pass-to-pass L2 = 202.35151410176172 |
| Soteria representation mask | defense-not-identity | PASS | L2 = 19.530426882988618; retained 154/768 units |
| Gradient pruning 70% | defense-not-identity | PASS | L2 = 5.956185012344075; retained coordinates = 4749 |
| Count-Sketch known hashes | defense-not-identity | PASS | sketch norm = 65.26268005371094 |
| Count-Sketch known hashes | server/decode knowledge | PASS | `sketch_hashes` present |
| Count-Sketch unknown hashes | server/decode knowledge | FAIL as expected | missing `sketch_hashes` |
| DNA v1 conservative | defense-not-identity | PASS | L2 = 2.2266035003589613 |
| DNA v2 0.95/0.01 | defense-not-identity | PASS | L2 = 6.315960465865565; metadata items = 8 |
| DNA v2 key-holder least squares | lossless sketch sanity | PASS | sketch residual = 1.4232001021297678e-31 |
| ATS policy 3-1-7 | defense-not-identity | PASS | raw-vs-ATS gradient L2 = 7.641888087133349 |

### TABULAR checks: Adult train rows 0--7 / official FullyConnected

| Adapter | Check | Result | Effect / note |
|---|---|---:|---:|
| PRECODE model-level bottleneck | defense-not-identity | PASS | mapped L2 = 1.654908701768933 |
| PRECODE stochasticity | stochastic forward/gradient | PASS | pass-to-pass L2 = 2.7832735270787436 |
| Gradient pruning 70% | defense-not-identity | PASS | L2 = 0.4504768635972922; retained coordinates = 6272 |
| Soteria Adult representation mask | defense-not-identity | PASS | L2 = 0.6953850085949526; retained 60/100 units |
| DNA v1 conservative | defense-not-identity | PASS | L2 = 0.1159971243505578 |
| DNA v2 0.95/0.01 | defense-not-identity | PASS | L2 = 0.2669967440722377; metadata items = 6 |
| Count-Sketch known hashes | defense-not-identity | PASS | sketch norm = 2.6562612056732178 |

## Part B — Priority 29 E1/E2/E3 matrix

### Stage S0 — end-to-end smoke test

Status: COMPLETE.

Purpose: exercise all image and Adult cells with 2 targets per cell, verify that
one result file is written per `(cell, target)`, and verify that rerunning with a
larger `--n` resumes/skips completed outputs rather than recomputing them.  Smoke
outputs are stored separately and are **not** used as confirmatory results.

Artifacts:

- `artifacts/priority30_native_defenses/audit/S0/summary.csv`
- `artifacts/priority30_native_defenses/audit/S0_progress.log`
- `artifacts/priority30_native_defenses/S0_nohup.log`

Stage result count: 52 result files, matching 2 targets across the smoke matrix.

Progress log terminal entry:

```json
{"event": "complete", "stage": "S0", "tasks": 52, "elapsed": 307.25130858295597}
```

Smoke summary:

| Status | Count |
|---|---:|
| `ran` | 28 |
| `NOT_ASSESSABLE` | 24 |

The `NOT_ASSESSABLE` rows correspond to cells where the current native official
attack loop cannot validly consume the defense payload or where an adaptive
native objective has not been implemented.  They are retained in the output
rather than silently dropped.

SHA-256:

| Artifact | SHA-256 |
|---|---|
| `experiments/priority30_native_defenses/run_audit.py` | `252654a6b5a2177e2edc10d3d7acbc51496e0cd323a395395d18ff4474567093` |
| `artifacts/priority30_native_defenses/audit/S0/summary.csv` | `de55559fc83143ed34b85792401c481569bf1aaa329e202d28d3b451366cc915` |
| `artifacts/priority30_native_defenses/audit/S0_progress.log` | `417b559f096ccaab4910290acc8c14850de1cd5259a5509e6f2e4b0a56e32036` |
| `artifacts/priority30_native_defenses/S0_nohup.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

### Remaining E1/E2/E3 stages

### Stage S1 — IMAGE E1 + E2, confirmatory n=39

Status: COMPLETE.

Command:

```bash
external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/run_audit.py --stage S1 --n 39 --workers 4
```

The requested `nohup` launch for S1 exited without producing result files in
this Codex shell environment, so the same command was run in a managed foreground
session.  The per-target resume mechanism remained active and every target wrote
its own `result.json` as soon as it finished.

Artifacts:

- `artifacts/priority30_native_defenses/audit/S1/summary.csv`
- `artifacts/priority30_native_defenses/audit/S1_progress.log`

Progress log terminal entry:

```json
{"event": "complete", "stage": "S1", "tasks": 546, "elapsed": 1367.006364874891}
```

Stage result count: 546 result files, matching 39 targets across 7 image
defenses and E1/E2 cells.

Stage summary:

| Status | Count |
|---|---:|
| `ran` | 351 |
| `NOT_ASSESSABLE` | 195 |

Breakdown:

| Domain | Evaluation | Status | Count |
|---|---|---|---:|
| image | E1 | `ran` | 234 |
| image | E2 | `ran` | 117 |
| unspecified unsupported cells | unspecified | `NOT_ASSESSABLE` | 195 |

Mean PSNR across the 351 image cells that ran: 14.90734331283789 dB.

SHA-256:

| Artifact | SHA-256 |
|---|---|
| `artifacts/priority30_native_defenses/audit/S1/summary.csv` | `0ba596505e21fe31ddd3a72cb18411b63d9f3e1dc42f8a6b780f551e6fc62cba` |
| `artifacts/priority30_native_defenses/audit/S1_progress.log` | `5f1a1fb0e78435ad55f9241eb9ee322ed8b7394507ae2fc197f3a6775d23a3e8` |

### Stage S2 — TABULAR Adult E1 + E2, confirmatory n=39

Status: COMPLETE.

Command:

```bash
external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/run_audit.py --stage S2 --n 39 --workers 4
```

As with S1, the stage was run in a managed foreground session because `nohup`
launches did not persist in this shell environment.  The per-target resume
mechanism remained active.

Progress log terminal entry:

```json
{"event": "complete", "stage": "S2", "tasks": 468, "elapsed": 4734.213759375038}
```

Stage result count: 468 result files, matching 39 targets across 6 Adult
defenses and E1/E2 cells.

Stage summary:

| Status | Count |
|---|---:|
| `ran` | 195 |
| `NOT_ASSESSABLE` | 273 |

Breakdown:

| Domain | Evaluation | Status | Count |
|---|---|---|---:|
| adult | E1 | `ran` | 195 |
| unspecified unsupported cells | unspecified | `NOT_ASSESSABLE` | 273 |

Mean Adult feature accuracy across the 195 rows that ran:
69.8489010989011%.

SHA-256:

| Artifact | SHA-256 |
|---|---|
| `artifacts/priority30_native_defenses/audit/S2/summary.csv` | `ff941b77844177833327145490ff553f019a0e4633d9da1c0c168019dc13efef` |
| `artifacts/priority30_native_defenses/audit/S2_progress.log` | `794df51a06b489c79a449fc9117078d598323c8542686e5a487d3bb660450cbb` |

The adapter checks now establish that many adapters are non-identity and that the
v2 least-squares sketch sanity condition is satisfied.  However, Priority 29's
E1/E2/E3 matrix still requires integrating each passing adapter into the native
Geiping and TabLeak target-generation/reconstruction loop at n=39 per cell,
including adaptive per-defense attacks and DP comparators.  Running a reduced
subset or reporting only adapter checks as E1/E2/E3 would violate the frozen
answer rule.  Therefore all RQ4 verdicts remain pending.

Cells currently eligible for future E1/E2/E3 integration based on adapter
checks:

- IMAGE: PRECODE, Soteria, gradient pruning, Count-Sketch known-hash, DNA v1
  conservative, DNA v2 0.95/0.01, ATS 3-1-7.
- TABULAR Adult: PRECODE, Soteria, gradient pruning, Count-Sketch known-hash,
  DNA v1 conservative, DNA v2 0.95/0.01.

Count-Sketch unknown-hash must be reported as a separate unknown-hash condition;
it cannot be evaluated with a known-hash adaptive decoder unless that knowledge
is explicitly granted.

## Part C — PaySim no-balance-feature ablation

Driver added:
`experiments/priority30_native_defenses/paysim_no_balance_ablation.py`

SHA-256:
`3f10801ade1555dc0d7d4c1052b3834647150b641a08748fedb36a1380091530`

The driver loads PaySim through the frozen `load_creditcard_data` function and
then projects out the six forbidden balance-derived features after loading,
without modifying `data/load_creditcard.py`.

Projection check command:

```bash
DATALOADER_NUM_WORKERS=0 NUM_ROUNDS=1 external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/paysim_no_balance_ablation.py --check-only
```

Projection check result:

```json
{
  "original_dim": 13,
  "projected_dim": 7,
  "removed": [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_diff_orig",
    "balance_diff_dest"
  ],
  "kept": [
    "step",
    "amount",
    "type=CASH_IN",
    "type=CASH_OUT",
    "type=DEBIT",
    "type=PAYMENT",
    "type=TRANSFER"
  ],
  "exact_removed_match": true,
  "projected_client0_shape": [1681315, 7]
}
```

The full frozen Part C run (baseline, v1 conservative, v2 0.95/0.01; with and
without balance features; 16 paired replicates) was **not** run in this pass.
The check shows the full dataset/client split is large (`client0` alone has
1,681,315 rows), so running the full 96 training jobs is a substantial workload.
No reduced or quick-mode result is reported.

## Continuation after S2: S1b/S2b adaptive objectives

After S2 completed, the supervisor authorized filling the cells that had been
reported as `NOT_ASSESSABLE` because the native adaptive objectives had not yet
been integrated into the official attack loops. This continuation keeps the
frozen Priority 29 E2 definition ("the adaptive attack for that defense") and
does not change target sets, budgets, metrics, or answer rules.

Implementation update:

- Image Geiping loop (`GradientReconstructor`): added adaptive objectives inside
  the official optimization loop for:
  - gradient pruning: mask-aware matching over retained coordinates only;
  - DNA v2 0.95/0.01: sketch-space matching against `R_s(grad(x_hat))` and the
    observed sketch, using the known v2 key and a differentiable SRHT projection
    (not the naive lift);
  - DNA v1 conservative: both plain transmitted-update matching and
    structure-aware debiasing `(T - m*mean(T))/(1-m)` per tensor, selecting the
    better branch by attacker-observable objective only;
  - Count-Sketch: E1 on the decoded unsketched estimate and E2 in sketch space
    with known hashes.
- Adult TabLeak loop: added the same adaptive objectives by wrapping the
  official TabLeak cosine loss inside the worker process and restoring it after
  each call.
- Count-Sketch E2 was implemented with a compact bucket representation. This is
  algebraically the same sketch-space cosine objective but avoids materializing
  the full 500k-column sketch table inside every optimizer closure.
- Added identity/lossless objective-equivalence assertions in
  `tests/test_priority30_adaptive_objectives.py`. The local environment does not
  expose `pytest` on the system Python, so the assertions were run directly with
  `external_defenses/.venv/bin/python`.

Commands:

```bash
external_defenses/.venv/bin/python -B -m py_compile \
  experiments/priority30_native_defenses/run_audit.py \
  tests/test_priority30_adaptive_objectives.py

external_defenses/.venv/bin/python -B - <<'PY'
# direct identity/lossless assertions for mask-aware, v2 sketch identity,
# Count-Sketch identity, and v1 debias mix=0
PY

DATALOADER_NUM_WORKERS=0 NUM_ROUNDS=1 external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/run_audit.py \
  --stage S1b --n 39 --workers 4 \
  --output-dir artifacts/priority30_native_defenses/audit

DATALOADER_NUM_WORKERS=0 NUM_ROUNDS=1 external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/run_audit.py \
  --stage S2b --n 39 --workers 4 \
  --output-dir artifacts/priority30_native_defenses/audit
```

Technical smoke/disclosure:

- A separate one-target smoke run was made under
  `artifacts/priority30_native_defenses/adaptive_smoke/`.
- The first Adult S2b smoke attempt was interrupted during Count-Sketch E2 after
  showing that a dense 500k-column loss implementation was computationally
  impractical. No scientific result from that interrupted smoke is reported.
- The Count-Sketch loss was then changed to the compact equivalent described
  above; the smoke completed, then the full S1b/S2b runs were executed.

Stage results:

| Stage | Domain | Cells filled | Result files | Status | Elapsed |
|---|---|---:|---:|---|---:|
| S1b | image | 5 cells × 39 targets | 195 | all `ran` | 590.67 s |
| S2b | Adult | 7 cells × 39 targets | 273 | all `ran` | 9888.14 s |

S1b filled exactly:

- image E2 for `dna_v1_conservative`;
- image E2 for `dna_v2_0p95`;
- image E2 for `gradient_pruning`;
- image E1 and E2 for `count_sketch`.

S2b filled exactly:

- Adult E1 for `count_sketch`;
- Adult E2 for `precode`, `soteria`, `gradient_pruning`, `count_sketch`,
  `dna_v1_conservative`, and `dna_v2_0p95`.

No S1b/S2b cell remains `NOT_ASSESSABLE`.

S3/S4 readiness check after S2b:

```text
build_tasks("S3", 39, audit_dir) -> 0 tasks
build_tasks("S4", 39, audit_dir) -> 0 tasks
```

Therefore S3/S4 were **not** launched in this continuation. Launching the
current stage labels would produce a no-op completion, not valid E3/utility
evidence. A separate implementation of the frozen E3 comparator stages, Adult
utility-matched grid, and Part C full ablation execution is still required
before the full RQ4 verdict can be run honestly.

## Required paired undefended comparator fix: S0u

An independent check found that the existing S1/S2 artifacts did not include a
paired undefended comparator on the same 39 confirmatory targets. This is part
of the frozen E2 primary test ("defense reduces reconstruction quality versus
undefended on the same target"), so adding it is recorded here as a required
completion of the existing design, not a new design change.

S0u command:

```bash
DATALOADER_NUM_WORKERS=0 NUM_ROUNDS=1 external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/run_audit.py \
  --stage S0u --n 39 --workers 4 \
  --output-dir artifacts/priority30_native_defenses/audit
```

S0u generated undefended results for exactly the same target IDs/source IDs as
S1/S2:

| Domain | Result files | Target/source match check |
|---|---:|---|
| image | 39 | `image_matches_S1: true` |
| Adult | 39 | `adult_matches_S2: true` |

The S0u result files also include data-free baselines:

- image: gray image and CIFAR-10 mean image;
- Adult: mean/mode and empirical-marginal guesses.

S0u completed with 78/78 result files and summary line count 79
(header + 78 rows). No S0u background workload remained after completion.

## Critical correction: image E2 adaptive attacks and Adult PRECODE E2

An independent review found that the image E2 adaptive objectives from S1b and
the Adult PRECODE E2 row from S2b were not valid for final analysis. The old
artifacts are kept for disclosure but are no longer used by S6.

Verified issues:

1. `AdaptiveImageReconstructor._adaptive_loss` checked canonical mode strings
   (`mask_aware`, `v2_sketch`, `count_sketch`), while the caller passed variant
   labels such as `gradient_pruning_mask_aware_e2`,
   `dna_v2_sketch_space_e2`, and `count_sketch_sketch_space_e2`. Unknown modes
   silently fell back to plain `reconstruction_costs`.
2. Image v2/count-sketch E2 passed the undefended gradient as `input_gradient`.
   With the silent fallback above, this made the purported adaptive attack match
   the raw gradient rather than the server-observable sketch payload.
3. Image Soteria/PRECODE/ATS E2 were falling through to the E1 path.
4. Adult PRECODE E2 used the PRECODE model but did not activate an adaptive
   PRECODE-specific matching objective.

Code fixes:

- Adaptive image modes are now canonical and fail-closed: `plain`,
  `mask_aware`, `v2_sketch`, and `count_sketch`. Any unknown mode raises.
- Image v2 and count-sketch E2 now pass only the server-observable sketch
  payload into the official Geiping loop.
- Image Soteria E2 omits the defended fully connected layer gradient.
- Image PRECODE E2 omits the stochastic bottleneck gradients.
- Image ATS E2 is labeled as an identity straight-through approximation because
  the official policy path used here is PIL/non-differentiable; scoring remains
  against the raw image.
- Image gradient-pruning E2 uses mask-aware matching.
- Image v1 E2 reports the better of plain transmitted-update matching and
  structure-aware debiasing, selected by attacker-observable objective only.
- Adult PRECODE E2 now uses mask-aware matching that omits the bottleneck
  gradients.
- S6 now includes an `E2 != E1` fail-closed check. Exact equality fails the
  cell unless there is a documented exception. Documented exceptions observed:
  ATS straight-through approximation and the v1 plain branch when it is selected
  by the pre-registered objective rule.

Corrected rerun commands:

```bash
DATALOADER_NUM_WORKERS=0 NUM_ROUNDS=1 external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/run_audit.py \
  --stage S1c --n 39 --workers 4 \
  --output-dir artifacts/priority30_native_defenses/audit

DATALOADER_NUM_WORKERS=0 NUM_ROUNDS=1 external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/run_audit.py \
  --stage S2c --n 39 --workers 4 \
  --output-dir artifacts/priority30_native_defenses/audit
```

Corrected rerun results:

| Stage | Scope | Result files | Status | Elapsed |
|---|---|---:|---|---:|
| S1c | image E2 for all 7 defenses | 273 | all `ran` | 1106.27 s |
| S2c | Adult PRECODE E2 | 39 | all `ran` | 1397.05 s |

S6 supersession rule:

- All old image E2 rows from S1/S1b are excluded and replaced by S1c.
- Old Adult PRECODE E2 from S2b is excluded and replaced by S2c.
- Other S1/S2/S2b rows remain as before.

## S6 paired analysis for E1/E2 cells

The S6 paired analysis compares every available S1/S1b/S2/S2b defense result
against the paired S0u undefended result on the same target, plus the relevant
data-free baselines. Score convention: larger score means worse reconstruction
/ stronger apparent defense. For image this is raw input-space MSE. For Adult,
the official TabLeak feature accuracy is converted to `100 - accuracy_percent`.

S6 command:

```bash
external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/analyze_s0u_pairs.py \
  --audit-dir artifacts/priority30_native_defenses/audit --n 39
```

S6 produced:

| Artifact | Rows |
|---|---:|
| `artifacts/priority30_native_defenses/audit/S6/paired_per_target.csv` | 3,042 comparisons + header |
| `artifacts/priority30_native_defenses/audit/S6/paired_sign_tests.csv` | 78 sign-test summaries + header |
| `artifacts/priority30_native_defenses/audit/S6/e2_not_e1_check.csv` | 312 E2-vs-E1 checks + header |

Primary E2/E1 paired defended-vs-undefended Holm family:

- 16 cells were significant after Holm in the direction "defended
  reconstruction score worse than paired undefended reconstruction score".
- These are analysis outputs only for the completed E1/E2 portions. They are
  **not** a final RQ4 verdict, because E3 comparators, Adult utility-matched
  DP, and Part C are still not implemented/executed.

Significant defended-vs-undefended rows after Holm:

| Domain | Stage | Eval | Defense | Wins / non-ties | Ties | Raw p | Holm p |
|---|---|---|---|---:|---:|---:|---:|
| Adult | S2 | E1 | `dna_v2_0p95` | 32/38 | 1 | 1.22e-05 | 1.46e-04 |
| Adult | S2 | E1 | `gradient_pruning` | 39/39 | 0 | 1.82e-12 | 4.73e-11 |
| Adult | S2 | E1 | `precode` | 39/39 | 0 | 1.82e-12 | 4.73e-11 |
| Adult | S2 | E1 | `soteria` | 39/39 | 0 | 1.82e-12 | 4.73e-11 |
| Adult | S2b | E2 | `dna_v1_conservative` | 29/35 | 4 | 5.84e-05 | 6.43e-04 |
| Adult | S2c | E2 | `precode` | 39/39 | 0 | 1.82e-12 | 4.73e-11 |
| image | S1 | E1 | `ats` | 39/39 | 0 | 1.82e-12 | 4.73e-11 |
| image | S1 | E1 | `dna_v1_conservative` | 36/39 | 0 | 1.80e-08 | 2.71e-07 |
| image | S1 | E1 | `dna_v2_0p95` | 35/39 | 0 | 1.68e-07 | 2.35e-06 |
| image | S1 | E1 | `gradient_pruning` | 39/39 | 0 | 1.82e-12 | 4.73e-11 |
| image | S1 | E1 | `precode` | 39/39 | 0 | 1.82e-12 | 4.73e-11 |
| image | S1 | E1 | `soteria` | 39/39 | 0 | 1.82e-12 | 4.73e-11 |
| image | S1c | E2 | `ats` | 39/39 | 0 | 1.82e-12 | 4.73e-11 |
| image | S1c | E2 | `dna_v1_conservative` | 35/39 | 0 | 1.68e-07 | 2.35e-06 |
| image | S1c | E2 | `gradient_pruning` | 38/39 | 0 | 7.28e-11 | 1.16e-09 |
| image | S1c | E2 | `precode` | 39/39 | 0 | 1.82e-12 | 4.73e-11 |

Data-free baseline paired comparisons:

- 10 rows were significant after Holm in the direction "defended
  reconstruction score worse than the data-free baseline".
- The detailed rows are in
  `artifacts/priority30_native_defenses/audit/S6/paired_sign_tests.csv`.

E2-vs-E1 fail-closed check summary:

| Domain/defense | PASS | Documented exception | FAIL |
|---|---:|---:|---:|
| image `precode` | 39 | 0 | 0 |
| image `soteria` | 39 | 0 | 0 |
| image `gradient_pruning` | 39 | 0 | 0 |
| image `count_sketch` | 39 | 0 | 0 |
| image `dna_v2_0p95` | 39 | 0 | 0 |
| image `dna_v1_conservative` | 26 | 13 (`dna_v1_plain_e2` selected) | 0 |
| image `ats` | 0 | 39 (PIL/non-differentiable straight-through approximation) | 0 |
| Adult `precode` | 39 | 0 | 0 |

## SHA-256 hashes

| Artifact | SHA-256 |
|---|---|
| `protocols/amendments/2026-09-30_priority30_native_defenses.md` | `13e201883219021a51d08d9326b582814b05b654ff0781a07a592a9199314b00` |
| `experiments/priority30_native_defenses/native_adapters.py` | `1f301235bfb7253757cc946459579e73a7e94061a0dbe6c7a46f27a21da956e5` |
| `experiments/priority30_native_defenses/run_adapter_checks.py` | `7cebff2304bfcc74342a31f4fc892b771eec98983f7a1f6a26c47ba3bc9fb455` |
| `experiments/priority30_native_defenses/run_audit.py` | `252654a6b5a2177e2edc10d3d7acbc51496e0cd323a395395d18ff4474567093` |
| `experiments/priority30_native_defenses/analyze_s0u_pairs.py` | `2b6b3f1cdb3e5807160f605935ecfbd052a822b99903b3902198ced402c12b2c` |
| `tests/test_priority30_adaptive_objectives.py` | `aaf9bf6d53fe93cc897bb0b59fae03c33bd8fa5b7e832392e99a3b6fd3472005` |
| `experiments/priority30_native_defenses/paysim_no_balance_ablation.py` | `3f10801ade1555dc0d7d4c1052b3834647150b641a08748fedb36a1380091530` |
| `artifacts/priority30_native_defenses/adapter_checks/adapter_checks.json` | `a8f97c93aac392af33b13f5679ec0100cd28ff45d1263b3ee9febfd96e8e4cad` |
| `artifacts/priority30_native_defenses/audit/S0/summary.csv` | `de55559fc83143ed34b85792401c481569bf1aaa329e202d28d3b451366cc915` |
| `artifacts/priority30_native_defenses/audit/S1/summary.csv` | `0ba596505e21fe31ddd3a72cb18411b63d9f3e1dc42f8a6b780f551e6fc62cba` |
| `artifacts/priority30_native_defenses/audit/S1b/summary.csv` | `a4921f458835383229d47650c7e96ed36d30a03902dcb4b52f68ebd8843d20ec` |
| `artifacts/priority30_native_defenses/audit/S1b_progress.log` | `8a838766b0d6e009da3476b09e41b124a42bf851ff8f7010285c985b9dcf0387` |
| `artifacts/priority30_native_defenses/audit/S2b/summary.csv` | `1f339af23da19a469f7ceba142dbbcfaae249007b8483034d6d3d218cb309d47` |
| `artifacts/priority30_native_defenses/audit/S2b_progress.log` | `beb715944547ca70dcd544f7359716f0f254ccb5525ea152b18c9464236c6932` |
| `artifacts/priority30_native_defenses/audit/S1c/summary.csv` | `b8af212a838c737fd42b95ad937431955b5f8696c3f5b554966151ce7bf427a4` |
| `artifacts/priority30_native_defenses/audit/S1c_progress.log` | `226f3e120dcff8ec635865f12c838dd2ff50ee2889fafec79a5fac9b7f36257b` |
| `artifacts/priority30_native_defenses/audit/S2c/summary.csv` | `895744d7ad9e91b5697a72f8e18551e33dbedb46e9d4c9253582b8e2f0b61e6e` |
| `artifacts/priority30_native_defenses/audit/S2c_progress.log` | `3fbca7e243b259568605a323828217e3617e7cd6a6b1d812f5d28e6475649fe0` |
| `artifacts/priority30_native_defenses/audit/S0u/summary.csv` | `03121d53df2f5d2d69285bab9ccafe673eb818db393a1560cf662b11e8681a2b` |
| `artifacts/priority30_native_defenses/audit/S0u_progress.log` | `00b14c4173669e8dcd435f21db177db6df667a7ae910d88c0a884707b600d80b` |
| `artifacts/priority30_native_defenses/audit/S0u/target_match_check.json` | `e6d2b027063a2cbdccd75efc5ffa52befdf77a598f4486ee85f525e3b8913823` |
| `artifacts/priority30_native_defenses/audit/S6/paired_per_target.csv` | `313d966c4039b132b6ed8502dee9a8fbde4d55949b282e1ed4c16d4377a3a0a4` |
| `artifacts/priority30_native_defenses/audit/S6/paired_sign_tests.csv` | `1bdd4f346263c3185327c54b4cb25524374a19f52930c6cb5efe36becf742d20` |
| `artifacts/priority30_native_defenses/audit/S6/e2_not_e1_check.csv` | `925c0f5f30f59d09038a24ea352062c3d917c92a2599907816eacbe1a07a52c8` |
| `artifacts/priority30_native_defenses/audit/S6/analysis_manifest.json` | `44a408a45caf7d16fa941c4cbccddc21227c88847105bf271bc945f21bb618d1` |

## Final checks

- `py_compile` passed for Priority 30 adapter/check/ablation modules, the
  S1b/S2b/S0u continuation driver/test, and the S6 paired-analysis script.
- `git diff --check` passed for the Priority 30 amendment, scripts, and report.
- No file under `external_defenses/` was edited.
- No file under `Latex/` was edited.
- After S1b/S2b/S0u/S6, no matching background workload remained. S3/S4 were
  not executed because the current runner has zero S3/S4 tasks. The S6 analysis
  here is the required paired E1/E2 analysis against S0u, not the full final
  RQ4 analysis with E3/utility/Part C.

---

## Completion addendum: Count-Sketch correction, E3 comparators, and Part C

Date appended: 2026-10-01.

This addendum supersedes the earlier "S3/S4 not executed" status above.  The
missing comparator stages were implemented in separate fail-closed runners,
Count-Sketch was rerun with a compressing sketch, and Part C completed all 16
paired replicates.

### Count-Sketch configuration correction before E3

Issue found before E3: the initial Priority 30 Count-Sketch adapter used the
absolute FetchSGD default `rows=5, columns=500000`.  For the Priority 30 models
this expanded the gradient rather than compressing it.  Those Count-Sketch rows
are therefore marked invalid as a "non-compressing sketch".

Correction applied before E3:

- keep FetchSGD's five-row structure;
- port the sketch size by compression ratio to the small audit models;
- use `columns = ceil((dimension / 10) / rows)` so that `rows*columns` is well
  below the gradient dimension;
- source note: `external_defenses/fetchsgd/PORTING_NOTES.md` documents the
  official large-model defaults; this audit ports their compression intent to
  LeNet-Zhu and Adult instead of reusing absolute dimensions.

Rerun commands:

```bash
DATALOADER_NUM_WORKERS=0 external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/run_audit.py \
  --stage S1d --n 39 --workers 4 \
  --output-dir artifacts/priority30_native_defenses/audit

DATALOADER_NUM_WORKERS=0 external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/run_audit.py \
  --stage S2d --n 39 --workers 4 \
  --output-dir artifacts/priority30_native_defenses/audit

external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/analyze_s0u_pairs.py \
  --audit-dir artifacts/priority30_native_defenses/audit --n 39
```

S1d/S2d produced 78/78 Count-Sketch result files each.  A technical replay was
needed for S2d E2 because the first S2d pass preserved the old `NOT_ASSESSABLE`
guard; those files were saved as `result.invalid_pre_fix.json` before replay.

Valid compressing Count-Sketch E2 results:

| Domain | Stage | Wins defense worse vs undefended | Losses | Ties | raw p | Holm p | Median score diff |
|---|---|---:|---:|---:|---:|---:|---:|
| image | S1d E2 | 39 | 0 | 0 | 1.82e-12 | 4.73e-11 | 0.003142 |
| Adult | S2d E2 | 37 | 0 | 2 | 7.28e-12 | 9.46e-11 | 9.821429 |

### E3(i)/(ii): paper Gaussian and distortion-matched clipped DP

Runner added:
`experiments/priority30_native_defenses/run_e3_dp_comparators.py`.

Commands:

```bash
DATALOADER_NUM_WORKERS=0 external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/run_e3_dp_comparators.py \
  --stage S3 --n 39 --workers 4 \
  --output-dir artifacts/priority30_native_defenses/audit

external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/run_e3_dp_comparators.py \
  --stage summarize --n 39 \
  --output-dir artifacts/priority30_native_defenses/audit
```

Disclosure: the S3 attack run completed all 936 result files, then the first
summary attempt failed because the CSV field list omitted image SSIM columns.
No attacks were rerun; the summarizer was patched to use the union of row keys
and replayed on the existing artifacts.

Artifacts:

- `artifacts/priority30_native_defenses/audit/S3/calibration.json`
- `artifacts/priority30_native_defenses/audit/S6/e3_per_target.csv`
- `artifacts/priority30_native_defenses/audit/S6/e3_sign_tests.csv`

Selected E3(i)/(ii) results.  Score convention: larger error means worse
reconstruction and therefore stronger apparent protection.  `defense_wins`
means the defense had larger error than DP; `dp_wins` means DP had larger error
than the defense.

| Domain | Defense | Comparator | Defense wins | DP wins | Ties | Holm p defense>DP | Holm p DP>defense | Median effect metric defense vs DP |
|---|---|---|---:|---:|---:|---:|---:|---:|
| image | PRECODE | paper unclipped | 33 | 6 | 0 | 1.14e-04 | 1.00 | PSNR 9.39 vs 23.49 |
| image | PRECODE | distortion-matched clipped | 32 | 7 | 0 | 4.92e-04 | 1.00 | PSNR 9.39 vs 11.45 |
| image | Soteria | paper unclipped | 0 | 39 | 0 | 1.00 | 3.27e-11 | PSNR 24.29 vs 18.26 |
| image | Soteria | distortion-matched clipped | 0 | 39 | 0 | 1.00 | 3.27e-11 | PSNR 24.29 vs 14.15 |
| image | Gradient pruning | paper unclipped | 0 | 39 | 0 | 1.00 | 3.27e-11 | PSNR 20.66 vs 18.27 |
| image | Gradient pruning | distortion-matched clipped | 31 | 8 | 0 | 1.91e-03 | 1.00 | PSNR 20.66 vs 22.18 |
| image | Count-Sketch | distortion-matched clipped | 0 | 39 | 0 | 1.00 | 3.27e-11 | PSNR 21.16 vs 7.96 |
| image | DNA v1 conservative | distortion-matched clipped | 33 | 6 | 0 | 1.14e-04 | 1.00 | PSNR 22.65 vs 23.55 |
| image | DNA v2 0.95/0.01 | distortion-matched clipped | 0 | 39 | 0 | 1.00 | 3.27e-11 | PSNR 23.49 vs 22.07 |
| Adult | PRECODE | paper unclipped | 39 | 0 | 0 | 3.27e-11 | 1.00 | accuracy 23.21 vs 75.00 |
| Adult | PRECODE | distortion-matched clipped | 39 | 0 | 0 | 3.27e-11 | 1.00 | accuracy 23.21 vs 100.00 |
| Adult | DNA v1 conservative | distortion-matched clipped | 26 | 7 | 6 | 7.91e-03 | 1.00 | accuracy 95.54 vs 99.11 |
| Adult | DNA v2 0.95/0.01 | distortion-matched clipped | 8 | 21 | 10 | 1.00 | 1.09e-01 | accuracy 100.00 vs 99.11 |
| Adult | Count-Sketch | distortion-matched clipped | 1 | 38 | 0 | 1.00 | 8.00e-10 | accuracy 88.39 vs 64.29 |

ATS E2 remains `NOT_ASSESSABLE` for the final verdict because the documented
straight-through exception is not a working adaptive attack.

### E3(iii): Adult utility-matched DP

Runner added:
`experiments/priority30_native_defenses/run_e3_utility_adult.py`.

Commands:

```bash
DATALOADER_NUM_WORKERS=0 external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/run_e3_utility_adult.py \
  --stage calibrate --workers 4 \
  --output-dir artifacts/priority30_native_defenses/audit

DATALOADER_NUM_WORKERS=0 external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/run_e3_utility_adult.py \
  --stage S4 --workers 4 \
  --output-dir artifacts/priority30_native_defenses/audit
```

Disclosure:

- First calibration attempt failed before scientific output because Tableak's
  Adult loader requires cwd `external_defenses/tableak`; the runner was fixed to
  chdir locally.
- The initial grid extension to `sigma=0.3` did not bracket; the unbracketed
  selection was preserved as
  `artifacts/priority30_native_defenses/audit/S4/adult_utility_selected_unbracketed_to_0p3.json`.
- The grid was extended upward to `[1.0, 3.0, 10.0]`, after which `sigma=0.3`
  was bracketed for both Adult PRECODE and Adult DNA v1 conservative.

Calibration:

| Item | Value |
|---|---:|
| Clip norm C | 0.7474398315 |
| Grid | 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10 |
| Selected sigma for PRECODE | 0.3 |
| Selected sigma for DNA v1 conservative | 0.3 |
| RDP epsilon, delta=1e-5, one release | 21.5506426840 |

E3(iii) confirmatory results:

| Domain | Defense | Comparator | Defense wins | DP wins | Ties | Holm p defense>DP | Holm p DP>defense | Median accuracy defense vs DP |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Adult | PRECODE | utility-matched clipped DP | 36 | 3 | 0 | 3.61e-08 | 1.00 | 23.21 vs 39.29 |
| Adult | DNA v1 conservative | utility-matched clipped DP | 0 | 39 | 0 | 1.00 | 3.64e-12 | 95.54 vs 38.39 |

### Part C: PaySim no-balance-feature ablation

Commands:

```bash
external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/paysim_no_balance_ablation.py --check-only

DATALOADER_NUM_WORKERS=0 external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/paysim_no_balance_ablation.py \
  --replicates 16 --workers 4 \
  --output-dir artifacts/priority30_native_defenses/paysim_no_balance_ablation

DATALOADER_NUM_WORKERS=0 external_defenses/.venv/bin/python -B \
  experiments/priority30_native_defenses/paysim_no_balance_ablation.py \
  --replicates 16 --workers 8 \
  --output-dir artifacts/priority30_native_defenses/paysim_no_balance_ablation
```

Projection check:

- original dimension: 13;
- projected dimension: 7;
- exactly removed:
  `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`,
  `balance_diff_orig`, `balance_diff_dest`;
- kept:
  `step`, `amount`, `type=CASH_IN`, `type=CASH_OUT`, `type=DEBIT`,
  `type=PAYMENT`, `type=TRANSFER`.

Disclosure: the first 4-worker run was technically interrupted after 6/96
result files to reduce wall time; four orphan worker processes were killed
explicitly.  The run was resumed with 8 workers using the script's per-file
resume logic.  Final output contains all 96 JSON files and `summary.csv`
(97 lines including header).

Absolute metric means over 16 paired replicates:

| Balance features | Method | F1 mean | AUC mean | PR-AUC mean |
|---|---|---:|---:|---:|
| included | baseline | 0.787144 | 0.997559 | 0.824455 |
| included | v1 conservative | 0.787021 | 0.997603 | 0.827056 |
| included | v2 0.95/0.01 | 0.747309 | 0.989657 | 0.766803 |
| removed | baseline | 0.302671 | 0.944995 | 0.249158 |
| removed | v1 conservative | 0.302712 | 0.944745 | 0.249138 |
| removed | v2 0.95/0.01 | 0.267497 | 0.941379 | 0.206972 |

Paired transform-vs-baseline deltas:

| Balance features | Method | ΔF1 mean | ΔAUC mean | ΔPR-AUC mean |
|---|---|---:|---:|---:|
| included | v1 conservative | -0.000123 | +0.000044 | +0.002601 |
| included | v2 0.95/0.01 | -0.039835 | -0.007902 | -0.057652 |
| removed | v1 conservative | +0.000041 | -0.000250 | -0.000020 |
| removed | v2 0.95/0.01 | -0.035174 | -0.003616 | -0.042186 |

### Final RQ4 verdict table for emitted Priority 30 cells

Rule: `SURVIVES` requires a significant adaptive E2 reduction vs paired
undefended and no significant loss to the relevant DP comparator.  Adult cells
with E3(iii) also require not losing to utility-matched DP.  `DOES NOT
SURVIVE` names the first failing check.  `NOT_ASSESSABLE` indicates no valid
adaptive cell.

| Domain | Defense | Verdict | Check that determines verdict |
|---|---|---|---|
| image | PRECODE | SURVIVES | E2 passes; defense beats distortion-matched clipped DP |
| image | Soteria | DOES NOT SURVIVE | E2 does not reduce reconstruction vs undefended; DP also beats defense |
| image | Gradient pruning | SURVIVES | E2 passes; defense beats distortion-matched clipped DP |
| image | ATS | NOT_ASSESSABLE | no working adaptive attack; straight-through exception only |
| image | Count-Sketch | DOES NOT SURVIVE | E2 passes, but distortion-matched clipped DP beats defense 39/39 |
| image | DNA v1 conservative | SURVIVES | E2 passes; defense beats distortion-matched clipped DP |
| image | DNA v2 0.95/0.01 | DOES NOT SURVIVE | E2 fails after Holm; distortion-matched clipped DP beats defense 39/39 |
| Adult | PRECODE | SURVIVES | E2 passes; defense beats distortion-matched and utility-matched DP |
| Adult | Soteria | DOES NOT SURVIVE | E2 does not reduce reconstruction vs undefended |
| Adult | Gradient pruning | DOES NOT SURVIVE | E2 does not reduce reconstruction vs undefended |
| Adult | Count-Sketch | DOES NOT SURVIVE | E2 passes, but distortion-matched clipped DP beats defense 38/1 |
| Adult | DNA v1 conservative | DOES NOT SURVIVE | E2 passes, but utility-matched DP beats defense 39/39 |
| Adult | DNA v2 0.95/0.01 | DOES NOT SURVIVE | E2 does not reduce reconstruction vs undefended |

Study-wide summary for emitted cells:

- SURVIVES: 4 cells (image PRECODE, image gradient pruning, image DNA v1
  conservative, Adult PRECODE).
- DOES NOT SURVIVE: 8 cells.
- NOT_ASSESSABLE: 1 cell (image ATS).

### Additional SHA-256 hashes for completion artifacts

| Artifact | SHA-256 |
|---|---|
| `experiments/priority30_native_defenses/native_adapters.py` | `ea1203e79c305fa2e43fbfccbb3f7a222d582aba1105559df409ccd3cb1ffd8f` |
| `experiments/priority30_native_defenses/run_audit.py` | `d2cfd765e18c1deda58e5c8b0208351342c70204d05821685ed13da7b4c23f34` |
| `experiments/priority30_native_defenses/analyze_s0u_pairs.py` | `78febad440a0babd6f2469a3bb85a20032fb3330f8f345e21fdda1d26342790b` |
| `experiments/priority30_native_defenses/run_adapter_checks.py` | `3d06632c22c76e4b16675b09fe93de2fe59d7f04eb5f95dab59c482e4faa1a0f` |
| `experiments/priority30_native_defenses/run_e3_dp_comparators.py` | `2ea05205089857c96202fc41c9fbb64063e5d82931baeac01ac0ea67a9fcb67b` |
| `experiments/priority30_native_defenses/run_e3_utility_adult.py` | `8af86f5fd4a8b6c1260d94689f1c4e8e525810a3d08f2af89a01c4dc502bb3d3` |
| `experiments/priority30_native_defenses/paysim_no_balance_ablation.py` | `f944964cd72049ee4d4b982db4f0e6b76b0f12e8fe47664cc114fb9687816874` |
| `artifacts/priority30_native_defenses/audit/S1d/summary.csv` | `605a7cf440ffae14aa2b7556f8d02690d3dcf88f1b69fe86ff309b95be0f5f5d` |
| `artifacts/priority30_native_defenses/audit/S2d/summary.csv` | `d55392ca65a70bf52d48201978ba9cbd1d6f377df7074c0afe0af81e38089a4e` |
| `artifacts/priority30_native_defenses/audit/S6/paired_sign_tests.csv` | `39b8108f34b7c61cd017050da02d9a7db6a86643c46aa3f5f4be47973c2f6100` |
| `artifacts/priority30_native_defenses/audit/S6/e3_sign_tests.csv` | `4deb8c01a0c01f7c4857c5f02c461f0c3fc51380d670bbb934b7485afdb3bb8f` |
| `artifacts/priority30_native_defenses/audit/S6/e3_per_target.csv` | `184b970e071b8e89a0798947b1a7c0862039aa66accd3075fba35ae1c9ae0480` |
| `artifacts/priority30_native_defenses/audit/S3/calibration.json` | `ca8c9585b0ee92e7ef7d261e4fa4522dca71c96a0e098aaaaa5f9f8338baf475` |
| `artifacts/priority30_native_defenses/audit/S4/adult_utility_selected.json` | `70cb03e257b529beee818f8067342239f707e2bf5d598c82f3bec4c894c64fde` |
| `artifacts/priority30_native_defenses/audit/S4/adult_utility_grid.csv` | `e80aa0f54b70e0673020270e53a6eec6f464c1b467300094e85b68b1b1791bb7` |
| `artifacts/priority30_native_defenses/audit/S6/e3_utility_adult_sign_tests.csv` | `0874b18f1d19c2760b9ca204de53f4b15c032a166b41b41cb8f23b4d1433391c` |
| `artifacts/priority30_native_defenses/audit/S6/e3_utility_adult_per_target.csv` | `4dd15e9418cd8ceacc557b0d0b0475ab974652a9d714d32281e8dc42a11ba53a` |
| `artifacts/priority30_native_defenses/paysim_no_balance_ablation/summary.csv` | `dba3dec97016de5f0c80c7b66024fbf53ff132bfb6a05cad7e70ed9e780304b9` |
| `artifacts/priority30_native_defenses/paysim_no_balance_ablation.log` | `5e29244d3503328196e0d32306c3ed7771769ba4bf5ac3bab94197b784d8db3a` |

### Completion-state checks

- E3(i)/(ii), E3(iii), and Part C all completed.
- All per-target CSVs requested for emitted tests are under
  `artifacts/priority30_native_defenses/audit/S6/`.
- No `Latex/` files were edited.
- No file under `external_defenses/` was edited.
