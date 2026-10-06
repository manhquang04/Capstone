# Priority 26 — Transform v2-SB server-blind aggregation

Date: 2026-09-29  
Scope: new mechanism/protocol evaluation; no earlier artifacts or reports were modified.  
Amendment: `protocols/amendments/2026-09-29_priority26_v2sb_server_blind.md`

## Registered question

RQ1-v2SB. Against an honest-but-curious server that does not hold the sketch key, and assuming no collusion between the server and any client, does server-blind sketch aggregation (v2-SB) reduce reconstruction quality of individual client updates, measured under an exact one-sided sign test on input-space reconstruction MSE, more than a distortion- or utility-matched DP baseline?

## Scope actually executed

The frozen amendment used the allowed reduction path because full Priority 26 would exceed the practical three-day budget:

1. Part 4 cost benchmark was dropped.
2. A4 cross-round attack was dropped.
3. All T2 gradient-only arms were dropped.

Executed components:

- Part 0 key-space audit for v1/v2.
- Part 1 v2-SB implementation and unit tests.
- Part 2 RQ2-style utility confirmatory for v2-SB, n=52.
- Part 3 T1-only privacy gate: A1 seed-leak control, A2 key-less direct attacks, A3 invariant-statistics ridge.

## Commands run

```bash
PYTHONPATH=. .venv-phase1/bin/python -m pytest tests/test_transform_defense_v2_server_blind.py -q

PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority26_v2sb/keyspace_targets_20260929 \
  --output-name paysim_priority26_keyspace_targets.pt \
  --groups 8 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092601 \
  --purpose 'Priority 26 v2-SB keyspace audit n8'

PYTHONPATH=. .venv-phase1/bin/python experiments/priority26_keyspace_audit.py \
  --target artifacts/priority26_v2sb/keyspace_targets_20260929/paysim_priority26_keyspace_targets.pt \
  --output-dir artifacts/priority26_v2sb/keyspace_audit_20260929_replay1 \
  --seed 2026092601 --wrong-seeds 1000

QUICK=1 MAX_ROWS=20000 NUM_ROUNDS=1 FL_RUN_SEED=260001 \
  DNA_TRANSFORM_V2SB_OUTPUT_PATH=artifacts/priority26_v2sb/smoke_v2sb_20260929/metrics.json \
  PYTHONPATH=. .venv-phase1/bin/python experiments/run_fraud_fl_dna_transform_v2_server_blind.py

PYTHONPATH=. .venv-phase1/bin/python experiments/run_rq2_v2sb_confirmatory.py \
  --config protocols/config/rq2_v2sb_confirmatory.json \
  --output-dir artifacts/priority26_v2sb/rq2_v2sb_confirmatory_20260929 \
  --workers 9

PYTHONPATH=. .venv-phase1/bin/python experiments/analyze_rq2_v2sb_confirmatory.py \
  --config protocols/config/rq2_v2sb_confirmatory.json \
  --run-dir artifacts/priority26_v2sb/rq2_v2sb_confirmatory_20260929 \
  --output-dir results/rq2_v2sb/confirmatory_20260929

PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority26_v2sb/t1_dev_targets_20260929 \
  --output-name paysim_priority26_t1_dev_targets.pt \
  --groups 8 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092602 \
  --purpose 'Priority 26 v2-SB T1 dev n8'

PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority26_v2sb/t1_confirm_targets_20260929 \
  --output-name paysim_priority26_t1_confirm_targets.pt \
  --groups 39 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092603 \
  --purpose 'Priority 26 v2-SB T1 confirm n39'

PYTHONPATH=. .venv-phase1/bin/python experiments/priority26_t1_v2sb_privacy.py \
  --dev-target artifacts/priority26_v2sb/t1_dev_targets_20260929/paysim_priority26_t1_dev_targets.pt \
  --confirm-target artifacts/priority26_v2sb/t1_confirm_targets_20260929/paysim_priority26_t1_confirm_targets.pt \
  --output-dir artifacts/priority26_v2sb/t1_privacy_20260929 \
  --seed 2026092604
```

One technical replay occurred: the first key-space audit output directory `keyspace_audit_20260929` failed before producing scientific output because the audit script cited a non-existent v1 helper (`transform_defense._rng`). The script was corrected to cite `transform_update_array` and `_dna_block_seed_from_float32_block`, then rerun into `keyspace_audit_20260929_replay1`.

## Part 0 — key-space audit of v1/v2

Code citations recorded in `artifacts/priority26_v2sb/keyspace_audit_20260929_replay1/keyspace_audit_summary.json`:

- v2 `_derive_seed`: `dna_encoder/transform_defense_v2.py:229-235`
- v2 `_rng`: `dna_encoder/transform_defense_v2.py:238-239`, using `int(seed) % (2**32)`
- v1 `transform_update_array`: `dna_encoder/transform_defense.py:36-92`
- v1 `_dna_block_seed_from_float32_block`: `dna_encoder/transform_defense.py:110-142`, returning `seed % (2**31)`
- project `derive_seed`: `privacy/seed_manager.py:16-20`, range `[1, 2**31 - 1]`

Measured v2 decode timing on the smallest informative BN tensor:

| Metric | Value |
|---|---:|
| Median time / candidate seed | 4.4958e-05 s |
| Mean time / candidate seed | 4.5415e-05 s |
| v2 2^32 enumeration, 1 process | 193,093 s |
| v2 2^32 enumeration, 9 processes | 21,455 s |
| v1 2^31 enumeration, 1 process | 96,547 s |
| v1 2^31 enumeration, 9 processes | 10,727 s |

Verification criterion tested: standardized T1 batch-mean MSE after reconstructing BN running_mean; lower is better. On 8 targets, true seed vs 1,000 random wrong seeds:

| Result | Value |
|---|---:|
| All true seeds rank 1 | false |
| Worst true-seed rank | 6 |
| Total wrong seeds better-or-equal across all targets | 13 |

Verdict: v1/v2 old seed paths are bounded by small 31/32-bit spaces, but the tested BN-running_mean T1 criterion does not perfectly separate true seed from 1,000 random wrong seeds on this sample.

## Part 1 — v2-SB implementation and tests

New module:

- `dna_encoder/transform_defense_v2_server_blind.py`

Key design:

- 256-bit round key (`secrets.token_bytes(32)` for generation; deterministic SHA-256 round-key derivation only for reproducible experiments).
- Randomness domain-separated through SHAKE-256 over `(K, round, tensor_index, purpose)`.
- Server aggregation object stores no `key` or `round_key`.
- Server aggregates only sketches; lift/decode happens outside the server object.

Unit tests:

```text
tests/test_transform_defense_v2_server_blind.py: 6 passed
```

Tests cover 256-bit key validation, no modulo seed truncation in v2-SB path, server no-key assertion, linearity of lifted aggregate, one-client lift consistency, and client-specific quantization delta compatibility.

## Part 2 — RQ2-style utility confirmatory

Run:

- Config: `protocols/config/rq2_v2sb_confirmatory.json`
- Execution: `artifacts/priority26_v2sb/rq2_v2sb_confirmatory_20260929`
- Analysis: `results/rq2_v2sb/confirmatory_20260929`
- Jobs: 104/104 success (52 baseline + 52 v2-SB)

| Endpoint | n | Mean Δ (v2-SB − baseline) | SD | 95% CI | Margin | Gate |
|---|---:|---:|---:|---:|---:|---|
| F1 | 52 | -0.006616 | 0.032837 | [-0.015758, 0.002526] | 0.020 | PASS |
| AUC-ROC | 52 | -0.002022 | 0.014514 | [-0.006063, 0.002019] | 0.005 | FAIL |
| PR-AUC (descriptive) | 52 | -0.005482 | 0.025189 | [-0.012495, 0.001530] | descriptive | — |

RQ2-style conclusion: v2-SB does **not** establish non-inferiority for both frozen RQ2 endpoints, because the AUC lower CI bound is slightly below the -0.005 margin.

## Part 3 — T1-only v2-SB privacy gate

Targets:

- Dev n=8: `artifacts/priority26_v2sb/t1_dev_targets_20260929/paysim_priority26_t1_dev_targets.pt`
- Confirm n=39: `artifacts/priority26_v2sb/t1_confirm_targets_20260929/paysim_priority26_t1_confirm_targets.pt`
- Both target-generation provenance files report `max_overlap_with_existing_targets = 0`.

Dev gate results:

| Attack | Wins vs Prior | p vs Prior | Wins vs Decoy | p vs Decoy | Qualified | Median standardized MSE |
|---|---:|---:|---:|---:|---|---:|
| A1 seed-leak control | 1/8 | 0.996094 | 4/8 | 0.636719 | false | 0.354465 |
| A2 identity/direct sketch | 0/8 | 1.000000 | 4/8 | 0.636719 | false | 6.811942 |
| A2 random-surrogate lift | 0/8 | 1.000000 | 4/8 | 0.636719 | false | 19.848870 |
| A3 invariant-statistics ridge | 4/8 | 0.636719 | 2/8 | 0.964844 | false | 0.232247 |

No Level-1 T1 attack qualified at n=8, so the frozen reduced protocol did not execute v2-SB-vs-DP confirmatory privacy tests.

Calibration values computed before this stop:

| Comparator | Clip norm | Noise multiplier | Median v2-SB lift distortion | BN dim |
|---|---:|---:|---:|---:|
| DP distortion-matched on BN vector | 3.915665 | 0.009884 | 0.444363 | 128 |
| DP utility-matched from Priority 25 v2 | 259.084113 | 0.000030 | — | full transmitted update |

## RQ1-v2SB answer under the frozen rule

Under the executed reduced plan, RQ1-v2SB is **not answerable / No under the registered decision rule**, because no pre-specified Level-1 T1 attack qualified on the n=8 development gate. Therefore no Holm-corrected v2-SB-vs-DP confirmatory family was run.

Conditions still required for the mechanism:

- K must remain secret from the server.
- No server-client collusion.
- Fresh K per round.
- v2-SB is not differentially private if K leaks.
- The sketch payload still leaks dimensions, sketch norms, quantization-level statistics, and other invariant summaries; A3 did not exploit these enough to qualify in this n=8 gate.

## Artifact hashes

Full hash manifest:

- `artifacts/priority26_v2sb/priority26_hash_manifest_20260929.txt`
- SHA-256: `526d4e44fbe999de90620b26cace21a41aa00f38c4d8c5f8cb34880e245bc49e`

Selected hashes:

| Artifact | SHA-256 |
|---|---|
| `protocols/amendments/2026-09-29_priority26_v2sb_server_blind.md` | `f01726f51a7887380e8911fdf76aa893309abcdca98bcbdd3fb127b90df1ff06` |
| `dna_encoder/transform_defense_v2_server_blind.py` | see hash manifest |
| `tests/test_transform_defense_v2_server_blind.py` | see hash manifest |
| `protocols/config/rq2_v2sb_confirmatory.json` | `4dcd51e1f1ac26f34dc0db58c8c1d62257a039dd84545728d12bb13097cb66bd` |
| `artifacts/priority26_v2sb/keyspace_audit_20260929_replay1/keyspace_audit_summary.json` | `61be875587eaaef2fadefe2e1bbe0640fe62f3b22234685de3f8a382189cfd92` |
| `artifacts/priority26_v2sb/rq2_v2sb_confirmatory_20260929/execution_manifest.json` | `9801386b54b9a17561fce22f4d64db982cbeecd2ebe645abf5ce048b41fc795b` |
| `artifacts/priority26_v2sb/rq2_v2sb_confirmatory_20260929/execution_summary.json` | `00bda1d5b51ca6ed2e208f70d062b245cf6148c32c99c82950a13ff212647d10` |
| `results/rq2_v2sb/confirmatory_20260929/summary.json` | `293021df2bca693be26a697dc546f36b931f93c6aef6c66c6d75d7aaf4ef3bfc` |
| `results/rq2_v2sb/confirmatory_20260929/per_seed.csv` | `fb6573e4f171a95a30dad4a3bc8ca60fe40a1f73c49156a1d38283aa26d88c6c` |
| `artifacts/priority26_v2sb/t1_privacy_20260929/priority26_t1_v2sb_summary.json` | `04062854469f8ae674cca492a85dea50003acd4a03f0583cd583bbd3f122142d` |
| `artifacts/priority26_v2sb/t1_privacy_20260929/dev_t1_v2sb_rows.csv` | `acb175fefee3af924721ea91818bbb2ce19b0fc139f473eefa24c4aa6f6aeae3` |
| `artifacts/priority26_v2sb/t1_privacy_20260929/confirm_t1_v2sb_rows.csv` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

## Final checks

- v2-SB unit tests: PASS (`6 passed`).
- RQ2-v2SB jobs: PASS (`104/104 success`).
- T1 privacy script: PASS; no confirmatory privacy run executed because no Level-1 attack qualified.
- No `Latex/` files edited.
- No earlier artifacts or reports modified.
