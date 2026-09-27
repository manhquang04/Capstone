# Priority 20 — multi-checkpoint DP-vs-transform head-to-head replication

**Date:** 2026-09-27  
**Status:** COMPLETE  
**Scope:** new pre-registered experiment for PaySim / DNA Transform v1-medium vs distortion-matched DP `noise_multiplier=0.000315`. No files under `Latex/` were edited.

## Purpose

Priority 18 showed that the original tabular DP-vs-transform head-to-head run
shared a single pre-local checkpoint across all 39 targets. This Priority 20
experiment tests whether the same DP-wins pattern replicates across genuinely
separate model-checkpoint realizations.

The experiment intentionally keeps the target count per checkpoint small
(`n=8`) and spans five independently initialized/warmup-trained checkpoints.
The checkpoint, not the within-checkpoint target, is the primary independence
unit for this analysis.

## Pre-registration

Protocol amendment written before any Priority 20 checkpoint, target, or attack
execution:

- `protocols/amendments/2026-09-27_priority20_multi_checkpoint_head_to_head_replication.md`
- SHA-256: `9fdcb76efa5a94552abeb7887537f5fe9d56f9b7bb0a9df9f9d8b82999a281d0`

Frozen design:

- Comparator cell: DNA Transform v1-medium vs DP `noise_multiplier=0.000315`.
- Checkpoints: `5`.
- Targets per checkpoint: `8`.
- Total target groups: `40`.
- Restarts per target: `4`.
- Primary targetwise rule: strict zero-threshold,
  `D = MSE_DNA - MSE_DP`; DP wins when `D < 0`.
- Checkpoint-level tests:
  - count checkpoints where DP wins a strict majority of targets;
  - count checkpoints where mean `D < 0`;
  - exact one-sided binomial test at checkpoint level under `p0 = 0.5`.
- No early stopping, no checkpoint addition/removal after seeing partial
  results, and all five checkpoint runs must be reported.

Runner:

- `experiments/priority20_multi_checkpoint_head_to_head.py`
- SHA-256: `d018b03f1b67375d0b4b5cb6f6e6dc3c2c427a55ba30c8473d6bb910d8678633`

## Commands run

### Checkpoint creation

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority20_multi_checkpoint_head_to_head.py \
  create-checkpoints \
  --output artifacts/priority20_multi_checkpoint/checkpoints_20260927 \
  --checkpoint-seeds 2026092801 2026092802 2026092803 2026092804 2026092805 \
  --data-seed 20260907 \
  --max-rows 500000 \
  --amendment protocols/amendments/2026-09-27_priority20_multi_checkpoint_head_to_head_replication.md
```

### Target generation

The following loop was run exactly once after the amendment was frozen:

```bash
for i in 0 1 2 3 4; do
  mkdir -p artifacts/priority20_multi_checkpoint/ckpt_${i}_targets_20260927
  seed=$((2026092811+i))
  PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
    artifacts/priority20_multi_checkpoint/ckpt_${i}_targets_20260927 \
    --output-name paysim_priority20_ckpt_${i}_targets.pt \
    --groups 8 --records-per-group 4 --fraud-per-group 1 \
    --seed ${seed} \
    --purpose "Priority 20 multi-checkpoint v1_medium vs DP checkpoint ckpt_${i}"
done
```

### Head-to-head execution

The following loop was run exactly once:

```bash
for i in 0 1 2 3 4; do
  run_seed=$((2026092821+i))
  PYTHONPATH=. .venv-phase1/bin/python experiments/priority20_multi_checkpoint_head_to_head.py \
    execute \
    --checkpoint-id ckpt_${i} \
    --checkpoint artifacts/priority20_multi_checkpoint/checkpoints_20260927/ckpt_${i}/pre_local.pt \
    --target artifacts/priority20_multi_checkpoint/ckpt_${i}_targets_20260927/paysim_priority20_ckpt_${i}_targets.pt \
    --output artifacts/priority20_multi_checkpoint/ckpt_${i}_run_20260927 \
    --seed ${run_seed} --groups 8 --restarts 4 --workers 8 \
    --amendment protocols/amendments/2026-09-27_priority20_multi_checkpoint_head_to_head_replication.md
done
```

### Aggregate analysis

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority20_multi_checkpoint_head_to_head.py \
  analyze \
  --reports \
    artifacts/priority20_multi_checkpoint/ckpt_0_run_20260927/priority20_probe_report.json \
    artifacts/priority20_multi_checkpoint/ckpt_1_run_20260927/priority20_probe_report.json \
    artifacts/priority20_multi_checkpoint/ckpt_2_run_20260927/priority20_probe_report.json \
    artifacts/priority20_multi_checkpoint/ckpt_3_run_20260927/priority20_probe_report.json \
    artifacts/priority20_multi_checkpoint/ckpt_4_run_20260927/priority20_probe_report.json \
  --output artifacts/priority20_multi_checkpoint/priority20_aggregate_analysis_20260927.json \
  --amendment protocols/amendments/2026-09-27_priority20_multi_checkpoint_head_to_head_replication.md
```

## Checkpoints

All five checkpoints were independently initialized and warmup-trained from the
frozen checkpoint seeds. The feature split/preprocessing seed was held fixed at
`20260907` to preserve the same PaySim feature coordinate system used by the
existing tabular target-generation pipeline.

| Checkpoint | Seed | Checkpoint SHA-256 | Warmup losses by client |
|---|---:|---|---|
| `ckpt_0` | 2026092801 | `12b01e8143c856ae3eac746fd68b56ff036fa0895fca14a6e3fee9be9c3912ac` | `[0.0033828354702560653, 0.0042360556844407895, 0.005829297677210283]` |
| `ckpt_1` | 2026092802 | `0fcf8a8d41bef4bd9359ce4320798bedd141baf2e804132482d7b9511a7c0433` | `[0.0041763885352825375, 0.005217982436121032, 0.006849392315930443]` |
| `ckpt_2` | 2026092803 | `affa96b6709f1d07761cfe36af899c322fffe5433e48075776b11709399f3381` | `[0.0019207139340112983, 0.0027704632700259746, 0.0037571419713196496]` |
| `ckpt_3` | 2026092804 | `fc26747761f8f5687aa033a87c39c90088d63b5624eb460ec533ccb9d783ad33` | `[0.0027341931539364667, 0.0033209539714203883, 0.004466830299371982]` |
| `ckpt_4` | 2026092805 | `1797e895ce4eac857add189f31201f0a227c67a84481f3ba9c41117638e7244e` | `[0.005064247495143181, 0.00614120893775788, 0.007739872558889275]` |

Checkpoint manifest:

- `artifacts/priority20_multi_checkpoint/checkpoints_20260927/priority20_checkpoint_manifest.json`
- SHA-256: `24c5a49fd20c07eebbb219af6927e330dd69f7bbd7b8c687cec4cd76e9d17197`

## Data firewall

All five target provenance files report `max_overlap_with_existing_targets = 0`.

| Checkpoint | Target path | Target SHA-256 | Provenance SHA-256 |
|---|---|---|---|
| `ckpt_0` | `artifacts/priority20_multi_checkpoint/ckpt_0_targets_20260927/paysim_priority20_ckpt_0_targets.pt` | `11d922ade890d26c2d79fd23642fef0ec95ae9f56ae50489f55a522a5506c772` | `7288c8e51c593493c6d29587227d23746863b9a8ea3ec1f705e1292f2c6f59c2` |
| `ckpt_1` | `artifacts/priority20_multi_checkpoint/ckpt_1_targets_20260927/paysim_priority20_ckpt_1_targets.pt` | `90973d3f6a9765cfe1c92521dc4bcbaf8e6bb93e9aa5318d3cf632791925cf86` | `002b58488623a15b79c0dc6d1abf78e44fcf7df7b3091ccd7237bf4c120d13c3` |
| `ckpt_2` | `artifacts/priority20_multi_checkpoint/ckpt_2_targets_20260927/paysim_priority20_ckpt_2_targets.pt` | `2767ec2424a0740450dc52a376a3f88c89d1259772c4d06d0988e1050be9af8b` | `12f1145696e818375436cb63aa1df5135586e98fa2ff57b5abfdbbba237bade7` |
| `ckpt_3` | `artifacts/priority20_multi_checkpoint/ckpt_3_targets_20260927/paysim_priority20_ckpt_3_targets.pt` | `b0a99591447ad25d1c06333e5473c1d07cb4e60e4baff524eaeb13f29c90cf89` | `54c71e5db00a76fd9bbcb5e5edbe045468d0169041f81d50c28fb437b9ead049` |
| `ckpt_4` | `artifacts/priority20_multi_checkpoint/ckpt_4_targets_20260927/paysim_priority20_ckpt_4_targets.pt` | `4f11d3f0e2e072e5e7dd2adf7966c9c7685865a7fb5d8d3e2392bc9685127dcd` | `f30f7308912086d0d11eefa3bce5b3128b44ce62ccc45ae66df1a386ea30dc88` |

Dataset SHA-256 in all target provenance files:

`16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`

## Per-checkpoint results

Definition:

```text
D = MSE(attacker, DNA-defended update) - MSE(attacker, DP-defended update)
```

DP wins when `D < 0`; DNA wins when `D > 0`; strict ties require `D == 0`.

| Checkpoint | Targets | DP wins | DNA wins | Ties | Mean D | SD(D) | Median D | DP won every target? | DP strict majority? |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `ckpt_0` | 8 | 8 | 0 | 0 | -0.0010120696780451157 | 0.000006460139836418253 | -0.0010105526943022516 | yes | yes |
| `ckpt_1` | 8 | 8 | 0 | 0 | -0.0010123102164284034 | 0.000006506696792026241 | -0.0010109707487387719 | yes | yes |
| `ckpt_2` | 8 | 8 | 0 | 0 | -0.0010119900820638487 | 0.000006799540297524608 | -0.0010112557650195598 | yes | yes |
| `ckpt_3` | 8 | 8 | 0 | 0 | -0.0010126253056152476 | 0.000006598636396146169 | -0.0010116989369094568 | yes | yes |
| `ckpt_4` | 8 | 8 | 0 | 0 | -0.0010122728877414592 | 0.000006375953661921601 | -0.0010113459329343872 | yes | yes |

Per-checkpoint report hashes:

| Checkpoint | Report path | SHA-256 |
|---|---|---|
| `ckpt_0` | `artifacts/priority20_multi_checkpoint/ckpt_0_run_20260927/priority20_probe_report.json` | `2c6063ff388d04ffda17cf9520d5c7113f7f8298d5bd5b7955ba4ddbb611cc69` |
| `ckpt_1` | `artifacts/priority20_multi_checkpoint/ckpt_1_run_20260927/priority20_probe_report.json` | `5c541c424908d951f73802475096277d3deb66961278649ff8922b787b85ec5a` |
| `ckpt_2` | `artifacts/priority20_multi_checkpoint/ckpt_2_run_20260927/priority20_probe_report.json` | `5735366eb5cfdc8172b471f574d18fe604533b884a8976db89c774740d8a9fd9` |
| `ckpt_3` | `artifacts/priority20_multi_checkpoint/ckpt_3_run_20260927/priority20_probe_report.json` | `96b8b5326ab1cbbfe1a32537ad90d4066d6f4e4b06c83f84cf8a76da61412000` |
| `ckpt_4` | `artifacts/priority20_multi_checkpoint/ckpt_4_run_20260927/priority20_probe_report.json` | `e2c8c1905d3c1a6ea38fb7a16d1f28553b16f44241c732cb27f915ca68faa72f` |

## Cross-checkpoint analysis

Frozen checkpoint-level analyses:

| Analysis | Result |
|---|---:|
| Checkpoints tested | 5 |
| Checkpoints where DP wins strict target majority | 5/5 |
| One-sided exact p for DP-majority checkpoints | 0.03125 |
| Checkpoints where mean D < 0 | 5/5 |
| One-sided exact p for negative-mean-D checkpoints | 0.03125 |
| DP won every target in every checkpoint | true |

Aggregate targetwise descriptive result:

| DP wins | DNA wins | Ties | Non-tied n | Targetwise p if targets were independent |
|---:|---:|---:|---:|---:|
| 40 | 0 | 0 | 40 | 9.094947017729282e-13 |

The targetwise p-value is reported as descriptive only. The purpose of Priority
20 is the checkpoint-level evidence: all five independently initialized
checkpoints reproduce the same DP-wins-majority and negative-mean-D pattern.

Aggregate analysis artifact:

- `artifacts/priority20_multi_checkpoint/priority20_aggregate_analysis_20260927.json`
- SHA-256: `1a8d8722b42a170e8c3dc0669907f1cc97cc1c0cb7b5fde7f3f889e03779f5e3`

## Honest verdict

The DP-wins pattern **does replicate across independently initialized
checkpoints** for the v1-medium vs DP `0.000315` cell:

- DP won every target in every checkpoint: `40/40` targetwise, across `5/5`
  checkpoints.
- Every checkpoint independently had DP win `8/8`.
- Every checkpoint had mean `D < 0`.
- The checkpoint-level one-sided exact p-value is `0.03125` for both the
  strict-majority and negative-mean-D analyses.

This directly addresses the strongest version of the Priority 18 effective-n
concern for this cell. The original single-checkpoint `39/39` result could be
criticized as one run-level realization; Priority 20 adds five independently
initialized/warmup-trained checkpoint realizations, and the sign pattern is
consistent across all five. The evidence no longer rests solely on within-run
target count for this comparator cell.

This result should still be described precisely: it is a five-checkpoint
replication for **one** comparator cell, v1-medium vs DP `0.000315`, not an
all-cell replication of every DP-vs-transform row.

## Final checks

- Pre-registration amendment written before Priority 20 execution: PASS.
- Fresh source-disjoint targets for all five checkpoints: PASS.
- All five pre-registered checkpoints reported; no early stopping: PASS.
- No checkpoint added/dropped after partial results: PASS.
- `torch.set_num_threads(1)` preserved in the runner.
- No files under `Latex/` edited.
