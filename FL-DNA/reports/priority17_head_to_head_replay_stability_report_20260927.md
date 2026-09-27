# Priority 17 — DP-vs-Transform head-to-head replay-stability check

**Date:** 2026-09-27
**Status:** COMPLETE
**Scope:** determinism replay of the existing confirmatory DP-vs-DNA head-to-head harness. No new targets were generated, and no files under `Latex/` were touched.

## Purpose

Round-6 review flagged that the head-to-head table reports small mean
differences:

- Transform v1 stronger: mean `D = MSE_DNA - MSE_DP` about `-0.0016`;
- Transform v1 medium: mean `D` about `-0.0010`;
- Transform v2 ratio0.95/eta0.01: mean `D` about `-0.0111`.

Earlier replay-discrepancy tie bands (`0.00390625` for v1-family and
`0.001953125` for v2-family) came from a different attacker-vs-control replay
pipeline, not from the DP-vs-DNA head-to-head harness. This Priority 17 check
directly replays the head-to-head harness itself using the exact frozen
confirmatory targets and the exact original CLI arguments.

## Replayed cells

| Cell | Original report JSON | Replay report JSON |
| --- | --- | --- |
| v1 stronger vs DP 0.0004 | `artifacts/priority14_fresh_probe/v1_stronger_confirmatory_20260921/priority14_probe_report.json` | `artifacts/priority17_replay_stability/v1_stronger_replay_20260927/priority14_probe_report.json` |
| v1 medium vs DP 0.000315 | `artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_run_20260927/priority14_probe_report.json` | `artifacts/priority17_replay_stability/v1_medium_replay_20260927/priority14_probe_report.json` |
| v2 ratio0.95/eta0.01 vs DP 0.00105 | `artifacts/priority14_fresh_probe/v2_ratio0p95_confirmatory_20260921/priority14_probe_report.json` | `artifacts/priority17_replay_stability/v2_ratio0p95_replay_20260927/priority14_probe_report.json` |

## Commands run

### v1 stronger replay

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority14_fresh_dna_vs_dp_probe.py \
  --cell v1_stronger_vs_dp_0p0004 \
  --target artifacts/priority14_fresh_probe/v1_stronger_confirmatory_targets_20260921/paysim_priority14_v1_stronger_confirmatory_targets.pt \
  --output artifacts/priority17_replay_stability/v1_stronger_replay_20260927 \
  --seed 2026092128 --groups 39 --restarts 4 --workers 8 \
  --amendment protocols/amendments/2026-09-21_priority14_confirmatory_escalation_draft.md
```

### v1 medium replay

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority16_v1_medium_dna_vs_dp_probe.py \
  --cell v1_medium_vs_dp_0p000315 \
  --target artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_targets_20260927/paysim_priority16_v1_medium_confirmatory_targets.pt \
  --output artifacts/priority17_replay_stability/v1_medium_replay_20260927 \
  --seed 2026092706 --groups 39 --restarts 4 --workers 8 \
  --amendment protocols/amendments/2026-09-27_v1_medium_priority14_style_dp_head_to_head.md
```

### v2 replay

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority14_fresh_dna_vs_dp_probe.py \
  --cell v2_ratio0p95_eta0p01_vs_dp_0p00105 \
  --target artifacts/priority14_fresh_probe/v2_ratio0p95_confirmatory_targets_20260921/paysim_priority14_v2_ratio0p95_confirmatory_targets.pt \
  --output artifacts/priority17_replay_stability/v2_ratio0p95_replay_20260927 \
  --seed 2026092129 --groups 39 --restarts 4 --workers 8 \
  --amendment protocols/amendments/2026-09-21_priority14_confirmatory_escalation_draft.md
```

## Summary results

Definition:

```text
D_i = MSE_DNA_i - MSE_DP_i
```

The replay discrepancy is computed as:

```text
|D_i(replay) - D_i(original)|
```

| Cell | Original mean D | Original mean \|D\| | Max replay \|ΔD\| | Mean replay \|ΔD\| | Signs reproduced? | MSE bit-exact? | Discrepancy vs mean \|D\| |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| v1 stronger vs DP 0.0004 | -0.0016166626669785634 | 0.0016166626669785634 | 0.0 | 0.0 | Yes, 39/39 DP wins reproduced | Yes | smaller |
| v1 medium vs DP 0.000315 | -0.0010026436558672064 | 0.0010026436558672064 | 0.0 | 0.0 | Yes, 39/39 DP wins reproduced | Yes | smaller |
| v2 ratio0.95/eta0.01 vs DP 0.00105 | -0.01114200033968249 | 0.01114200033968249 | 0.0 | 0.0 | Yes, 39/39 DP wins reproduced | Yes | smaller |

## Zero-threshold sign-test reproduction

| Cell | Original DNA wins | Original DP wins | Original ties | Replay DNA wins | Replay DP wins | Replay ties | Replay DP p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 stronger vs DP 0.0004 | 0 | 39 | 0 | 0 | 39 | 0 | 1.8189894035458565e-12 |
| v1 medium vs DP 0.000315 | 0 | 39 | 0 | 0 | 39 | 0 | 1.8189894035458565e-12 |
| v2 ratio0.95/eta0.01 vs DP 0.00105 | 0 | 39 | 0 | 0 | 39 | 0 | 1.8189894035458565e-12 |

## Per-target comparison CSVs

Each CSV lists per-target original and replay values for `MSE_DNA_i`,
`MSE_DP_i`, `D_i`, winner, and replay-minus-original deltas.

| Cell | CSV |
| --- | --- |
| v1 stronger vs DP 0.0004 | `artifacts/priority17_replay_stability/comparisons_20260927/v1_stronger_replay_comparison.csv` |
| v1 medium vs DP 0.000315 | `artifacts/priority17_replay_stability/comparisons_20260927/v1_medium_replay_comparison.csv` |
| v2 ratio0.95/eta0.01 vs DP 0.00105 | `artifacts/priority17_replay_stability/comparisons_20260927/v2_ratio0p95_replay_comparison.csv` |

## Interpretation

All three cells replayed bit-for-bit for the stored MSE values and D values:

- `max_i |D_i(replay) - D_i(original)| = 0.0` for all three cells;
- `mean_i |D_i(replay) - D_i(original)| = 0.0` for all three cells;
- the 39/39 DP-win sign pattern reproduced exactly for all three cells.

Therefore, the DP-vs-DNA head-to-head harness's own replay-discrepancy is
smaller than each cell's reported mean `|D|`; in this replay it is exactly
zero. The small v1 mean `|D|` values are not due to replay nondeterminism in
this harness under fixed targets, fixed seeds, fixed restarts, and fixed
worker count.

## Artifact hashes

| Artifact | SHA-256 |
| --- | --- |
| `artifacts/priority17_replay_stability/v1_stronger_replay_20260927/priority14_probe_report.json` | `b94e65b7f8652820f78ffb38a0e7660a650cdf6025202017887ee07a0ef6874c` |
| `artifacts/priority17_replay_stability/v1_medium_replay_20260927/priority14_probe_report.json` | `faaa8eb8ef9846092a838987999b60ec38227618492354f33ac87c79dd32cd67` |
| `artifacts/priority17_replay_stability/v2_ratio0p95_replay_20260927/priority14_probe_report.json` | `7b214356cdb55fc9165862e0cf8e4497f998a04b25d5a5da9499961801b191fd` |
| `artifacts/priority17_replay_stability/comparisons_20260927/v1_stronger_replay_comparison.csv` | `c89a5c99c6ab8a6d2f998c1b67d4f033a1f0a18f51145c14ae23a9b20dad5ae3` |
| `artifacts/priority17_replay_stability/comparisons_20260927/v1_medium_replay_comparison.csv` | `09d3bafe7a3b5e303610d6e855222383a24722ba5418e8b3b85cf9e6f0d55a5a` |
| `artifacts/priority17_replay_stability/comparisons_20260927/v2_ratio0p95_replay_comparison.csv` | `1f311a74a18be18349b4bdaf764d4c8a552f640740c368bdff477f4be209bdb1` |
| `artifacts/priority17_replay_stability/comparisons_20260927/replay_stability_summary.json` | `80c9013320b5e2d6f9d94bc624d31ee5a9954e33e3218a2be8e539be060914ec` |
| `experiments/priority14_fresh_dna_vs_dp_probe.py` | `b07ef173308d211833185ceed762cb40602a4c5c8a4f89efc6a0359067cd23d1` |
| `experiments/priority16_v1_medium_dna_vs_dp_probe.py` | `1a12aecd274c6fc9b2fca86df58405cca202e7e6b62dcb96e6362a257789c152` |

## Final check

No new target artifacts were generated. No protocol amendment was created. No
file under `Latex/` was edited.
