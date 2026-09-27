# Priority 8 v2 ratio0.90 parameter-sensitivity report

**Date:** 2026-09-17  
**Scope:** IEEE-CIS development-gate parameter-sensitivity check only.  
**Status:** STOPPED AT n=8 because the development gate did not pass.

## Purpose

`GEN_COSINE_TV` has confirmed attacks against DNA Transform v2 at
`compression_ratio=0.95, quantization_eta=0.01` on IEEE-CIS and CIFAR-10.
This report tests whether the same attacker also works at an existing v2
configuration:

- `compression_ratio=0.90`;
- `quantization_eta=0.01`.

This is not a confirmatory run and does not change any frozen RQ1/RQ2/RQ3
conclusion.

## Pre-run amendment

Before target generation or execution, the following amendment was written:

`protocols/amendments/2026-09-17_priority8_v2_ratio0p90_sensitivity.md`

The amendment authorized:

- IEEE-CIS first;
- scope 4 rows/group, 1 fraud row/group, 1 client step;
- n=8 development gate;
- defense `v2_ratio0p9_eta0p01`;
- attacker `GEN_COSINE_TV`;
- Prior and Zero-update controls;
- exact one-sided sign-test gate.

No CIFAR-10 run and no confirmatory run were authorized.

## Implementation note

`experiments/run_priority6_sota_style_attackers.py` was extended to accept a new
defense label:

- `v2_ratio0p9_eta0p01`.

The existing `v2_ratio0p95_eta0p01` behavior remains unchanged. The patch only
routes the correct `compression_ratio`/`quantization_eta` to the existing v2
sketch-space attacker machinery.

## Target generation

Command:

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority6_sota_style_attackers.py \
  prepare-ieee \
  --output artifacts/priority8_v2_ratio0p90/ieee_n8_targets_20260917 \
  --seed 2026091801 \
  --groups 8 \
  --max-rows 50000 \
  --amendment protocols/amendments/2026-09-17_priority8_v2_ratio0p90_sensitivity.md
```

Data firewall:

- historical IEEE target files checked: `4`;
- maximum overlap: `0`;
- warmup/target overlap: `0`;
- disjointness gate: `PASS`.

Historical target files checked:

- `artifacts/priority5_ieee_cis/phase53_rq1_development_gate_20260916/development_targets.pt`;
- `artifacts/priority5_ieee_cis/phase53_rq1_development_gate_20260916_rerun1/development_targets.pt`;
- `artifacts/priority6_sota_attackers/targets_ieee/ieee_priority6_bundle.pt`;
- `artifacts/priority6_sota_attackers/targets_ieee_n24/ieee_priority6_bundle.pt`.

Target bundle SHA-256:

`02b8cd828833662792c47f1f02fa20814e0a0f22f44ba9cbd8322e1aa4eb2871`

## Gate command

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_priority6_sota_style_attackers.py \
  execute \
  --dataset ieee \
  --bundle artifacts/priority8_v2_ratio0p90/ieee_n8_targets_20260917/ieee_priority6_bundle.pt \
  --output artifacts/priority8_v2_ratio0p90/ieee_n8_gate_20260917 \
  --seed 2026091802 \
  --workers 8 \
  --groups 8 \
  --defenses v2_ratio0p9_eta0p01 \
  --generations GEN_COSINE_TV \
  --amendment protocols/amendments/2026-09-17_priority8_v2_ratio0p90_sensitivity.md
```

Completed jobs: `32/32`.

## Results

| Control | Wins/non-ties | Mean ΔMSE | Median ΔMSE | One-sided exact sign-test p | Gate |
|---|---:|---:|---:|---:|---|
| Prior | 6/8 | -10.7867475326 | -1.1232427303 | 0.14453125 | FAIL |
| Zero-update | 6/8 | -10.9828274341 | -1.0643864858 | 0.14453125 | FAIL |

Overall development gate: **FAIL**.

## Interpretation

The direction of effect is favorable on average, but the exact sign-test gate
does not pass at n=8. Per the pre-run amendment, escalation stops here:

- no n=24 pilot;
- no confirmatory amendment;
- no CIFAR-10 repeat in this turn.

This is useful negative development evidence. It suggests the confirmed v2
attackability observed at `compression_ratio=0.95, eta=0.01` may not
automatically generalize across the whole v2 parameter space. In particular,
the v2 vulnerability appears potentially sensitive to the compression ratio
and/or the resulting sketch dimension. This is not evidence that
`ratio=0.90, eta=0.01` is safe; it only says this attacker/gate did not meet
the pre-registered development threshold at n=8.

## Artifact hashes

| File | SHA-256 |
|---|---|
| `protocols/amendments/2026-09-17_priority8_v2_ratio0p90_sensitivity.md` | `84861cf04693fc7c9b0ecefa34247a94b7ac3d9e5e6b706b108b66975019633c` |
| `experiments/run_priority6_sota_style_attackers.py` | `bbcfe326bbf49c74888b18049b9154e9e10ff66f461b03cc0966c425b5169eb2` |
| `ieee_n8_targets_20260917/ieee_priority6_bundle.pt` | `02b8cd828833662792c47f1f02fa20814e0a0f22f44ba9cbd8322e1aa4eb2871` |
| `ieee_n8_targets_20260917/ieee_target_firewall.json` | `203245e8621e6782fb38d48ed8a17dc370d73457a8bd6ff5d5f0981005eb414a` |
| `ieee_n8_gate_20260917/priority6_gate_report.json` | `01b648290acf4314b02c13c7cc2001149848a187555014f7c3a6114c805db311` |

## Checks

- `torch.set_num_threads(1)`: preserved in worker.
- Source-disjoint target set: PASS.
- Development n=8 gate: completed.
- n=24 escalation: not run.
- Confirmatory: not run.
- CIFAR-10 repeat: not run.

