# Priority 14 — fresh clean-vector DNA-vs-DP probe

Date: 2026-09-21  
Status: n=8 DEVELOPMENT PROBE COMPLETE; NO ESCALATION RUN

## Scope

This probe tests whether the Priority 13 zero-threshold pattern (DP beating DNA
after BatchNorm-buffer exclusion) appears when the attacker is optimized directly
on the corrected clean vector from the start.

This is a cheap development probe only:

- no n=24 pilot was run;
- no confirmatory run was run;
- no old candidate was reused;
- every target set is new and source-disjoint from historical PaySim targets.

## Amendments

- Main amendment:
  `protocols/amendments/2026-09-21_priority14_fresh_dna_vs_dp_probe.md`
- Technical replay amendment:
  `protocols/amendments/2026-09-21_priority14_technical_replay.md`

The technical replay amendment records two implementation issues:

1. the first launch failed before scientific jobs due to variable shadowing;
2. a first successful run used a generic nonnegative penalty instead of the
   original PaySim `_nonnegative_penalty(latent, meta)` helper.

Only the final `rerun2` artifacts, using the original nonnegative penalty, are
reported as scientific probe results below. Earlier launch/generic-penalty
output directories are retained as technical artifacts and are not interpreted.

## Corrected vector definition

Both DNA and DP branches used only trainable parameters:

```text
model.named_parameters() where requires_grad=True
```

All buffers were excluded before constructing defended signals and before
computing matching loss:

```text
running_mean, running_var, num_batches_tracked, and any other named_buffers()
```

## Probe design

| Cell | DNA config | DP comparator | Attacker | n | Restarts |
|---|---|---|---|---:|---:|
| v1 stronger | mix=0.12, keep=0.82, shrink=0.35 | clipping/noise MC, noise_multiplier=0.0004 | simple balanced-tensor/raw-lift harddiff attacker | 8 | 4 |
| v2 ratio0.95/eta0.01 | compression_ratio=0.95, eta=0.01 | clipping/noise MC, noise_multiplier=0.00105 | simple balanced-tensor/raw-lift harddiff attacker in defended signal space | 8 | 4 |

Head-to-head statistic:

```text
D_i = MSE_DNA_i - MSE_DP_i
```

DNA win means `D_i > 0`. DP win means `D_i < 0`.

## Data firewall

| Cell | Target artifact | Provenance SHA-256 | max overlap | Dataset SHA-256 |
|---|---|---:|---:|---|
| v1 stronger | `artifacts/priority14_fresh_probe/v1_stronger_n8_targets_20260921/paysim_priority14_v1_stronger_n8_targets.pt` | `abb4acdbdcd8453a2bdc2a33f44c0bac3486254cd733f619db41b4d993477534` | 0 | `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b` |
| v2 ratio0.95 | `artifacts/priority14_fresh_probe/v2_ratio0p95_n8_targets_20260921/paysim_priority14_v2_ratio0p95_n8_targets.pt` | `7d672057ec9ad16e3875b3f13692466d98d14eea7993441441b71f964fc4f563` | 0 | `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b` |

## Commands run

Target generation:

```bash
mkdir -p FL-DNA/artifacts/priority14_fresh_probe/v1_stronger_n8_targets_20260921 \
  FL-DNA/artifacts/priority14_fresh_probe/v2_ratio0p95_n8_targets_20260921

PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python \
  FL-DNA/experiments/create_phase4_source_disjoint_targets.py \
  FL-DNA/artifacts/priority14_fresh_probe/v1_stronger_n8_targets_20260921 \
  --output-name paysim_priority14_v1_stronger_n8_targets.pt \
  --groups 8 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092118 \
  --purpose 'Priority 14 fresh clean-vector DNA-vs-DP probe v1_stronger n8'

PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python \
  FL-DNA/experiments/create_phase4_source_disjoint_targets.py \
  FL-DNA/artifacts/priority14_fresh_probe/v2_ratio0p95_n8_targets_20260921 \
  --output-name paysim_priority14_v2_ratio0p95_n8_targets.pt \
  --groups 8 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092119 \
  --purpose 'Priority 14 fresh clean-vector DNA-vs-DP probe v2_ratio0p95 n8'
```

Final valid probe runs:

```bash
PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python \
  FL-DNA/experiments/priority14_fresh_dna_vs_dp_probe.py \
  --cell v1_stronger_vs_dp_0p0004 \
  --target FL-DNA/artifacts/priority14_fresh_probe/v1_stronger_n8_targets_20260921/paysim_priority14_v1_stronger_n8_targets.pt \
  --output FL-DNA/artifacts/priority14_fresh_probe/v1_stronger_n8_probe_20260921_rerun2 \
  --seed 2026092120 --groups 8 --restarts 4 --workers 8

PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python \
  FL-DNA/experiments/priority14_fresh_dna_vs_dp_probe.py \
  --cell v2_ratio0p95_eta0p01_vs_dp_0p00105 \
  --target FL-DNA/artifacts/priority14_fresh_probe/v2_ratio0p95_n8_targets_20260921/paysim_priority14_v2_ratio0p95_n8_targets.pt \
  --output FL-DNA/artifacts/priority14_fresh_probe/v2_ratio0p95_n8_probe_20260921_rerun2 \
  --seed 2026092121 --groups 8 --restarts 4 --workers 8
```

## Results

### Summary

| Cell | DNA wins | DNA losses / DP wins | Exact ties | DP-advantage descriptive p | Mean D = MSE_DNA - MSE_DP | Median D |
|---|---:|---:|---:|---:|---:|---:|
| v1 stronger vs DP 0.0004 | 0 | 8 | 0 | 0.00390625 | -0.0016328180 | -0.0016302376 |
| v2 ratio0.95 vs DP 0.00105 | 0 | 8 | 0 | 0.00390625 | -0.0112495234 | -0.0112349482 |

The n=8 fresh clean-vector probe reproduces the Priority 13 direction in both
cells: the DP branch has larger clean-vector defended-space MSE than DNA for
every group. This remains development-only evidence.

### Generic-penalty vs corrected-penalty raw comparison

For transparency, the table below reports the raw scientific counts from the
generic-penalty run that was later superseded, next to the corrected
`_nonnegative_penalty(latent, meta)` run used as the valid result.

| Cell | Penalty implementation | Artifact | DNA wins | DNA losses / DP wins | Exact ties | DP-advantage descriptive p | Mean D | Median D |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v1 stronger vs DP 0.0004 | generic `relu(-latent)` penalty | `v1_stronger_n8_probe_20260921_rerun1` | 0 | 8 | 0 | 0.00390625 | -0.0016329897 | -0.0016307065 |
| v1 stronger vs DP 0.0004 | corrected `_nonnegative_penalty(latent, meta)` | `v1_stronger_n8_probe_20260921_rerun2` | 0 | 8 | 0 | 0.00390625 | -0.0016328180 | -0.0016302376 |
| v2 ratio0.95 vs DP 0.00105 | generic `relu(-latent)` penalty | `v2_ratio0p95_n8_probe_20260921` | 0 | 8 | 0 | 0.00390625 | -0.0112504043 | -0.0112347354 |
| v2 ratio0.95 vs DP 0.00105 | corrected `_nonnegative_penalty(latent, meta)` | `v2_ratio0p95_n8_probe_20260921_rerun2` | 0 | 8 | 0 | 0.00390625 | -0.0112495234 | -0.0112349482 |

### Per-group selected results

v1 stronger:

| Group | DNA restart | DP restart | DNA MSE | DP MSE | D | Winner |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 1.322200e-06 | 1.616435e-03 | -1.615113e-03 | DP |
| 1 | 2 | 3 | 1.069542e-06 | 1.630887e-03 | -1.629817e-03 | DP |
| 2 | 2 | 3 | 8.926008e-07 | 1.627539e-03 | -1.626647e-03 | DP |
| 3 | 3 | 2 | 1.007861e-06 | 1.629771e-03 | -1.628763e-03 | DP |
| 4 | 2 | 1 | 9.068566e-07 | 1.642051e-03 | -1.641144e-03 | DP |
| 5 | 1 | 0 | 1.199296e-06 | 1.644333e-03 | -1.643133e-03 | DP |
| 6 | 0 | 1 | 9.536853e-07 | 1.648223e-03 | -1.647269e-03 | DP |
| 7 | 3 | 0 | 1.118180e-06 | 1.631776e-03 | -1.630658e-03 | DP |

v2 ratio0.95:

| Group | DNA restart | DP restart | DNA MSE | DP MSE | D | Winner |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 | 2 | 1.238948e-06 | 1.112475e-02 | -1.112351e-02 | DP |
| 1 | 0 | 0 | 1.538355e-06 | 1.121999e-02 | -1.121845e-02 | DP |
| 2 | 2 | 3 | 1.432215e-06 | 1.120657e-02 | -1.120513e-02 | DP |
| 3 | 0 | 3 | 1.532795e-06 | 1.122731e-02 | -1.122578e-02 | DP |
| 4 | 2 | 2 | 1.170779e-06 | 1.130728e-02 | -1.130611e-02 | DP |
| 5 | 1 | 1 | 1.385611e-06 | 1.132575e-02 | -1.132437e-02 | DP |
| 6 | 2 | 1 | 1.305432e-06 | 1.135003e-02 | -1.134873e-02 | DP |
| 7 | 3 | 3 | 1.520899e-06 | 1.124564e-02 | -1.124412e-02 | DP |

## Artifact IDs

| Cell | Final valid report artifact | SHA-256 |
|---|---|---|
| v1 stronger | `artifacts/priority14_fresh_probe/v1_stronger_n8_probe_20260921_rerun2/priority14_probe_report.json` | `6a7759b44c6395dd3ad672a1c4a76108a54bd4e1b983a3e1be279242be4c7036` |
| v2 ratio0.95 | `artifacts/priority14_fresh_probe/v2_ratio0p95_n8_probe_20260921_rerun2/priority14_probe_report.json` | `d53f9e78672782970d7f27e69fb9ca86075b23e39608a88345881d3727cead97` |

Technical artifacts not interpreted:

- `artifacts/priority14_fresh_probe/v1_stronger_n8_probe_20260921`
  launch failed before results due to variable shadowing.
- `artifacts/priority14_fresh_probe/v1_stronger_n8_probe_20260921_rerun1`
  and `artifacts/priority14_fresh_probe/v2_ratio0p95_n8_probe_20260921`
  used the generic penalty and are superseded by `rerun2`.

## Gate / escalation status

The n=8 probe shows the same qualitative direction as Priority 13:

```text
DP wins all 8/8 groups for both v1 and v2.
```

However, this report does not run escalation. Priority 14's amendment explicitly
authorizes only the n=8 probe. Any n=24 pilot or confirmatory rerun requires a
separate supervisor decision/amendment.

## Checks

- No old target set was reused.
- No n=24 or confirmatory run was executed.
- `torch.set_num_threads(1)` was not changed.
- The final runner passed `py_compile`.
