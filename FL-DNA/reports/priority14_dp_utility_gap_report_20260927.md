# Priority 14 DP utility/accounting gap report

**Date:** 2026-09-27
**Status:** COMPLETE
**Scope:** fill the missing accounting + RQ2-style utility evidence for the exact DP comparators used in Priority 14 Table-II-style DNA-vs-DP head-to-head tests.  No paper files were edited.

## Task 1 — Exact Priority 14 DP configuration match

Priority 14 used these exact DP-style clipping/noise comparators:

| Priority 14 cell | Clip norm | Noise multiplier | Evidence |
| --- | ---: | ---: | --- |
| v1 stronger vs DP | 100.0 | 0.0004 | `experiments/priority14_fresh_dna_vs_dp_probe.py`, `CELLS["v1_stronger_vs_dp_0p0004"]`; `reports/priority14_confirmatory_final_20260921.md` |
| v2 ratio0.95/eta0.01 vs DP | 100.0 | 0.00105 | `experiments/priority14_fresh_dna_vs_dp_probe.py`, `CELLS["v2_ratio0p95_eta0p01_vs_dp_0p00105"]`; `reports/priority14_confirmatory_final_20260921.md` |

Comparison against Priority 3:

| Priority 14 comparator | Priority 3 coverage | Match status |
| --- | --- | --- |
| v1 stronger DP 0.0004 | `RQ1 stronger distortion-matched`, clip 100.0, multiplier 0.0004 | IDENTICAL |
| v2 ratio0.95 DP 0.00105 | Not listed in `reports/priority3_dp_accounting_report.md` by name | NOT COVERED BY PRIORITY 3 |

Note: an older v2 accounting artifact also exists at
`artifacts/rq1_v2/confirmatory_freeze_20260916/dp_accounting_v2_confirmatory.json`
and matches the recomputed value below, but it was not part of the Priority 3
report cited in the paper-review gap.

## Task 2 — Missing epsilon values

The v1 stronger DP comparator is already covered by Priority 3 and was not
recomputed:

| Comparator | Scenario | δ | Adjacency | ε |
| --- | --- | ---: | --- | ---: |
| v1 stronger DP 0.0004 | one release | 1e-5 | add/remove | 3,136,996.3153463383 |
| v1 stronger DP 0.0004 | 50 releases / same client | 1e-5 | add/remove | 156,334,827.96468508 |

The v2 ratio0.95/eta0.01 DP comparator was computed with the same RDP
accountant and convention as Priority 3:

| Comparator | Scenario | δ | Adjacency | ε |
| --- | --- | ---: | --- | ---: |
| v2 ratio0.95 DP 0.00105 | one release | 1e-5 | add/remove | 458,084.7641228683 |
| v2 ratio0.95 DP 0.00105 | 50 releases / same client | 1e-5 | add/remove | 22,708,052.122672714 |

These are update-level Gaussian-mechanism diagnostics only.  They are not
record-level DP guarantees because the implementation clips full client/model
updates, not per-example gradients.

Artifacts:

| Artifact | SHA-256 |
| --- | --- |
| `artifacts/dp_accounting/priority14_dp_utility_gap_20260927/dp_update_accounting.json` | `ce730b5694f02a584527a5f21d5e753dcd5324b9b11714c97901e12fdf99c8c2` |
| `artifacts/dp_accounting/priority14_dp_utility_gap_20260927/dp_update_accounting.csv` | `e6b166695da03f80dad75f35dac9cf023e91197d70be7eeeee18f4d254d9f891` |
| `experiments/dp_update_accounting.py` | `031ed33296777040704c2b4bede786d991140af9339f704a0e49b54a43967215` |

## Task 3 — Missing RQ2-style utility test

### Protocol

Wrote and froze:

- Amendment: `protocols/amendments/2026-09-27_priority14_dp_utility_gap_protocol.md`
- Config: `protocols/config/rq2_priority14_dp_utility_confirmatory.json`

Frozen utility rule:

- paired non-inferiority, DP comparator minus unprotected FL baseline;
- alpha `0.05`, 95% paired CI;
- F1 non-inferiority margin `0.02`;
- AUC-ROC non-inferiority margin `0.005`;
- both endpoints must pass.

The test reused the already frozen unprotected baseline from RQ2-v2
confirmatory:

`artifacts/rq2_v2/confirmatory_20260916/`

Rationale: the baseline run has the same 52 seeds, same training configuration,
and same paired seed contract.  Fresh DP jobs were run for both Priority 14 DP
comparators.

### Commands

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_rq2_priority14_dp_utility.py \
  --config protocols/config/rq2_priority14_dp_utility_confirmatory.json \
  --output-dir artifacts/rq2_priority14_dp_utility/confirmatory_20260927 \
  --workers 9

PYTHONPATH=. .venv-phase1/bin/python experiments/analyze_rq2_priority14_dp_utility.py \
  --config protocols/config/rq2_priority14_dp_utility_confirmatory.json \
  --run-dir artifacts/rq2_priority14_dp_utility/confirmatory_20260927 \
  --output-dir results/rq2_priority14_dp_utility/confirmatory_20260927
```

Execution summary: `104/104` DP jobs succeeded, `0` failed.

### Results

| DP comparator | Endpoint | n | Mean paired Δ(DP − Baseline) | SD | Median Δ | 95% CI | Margin | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| DP 0.0004 (v1 stronger comparator) | F1 | 52 | -0.127556 | 0.039056 | -0.124675 | [-0.138430, -0.116683] | -0.020000 | FAIL |
| DP 0.0004 (v1 stronger comparator) | AUC-ROC | 52 | -0.004227 | 0.002012 | -0.003993 | [-0.004787, -0.003667] | -0.005000 | PASS |
| DP 0.00105 (v2 comparator) | F1 | 52 | -0.316046 | 0.203003 | -0.228346 | [-0.372563, -0.259530] | -0.020000 | FAIL |
| DP 0.00105 (v2 comparator) | AUC-ROC | 52 | -0.140034 | 0.195210 | -0.060969 | [-0.194380, -0.085687] | -0.005000 | FAIL |

Primary utility conclusion under the frozen endpoint rule:

- `dp_v1_stronger_0p0004`: FAIL overall, because F1 non-inferiority fails.
- `dp_v2_ratio0p95_0p00105`: FAIL overall, because both F1 and AUC-ROC non-inferiority fail.

Artifacts:

| Artifact | SHA-256 |
| --- | --- |
| `protocols/amendments/2026-09-27_priority14_dp_utility_gap_protocol.md` | `3875bb32c7639318583d5f7d8b9ec30b23c85a86e4d097bfb387bb8774606c1b` |
| `protocols/config/rq2_priority14_dp_utility_confirmatory.json` | `689a651d8436f15e9caa0145641e9b5ccb74b37085aaa7c8d2ddd0fdb69ef966` |
| `experiments/run_rq2_priority14_dp_utility.py` | `74dffd8c9c3d82c408b87a5de5f99cb548994bddb4d3f73f7aaa7fb9476257ec` |
| `experiments/analyze_rq2_priority14_dp_utility.py` | `289bfe4bd4729ba83fbba4be13b433ff2d950535bc191da356acfa788ae3efd0` |
| `artifacts/rq2_priority14_dp_utility/confirmatory_20260927/execution_manifest.json` | `97b72a06a9eb6260e6f76d8eb8ab4596785fc34a5f8d67bc66bd8ae5b71b4b10` |
| `artifacts/rq2_priority14_dp_utility/confirmatory_20260927/execution_summary.json` | `944ae62d10ae8f5326bfb69a7981f6e782f2abd729e3b8811d11bfe024bd11c4` |
| `results/rq2_priority14_dp_utility/confirmatory_20260927/summary.json` | `6a2af80aa508e1db37b129208e7161e3854506beba27c42822377a7206c761fd` |
| `results/rq2_priority14_dp_utility/confirmatory_20260927/per_seed.csv` | `6be07c0274168f513aa9a186cb1f38dc5e782d40864d14c0a8a6890cfb64532f` |

## Task 4 — Strong-DP attack-gate ambiguity

`reports/strong_update_dp_work3_attack_gate_report.md` states that the raw
branch failed its Prior control (`6/8`, p=`0.1445`), so no valid attacker gate
was established.

The Work-3 attacker is not the later validated Priority 6-9
literature-standard attacker family (`GEN_IDLG_STYLE` / `GEN_COSINE_TV`).
The runner `experiments/run_strong_update_dp_attack.py` invokes:

- raw branch: `experiments/run_phase4_harddiff_reparam_for_misselected.py`;
- DP branches: `experiments/run_phase4_simple_defense_attack.py` with
  `--defense clipping_noise_mc`, `--restarts 8`, `--iterations 600`,
  `--attack-lr 0.1`, `--mc-noise-samples 100`, and fixed defense seed
  `314159265`.

This matches the Work-3 amendment language, which froze the "existing
clipping/noise Monte-Carlo attacker" for that exploratory study.  It is a
different, less-validated attacker path than the Priority 6-9 SOTA-style
attackers.

No rerun was attempted.  Swapping in `GEN_IDLG_STYLE` or `GEN_COSINE_TV` would
not be a clean technical replay, because it would change the attacker
generation/objective, not merely repair an implementation error while holding
the scientific design fixed.  The Work-3 conclusion should remain: strong-DP
attack resistance is unresolved under that failed raw gate; it should not be
worded as validated evidence that strong DP by itself prevents attack success.

Relevant hashes:

| Artifact | SHA-256 |
| --- | --- |
| `reports/strong_update_dp_work3_attack_gate_report.md` | `f7c87e9aa9ee6d8f093a072f30e4d159dd175862311d78951e10b31501b2e318` |
| `experiments/run_strong_update_dp_attack.py` | `b6978642c68b53ace139dac8b9f36e3c52d8142e6eba1a4fbf01ea475c79fa53` |
| `protocols/amendments/2026-09-16_strong_update_dp_exploratory_pareto.md` | `c004d9d1c45a00781bae16a917c04b75fd3d3e1228c0cb3924fb128b07568b53` |

## Final check

This report did not edit `main.tex` or any file under `Latex/`.  It adds only
the pre-run amendment/config/scripts plus the generated accounting and utility
artifacts described above.
