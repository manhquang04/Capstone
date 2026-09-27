# Priority 3 — Formal DP accounting diagnostic

**Date:** 2026-09-16  
**Status:** `COMPLETE — EPSILON FINITE BUT PRACTICALLY MEANINGLESS`  
**Scope:** accounting-only analysis for existing clipping + Gaussian-noise
settings.  No DP mechanism or previous experiment was changed.

## 1. Đã làm gì

1. Read the implemented DP-style mechanism in `privacy/dp_engine.py`.
2. Verified that clipping is applied to the full client model update, not to
   per-example gradients.
3. Wrote a pre-run accounting amendment defining the exact assumptions and RDP
   conversion rule.
4. Implemented a versioned RDP accountant for the Gaussian mechanism.
5. Ran the accountant for the existing distortion-matched and utility-matched
   noise multipliers.

## 2. File/config thay đổi

- `protocols/amendments/2026-09-16_priority3_dp_accounting.md`
- `experiments/dp_update_accounting.py`
- `artifacts/dp_accounting/priority3_20260916_v2/dp_update_accounting.json`
- `artifacts/dp_accounting/priority3_20260916_v2/dp_update_accounting.csv`

The first accountant output directory,
`artifacts/dp_accounting/priority3_20260916/`, is superseded by `_v2` because
the alpha grid was refined near `alpha=1` before writing this report.  This was
a numerical-resolution correction only; no DP configuration was changed.

## 3. Lệnh thực sự đã chạy

```bash
python3 -m py_compile experiments/dp_update_accounting.py

PYTHONPATH=. .venv-phase1/bin/python \
  experiments/dp_update_accounting.py \
  --output-dir artifacts/dp_accounting/priority3_20260916

PYTHONPATH=. .venv-phase1/bin/python \
  experiments/dp_update_accounting.py \
  --output-dir artifacts/dp_accounting/priority3_20260916_v2
```

## 4. Accounting assumptions

The implemented mechanism is:

```text
full client update -> L2 clip at C -> add Gaussian noise N(0, (sigma*C)^2 I)
```

Therefore the accountant is only a client/update-level Gaussian-mechanism
diagnostic.  It is not record-level DP for individual PaySim transactions,
because the implementation does not perform per-example gradient clipping.

The report includes two sensitivity conventions:

- `add_remove`: neighboring update differs by at most `C`;
- `replace_one`: neighboring update differs by at most `2C`.

Primary displayed values below use `delta = 1e-5` and the more optimistic
`add_remove` convention.  The full CSV also includes `delta = 1e-6`, `1e-8`,
and the stricter `replace_one` convention.

## 5. Kết quả kèm uncertainty

This is deterministic accounting, so there is no sampling uncertainty.  The
dominant uncertainty is semantic: these are update-level assumptions, not
record-level DP guarantees.

| Config | Noise multiplier | Noise std (`C=100`) | Scenario | ε at δ=1e-5, add/remove |
| --- | ---: | ---: | --- | ---: |
| RQ1/RQ2 conservative distortion-matched | 0.00025 | 0.025 | one release | 8.019e6 |
| RQ1/RQ2 conservative distortion-matched | 0.00025 | 0.025 | 50 releases / same client | 4.001e8 |
| RQ1 medium distortion-matched | 0.000315 | 0.0315 | one release | 5.054e6 |
| RQ1 medium distortion-matched | 0.000315 | 0.0315 | 50 releases / same client | 2.521e8 |
| RQ1 stronger distortion-matched | 0.0004 | 0.04 | one release | 3.137e6 |
| RQ1 stronger distortion-matched | 0.0004 | 0.04 | 50 releases / same client | 1.563e8 |
| RQ2 development utility-matched extension | 0.00001 | 0.001 | one release | 5.000e9 |
| RQ2 development utility-matched extension | 0.00001 | 0.001 | 50 releases / same client | 2.500e11 |
| Historical default `medium` preset | 0.005 | 0.5 | one release | 2.098e4 |
| Historical default `medium` preset | 0.005 | 0.5 | 50 releases / same client | 1.007e6 |

Under the stricter `replace_one` convention, epsilon is approximately four
times larger for the same noise multiplier and composition count.

## 6. Gate đạt/chưa đạt

| Gate | Status | Reason |
| --- | --- | --- |
| Accountant implemented | PASS | RDP-to-DP script runs and emits JSON/CSV |
| Existing configs preserved | PASS | No noise multiplier, clip norm, or experiment output changed |
| Finite epsilon computed | PASS | All listed configurations produce finite epsilon |
| Meaningful formal DP baseline | FAIL | Epsilon values are extremely large |
| Record-level DP claim | FAIL | Whole-update clipping is not per-example clipping |

## 7. Interpretation

The existing comparator should not be upgraded to a strong formal DP baseline.
The most accurate wording is:

```text
DP-style full-client-update clipping/noise with weak update-level accounting
under stated Gaussian-mechanism assumptions.
```

For RQ1, this means the DNA Transform was compared against a distortion-matched
noise baseline, but that baseline has extremely weak formal DP privacy at the
noise levels required to preserve comparable utility or relative-L2 distortion.

For RQ2, the utility-matched extension at multiplier `0.00001` is essentially
non-private under this accountant.  It remains useful only as an engineering
utility-matching point, not as a meaningful privacy comparator.

## 8. Artifact/run ID

- Amendment:
  `protocols/amendments/2026-09-16_priority3_dp_accounting.md`
- Accountant:
  `experiments/dp_update_accounting.py`
- Final accounting artifacts:
  `artifacts/dp_accounting/priority3_20260916_v2/`

Important hashes:

| Artifact | SHA-256 |
| --- | --- |
| accountant script | `031ed33296777040704c2b4bede786d991140af9339f704a0e49b54a43967215` |
| amendment | `398c9b296796c0a23fcd8b91d52d38ce4d65864812fe239ff2d2029363076f96` |
| accounting JSON | `bddc29d248541153991447fc61000d2181f7dc3fde5106e39d955d5b0cbd7343` |
| accounting CSV | `d44efca06996e38325ba0438d8d9580d5562fb25bfc502453cfb9ef31f6cac3c` |

## 9. Bước tiếp theo được phép

Priority 3 is complete.  The next supervisor-ordered item is Priority 2: a new,
clean RQ1 protocol with larger power for a smaller practical effect and a
development-only Pareto sweep before selecting 1-2 confirmatory points.
