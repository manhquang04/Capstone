# Priority 5.3 IEEE-CIS RQ1 development gate report

**Date:** 2026-09-16  
**Status:** COMPLETE — raw attacker gate FAILED at the smallest IEEE-CIS scope

## Đã làm gì

Ran the frozen Priority 5.3 IEEE-CIS development gate at the smallest scope:
four records per group, one fraud record per group, and one Adam local step.
The run used the transaction-only IEEE-CIS feature contract from Priority
5.1/5.2, with 29 numeric columns and 11 categorical columns expanded to a
177-dimensional transformed feature vector.

The attacker was implemented as a new IEEE-CIS-specific label-free development
runner. It optimizes scaled numeric features, categorical softmax variables,
and soft labels jointly. It does not use true labels as an attacker oracle and
does not use PaySim balance constraints. True labels are used only to create
the one-fraud target groups and to report class-wise MSE after the gate.

The first run at
`artifacts/priority5_ieee_cis/phase53_rq1_development_gate_20260916/` stopped
with a PyTorch autograd graph-reuse error before producing any gate result.
This was a technical pre-result failure. The runner was fixed by using
`torch.autograd.grad(...)` for optimizer gradients, matching the existing
Phase 3/4 runner pattern, and was rerun into a separate artifact directory.

## Lệnh thực sự đã chạy

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_ieee_cis_rq1_development_gate.py \
  --output artifacts/priority5_ieee_cis/phase53_rq1_development_gate_20260916

python3 -m py_compile experiments/run_ieee_cis_rq1_development_gate.py

PYTHONPATH=. .venv-phase1/bin/python experiments/run_ieee_cis_rq1_development_gate.py \
  --output artifacts/priority5_ieee_cis/phase53_rq1_development_gate_20260916_rerun1
```

## Kết quả kèm uncertainty

The valid rerun completed 8 development groups and 4 restarts per group
(32 group/restart jobs). The exact one-sided sign-test gate failed against
both controls.

| Control | Wins / non-ties | Mean Δ raw-control | Median Δ raw-control | One-sided sign p | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Prior | 4 / 8 | -0.05329 | -0.02094 | 0.63672 | FAIL |
| Zero-update | 6 / 8 | -0.14217 | -0.02752 | 0.14453 | FAIL |

Negative deltas mean the raw attacker had lower feature-MSE than the control,
but the win counts were not strong enough to pass the predeclared gate.

Source-disjointness checks inside the protocol lock:

| Check | Result |
| --- | --- |
| Target rows unique | PASS |
| Warmup-target overlap | 0 rows |

## Gate đạt/chưa đạt

| Gate | Status | Reason |
| --- | --- | --- |
| Development target creation | PASS | 8 groups created; 4 records/group; 1 fraud/group |
| Runner completion | PASS | valid rerun completed 32/32 group/restart jobs |
| Label-free attacker contract | PASS | optimized soft labels; true labels evaluation-only |
| Raw vs Prior gate | FAIL | 4/8 wins, p=0.63672 |
| Raw vs Zero-update gate | FAIL | 6/8 wins, p=0.14453 |
| Eight-record screen | NOT AUTHORIZED | amendment says stop if 4-record raw gate fails |
| DP/DNA calibration or confirmatory | NOT AUTHORIZED | no IEEE-CIS raw attacker validation at smallest scope |

## Artifact/run ID

- Amendment: `protocols/amendments/2026-09-16_priority5_ieee_cis_rq1_development_gate.md`
- Runner: `experiments/run_ieee_cis_rq1_development_gate.py`
- Excluded technical-failure run: `artifacts/priority5_ieee_cis/phase53_rq1_development_gate_20260916/`
- Valid run: `artifacts/priority5_ieee_cis/phase53_rq1_development_gate_20260916_rerun1/`
- Report JSON: `artifacts/priority5_ieee_cis/phase53_rq1_development_gate_20260916_rerun1/baseline_gate.json`

Hashes recorded after the valid rerun:

- `experiments/run_ieee_cis_rq1_development_gate.py`: `988410405dcd77649ed19b23b17022892e1e95497e8d077fa5b6ea85393e1340`
- `protocol_lock.json`: `e1c8db9423410bd6ac4fc2c9c3f5207ebae134f61a8a6ad60c3b28217b74a408`
- `development_targets.pt`: `fdff672ab423f4f760191c627ad6e42b1998dfd0426ca6463983d31c11394e84`
- `baseline_gate.json`: `64167ab4f8fbb388ee3c948bf2870c76773ec0528c348665747d76e1d1abe97b`

## Conclusion and next permitted step

The present IEEE-CIS label-free raw attacker did not validate even at the
smallest 4-record/1-fraud/1-step scope. Per the frozen amendment, Priority 5.3
stops here: no 8-record screen, DP distortion calibration, DNA-vs-DP run, or
confirmatory IEEE-CIS target may be created from this protocol.

This is not evidence that IEEE-CIS is secure against all inversion attacks. It
means the currently implemented, protocol-compliant attacker was not strong
enough to pass its own development controls on IEEE-CIS.

The result also narrows the earlier cross-dataset hypothesis. IEEE-CIS has a
less extreme fraud rate than PaySim, but the raw attacker still failed at the
smallest tested scope. That weakens a simple explanation that PaySim failures
were driven only by class imbalance. A more consistent interpretation is that
the PaySim-specific data structure, especially the balance-difference
constraints exploited by earlier PaySim parameterizations, may be a major part
of why the attacker could validate there. IEEE-CIS lacks an analogous hard
balance-conservation constraint, so the same reconstruction approach loses an
important source of structure even though the class imbalance is milder.
