# Priority32: multi-dataset RQ2 and RQ3

Status: IN PROGRESS. No confirmatory outcome available yet.

Goal/checklist: stages0–3 complete; Stage4 has a numerical failure while
already-submitted jobs continue; stages5–7 pending. Complete protocol:
protocols/amendments/2026-10-02_priority32_multidataset_rq2_rq3.md.
No earlier artifact/report is replaced, no Latex/external_defenses edits,
and no reconstruction attacks.

## Data audit

| Dataset | Raw rows | Raw fraud rate | Features | Selected train/validation/test | Split overlap |
| --- | --- | --- | --- | --- | --- |
| PaySim | 6362620 | See audit.json | 13 | 325000/75000/100000 | 0 |
| IEEE-CIS | 590540 | 0.03499000914417313 | 476 | 325000/75000/100000 | 0 |
| BAF | 1000000 | 0.011029 | 58 | 337833/59662/102505 | 0 |

Full hashes, train-only preprocessing parameters, source IDs and category
mappings: artifacts/priority32_multidataset/prepared/<dataset>/.
IEEE left join includes identity; temporal split;12 >90%-missing columns
dropped, documented in its audit. BAF exact -1 indicators are recorded;
other negative values (including intended_balcon_amount when not exactly -1)
are retained. All datasets capped at500k within their partitions.

## Commands and disclosure so far

```text
.venv-phase1/bin/python -B tests/test_priority32_multidataset.py
.venv-phase1/bin/python -B experiments/priority32_multidataset.py --prepare-only
```

Preparation executed once, completed without interruption; outputs subsequently
validated, not regenerated. Five synthetic contract tests passed. Initial
py_compile and git diff --check passed. This preliminary report is replaced only by Priority32's own final
report, never by editing an earlier priority's report.

## Progress

All three validation baseline gates passed on the first frozen configuration
(rounds50, lr.001, focal alpha.95/gamma2):

| Dataset | Validation F1 | Validation ROC-AUC | Quality gate |
| --- | --- | --- | --- |
| PaySim | 0.7457627118644068 | 0.9953917031663356 | PASS |
| IEEE-CIS | 0.40025948751216345 | 0.8525038506570792 | PASS |
| BAF | 0.23506988564167725 | 0.8804120881813815 | PASS |

All-fraud F1 and exact outputs: quality/<dataset>/attempt_1.json. Those jobs
did not evaluate test. execution_freeze.json was saved before Stage4 launch.

### Numerical failure, not silently replayed

At 2026-10-01T20:18:19.540099Z (2026-10-02 local), BAF / DNA v1 conservative /
seed321001 finished50 rounds but validation probability contained NaN;
sklearn.roc_auc_score raised `ValueError: Input contains NaN`.
Exact traceback and job config hash are retained in runs.jsonl. No valid
scientific result JSON was emitted for that job; it must NOT be automatically
retried or excluded from the21-replicate analysis. No clamping, imputation,
configuration change, or result-dependent tuning has occurred.

The supervisor's ProcessPoolExecutor shutdown waits for already-submitted
futures after this exception. Consequently the authorized jobs continue;
no early stop or second training workload was launched. A telemetry-only
sidecar, experiments/observe_priority32_drain.py, records finished output files
and updates progress while the original supervisor drains its queue. It runs
no training and creates no targets. Its exact command/PID are in
observer_launch.json and its observations in observer.jsonl. Other absent
results are not automatically assumed to have this same cause.

Root cause has not yet been established from saved model-state evidence.
In particular, no claim is made that nonfinite output is merely an
infrastructure failure. RQ2 statistics cannot treat it as a valid finite score.
Any scientific repair or missing-result treatment requires explicit direction;
there will be no automatic resume/reoptimization of this failed seed.

### Read-only reporting-path audit

The existing `experiments/fraud_fl_common.py:171-179` catches a ValueError
from ROC-AUC/PR-AUC evaluation and returns `None` for those endpoints.
The new `experiments/priority32_multidataset.py:305-308` evaluates the same
metrics without that catch, so nonfinite probabilities raise an exception
instead of being saved as unavailable endpoints. This is a reporting-path
difference, not evidence that the nonfinite model prediction itself is an
infrastructure error. Neither a `None` nor a NaN is a valid finite input to
the21-pair non-inferiority CI. No probabilities were replaced and no historical
metrics were re-scored. The missing completed results and50-round journals
remain preserved for later diagnosis.

Machine checklist/progress and all run disclosures:
artifacts/priority32_multidataset/checklist.json, progress.json, progress.log,
runs.jsonl. Four single-thread CPU worker processes; validated per-job resume.
RQ2 remains189 jobs with21 paired seeds/dataset, no scope reduction.
