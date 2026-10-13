# Priority35B: append-only report integration

2026-10-12 (Asia/Ho_Chi_Minh). Additive infrastructure implementation before
production analysis/publication; no scientific measurement or rule changes.

The new publication runner accepts the stable preparation receipt, the actual
render receipt and measured visual QA, the actual final test/compile/diff
receipt and a result-specific descriptive reading explicitly bound to the full
frontier and Holm CSV hashes. It rechecks all evidence hashes and CSV/figure
joins and rejects live preparation, rendering or validation actors. Test logs
are reconciled with counts and commands; synthetic QA cannot be substituted.

It builds a new proposal under a new publication attempt, verifies all checks,
then appends only the authorized suffix to the unchanged 4,793-byte historical
report. Every command and amendment digest is recorded. No measurement is
repeated. Failures are preserved and require human direction; no auto replay.

Success is REPORT_APPENDED_PENDING_POSTEXIT_FULL_SEAL, not COMPLETE. A separate
runner after this writer exits must verify the full closed manifest, report
prefix, historical inputs and fresh no-live state before writing COMPLETE.

Command after all prior audits:
`.venv-phase1/bin/python -B -m experiments.priority35b_leakage_publish_report --ready <receipt> --render <receipt> --qa <measured QA> --validation <receipt> --interpretation <full-CSV-bound reading>`
