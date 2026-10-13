# Priority35B: additive publication receipt contracts

2026-10-11. Infrastructure implementation before production analysis/publication.
No change to measurements, interpretation rules, estimators, budgets, numerical
tolerances, arms, gates, matching, or the fixed Holm family.

This pure new module rejects partial CSV inventories, incomplete utility audits,
synthetic figure QA substituted for measured figure inspection, missing visual
checks, and QA that is not bound to the exact PDF/PNG/data/render hashes. Actual
tests, compilation, diff checks, source/input hashes and no-live checks remain
mandatory. The caller must hash-verify receipts and files; these contracts do not
perform independent verification merely because a receipt says PASS.

A publication-stage receipt is not COMPLETE. Completion requires a separate
post-exit seal bound to the full report and manifest digests. This module neither
writes a report nor launches measurements nor marks the project complete.
