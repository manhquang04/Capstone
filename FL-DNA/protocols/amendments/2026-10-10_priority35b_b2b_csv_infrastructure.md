# B2b final CSV serialization — 2026-10-10, post hoc infrastructure note

Written after launch but before any utility-table/matching analysis or inspection
of measured utility outcomes. Static code inspection showed that final matching
rows have different optional metadata keys for MATCHED versus NOT_ASSESSABLE.
The generic CSV writer takes fieldnames from the first row and would reject a
later row with different keys. No verifier failure has occurred and no result,
gate, numeric computation, interval, matching decision or Holm family changes.

Keep all frozen sources unchanged. An additive finalization adapter
`experiments/priority35b_b2b_finish.py` changes ONLY the serializer's fieldnames
to the ordered union of metadata keys when writing `matching.csv`; missing
metadata is an empty CSV field. JSON decisions and every numerical value remain
unchanged. Every other file uses the original writer. The adapter invokes the
original frozen independent verifier only after all scientific processes exit.

Freeze this note and adapter in
`artifacts/priority35b/b2b/arms/finalizer_infrastructure_freeze.json` before
running the adapter. Preserve the original execution freeze, all results,
attempts and logs. No measurement replay, tolerance change or new arm.
