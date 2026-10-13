# Priority35B B3: final post-publication closure integration

2026-10-12, Asia/Ho_Chi_Minh. Infrastructure implementation before production
analysis/publication. No measurement or scientific decision/rule changes.

The fresh sealing process requires the report-publication writer and all earlier
recorded actors to have exited. It verifies the original report prefix and exact
proposal, prepared inputs/outputs, every 49,920 scientific evidence-file hash,
the historical utility manifest, full CSV/figure joins and source freezes. Two
new final integration suites run separately, with py_compile outputs in the new
seal attempt and a fresh diff check; the earlier twenty-suite runner is unchanged.

The final manifest covers the closed B3 and figure directories plus the complete
report, with only three explicitly named exclusions: manifest self-reference,
COMPLETE receipt, and the seal attempt's failure log. No scientific evidence or
report is excluded. Manifest hashes are verified in full before a COMPLETE
receipt is written. Failures are preserved and require direction; no auto replay.

Command, only after actual publication exits:
`.venv-phase1/bin/python -B -m experiments.priority35b_leakage_final_seal --published <relative PUBLISHED.json receipt>`

The calling agent must wait for this process to exit, then independently recheck
receipt digests, manifest closure, report prefix and no-live state before
announcing verified COMPLETE, updating administrative progress and pausing the
heartbeat. Existing Goals are unchanged. No measurement is repeated.
