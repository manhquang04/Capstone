# Priority35B — post-hoc report serialization infrastructure note

Date: 2026-10-12. The report renderer stopped before writing the report because
NOT_ASSESSABLE matching rows omit optional fields present in MATCHED rows.
Preserve the original error record and all frozen sources/data. A new additive
table-only adapter displays missing optional metadata as NA; it never supplies
a numeric result, computes a test, changes any value or edits matching.csv.
Required scientific inventories, all CSV recomputation and hash gates remain
unchanged. No report bytes or measurement files were changed by the failed
render. Include the adapter, note, error and freeze in the final closure.
