# Priority 35B B3 final manifest hash-audit infrastructure

Date: 2026-10-11. Additive post-launch infrastructure before final analysis.
No measurements, scientific sources, rules, budgets or tolerances change.

The new read-only hash-audit module requires all 9,984 result documents and
their result/process/three-bundle paths (49,920 paths), never sampled entries.
It verifies every input and output SHA-256 and rejects extra or missing files
within explicitly closed artifact output directories. Scientific evidence
and the full report cannot be excluded from the manifest.

The one authorized changed historical input is the report after append:
its original 4,793-byte snapshot must have the frozen digest, remain an exact
prefix, and satisfy the append-only report guard. Every other historical
input must still match its original digest. The current full report receives
an output digest. This does not replace the separate scientific, statistics,
figure or source-ID verification.

Manifest self-reference and completion/active publication logs require
explicitly named exclusions and a separate post-exit infrastructure seal;
no measurement or report exclusion is allowed. Logs must be stable after
their writers exit before being sealed, preventing the earlier class of
hash-before-traceback errors. Original manifests/logs remain untouched.

Functions perform no writes and never mark COMPLETE. Synthetic hash mismatch
and complete-inventory tests run before freezing. Actual final integration,
post-publication no-live audit and completion receipt remain pending.
