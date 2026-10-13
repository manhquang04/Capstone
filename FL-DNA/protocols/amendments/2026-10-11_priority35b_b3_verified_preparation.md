# Priority 35B B3 stable-evidence audit wrapper

Date: 2026-10-11. Additive post-launch infrastructure, frozen before final
protected-data analysis. No scientific rule, measurement, threshold, source,
target or optimization changes. No measurements repeated.

After actual supervisor/all-worker exit and complete 9,984-job status, use:

```text
.venv-phase1/bin/python -B -m experiments.priority35b_leakage_verified_preparation --prepare
```

This wraps the already frozen publication-preparation command rather than
editing it. Before invoking any scientific verifier, it records SHA-256 of
all 49,920 result/process/tensor evidence files. After complete scientific,
CSV and figure-source-data verification, it recomputes every hash and requires
the whole map to be unchanged. The resulting ledger must also be reverified
before final publication, so a later manifest cannot silently bless evidence
changed after its independent scientific audit.

The wrapper additionally audits all original additive sources and their
dependency seals and rechecks no-live conditions. A new exclusive-create
attempt records command, PID, ledger, successful preparation receipt digest
and stability receipt. Failures are preserved and require human direction;
there is no automatic retry or measurement replay. Tracking remains at
analysis-verified/figure-pending, never COMPLETE.

The original preparation command still exists unchanged, but this wrapper
is the intended final workflow entry point. Rendering/visual QA, append-only
report interpretation, full manifest and post-publication exit seal remain
separate pending stages. Unit tests use synthetic hash maps only.
