# Priority 35B B3 post-exit publication preparation runner

Date 2026-10-11; additive post-launch infrastructure frozen before final
analysis. No measurement, scientific source, tolerance, gate, pairing,
matching rule, interval or Holm family is changed. No measurements repeated.

New command, only AFTER supervisor and every recorded worker have exited:

```text
.venv-phase1/bin/python -B -m experiments.priority35b_leakage_prepare_publication --audit
```

The command first checks actual processes, the scientific socket, locks,
finished 9,984-job tracking, failures and all additive source/dependency
seals. It then invokes the complete independent scientific coordinator and
historical utility manifest validation, serializes full per-target/frontier/
Holm CSVs and independently recomputes them from their written contents.
It joins exact figure source data and independently verifies that join.
It snapshots the original report prefix, without editing the report.

Outputs use a new exclusive-create publication attempt directory. Existing
attempts, measurements, reports, manifests and logs are never overwritten.
Failure records remain preserved and require human direction, never a
measurement replay or automatic retry. Root administrative progress changes
only after a successful audit and fresh no-live check, when no scientific
process can race those tracking writes.

Successful preparation is explicitly FIGURE_AND_PUBLICATION_PENDING, NOT
COMPLETE. PDF/PNG rendering and visual QA, tests/compile/diff receipts,
result-specific interpretation, append-only report publication, full final
manifest re-verification and a fresh no-live seal remain required. This
runner does not perform those remaining operations or mark COMPLETE.

Preparation unit tests exercise only synthetic stage gates and historical
CSV schema/type conversion. The production command has NOT been invoked
while measurements are running. No partial protected quality is read.
