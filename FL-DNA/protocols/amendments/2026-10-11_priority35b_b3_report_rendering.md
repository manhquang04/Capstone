# Priority 35B B3 report rendering infrastructure

Date 2026-10-11; additive post-launch infrastructure, before final protected
analysis. The new pure Markdown renderer consumes already verified tables and
actual provenance; it never reads incomplete measurements or writes a report.
It changes no measurement, qualification, matching rule, interval, sign test,
Holm family, tolerance or scientific decision. The report remains unchanged
until the post-exit publication workflow completes its mandatory checks.

The rendering specification includes all 288 utility rows, 81 decomposition
rows, 12 matching entries, 768 frontier summaries and 144 Holm entries, not
selected favorable comparisons. It distinguishes 11 utility seed units from
39 paired leakage batch units and includes explicit clip-only NO-DP and
different-configuration-axis disclosures. Exact float representations are
retained in tables; Markdown cells escape separators. Missing evidence or
non-finite numeric cells fail closed.

Nature-statistics reporting guidance informs explicit experimental units,
uncertainty, tie handling and avoidance of equivalence/privacy claims from
non-significance. Unit tests use synthetic strings only. Final result-specific
interpretation will be based on complete independently recomputed tables.
No existing report section or historical evidence is changed.
