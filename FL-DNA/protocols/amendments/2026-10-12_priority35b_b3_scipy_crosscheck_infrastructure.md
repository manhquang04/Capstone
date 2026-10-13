# Priority35B B3 — post-hoc SciPy cross-check infrastructure addendum

Date: 2026-10-12. Written and frozen before re-verification, following explicit
user approval. No measurement, score, exact sign count, p-value, Holm family,
threshold, matching result, figure datum or earlier report text is changed.

## Authority and numerical rule

The exact integer-binomial tails already computed by the frozen design are
authoritative: sum(comb(n,k), k=w..n) / 2**n, with n<=39. These dyadic
probabilities are exactly representable in float64. The cross-check must first
recompute wins/losses/ties and this integer-binomial value exactly. SciPy
binomtest is a secondary independent cross-check, not the published authority.
Its float64 internal numerical algorithm differed by at most
2.220446049250313e-16 in five comparisons; no threshold decision changed.

Before the new audit, fix SciPy cross-check acceptance to

    abs(exact_p - scipy_p) <= max(1e-15, 1e-12*max(abs(exact_p),abs(scipy_p))).

This is a conservative float64 cross-check resolution, allowing a handful of
algorithmic rounding ULPs without rewriting any p-value. All integer counts,
exact-binomial values, and separate NumPy Holm recomputation must still match
exactly. No p-value is rounded or substituted. Before accepting any row, check
that neither authoritative raw p nor Holm p lies within this tolerance envelope
of 0.05, the only p-value decision threshold used here. Fail closed otherwise.

## Before-run threshold-distance check

All 144 raw p-values and all 144 Holm-adjusted values, including NA p=1
reservations, were checked read-only. Minimum raw distance to 0.05 is
0.012499999999999997 (raw p=0.0625). Minimum Holm distance is
0.030309277074411514 (Holm p=0.01969072292558849). None lies within the
tolerance envelope of 0.05. No other p-value threshold is used. The verifier
will repeat this check from the CSVs. Thus no decision depends on the numerical
cross-check difference. The previously approved MSE audit rule remains in force.

## Preservation and closure

Keep every byte of the four original failed-verifier FAILURE.json files and
their START/ledger logs unchanged in the two verified_preparation_attempt and
two publication_attempt directories dated before this addendum. A checksum
allowlist identifies precisely these four known infrastructure failures; no
scientific job failure, new failure or unrecognized file is waived. Include them
in the final manifest, along with the new adapter, tests, addenda and freezes.
The earlier read-only diagnostic failure is preserved too.

Use only additive adapters; frozen sources remain unchanged. Re-run only the
post-exit verifier, then its CSV/figure/report/closure verification stages.
Synthetic tests are permitted; no scientific measurement, optimizer job,
training run or seed replacement is repeated. Preserve the original report as
an exact byte prefix and append the fully verified report section and explicit
infrastructure disclosures. Do not mark COMPLETE until all scientific audits,
CSV recomputation, visual QA, tests, py_compile, git diff --check, full manifest,
historical hashes and fresh post-exit checks pass. Any new discrepancy stops work.
