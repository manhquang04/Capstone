# Priority35B B3 — post-hoc infrastructure addendum, 2026-10-12

Written before amended re-verification. This changes only the independent MSE
verifier's numerical acceptance rule. No measurement, stored score, control,
matching rule, gate, Holm family, sign-test tie rule, plot value or interpretation
is changed. All scientific and previously frozen verifier sources stay immutable;
an additive runtime adapter implements the rule.

## Preserved failures and read-only evidence

Keep unchanged both FAILURE.json files and all other files in
`artifacts/priority35b/b3/verified_preparation_attempt_1791783434043922000/`
and `publication_attempt_1791783453466824000/`. Their tracebacks record the
original `independent float64 MSE reduction mismatch`. Also preserve the
read-only diagnosis failure `precision_diagnosis_first_scan_failure.json`.

The complete read-only scan is
`artifacts/priority35b/b3/precision_diagnosis_1791784605000547000.json`.
Among 9984 outputs, ten exceed the former absolute 1e-12 rule: IEEE-CIS v1
(four) and v2 (six). Maximum absolute discrepancy is 1.4901161193847656e-08,
median over all outputs zero; median among 3563 nonzero discrepancies is
8.881784197001252e-16. Stored scores equal native Torch recomputation exactly
for every output. Accuracy and assignment equal the independent result exactly.

Both reductions use float64, starting from the same float32 tensors and same
clamping/alignment. This is NOT a comparison of float32 MSE to float64 MSE.
Torch and NumPy use different summation orders. A fixed absolute rule does not
scale with large squared residuals.

## Rule fixed before re-verification

Let u = numpy.finfo(float64).eps = 2^-52, N be the number of encoded scalar
entries in the aligned batch, and gamma_N = N*u/(1-N*u). Require N*u < 1,
finite nonnegative MSEs, exact accuracy/assignment, and exact equality of the
stored score with a native Torch recomputation. Accept the independent MSE iff

    abs(stored - independent) <= 1e-12 + 2*gamma_N*max(abs(stored),abs(independent)).

The conservative gamma_N summation-error bound allows two differently ordered
float64 reductions of nonnegative squared residuals, with u larger than the
usual half-epsilon unit roundoff. The existing absolute floor is retained for
near-zero values. No values are rounded, clamped, replaced or excluded in the
published statistics. Bounded-score equality with the stored prior remains exact.

## Decision invariance condition and result

Before this addendum, the scan recomputed both the recovered-row MSE and
mean/mode prior MSE using independent NumPy float64 reductions, then bounded
scores and all fixed-144 family sign tests. There were zero changes in
wins/losses/ties, raw p values, Holm-adjusted p values or p-threshold decisions.
Feature accuracies are exactly unchanged, hence the unprotected estimator
selection and all figure y coordinates and intervals are unchanged. Utility
data determine matching and figure x coordinates and are not modified; all
matched arms and NOT_ASSESSABLE statuses are unchanged. P34C qualification
gates remain historical immutable inputs. Every discrepancy satisfies the
gamma_N bound above. The native equality check additionally guards against
differences arising from anything other than the independent reduction.

## Execution and stop rule

Run tests of the additive adapter, freeze its sources/tests/addendum and the
preserved failure/diagnosis hashes, reconfirm no live scientific processes, then
re-run ONLY the post-exit verifier through the additive adapter. Preserve all
new attempt outputs. Any additional discrepancy stops publication and must be
reported. In particular, the separate SciPy exact-sign bit-equality check was
encountered by the read-only diagnosis and is NOT relaxed by this addendum.
No measurement may be replayed. Full Priority35B remains incomplete until all
independent audits, figure QA, report, manifest and post-exit seals pass.
