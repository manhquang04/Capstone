# Priority35A: post-hoc infrastructure-only cosine verification addendum

Date: 2026-10-10. Written after the failed final verifier and direct user approval, before the amended verifier runs. This is not a scientific design amendment.

The original failed verifier record, including its command, output summary and traceback, remains unchanged at `artifacts/priority35a/final/REQUIRES_DIRECTION_cosine_audit_20261010.json`. Its original `final_verifier_requires_direction` event remains in the append-only `progress.log`. No original frozen verifier, measurement, result, or log is edited.

Only the verification rule for the descriptive sign cosine changes. Reload stored float32 gradients and Adam uploads. Independently recompute the cosine using float32 torch reductions: the stored float32 value must match exactly, including zero-norm flags. Independently compute dot products and norms in NumPy float64, retaining the original epsilon clamps of 1e-12. Require absolute discrepancy <= 1e-5 with relative tolerance zero. Reject non-finite values, incorrect schemas or zero flags, and any exact float32 mismatch. Report the maximum float64 discrepancy and the observed range of each cosine.

Float32 machine epsilon is 2^-23 (approximately 1.19209e-7). The predeclared 1e-5 absolute guard is approximately 84 such epsilons, accommodating accumulated dot/norm reduction rounding while exact float32 reproduction separately detects stored-value discrepancies. This is an engineering cross-precision audit threshold, not a universal mathematical error bound for arbitrary vectors, nor a measurement or statistical tolerance. The old read-only diagnosis already recorded a maximum difference of 2.614711619775001e-6; this addendum is explicitly post hoc and does not conceal that observation.

An additive verifier substitutes only the cosine-audit callable during the unchanged independent tensor audit, then reuses the existing CSV export and independent recomputation. Original sources and failed outputs remain hash-checked. No measurement is rerun. The final manifest also covers this addendum and the additive verifier/tests.

Qualification remains 0/72; every gate, sample, target, seed, budget, metric, exact tie, Holm family of 6048 and diagnostic interpretation rule remains unchanged. Cosine is descriptive only: conclusions use reconstruction feature accuracy against controls, not cosine or its 1e-5-level precision. The report must state the observed cosine ranges and explicitly confirm this independence. All six D1 failures must retain the locked interpretation that the instruments or budgets are inadequate and negative qualification says nothing about privacy.

Commands: `.venv-phase1/bin/python -B experiments/priority35a_cosine_verifier.py --prepare`, then the same script with `--finish`. Neither command launches scientific measurement workers.
