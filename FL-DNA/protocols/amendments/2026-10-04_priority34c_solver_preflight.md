# Priority 34C — payload-only solver preflight

Written before synthetic solver tests and before any P34C target access or
scientific run. This additive implementation specification does not authorize
a research launch or replace the still-required complete execution freeze.

For the ratio instrument's key-known v2 arm, solve R x = q in least-squares
sense, where R is the original normalized signed, sampled Hadamard projection,
restricted to the original (unpadded) coordinates. Use scipy.sparse.linalg.lsmr
with zero initialization, atol=btol=1e-10, conlim=1e12, maxiter=4000, no damping.
Use float64 internally. Store convergence code, iterations, residual/normal
residual, norm/condition estimates and dimensions. Stop codes 0,1,2,4,5 are
accepted; other codes are a solver failure requiring direction, not an invitation
to tune the budget. Nonfinite input/output or diagnostics fail closed.

The solver receives only transmitted sketch and public/key-known projection
metadata; never the true gradient, target record or realized DP noise. The
transpose is an adjoint, not a pseudoinverse. k=ceil(.95*padded_dimension) may
exceed the original dimension; do not label every tensor underdetermined or
claim actual rank from dimensions alone. Record min(k,d) as an upper bound.
Small synthetic audits explicitly materialize R and compare numpy.linalg.lstsq,
including truncated-padding, underdetermined and quantized observations.

Ratio decoding chooses the first maximum-absolute bias-gradient row, using
observed payload only. An exactly zero denominator is an unresolved observation
and raises a recorded failure; never clamp or replace the target. Synthetic
identity and zero/nonfinite checks precede research use. No changes to original
DNA code, preprocessing, attack configuration or earlier artifacts.

Other execution seeds, source-disjoint firewall, dataset adapters, worker and
gate drivers remain pending. No P34C target or confirmatory job is launched by
this preflight.
