# P34D independent BN result verifier precision

Written before BN capture/qualification/confirmatory execution and before audit
access to any new measured BN target. New audit driver requires BN_COMPLETE and
no live P34D scientific workers before it loads measured batches. Science stays
frozen and unchanged. Independent public-checkpoint decoder uses NumPy least
squares and dense SciPy Hadamard, not the experiment's Torch/FWHT solver.

Fixed numeric comparison tolerances: recovered feature arrays rtol1e-6/atol1e-8;
standardized MSE rtol1e-6/atol1e-10; float32 actual BN-delta identity from direct
first-linear batch mean rtol5e-5/atol5e-6; calibration norm/ratio rtol2e-6/atol1e-8.
These are audit floating-point tolerances, not science clamps or gates. Every
value remains finite; negative actual BN variance fails. No tolerance changes
based on confirmatory outcomes. Mismatch preserves diagnostic evidence and
requires an additive technical repair or scientific direction, never exclusion.

Independently reload prepared source rows, true means, exact uploaded vectors,
public fixed checkpoints, per-target receipt hashes, qualification controls and
development matching. Recreate DP payloads bit-exact using private seeds only
inside verifier; never print/expose them to protected recovery interface. Exact
control sign counts/p must agree. Completeness: all39 confirmatory pairs for
every qualified checkpoint, failed checkpoints remain NOT_ASSESSABLE. Verifier
PASS is BN-stage evidence only, not the full P34D COMPLETE/report gate.
