# Amendment: Priority 8 DNA Transform v2 ratio0.90 Parameter-Sensitivity Check

**Timestamp:** 2026-09-17, before target generation or execution.  
**Status:** DEVELOPMENT-GATE ONLY — no confirmatory execution authorized.  
**Applies to:** IEEE-CIS parameter-sensitivity check for DNA Transform v2.

## Purpose

`GEN_COSINE_TV` has confirmed attacks against DNA Transform v2 at
`compression_ratio=0.95, quantization_eta=0.01` on both IEEE-CIS and CIFAR-10.
This amendment tests whether the same attacker still works at an existing v2
configuration that has not previously been used for an attack gate:

- `compression_ratio = 0.90`;
- `quantization_eta = 0.01`.

This is a parameter-sensitivity check for v2. It does not change any frozen
RQ1/RQ2/RQ3 conclusion.

## Dataset and initial gate

Run IEEE-CIS first because it is cheaper than CIFAR-10:

- dataset: IEEE-CIS Fraud Detection;
- scope: 4 rows/group, 1 fraud row/group, 1 client step;
- groups: 8;
- defense: `v2_ratio0p9_eta0p01`;
- attacker generation: `GEN_COSINE_TV`;
- controls: Prior and Zero-update;
- gate: exact one-sided sign test, `alpha=0.05`, must beat both controls.

The target set must be newly generated and source-disjoint with all prior
IEEE-CIS development, post-hoc, Priority 5, Priority 6, and confirmatory target
sets detected by the existing IEEE target-firewall logic.

## Escalation rule

If the n=8 IEEE-CIS development gate passes, escalate to an independent n=24
development pilot with a new source-disjoint target set, using the same defense,
attacker, and gate rule.

If n=24 also maintains a strong pass, stop and request supervisor confirmation
before writing any confirmatory amendment. A future confirmatory run must use
the established exact-binomial design with `p1=0.70`, required non-tied `n=37`,
draw `n=39`; do not refit `p1` from pilot results.

If the n=8 or n=24 gate fails, stop and report that v2 attackability may be
sensitive to the `compression_ratio`/`eta` setting. Do not escalate further.

## Out of scope

- No CIFAR-10/image-domain repeat is authorized in this amendment.
- No confirmatory run is authorized.
- No other v2 parameter setting is authorized.
- No tuning after seeing results is authorized.

