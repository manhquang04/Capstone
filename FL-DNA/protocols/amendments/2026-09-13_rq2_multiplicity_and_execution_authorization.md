# RQ2 multiplicity and RQ1/RQ2 execution authorization

**Timestamp:** 2026-09-13T00:41:34+07:00  
**Approved by:** Manh Quang (supervisor and research lead)  
**Timing:** recorded before confirmatory execution and before any confirmatory
outcome was generated or inspected

The RQ2 method-family rule is frozen as follows:

- `DNA Transform - Baseline` is the sole primary RQ2 hypothesis and retains
  the full one-sided `alpha = 0.05`.
- `DNA lossless - Baseline` is a secondary technical/sanity check of transport
  invariance. Its paired confidence intervals and results are reported
  separately, but it is not a member of the primary hypothesis family and is
  not adjusted jointly with DNA Transform by Holm or Bonferroni.
- The existing intersection-union endpoint rule remains unchanged: both F1
  and AUC-ROC must establish non-inferiority for the primary DNA Transform
  claim.
- The approved `required_confirmatory_seed_count = 21` remains unchanged. It
  was driven by DNA Transform F1, and the primary hypothesis continues to use
  the same unpartitioned alpha 0.05 used in that calculation.

With this final rule recorded, confirmatory execution is authorized for RQ1
and RQ2 using only the already frozen target set, seed list, method configs and
budgets. RQ3 remains held. Unfavorable outcomes do not authorize a rerun,
reconfiguration, new target draw or replacement seed.

The RQ1 surrogate attacker requires a concrete run seed. Before execution it
is deterministically materialized as `1184685071`, the first 31 bits of
`SHA256('RQ1-CONFIRMATORY-V1|DNA-SURROGATE|2026-09-13')`. This technical seed
does not depend on any privacy outcome and changes no scientific parameter.
