# B3 objective audit building block — 2026-10-11, post-launch infrastructure

Before protected quality analysis, add a read-only audit that recomputes the
observable projected-gradient objective of every saved native ensemble member.
It repeats no optimization. Plain cosine and block debiasing are separately
implemented with the same float32 operation order; exact agreement is required.
V2 uses the frozen sketch helper with the separately audited transmitted plan.
The minimum member objective must match exactly; do not incorrectly substitute
the pooled reconstruction's objective. No ground-truth row enters objective
evaluation. This changes no measurement, selection, budget, test or Holm family.

Synthetic tests compare plain objective and debiasing and reject corrupted saved
losses. Ratio solver residual verification, pooling verification and overall
integration remain pending; this module alone does not mark COMPLETE.
