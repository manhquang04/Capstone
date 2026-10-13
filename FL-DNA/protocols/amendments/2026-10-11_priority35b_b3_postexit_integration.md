# B3 post-exit audit integration — 2026-10-11, post-launch infrastructure

This new module joins the immutable integrity, payload and independent metric
building blocks. It refuses analysis until the actual supervisor, every recorded
worker PID and their Python descendants have exited; it checks socket lock and
lock files too. PID reuse is conservatively a stop, not an automatic override.
No optimizer or measurement is rerun. No rule, target, gate or Holm family changes.

Only after the no-live gate, it audits all 9,984 bundles, all 312 paired control
sets and plotted-estimator selection from unprotected uploads. The 30 empirical
controls are independently resampled from the frozen control seed and scored.
Their percent values permit absolute 1e-12 percentage-point reduction difference,
declared before running this verifier; stored measurement scores remain unchanged
for all inference. Report the maximum metric/control differences in the audit.

This is not COMPLETE: objective recomputation, statistical CSV regeneration,
figure/report verification and full final seals remain required. Synthetic tests
exercise the process gate, not incomplete protected measurements. The original
measurement driver and every earlier artifact/report remain immutable.
