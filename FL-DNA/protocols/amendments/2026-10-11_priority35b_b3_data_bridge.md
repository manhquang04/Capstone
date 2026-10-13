# Priority 35B B3 complete-only publication data bridge

Date 2026-10-11. Additive post-launch infrastructure, before final protected
analysis. This note changes no arm, target, budget, measurement, qualification,
matching rule, estimator-selection rule, test, tolerance or Holm reservation.

The pure bridge accepts all 9,984 already independently audited result
documents and checks inventory, source-ID pairing and reconstruction-seed
pairing against each unprotected target. It emits per-target rows without
reading files or writing evidence. Prior MSE and data-free reference scores
come only from the matching unprotected target, not protected outcomes.

Figure data join the complete 768 leakage summaries to all 288 utility rows.
The unprotected leakage arm joins the baseline utility method; all other
methods join by exact names. The 96 figure rows use utility loss = negative
Delta-F1, and negate/reverse its t-interval endpoints. Feature accuracy is
kept on [0,1] in source data and shown as percent by the renderer. Metadata
retain the exact epsilon, C, SD, clip-only status and preselected estimator.
No protected result is used to select the panel estimator. The existing
unprotected-only selection receipt and fixed tie rule must match.

For the data-free horizontal lines, mean/mode is the median of its saved
39 target scores; empirical marginal is the median across targets of the
saved 30-draw empirical mean score for each target. Those saved scores are
independently audited by the control verifier. No additional reference
measurement or random draw is made by this bridge.

A separate direct NumPy check recomputes medians/ranks from raw per-target
rows, sign/reversal of utility intervals and data-free lines. It does not
replace independent utility/source verification or figure visual QA. Tests
use synthetic complete inventories only, including utility gains, missing
results, changed pairing, wrong estimator choice and changed figure data.
No measured CSV, figure or report is written at this preparation stage.

Nature-statistics guidance informs explicit units and avoidance of treating
39 batches, their individual features or 30 ensemble members interchangeably.
The axes continue to describe different configurations, and the family is
still fixed at 144 tests including non-assessable reservations.
