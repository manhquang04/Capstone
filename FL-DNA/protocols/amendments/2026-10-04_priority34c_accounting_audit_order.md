# Priority34C — synthetic accounting audit order correction

Before development/confirmatory freeze or n39 access, the synthetic audit test
found a mismatch: evaluating the P34B analytic sigma only on an older discrete
RDP order grid gives10.0000138176 instead of10. P34B used the continuous optimal
order, not that grid. Preserve this preflight failure receipt. Correct only the
new audit to independently derive rho=50/(2sigma²), alpha=1+sqrt(log(1/delta)/rho)
and evaluate the standard Gaussian RDP conversion at that order. No sigma,
clip, adjacency, noise, seeds, targets or existing sources/results change.
This is a test/audit implementation correction, not outcome-based tuning.
