# Amendment: Priority 3 DP accounting for existing clipping/noise settings

**Date:** 2026-09-16  
**Supervisor:** Manh Quang  
**Status:** APPROVED FOR ACCOUNTING-ONLY ANALYSIS  
**Scope:** compute privacy-accounting diagnostics for existing clipping +
Gaussian-noise configurations.  This amendment does not change any noise
multiplier, clipping norm, training seed, attacker parameter, or prior
scientific result.

## Locked accounting assumptions

The implemented defense clips a full client model update and then adds
independent Gaussian noise to every floating tensor:

```text
update -> L2 clip at C -> add N(0, (noise_multiplier * C)^2 I)
```

Because clipping is applied to the whole client update rather than per-example
gradients, this accounting is only a client/update-level Gaussian-mechanism
diagnostic under explicit sensitivity assumptions.  It is not record-level DP
for individual transactions.

Report two sensitivity conventions:

1. `add_remove`: neighboring update changes by at most `C`;
2. `replace_one`: neighboring update changes by at most `2C`.

Primary displayed delta values:

```text
delta = 1e-5
secondary deltas = 1e-6, 1e-8
```

Composition scenarios:

1. `single_release`: one protected update release, relevant to one isolated RQ1
   target update.
2. `rq2_same_client_50_rounds`: 50 sequential releases for the same client in
   the RQ2-style 50-round setup.

The accountant uses Rényi Differential Privacy for the Gaussian mechanism:

```text
RDP_alpha = alpha * sensitivity_ratio^2 / (2 * noise_multiplier^2)
epsilon(delta) = min_alpha RDP_alpha + log(1/delta)/(alpha - 1)
```

The order grid is deterministic:

```text
alpha ∈ {1 + k * 1e-5 for k=1..1000}
        ∪ {1.02, 1.03, ..., 10.00}
        ∪ {10.50, 11.00, ..., 512.00}
```

Technical correction note: the first implementation used `1.01` as the
smallest alpha.  Because the approved noise multipliers are very small, the
optimal order can be closer to 1.  The accountant was corrected before writing
the report; no DP mechanism, experimental output, or interpretation rule was
changed.

If epsilon is extremely large, report it as-is.  Do not relabel the existing
defense as strong DP and do not change any experiment outcome.

## Configurations to account

- RQ1/RQ2 conservative distortion-matched: `clip_norm=100`,
  `noise_multiplier=0.00025`.
- RQ1 medium distortion-matched: `clip_norm=100`,
  `noise_multiplier=0.000315`.
- RQ1 stronger distortion-matched: `clip_norm=100`,
  `noise_multiplier=0.0004`.
- Development-only utility-matched extension: `clip_norm=100`,
  `noise_multiplier=0.00001`.
- Historical default preset `medium`: `clip_norm=100`,
  `noise_multiplier=0.005`.

## Interpretation rule

If epsilon values are large enough to be practically meaningless, the report
must state that the clipping/noise comparator remains best described as
`DP-style clipping/noise with weak formal accounting under the stated
client/update-level assumptions`, not as a strong record-level DP baseline.
