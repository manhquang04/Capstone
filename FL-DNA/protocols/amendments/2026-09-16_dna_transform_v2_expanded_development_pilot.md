# Amendment: DNA Transform v2 expanded development pilot

**Timestamp:** 2026-09-16, before expanded-development target generation or
results are observed.  
**Status:** DEVELOPMENT STABILITY CHECK ONLY — no confirmatory target generation
authorized by this amendment.  
**Applies to:** DNA Transform v2, compressed-sensing/IHT attacker family.

## 1. Motivation

The previous v2 attacker line tested three controlled attacker generations on
the same n=8 development pilot:

1. raw-lift attacker;
2. sketch-space STE attacker;
3. compressed-sensing/IHT attacker.

All failed the Prior and/or Zero-update development controls. Supervisor
authorizes one follow-up development stability check to reduce the possibility
that the n=8 result was only small-sample noise.

## 2. Development target expansion

Create exactly one new development target set:

```text
groups = 24
records_per_group = 4
fraud_records_per_group = 1
```

The new target set must be source-disjoint from:

- the old `development_gate_targets.pt` used by all three prior v2 attacker
  generations;
- every existing `*targets.pt` artifact registered under `artifacts/`;
- all post-hoc, replication, clean Priority-2, and confirmatory target sets.

Historical/post-hoc targets may only be read for source-ID overlap checking.
Their outcomes, reconstructions, or metrics must not be inspected for this
amendment.

## 3. Frozen attacker variant

Run only one attacker variant:

```text
attacker = compressed-sensing/IHT
sparsity_fraction = 0.20
iht_iterations = 80
iht_step_size = 1.0
restarts = 4
attack_iterations = 600
attack_lr = 0.1
compression_ratio = 0.95
quantization_eta = 0.01
v2_base_seed = 20260916
nonnegative_lambda = 0.001
init_mode = standard
```

Rationale for choosing `sparsity_fraction=0.20` before seeing expanded-pilot
results:

- IHT-0.10 and IHT-0.20 were tied as the closest compressed-sensing variants on
  the n=8 pilot with respect to Zero-update wins (`6/8`, p=0.1445).
- IHT-0.20 uses the weaker sparsity prior and is therefore the more conservative
  attacker choice: it gives the attacker more degrees of freedom and is less
  dependent on an aggressive sparsity assumption.

No other attacker hyperparameter may be changed under this amendment.

## 4. Gate rule

Use the same development control rule as before:

```text
mean difference < 0
median difference < 0
one-sided sign-test p < 0.05
```

The attacker must pass both controls:

```text
Prior control gate = PASS
Zero-update control gate = PASS
Overall gate = Prior PASS AND Zero PASS
```

For `n=24`, the first one-sided exact sign-test rejection count at alpha 0.05
is:

```text
wins >= 17 / 24
p = 0.0319573283
```

## 5. Interpretation rule

If the attacker still fails both controls, or clearly fails to pass the overall
gate, the project may proceed to a separate theoretical-limit analysis
amendment. The allowed wording is:

```text
The three-generation v2 attacker failure was stable under an independent n=24
development expansion for the strongest pre-selected compressed-sensing/IHT
variant.
```

If the attacker passes both controls at n=24, then the previous n=8 failures
are treated as small-sample instability. In that case, the attacker is validated
for development purposes and the next step is a separate full v2 confirmatory
protocol, not theoretical-limit analysis.

If the result is mixed, for example one control passes and the other fails,
stop and report the status as unresolved at this development scale. Do not
interpret it in the direction favorable to v2.

## 6. Prohibitions

- Do not use the old n=8 target groups in the expanded-pilot analysis.
- Do not use post-hoc targets except for source-ID overlap checking.
- Do not tune attacker parameters after seeing expanded-pilot results.
- Do not modify `torch.set_num_threads(1)`.
- Do not modify v1 code.
- Do not create any v2 confirmatory target set under this amendment.
- Do not start theoretical-limit analysis until the expanded-pilot report is
  written and the result clearly supports the "still fail" branch.
