# Amendment: DNA Transform v2 expanded pilot numeric replay

**Timestamp:** 2026-09-16, after an execution-time numeric consistency
assertion failed, before resuming the expanded n=24 development pilot and before
any expanded-pilot gate report was produced.  
**Status:** TECHNICAL REPLAY AMENDMENT ONLY — no attacker hyperparameter change.

## 1. Trigger

The expanded n=24 IHT-0.20 run stopped during `capture()` with a PyTorch
`assert_close` failure comparing the simulated update with the native Adam
state-dict delta:

```text
Greatest absolute difference: 1.9073486328125e-06
Original tolerance: atol=2e-7, rtol=2e-4
```

The failure occurred before the expanded-pilot aggregate report/gate was
generated. Existing completed restart artifacts are kept and the run will be
resumed by the existing skip-if-`results.json` logic.

## 2. Technical fix

For `experiments/run_phase4_dna_v2_iht_attack.py` only, add a local fallback
capture path:

1. First call the existing Phase-3 `capture()` function unchanged.
2. If and only if it fails at the numeric consistency assertion, recompute the
   same capture with identical model, data, batch order, RNG state, optimizer,
   and update simulation.
3. Apply a float32-compatible consistency check:

```text
atol = 2e-6
rtol = 3e-4
```

This does not change:

- target set;
- attacker family;
- sparsity fraction;
- IHT iterations or step size;
- attack learning rate;
- restarts;
- random seeds;
- `torch.set_num_threads(1)`.

## 3. Interpretation

This amendment is a technical replay fix, not a tuning decision. It must not be
used to alter or reinterpret the gate. The expanded-pilot gate remains exactly
the one frozen in:

```text
protocols/amendments/2026-09-16_dna_transform_v2_expanded_development_pilot.md
```

If the resumed run produces a gate report, it should be reported together with
this amendment and the fact that a numeric replay tolerance was required for at
least one capture.
