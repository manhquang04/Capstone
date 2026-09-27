# Technical replay amendment: Priority 14 launch-time variable-shadowing fix

**Date:** 2026-09-21  
**Scope:** Priority 14 fresh clean-vector n=8 probe runner only.

## Issue

The first launch of the v1 probe failed before any scientific job completed due
to a Python variable-shadowing error:

```text
UnboundLocalError: local variable 'common' referenced before assignment
```

The imported module `experiments.fraud_fl_common as common` was shadowed by a
local dictionary variable named `common` inside `_job()`.

## Fix

Rename the local dictionary from `common` to `job_context`. This does not change:

- target set;
- random seeds;
- DNA/DP parameters;
- vector definition;
- attacker objective;
- optimization budget;
- statistical decision rule.

The failed output directory is retained as a launch-failure artifact. The replay
uses a new output directory and the same frozen Priority 14 settings.

## Second implementation correction

After the first successful probe execution, audit of the runner found that the
nonnegative penalty was implemented as a generic `relu(-latent)` penalty instead
of the original PaySim hard-balance attacker penalty:

```text
_nonnegative_penalty(latent, meta)
```

This is an implementation mismatch with the frozen Priority 14 attacker
definition ("nonnegative penalty 0.001" following the original attacker). The
generic-penalty output directories are retained as invalid technical artifacts
and are not used for the scientific report. The runner is corrected to call the
original `_nonnegative_penalty` helper, and both cells are replayed into new
output directories with the same targets, seeds, DNA/DP parameters, objective,
and optimization budget.

## Authorization

This is a technical replay because the failure occurred before scientific
outcomes were produced.

The second correction is also a technical replay because it restores the
pre-specified original attacker penalty rather than changing a scientific
parameter after seeing outcomes.
