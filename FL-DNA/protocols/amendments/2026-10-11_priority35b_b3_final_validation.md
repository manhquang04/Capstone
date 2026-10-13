# Priority35B B3: additive final validation runner

2026-10-11, infrastructure implementation before production analysis. No
measurement, scientific rule, tolerance, target, seed, budget, matching or Holm
family changes. Frozen scientific sources are immutable.

After scientific processes AND the preparation wrapper exit, the new runner
requires its hash-verified stable preparation receipt. It runs each of twenty
test modules in a separate .venv-phase1 interpreter (CPU, one thread environment)
to avoid the preserved historical models import collision. Every subprocess
command, return code, stdout and stderr is saved exclusively under a new
validation attempt. Any failure is kept and requires direction; no automatic
retry. No scientific measurement is repeated.

It uses py_compile with bytecode outputs only in the new validation attempt,
checks git diff --check, rechecks source/input and all scientific-file hashes and
the no-live condition, then writes VALIDATION_PASS. This is not COMPLETE and
cannot substitute for measured figure visual QA, report-prefix checks or final
post-publication closure sealing.

Post-exit command:
`.venv-phase1/bin/python -B -m experiments.priority35b_leakage_final_validation --ready <relative READY_FOR_RENDER.json path>`
