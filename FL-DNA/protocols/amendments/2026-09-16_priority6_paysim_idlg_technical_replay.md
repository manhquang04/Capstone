# Technical replay amendment: Priority 6 PaySim v1 IDLG tensor filtering

Status: `AUTHORIZED_BEFORE_REPLAY`

Written at: 2026-09-16T17:02:00Z, after a type error but before replaying the
affected cell.

The first PaySim execution completed 96/128 jobs: both v2 cells and the v1
`GEN_COSINE_TV` cell.  The v1 `GEN_IDLG_STYLE` cell produced no completed job.
Its balanced-tensor objective was accidentally passed non-floating state
tensors, whereas all established project runners filter objective keys to
floating tensors.  `torch.finfo` therefore raised a type error before an
attacker result existed for this cell.

The technical correction is limited to restoring the established floating-key
filter.  The runner may resume in the same artifact directory, loading the 96
completed jobs without rerunning them and executing only the 32 missing v1
`GEN_IDLG_STYLE` jobs.  No target, seed, loss formula, optimizer, iteration
budget, restart count, gate, or selection rule changes.

