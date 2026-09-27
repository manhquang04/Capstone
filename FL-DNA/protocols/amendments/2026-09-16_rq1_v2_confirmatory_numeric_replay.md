# Amendment: RQ1-v2 confirmatory numeric replay for capture-consistency failures

Status: APPROVED TECHNICAL REPLAY — no scientific parameter changes  
Timestamp: 2026-09-16, before any aggregate RQ1-v2 confirmatory analysis  
Approver: Manh Quang / Supervisor instruction via chat  

## Scope

This amendment authorizes replaying only the RQ1-v2 confirmatory jobs that failed before optimization due to the shared local-update capture consistency assertion.

Authorized failed jobs:

- `raw`, group `116`
- `raw`, group `165`
- `dna_v2`, group `116`
- `dp_v2`, group `116`
- `dp_v2`, group `165`

No additional target groups, branches, seeds, defense parameters, attacker parameters, or interpretation rules may be changed under this amendment.

## Trigger

The first official RQ1-v2 confirmatory execution completed all 176 groups for all three branches, but the jobs listed above failed before producing reconstruction results. All failures were `torch.testing.assert_close` failures inside the capture/replay consistency check inherited from `experiments/run_phase3_adam_ladder.py`.

Observed discrepancy pattern:

- exactly `1 / 128` tensor elements mismatched;
- maximum absolute discrepancy was either `3.814697265625e-06` or `7.62939453125e-06`;
- the failure occurred before reconstruction optimization and before any aggregate RQ1-v2 result was analyzed.

This is treated as a numerical replay/checking issue, not as a privacy/security outcome.

## Authorized technical change

For the failed jobs only, use a float32-compatible capture consistency tolerance:

- `atol = 1e-5`
- `rtol = 5e-3`

This tolerance is used only to validate that the separately simulated update and native Adam replay are numerically consistent enough to proceed. It does not alter:

- source target set;
- group membership;
- random seeds;
- local training step;
- DNA Transform v2 configuration;
- DP noise multiplier or MC noise sample count;
- attacker objective;
- optimizer budget;
- candidate selection rule;
- branch-gate rule;
- primary contrast or sign-test threshold.

For raw-branch partial failed restart directories, the runner may resume a failed restart directory that lacks `results.json`; this is a technical resume fix for an interrupted pre-result folder and does not alter the attack.

## Audit requirement

The original failed `job_record.json` and `stderr.log` files must remain preserved. Replayed jobs must write separate replay logs or records, and the final RQ1-v2 report must explicitly state:

- which jobs were replayed;
- which code files changed;
- the updated implementation hashes;
- that replay occurred before aggregate confirmatory analysis;
- that no additional confirmatory target draw or parameter tuning occurred.
