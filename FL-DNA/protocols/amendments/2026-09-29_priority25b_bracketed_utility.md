# Amendment — Priority 25b bracketed utility-matched DP and T2 gate attempt

Date: 2026-09-29  
Status: FROZEN / AUTHORIZED FOR PRIORITY 25b EXECUTION  
Applies to: original RQ1 utility-matched DP arm and gradient-only instrument gap  
No earlier Priority 24/25 artifacts are modified by this amendment.

## Motivation

Priority 25's utility-matched DP grid did not satisfy the intended bracketing requirement. All four grid points `[1e-6, 3e-6, 1e-5, 3e-5]` were utility-neutral on n=8, and v2 selected the top edge of the grid. The original Priority 25 prompt required the grid to plausibly span mean ΔF1 from approximately 0 down to approximately -0.05. The next frontier is expected between `3e-5` and roughly `3e-4`.

Priority 25 also did not execute a concrete T2 gradient-only gate attempt. Priority 25b adds exactly one bounded T2 attempt plus one pre-declared retry.

## Pre-declared handling

The following handling is frozen before any Priority 25b data are generated:

> The Priority 25 T1 × utility-matched results stay reported, labeled "comparator failed the bracketing requirement (selected σ at the grid edge / inside the utility-neutral region)".
> In the combined RQ1 primary family, they are REPLACED by the Priority 25b T1 × utility-matched tests.
> Priority 24 (distortion-matched) is unchanged.

## Part 1 — bracketed utility grid

Mechanism:

- DP implementation: `privacy/dp_engine.py::apply_dp_to_local_state`
- Clip norm: `C = 259.0841131896973`, unchanged from Priority 25.
- DP applies to the whole floating transmitted client update, including BatchNorm buffers, matching the RQ2/utility protocol.
- Training contract: PaySim, 500k rows, 50 rounds, 3 clients, 1 local epoch, batch 1024, Adam, focal loss, paired seed contract exactly as Priority 25.

Grid:

- Existing grid: `[1e-6, 3e-6, 1e-5, 3e-5]`
- New appended grid: `[1e-4, 3e-4, 1e-3, 3e-3]`
- Full initial grid: `[1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]`

Replicates:

- Reuse Priority 25 utility seeds `[2501101, 2501102, 2501103, 2501104, 2501105, 2501106, 2501107, 2501108]`.
- For those eight seeds, reuse existing baseline and existing four old DP grid-point metrics; run only the four new grid points.
- Add eight new utility seeds `[2501201, 2501202, 2501203, 2501204, 2501205, 2501206, 2501207, 2501208]`, disjoint from known RQ2/Priority 25 seeds.
- For the eight new seeds, run baseline plus all eight grid points.

Matching rule:

- v1 conservative target threshold: `RQ2 mean ΔF1 +0.0014 - 0.005 = -0.0036`.
- v2 target threshold: `RQ2 mean ΔF1 -0.0117 - 0.005 = -0.0167`.
- `σ_T` is the largest grid `σ` whose 16-seed mean ΔF1 is greater than or equal to the transform-specific threshold.

Bracketing requirement:

- `σ_T` must not be the largest grid point.
- The next-larger grid point must have mean ΔF1 below the threshold.

If the initial 8-point grid does not bracket a transform, add exactly one pre-declared extension `[1e-2, 3e-2]`, run it on all 16 seeds, and re-apply the same matching/bracketing rule. If still unbracketed, report `NOT_BRACKETED` and do not run the corresponding T1 utility confirmatory test.

Report ΔF1, ΔAUC, ΔPR-AUC mean and SD for every grid point, plus RDP ε at δ=1e-5 for selected σ values.

## Part 2 — T1 × utility-matched DP confirmatory

Only run a transform's T1 utility confirmatory if its utility grid is bracketed.

Development gate:

- Fresh n=8 source-disjoint targets.
- T1 must beat both Prior and decoy with exact one-sided sign test at α=0.05.

Confirmatory:

- Fresh n=39 source-disjoint targets.
- Branches: none, v1 conservative, v2 decoded with seed, DP at `σ_v1`, DP at `σ_v2`.
- DP uses the same clip norm `C = 259.0841131896973` and full floating transmitted 4-record update.
- T1 reads the BN running_mean channel.
- Test DNA-vs-DP utility-matched in both directions via exact one-sided sign test. Tie band comes from deterministic development replay; if replay is exact, tie band is 0.
- Report every branch vs none, vs Prior, and vs decoy.

Interpretation note frozen from Priority 26:

- T1 cannot read the v2 path even with the key in the Priority 26 v2-SB check (`1/8` vs Prior). In this Priority 25b v2 comparison, T1's v2 result measures v2's lossy compression of the BN statistic, not key secrecy. Report v2-with-key vs Prior alongside.

## Part 3 — T2 gradient-only gate attempt

Instrument:

- T2 is a concrete TabLeak-style attacker on the trainable-parameter update only; it does not use BatchNorm buffers.
- Local step: one Adam local step from fresh optimizer state, simulated exactly as the protocol capture path.
- Objective: cosine loss on the defended candidate trainable update vs observed trainable update.
- Input relaxation: transaction type represented by a differentiable softmax relaxation over the five one-hot type features; continuous numeric features are optimized directly with a nonnegative/range penalty.
- Labels: known labels.
- Restarts: 8.
- Steps: 2,000 Adam steps.
- Attacker learning rate: `0.05`.
- Candidate selection: attacker-observable objective only.
- Reconstruction score: standardized input-space MSE with Hungarian alignment where needed.
- Ensemble/pooling: report the median-MSE restart and the coordinate median pooled reconstruction descriptively; gate uses the attacker-observable selected candidate.

Gate:

- Fresh n=8 targets.
- Undefended branch must beat both Prior and decoy via exact one-sided sign test at α=0.05.

Pre-declared retry:

- If the first n=8 gate fails, exactly one retry is allowed on a fresh n=8 target set with:
  - steps: 8,000
  - restarts: 16
  - same objective and attacker lr unless numerical instability occurs before scientific output; if numerical instability occurs, abort and report rather than tune.

If T2 passes either gate, stop and report `T2 passed its gate; confirmatory pending`. No T2 confirmatory test is authorized in Priority 25b.

Expected outcome note:

- Priority 25 diagnostic found one Adam step is almost a sign update: mean cosine with `-lr·sign(g)` was 0.992 and mean corr(|step|, |g|) was 0.047. T2 failure is therefore expected and remains a valid result.

## Step 4 — combined answer rule

Primary family:

- Priority 24 T1 × distortion-matched tests (2 fixed tests);
- Priority 25b T1 × utility-matched tests (2 tests, only if bracketed).

Apply Holm separately for DNA>DP and for DP>DNA.

Answer rule:

- "Yes": at least one DNA>DP test is significant, and no DP>DNA test is significant.
- "No": otherwise.
- If the grid is not bracketed, the utility-matched arm is "not available", and the answer rests on Priority 24 alone.

Gradient-only statement:

- "T2 attempted; gate failed (n=8, with one retry)", or
- "T2 passed its gate; confirmatory pending".

## Execution constraints

- No edits under `Latex/`.
- Do not modify earlier artifacts or reports.
- No early stopping.
- No change to the rule, grid, C, T2 budget, or answer rule after seeing Priority 25b data.
- Parallel worker processes are allowed; each process must keep `torch.set_num_threads(1)` and environment thread caps consistent with prior RQ2 runs.
