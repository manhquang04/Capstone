# RQ1 Confirmatory Protocol — Gradient-Inversion Resistance

**Protocol ID:** `RQ1-CONFIRMATORY-V1`  
**Version:** `1.2-confirmatory-execution-authorized`  
**Status:** `CONFIRMATORY COMPLETE — SINGLE EXECUTION CONSUMED`  
**Parent plan:** `../../PROJECT.md`  
**Required approval:** research team and supervisor  

## 1. Research question

> Does DNA encoding protect gradients against gradient inversion more
> effectively than Differential Privacy, measured quantitatively by PSNR and
> SSIM of the reconstructed data?

This protocol evaluates `DNA Transform`, not merely the bit-exact lossless DNA
transport. The comparison called “DP” in the research question is implemented
as DP-style clipping/noise. It must not be described as formal differential
privacy unless a valid privacy accountant and all required assumptions are
added in a separately approved amendment.

## 2. Confirmatory scope

The primary scope is the already validated bounded local-update setting:

```text
dataset: PaySim-derived fraud dataset
records per target group: 4
fraud records per group: 1
local optimizer: Adam
local steps: 1
attacker architecture knowledge: known architecture
oracle class-decomposed labels: unavailable
```

The attacker observes the transmitted update for each method and knows the
model, preprocessing, optimizer family/configuration and defense algorithm. For
stochastic defenses it knows the distribution, but not the realized defense
seed/noise. Claims do not extend to a full client, full local epoch or complete
FedAvg round.

## 3. Methods and frozen comparator identities

Every confirmatory target group is evaluated under the same pre-local
checkpoint and local data realization for:

1. `RAW`: unprotected bounded local update.
2. `DNA_TRANSFORM`: one configuration frozen after development calibration.
3. `DP_DISTORTION_MATCHED`: clipping/noise configuration matched to DNA by
   relative-L2 update distortion; primary DP-style comparator.
4. `DP_UTILITY_MATCHED`: clipping/noise configuration matched to DNA by
   utility degradation; secondary comparator if successfully calibrated.

`DP_DISTORTION_MATCHED` and `DP_UTILITY_MATCHED` must have different config IDs
even if calibration happens to choose identical numerical parameters. Lossless
DNA may be reported as a transport sanity check, but is not a primary privacy
branch because decode is bit-exact before aggregation.

## 4. Outcomes and hypotheses

### 4.1 Primary outcome

The primary reconstruction outcome is normalized feature-MSE. For target group
`i`:

```text
Delta_DNA_i = MSE_DNA_i - MSE_RAW_i
Delta_DP_i  = MSE_DP_i  - MSE_RAW_i
D_i         = MSE_DNA_i - MSE_DP_i
```

Larger MSE means worse reconstruction and therefore more resistance. The
primary research contrast is `D_i`: DNA Transform versus
`DP_DISTORTION_MATCHED`.

```text
H0: P(D_i > tie_threshold | non-tie) <= 0.50
H1: P(D_i > tie_threshold | non-tie) > 0.50
```

### 4.2 Secondary outcomes

- DNA Transform versus raw: `Delta_DNA_i`.
- Each DP-style comparator versus raw: `Delta_DP_i`.
- DNA Transform versus `DP_UTILITY_MATCHED`, if calibration succeeds.
- PSNR and SSIM on frozen deterministic pseudo-images.
- Original-unit and per-feature MSE/MAE, cosine similarity, Pearson
  correlation, sign match, transaction-type accuracy and plausibility errors.

Lower PSNR and lower SSIM mean worse reconstruction. PSNR/SSIM are required by
the wording of RQ1 but are secondary because PaySim is tabular data.

### 4.3 Values that must be approved before `PRE-PILOT FROZEN`

```yaml
primary_metric: normalized_feature_mse
primary_contrast: dna_transform_vs_dp_distortion_matched
hypothesis_direction: greater
alpha: 0.05
target_power: 0.80
null_win_probability: 0.50
minimum_meaningful_win_probability: 0.70
tie_threshold_mse: 0.0390625
measured_development_tie_rate: 0.0  # 0/8 groups
expected_technical_dropout_rate: 0.05
multiple_comparison_rule: holm_for_secondary_family
maximum_feasible_target_count: 150
required_confirmatory_target_count: 39
planning_target_count_at_10pct_ties: 44
```

These values must be justified scientifically and by compute constraints, not
selected from confirmatory outcomes.

## 5. Data firewall

### 5.1 Development pool

Development targets or synthetic fixtures may be used to audit artifacts,
develop attackers, test gates, calibrate distortion, estimate runtime and lock
the attack budget. Their row IDs and checksums must be registered.

### 5.2 Post-hoc pool

`level1_fresh_final_targets.pt` and every previously reported target set are
post-hoc. They may only produce outputs labeled
`POST_HOC_SENSITIVITY_ONLY`. They must not be used to:

- choose or extend a defense grid;
- select DNA/DP parameters;
- choose attack iterations, restarts or candidate-selection rules;
- select metrics, tie thresholds, normalization or pseudo-image layout;
- determine whether the confirmatory experiment will be run.

Post-hoc targets are not members of the development pool.

### 5.3 Confirmatory target set

After this protocol reaches `CONFIRMATORY FROZEN`, draw one new
target set that is source-row-disjoint from all registered development,
post-hoc and historical evaluation targets. Save:

- dataset and preprocessing checksums;
- source row IDs and class/type composition;
- generation seed and exact sampling algorithm;
- overlap matrix against every known target pool;
- target-set checksum and generation log.

No redraw is permitted because outcomes are weak or inconvenient.

## 6. Sample-size determination

RQ1 uses exact one-sided sign-test/binomial power analysis, not a continuous
variance formula. Before target generation:

1. Freeze null win probability `p0 = 0.50` unless an approved justification
   changes it.
2. Freeze the minimum scientifically meaningful win probability `p1`.
3. Freeze alpha, target power and the handling of ties.
4. Compute the minimum effective non-tied count using the exact binomial
   rejection region.
5. Inflate the draw count for the predeclared tie and technical-dropout rates.
6. Publish code and a sensitivity table over plausible tie/dropout rates.

Historical near-null results may inform tie/dropout feasibility but must not be
used as the desired effect. If the required count exceeds the frozen compute
ceiling, reduce the claim or report the design as underpowered; do not alter
alpha, power or `p1` after viewing outcomes.

## 7. Development calibration after pre-pilot freeze

### 7.1 DNA Transform

Freeze one DNA configuration using development-only evidence. Record block
size, mix ratio, keep ratio, shrink factor and seed derivation. Selection must
not use post-hoc or confirmatory privacy outcomes.

### 7.2 `DP_DISTORTION_MATCHED`

1. Freeze clipping rule, multiplier grid, calibration groups/seeds, number of
   noise realizations, matching statistic and tolerance.
2. Compute DNA relative-L2 distortion on the same development groups.
3. Run every DP-style candidate on every calibration group.
4. Choose the candidate minimizing absolute distance to the frozen DNA target,
   provided it lies within tolerance.
5. Resolve ties using a predeclared numerical rule, never privacy outcome.
6. Save the full grid and all per-group and aggregate relative-L2 values.

If no candidate matches, a dated amendment may expand the grid before target
generation. The amendment must retain and report the original grid.

### 7.3 `DP_UTILITY_MATCHED`

Use the single shared 5–8-seed development batch defined in the RQ2 protocol.
Freeze the candidate grid, distance function, tolerances and tie-breaker before
running it. This batch is also used to estimate paired RQ2 variance but may
never enter the RQ2 confirmatory seed set.

If no candidate satisfies the frozen utility rule, omit the utility-matched
contrast and present privacy/utility as a Pareto analysis. Do not relabel the
nearest candidate as utility-matched.

## 8. Attack branches

All branches use the frozen attack budget and candidate-selection rule. Restart
is optimization budget, not an independent statistical observation.

- `RAW`: hard balance-difference reparameterization already validated.
- `DNA_TRANSFORM`: surrogate-forward adaptive attacker; knows transform
  algorithm/distribution but not the exact realization.
- DP-style branches: Monte Carlo expectation attacker; knows clipping and noise
  distribution but not realized noise. It must not insert realized target noise
  into candidate updates.
- Controls: prior-only reconstruction and zero-update reconstruction, generated
  according to the same frozen artifact and selection contract.

The selected candidate must minimize an attacker-visible objective. Ground
truth reconstruction error must never choose a restart, step or candidate.

## 9. Branch-specific validity gates

Each attack branch must beat both its prior and zero-update controls under the
frozen gate rule. The exact gate statistic, direction, tie rule and threshold
must be placed in the machine-readable pre-pilot config before
`PRE-PILOT FROZEN`.

- Raw + DNA pass: interpret DNA versus raw.
- Raw + a DP branch pass: interpret that DP branch versus raw.
- DNA + a DP branch pass: interpret DNA versus that DP branch on common targets.
- Failure of one branch does not invalidate an otherwise valid contrast.
- Attack failure is not evidence of privacy.

## 10. Predeclared branch-failure procedure

If raw or a defense branch fails its confirmatory gate:

1. Withhold every contrast requiring that branch.
2. Audit implementation, artifact integrity, numerical stability and control
   construction without using privacy direction to guide changes.
3. If a demonstrable technical error exists, issue an amendment, retain the
   failed run and rerun only the affected scope with otherwise identical rules.
4. If no technical error exists, allow exactly one new source-disjoint
   robustness-check target set using the same sampling contract, target count,
   configs, budget, metrics and gates.
5. Never replace the primary set with the robustness set. Report all gates,
   outcomes and exclusions for both, explicitly labeled `CONFIRMATORY` and
   `ROBUSTNESS_CHECK`.

No second robustness set is permitted.

## 11. Artifact and metric contract

For every target, method, restart and noise realization, store enough data to
recompute results without rerunning the attacker:

- normalized ground truth and selected reconstruction;
- original-unit vectors where inverse transformation is defined;
- feature order, normalization metadata and pseudo-image layout/version;
- labels/types used only for evaluation;
- source row, checkpoint, observed-update and defended-update hashes;
- target, training, defense, noise, restart and selection seeds;
- full loss history, selected step and attacker-visible objective;
- control artifacts, environment manifest and command line.

PSNR/SSIM must be computed from retained vectors, not inferred from aggregate
MSE. The pseudo-image builder must be deterministic and identical across
methods. Reload tests must reproduce objectives and metrics within frozen
tolerances before statistical analysis.

## 12. Statistical reporting

The target group is the independent unit. For each permitted contrast report:

- target count, missing jobs, ties and effective non-tied count;
- every paired difference, wins/losses, mean, median, SD and IQR;
- exact sign-test p-value and confidence interval for win probability;
- confidence interval for the paired metric difference using the predeclared
  method;
- feature-MSE, PSNR and SSIM without cherry-picking among them;
- branch-gate table and sensitivity analyses clearly separated from primary
  confirmatory results.

A nonsignificant result means no detected difference under this protocol and
sample size. It does not prove equivalence.

## 13. Execution order and stop rules

1. Resolve scientific thresholds, freeze compute ceiling, grids, seeds, gates,
   metrics and budgets in the pre-pilot config.
2. Approve/hash protocol and pre-pilot config; set `PRE-PILOT FROZEN`.
3. Complete development audit, smoke tests and comparator calibration.
4. Compute `required_confirmatory_target_count` and compare it with the frozen
   `maximum_feasible_target_count`.
5. Freeze DNA/DP comparators, target count and all final rules in a separate
   confirmatory config; approve/hash it and set `CONFIRMATORY FROZEN`.
6. Generate and verify the one confirmatory target set. This step was completed
   once on 2026-09-13; its checksum and overlap matrix are frozen in
   `config/rq1_confirmatory.yaml`.
7. Run raw and controls, then DNA and DP-style branches and their controls.
8. Verify artifacts, apply gates, compute permitted contrasts and report all.

Stop immediately for checksum mismatch, target overlap, missing frozen config,
candidate selection using ground truth, or irreproducible artifact reload. Do
not raise budgets or redraw targets because a p-value is near a threshold.

## 14. Required outputs

```text
protocols/rq1_confirmatory_protocol.md
protocols/config/rq1_pre_pilot.yaml
protocols/config/rq1_confirmatory.yaml
artifacts/rq1/development_audit/
artifacts/rq1/confirmatory_<run_id>/
results/rq1/rq1_paired_metrics.csv
results/rq1/rq1_summary.json
reports/rq1_report.md
```

## 15. Two-gate freeze checklist

### 15.1 `PRE-PILOT FROZEN`

- [x] All pre-pilot scientific values resolved and justified; the numerical tie
  threshold is explicitly deferred to the approved Stage 1 study.
- [x] Development/post-hoc registries and source-disjoint confirmatory rule frozen.
- [x] DNA and comparator grids/selection rules frozen.
- [x] Alpha, power, meaningful win probability and tie/dropout rules frozen.
- [x] `maximum_feasible_target_count` approved from resource constraints.
- [x] Pre-pilot config and exact RQ1 power-analysis code hashed.
- [x] Attack budgets, selectors, gates and controls machine-readable.
- [x] Pseudo-image builder and feature normalization hashed.
- [x] Robustness-check rule accepted before outcomes are seen.
- [x] Team and supervisor pre-pilot approvals recorded below.

### 15.2 `CONFIRMATORY FROZEN`

- [x] Development audit/calibration outputs archived; every Stage-1 technical
  change/grid expansion is retained in dated amendments.
- [x] `required_confirmatory_target_count` computed and compared with the
  previously frozen feasible ceiling.
- [x] DNA and DP comparator identities/configurations selected by frozen rules.
- [x] Exact target count, target-generation rule and final analysis plan frozen.
- [x] Confirmatory config plus code/protocol/environment hashes recorded in the
  confirmatory-freeze manifest.
- [x] Team and supervisor confirmatory-freeze approvals recorded below;
  execution authorization was subsequently granted and recorded in the dated
  multiplicity/execution amendment.

## 16. Approval record

| Gate | Role | Name | Decision | Timestamp | Signature/reference |
| --- | --- | --- | --- | --- | --- |
| Pre-pilot | Research lead | Manh Quang | Approved | 2026-09-12 | Chat session approval; decision package v1.0 |
| Pre-pilot | Supervisor | Manh Quang | Approved | 2026-09-12 | Chat session approval; decision package v1.0 |
| Confirmatory freeze | Research lead | Manh Quang | Approved | 2026-09-13 | Chat approval; amendment `2026-09-13_stage1_supervisor_approvals.md` |
| Confirmatory execution | Supervisor | Manh Quang | Approved | 2026-09-13 | Single execution authorization; multiplicity/execution amendment |

`DRAFT` authorizes writing only. `PRE-PILOT FROZEN` authorizes Stage 1
development/audit/calibration only. This confirmatory freeze authorized the
single target draw requested by the supervisor; the one confirmatory attack
execution was subsequently authorized before outcomes were generated.
