# Group 3 — Conservative replication report

**Execution date:** 2026-09-13; amended 2026-09-14  
**Status:** `COMPLETE`  
**Scope:** two independently frozen conservative replications for RQ1 and RQ2,
plus the final supervisor-authorized RQ1 conservative replication 3 and
post-hoc pooled analysis.

## 1. Đã làm gì

1. Froze two source-disjoint RQ1 target sets before observing either
   replication outcome.
2. Ran RQ1 conservative Level-1 attacks once for each replication: raw, DNA
   Level-1 and distortion-matched clipping/noise-MC.
3. Ran two independent RQ2 multi-seed replications, each with 21 new seeds and
   the frozen per-seed contract. Methods were baseline, DNA lossless, DNA
   Transform and distortion-matched DP contextual branch.
4. Analyzed all runs with the pre-existing analyzers and frozen decision rules.
5. On 2026-09-14, after supervisor approval, froze and executed exactly one
   final RQ1 conservative draw, replication 3, with the same design as
   replications 1/2. No further draws are permitted for this comparison.
6. Ran the pre-amended pooled analysis across original Group 2 conservative
   plus Group 3 replications 1/2/3.

## 2. File/config thay đổi

- `protocols/amendments/2026-09-13_group3_replication_1.md`
- `protocols/amendments/2026-09-13_group3_replication_2.md`
- `protocols/config/rq1_group3_replication_1.json`
- `protocols/config/rq1_group3_replication_2.json`
- `protocols/config/rq2_group3_replication_1.json`
- `protocols/config/rq2_group3_replication_2.json`
- `experiments/materialize_rq2_replication_contract.py`
- `experiments/run_rq2_replication.py`
- `experiments/analyze_rq2_replication.py`
- `protocols/amendments/2026-09-14_group3_replication_3_and_pooled_analysis.md`
- `protocols/config/rq1_group3_replication_3.json`
- `experiments/analyze_rq1_conservative_pooled.py`

## 3. Lệnh thực sự đã chạy

```bash
.venv-phase1/bin/python experiments/run_rq1_variant_confirmatory.py \
  --config protocols/config/rq1_group3_replication_1.json \
  --output-dir artifacts/rq1/group3_replication_1_run_20260913 \
  --workers 9
.venv-phase1/bin/python experiments/analyze_rq1_variant_confirmatory.py \
  --config protocols/config/rq1_group3_replication_1.json \
  --run-dir artifacts/rq1/group3_replication_1_run_20260913 \
  --output-dir results/rq1/group3_replication_1_20260913

.venv-phase1/bin/python experiments/run_rq1_variant_confirmatory.py \
  --config protocols/config/rq1_group3_replication_2.json \
  --output-dir artifacts/rq1/group3_replication_2_run_20260913 \
  --workers 9
.venv-phase1/bin/python experiments/analyze_rq1_variant_confirmatory.py \
  --config protocols/config/rq1_group3_replication_2.json \
  --run-dir artifacts/rq1/group3_replication_2_run_20260913 \
  --output-dir results/rq1/group3_replication_2_20260913

.venv-phase1/bin/python experiments/run_rq2_replication.py \
  --config protocols/config/rq2_group3_replication_1.json \
  --output-dir artifacts/rq2/group3_replication_1_run_20260913 \
  --workers 9
.venv-phase1/bin/python experiments/analyze_rq2_replication.py \
  --config protocols/config/rq2_group3_replication_1.json \
  --run-dir artifacts/rq2/group3_replication_1_run_20260913 \
  --output-dir results/rq2/group3_replication_1_20260913

.venv-phase1/bin/python experiments/run_rq2_replication.py \
  --config protocols/config/rq2_group3_replication_2.json \
  --output-dir artifacts/rq2/group3_replication_2_run_20260913 \
  --workers 9
.venv-phase1/bin/python experiments/analyze_rq2_replication.py \
  --config protocols/config/rq2_group3_replication_2.json \
  --run-dir artifacts/rq2/group3_replication_2_run_20260913 \
  --output-dir results/rq2/group3_replication_2_20260913

PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/rq1/group3_replication_3_freeze_20260914 \
  --output-name rq1_replication_3_targets.pt \
  --groups 39 --records-per-group 4 --fraud-per-group 1 \
  --seed 655946849 \
  --purpose rq1_group3_conservative_replication_3_final_draw
PYTHONPATH=. .venv-phase1/bin/python experiments/verify_rq1_target_disjointness.py \
  --confirmatory-target artifacts/rq1/group3_replication_3_freeze_20260914/rq1_replication_3_targets.pt \
  --output artifacts/rq1/group3_replication_3_freeze_20260914/source_overlap_matrix.json
.venv-phase1/bin/python experiments/run_rq1_variant_confirmatory.py \
  --config protocols/config/rq1_group3_replication_3.json \
  --output-dir artifacts/rq1/group3_replication_3_run_20260914 \
  --workers 9
.venv-phase1/bin/python experiments/analyze_rq1_variant_confirmatory.py \
  --config protocols/config/rq1_group3_replication_3.json \
  --run-dir artifacts/rq1/group3_replication_3_run_20260914 \
  --output-dir results/rq1/group3_replication_3_20260914
.venv-phase1/bin/python experiments/analyze_rq1_conservative_pooled.py \
  --output-dir results/rq1/conservative_pooled_20260914
```

## 4. Kết quả kèm uncertainty

### RQ1 conservative replications

Primary win means `feature-MSE(DNA) > feature-MSE(DP)` outside the frozen tie
threshold. Both replications used `n=39` targets and had zero ties.

| Replication | Raw/DNA/DP gates | DNA wins / non-tied | win probability (exact 95% CI) | one-sided p | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Replication 1 | PASS / PASS / PASS | 26/39 | 0.667 [0.498, 0.809] | 0.02663 | Significant DNA advantage |
| Replication 2 | PASS / PASS / PASS | 26/39 | 0.667 [0.498, 0.809] | 0.02663 | Significant DNA advantage |
| Replication 3 | PASS / PASS / PASS | 25/39 | 0.641 [0.472, 0.788] | 0.05406 | No significant DNA advantage |

Descriptive mean DNA-minus-DP feature-MSE was `-272.06` with 95% t-CI
`[-523.19, -20.94]` for replication 1 and `-375.00` with 95% t-CI
`[-890.86, 140.86]` for replication 2. These descriptive magnitude summaries
do not override the frozen sign-test decision.

Secondary DNA-minus-DP image-style metric CIs did not establish a stable
direction: replication 1 PSNR `[-0.823, 0.666]`, SSIM `[-0.056, 0.075]`;
replication 2 PSNR `[-0.494, 1.085]`, SSIM `[-0.030, 0.094]`; replication 3
PSNR `[-0.902, 1.289]`, SSIM `[-0.045, 0.096]`.

### RQ1 conservative pooled analysis

This pooled analysis was pre-amended on 2026-09-14 but remains
exploratory/post-hoc relative to the original RQ1 protocol. It does not replace
the decision of any individual confirmatory run and does not reverse the
original Group 2 conservative conclusion.

| Batch | Wins / non-tied | one-sided p | Individual decision |
| --- | ---: | ---: | --- |
| Original Group 2 conservative | 19/39 | 0.62537 | Not rejected |
| Group 3 replication 1 | 26/39 | 0.02663 | Rejected |
| Group 3 replication 2 | 26/39 | 0.02663 | Rejected |
| Group 3 replication 3 | 25/39 | 0.05406 | Not rejected |

Pooled over the four batches, DNA won `96/156` non-tied comparisons, with
win probability `0.615` and exact 95% CI `[0.534, 0.692]`. The exact one-sided
sign-test p-value against `p0=0.50` was `0.002456`, so the pooled exploratory
summary is statistically above chance. It does not meet the pre-specified
practical-effect reference `p1=0.70`, because the point estimate is below
`0.70` and the 95% CI upper bound is `0.692`.

The heterogeneity check across the 4x2 wins/losses table gave chi-square
`3.683` with `df=3`, `p=0.2978`; the minimum expected cell count was `15.0`.
This does not show strong evidence of between-batch heterogeneity, but the
batch-level pattern remains practically unstable: one original negative run,
two significant positive replications, and one borderline positive replication.

### RQ2 conservative replications

Primary rule: DNA Transform vs baseline must establish non-inferiority for
both F1 and AUC. The F1 margin is `0.02`; the AUC margin is `0.005`.

| Replication | F1 delta mean (95% CI) | AUC delta mean (95% CI) | Primary decision |
| --- | ---: | ---: | --- |
| Replication 1 | -0.00651 [-0.01622, 0.00319] | -0.000285 [-0.000939, 0.000368] | PASS |
| Replication 2 | -0.000513 [-0.01082, 0.00979] | -0.000223 [-0.000907, 0.000460] | PASS |

The contextual distortion-matched DP branch showed larger F1 loss in both
replications: replication 1 mean F1 delta `-0.07416` with 95% CI
`[-0.08604, -0.06228]`; replication 2 mean F1 delta `-0.05722` with 95% CI
`[-0.07221, -0.04224]`.

## 5. Gate đạt/chưa đạt

| Gate | Status | Reason |
| --- | --- | --- |
| RQ1 source disjointness | PASS | All three replication target sets were frozen separately and checked for zero overlap |
| RQ1 execution completeness | PASS | 351/351 replication attack jobs successful across three RQ1 replications |
| RQ1 branch validity | PASS | Raw, DNA and DP each beat Prior and Zero controls in all three RQ1 replications |
| RQ1 primary hypothesis | MIXED | Replications 1/2 rejected; replication 3 did not reject at alpha 0.05 |
| RQ1 pooled exploratory test | STATISTICALLY POSITIVE / PRACTICALLY BELOW P1 | 96/156, p=0.002456, CI upper bound 0.692 < 0.70 |
| RQ2 execution completeness | PASS | 168/168 training jobs successful across two replications |
| RQ2 primary non-inferiority | PASS in both replications | F1 and AUC CI lower bounds were above the frozen margins |

## 6. Artifact/run IDs

- `artifacts/rq1/group3_replication_1_run_20260913/`
- `results/rq1/group3_replication_1_20260913/`
- `artifacts/rq1/group3_replication_2_run_20260913/`
- `results/rq1/group3_replication_2_20260913/`
- `artifacts/rq1/group3_replication_3_freeze_20260914/`
- `artifacts/rq1/group3_replication_3_run_20260914/`
- `results/rq1/group3_replication_3_20260914/`
- `results/rq1/conservative_pooled_20260914/`
- `artifacts/rq2/group3_replication_1_run_20260913/`
- `results/rq2/group3_replication_1_20260913/`
- `artifacts/rq2/group3_replication_2_run_20260913/`
- `results/rq2/group3_replication_2_20260913/`

## 7. Amendment

The two original replication amendments and the RQ1/RQ2 config files were
created before observing replication outcomes. The replication 3 / pooled
analysis amendment was created on 2026-09-14 before target generation and
before observing replication 3. No parameter was tuned after seeing results.
Optional FL-configuration variation was not started in this group report; the
two full independent replications were prioritized before the later supervisor
decision to add exactly one final RQ1 conservative draw.

## 8. Bước tiếp theo được phép

Group 3 plus the final RQ1 conservative replication 3 is complete and reported.
No further RQ1 conservative target draw is permitted for this comparison.
Group 4 may begin only under a pre-run, time-boxed attacker-improvement
amendment and only after preserving the same constraints: no post-hoc targets,
no `torch.set_num_threads(1)` changes, and no parameter tuning after outcomes.

## 9. Post-report wording note from Priority 3 DP accounting

Priority 3 DP accounting, completed on 2026-09-16, found that the
distortion-matched clipping/noise comparator used in these RQ1 replications has
finite but extremely large epsilon under the approved update-level Gaussian/RDP
accountant.  Therefore these comparisons should be described as DNA Transform
versus `DP-style full-client-update clipping/noise with weak formal accounting`
at comparable update distortion.  They should not be described as comparisons
against strong record-level Differential Privacy.  This note does not alter any
historical result, p-value, confidence interval or stop rule in this report.
