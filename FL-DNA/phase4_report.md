# Phase 4 Report

## Status

**Current state: PHASE 4 CLOSED FOR THIS SCOPE. Decision: STOP_OR_REDESIGN_DNA_FOR_THIS_SCOPE.**

Phase 4 requires a baseline adaptive attack that first beats both paired
controls in the exact scope being evaluated. The first implementation step was
therefore a fraud-focused baseline gate. After that gate and Gate B passed on a
fresh target set, the first DNA-specific check was the attacker-favorable
Level 2 direct inversion case. The phase then ran a Level 1 (realization-unknown)
paired comparison against the raw update, four same-scope simple-defense
controls, and two narrow checks of a weaker (architecture-uncertain) threat
model. See `## Decision` and `## Conclusion` at the end of this report for the
final answer to RQ1/RQ2 in this bounded scope.

## Protocol Used

The runner is:

```bash
python -m experiments.run_pre_phase4_fraud_baseline_gate
```

Artifact directory:

```text
artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z
```

This run used a bounded Adam local-update scope:

| Field | Value |
| --- | --- |
| Records per group | 4 |
| Fraud records per group | 1 |
| Local optimizer steps | 1 |
| Development groups | 4 |
| Evaluation groups | 12 |
| Restarts per evaluation group | 3 |
| Objective | Balanced per-tensor objective over the full transmitted update |
| DNA used? | No |
| Candidate selection | Attacker objective only |
| Gate target | Fraud records |
| Reported diagnostic class | Non-fraud records |

The runner does not use a class-decomposed oracle gradient. It attacks the full
observed update and reports reconstruction metrics separately by class. The
labels used here are the supervised labels required by the local training
replay; they are not used to split the observed update into oracle class
components.

## Hyperparameter Freezing

Development data were used only to choose the attack budget before held-out
evaluation. The selected attack configuration was:

| Parameter | Value |
| --- | --- |
| Attack learning rate | 0.1 |
| Attack iterations | 600 |
| Restarts | 3 |
| Selected before evaluation | Yes |

Development sweep:

| Iterations | LR | Selection score | Fraud wins vs prior | Fraud mean delta vs prior | Fraud wins vs zero | Fraud mean delta vs zero |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 0.05 | 1.3369 | 4 | -5.0049 | 3 | 1.3369 |
| 300 | 0.10 | -2.5603 | 3 | -7.2974 | 2 | -2.5603 |
| 600 | 0.05 | 10.1388 | 4 | -5.0049 | 3 | 10.1388 |
| 600 | 0.10 | -13.1199 | 3 | -17.5889 | 3 | -13.1199 |

Lower selection score is better because it is the worse of the two fraud-control
mean deltas. The final evaluation used the best development setting only.

## Held-Out Evaluation

Negative deltas mean the baseline attack reconstructed closer to the target than
the control. A gate requires negative mean, negative median, and one-sided
sign-test p < 0.05 against both controls.

| Class | Control | Groups | Wins | Mean delta MSE | Median delta MSE | One-sided p | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Fraud | Prior | 12 | 8 | -30.8661 | -4.9110 | 0.1938 | FAIL |
| Fraud | Zero-update | 12 | 9 | -20.4938 | -1.8196 | 0.0730 | FAIL |
| Non-fraud | Prior | 12 | 7 | -7.3798 | -1.5239 | 0.3872 | FAIL |
| Non-fraud | Zero-update | 12 | 9 | -4.7901 | -1.6389 | 0.0730 | FAIL |

## Interpretation

The fraud rows move in the expected direction: the baseline attack beats both
controls on average and wins most groups. The result is still not strong enough
to open Phase 4 DNA evaluation. The prior comparison is weak statistically
(8/12 wins, p = 0.1938), and the zero-update comparison is close but still above
the locked threshold (9/12 wins, p = 0.0730).

This is a useful negative result. It says the current baseline attack has signal,
but the signal is not stable enough in this scope to serve as the yardstick for
measuring DNA protection. Running DNA now would risk ranking defenses using an
attacker whose own baseline is not yet validated.

## Follow-Up Diagnostic

After the closed gate, a cheap diagnostic was run on the same held-out
`evaluation_targets.pt`:

```bash
python -m experiments.analyze_class_gradient_contribution \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --targets evaluation_targets.pt \
  --modes eval train_shared_bn
```

This checks whether the current scope still contains strong fraud-gradient
signal. It does not run inversion and does not change the gate decision.

| Mode | Groups | Fraud weighted norm > non-fraud | Mean weighted ratio | Median weighted ratio | Mean fraud cosine/full | Mean non-fraud cosine/full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eval | 12 | 8/12 | 22.9814 | 17.9245 | 0.6665 | 0.1640 |
| Train shared BN | 12 | 12/12 | 148.0110 | 121.1114 | 1.0000 | 0.4258 |

The train-shared-BatchNorm mode is the closer match to the attack simulator
because it uses one train-mode forward pass and splits the per-sample loss on
the same graph. Under that mode, the fraud contribution still dominates the
full update: every held-out group has larger fraud weighted norm than non-fraud,
and the fraud contribution is almost collinear with the full gradient.

This explains the gap between the oracle Phase 3 diagnostics and the no-oracle
pre-Phase-4 gate. The signal exists, but the current official attacker optimizes
the whole update jointly and selects candidates by the full-update objective.
It does not use a class-decomposed gradient, a fraud-weighted objective, or an
oracle saying which part of the update came from fraud. The next change should
therefore target candidate selection/objective design on development data, not
blindly increase iterations.

## Candidate Alignment Diagnostic

A second diagnostic reused the saved pre-gate candidates and did not rerun any
attack:

```bash
python -m experiments.analyze_phase4_candidate_alignment \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z
```

It compares the attacker objective against reconstruction MSE for every saved
candidate/restart. This checks whether the lowest-objective candidate is also
the best fraud reconstruction.

| Split | Baseline candidates | Objective-best = fraud-MSE-best | Pearson objective/fraud MSE | Spearman objective/fraud MSE | Mean extra fraud MSE from objective selection |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development | 8 | 4/4 groups | 0.9032 | 0.9049 | 0.0000 |
| Evaluation | 36 | 8/12 groups | 0.5166 | 0.4080 | 36.1490 |

The development split looked clean, but the held-out split did not. In 4 of 12
evaluation groups, the objective-selected restart was not the lowest-fraud-MSE
restart. This is enough to change the gate outcome:

| Selection rule | Prior wins | Prior p | Zero wins | Zero p | Fraud gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Official objective selection | 8/12 | 0.1938 | 9/12 | 0.0730 | FAIL |
| Fraud-MSE oracle selection, diagnostic only | 10/12 | 0.0193 | 10/12 | 0.0193 | PASS |

The oracle row is not a valid attack protocol because it uses ground-truth
reconstruction error to choose candidates. Its value is diagnostic: good fraud
reconstructions exist among the saved candidates, but the current objective
does not select them reliably on held-out groups.

This narrows the next engineering target. A better Phase 4 baseline should not
start by increasing compute. It should first improve the link between the
attacker-visible objective and fraud reconstruction, using only development
data and without using class-separated observed gradients or ground-truth MSE at
evaluation time.

## Misselection Pattern Diagnostic

The four held-out groups where objective selection missed the best fraud
candidate were then compared against the eight correctly selected groups:

```bash
python -m experiments.analyze_phase4_misselection_patterns \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z
```

| Bucket | Groups | Train shared BN median ratio | Eval median ratio | Non-fraud pairwise distance mean | Fraud/non-fraud centroid distance mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Objective selected correct fraud candidate | 8 | 158.5185 | 34.4011 | 39.5350 | 98.7639 |
| Objective selected wrong fraud candidate | 4 | 104.5643 | 1.2430 | 65.0852 | 241.9039 |

The wrong groups still have strong fraud signal in train-shared-BN mode, but the
signal is weaker and less clean than in the correctly selected groups. Their
eval-mode ratio is much lower, and their average fraud/non-fraud centroid
distance is higher. This supports the interpretation that the full-update
objective is noisy around the fraud candidate in some held-out groups rather
than completely blind to it.

The objective margins also show why this is a selection problem:

| Wrong group | Objective-best restart | Fraud-MSE-best restart | Objective margin | Extra fraud MSE |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 1 | -0.0908 | 1.9294 |
| 2 | 0 | 2 | -0.2027 | 398.4545 |
| 6 | 1 | 0 | -0.0328 | 31.3748 |
| 9 | 0 | 1 | -0.0859 | 2.0293 |

Negative objective margin means the official objective genuinely preferred the
wrong candidate, but the margins are small. The restart pattern is not fixed:
the misses are split across different restart pairs. This makes a single bad
restart seed unlikely as the main explanation.

The next attack-development step should therefore test a small
magnitude-weighted or high-signal-tensor objective on development data. That
would still respect the no-class-oracle threat model because it uses only the
observed update magnitude, not ground-truth labels or class-separated observed
gradients. Increasing restarts can be used as a diagnostic, but the current
evidence points first to objective ranking noise.

## Extra Restart Diagnostic on Misselected Groups

To separate objective-ranking failure from insufficient restart coverage, seven
additional restarts were run only for the four already identified misselected
groups:

```bash
python -m experiments.run_phase4_extra_restarts_for_misselected \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z
```

This is a post-hoc diagnostic on held-out groups. It cannot replace the official
pre-gate result, but it helps decide what to fix next.

| Group | Restarts | Objective-best restart | Fraud-MSE-best restart | Objective margin | Extra fraud MSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 10 | 0 | 3 | -0.1639 | 14.0170 |
| 2 | 10 | 0 | 6 | -0.2087 | 457.7741 |
| 6 | 10 | 9 | 0 | -0.1428 | 63.8640 |
| 9 | 10 | 7 | 7 | 0.0000 | 0.0000 |

Extra restarts helped find better fraud reconstructions in the difficult
groups. Group 9 was resolved: restart 7 became both objective-best and
fraud-MSE-best. Groups 0, 2, and 6 still show the same pattern: the official
objective prefers a lower-objective candidate, but a different restart has much
lower fraud MSE.

Across all 12 held-out groups after adding these restarts to the four difficult
groups:

| Selection rule | Prior wins | Prior p | Zero wins | Zero p | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Objective selection, post-hoc extra restarts | 9/12 | 0.0730 | 8/12 | 0.1938 | FAIL |
| Fraud-MSE oracle selection, diagnostic only | 11/12 | 0.0032 | 11/12 | 0.0032 | PASS |

The diagnostic strengthens the conclusion from candidate alignment. More
restarts can expose better fraud reconstructions, but the current
attacker-visible objective still does not rank them reliably. The next official
development run should therefore test objective changes, such as
magnitude-weighted or high-signal-tensor matching, on development groups only
and then freeze the rule before a new held-out gate.

## Candidate Plausibility Diagnostic

Group 2 had the largest post-hoc gap: the objective-selected candidate had
457.7741 higher fraud MSE than the fraud-MSE-best candidate. The two saved
fraud reconstructions were inspected directly first:

```bash
python -m experiments.analyze_phase4_candidate_plausibility \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --group-id 2 \
  --objective-restart 0 \
  --fraud-best-restart 6
```

| Candidate | Amount negative | Amount | Out-of-dataset features | Orig-diff residual | Dest-diff residual | Type max prob | Type entropy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original fraud record | No | 6063656.98 | 0 | 0.26 | 0.00 | 1.0000 | 0.0000 |
| Objective-best candidate | Yes | -1080477.70 | 4 | -30281.77 | 7199115.89 | 0.6461 | 0.9546 |
| Fraud-MSE-best candidate | No | 149367.92 | 2 | -279737.13 | 2348849.64 | 0.4698 | 1.3904 |

The objective-best candidate is clearly implausible: it reconstructs a negative
transaction amount, negative balances, a negative destination balance, and a
large destination balance-difference residual while still having the lowest
gradient objective. The fraud-MSE-best candidate is also imperfect, but it
removes the negative amount and gives a lower fraud reconstruction error.

The same plausibility check was then applied to groups 0, 6, and 9:

| Group | Candidate | Amount negative | Amount | Out-of-dataset features | Orig-diff residual | Dest-diff residual | Type max prob |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Objective-best | Yes | -139832.48 | 3 | -616847.05 | -1550581.49 | 0.9526 |
| 0 | Fraud-MSE-best | Yes | -37924.80 | 3 | 38626.31 | -768252.41 | 0.3603 |
| 6 | Objective-best | Yes | -99590.78 | 3 | 84185.16 | 420138.38 | 0.8116 |
| 6 | Fraud-MSE-best | Yes | -77234.27 | 4 | -612315.05 | -2688290.78 | 0.9737 |
| 9 | Objective/Fraud best | Yes | -225075.15 | 4 | -219713.78 | 1920543.67 | 0.9054 |

The pattern is broader than group 2: all inspected objective-best fraud
reconstructions have negative amounts. However, the fraud-MSE-best candidates
for groups 0, 6, and 9 also remain implausible. This means validity constraints
are not merely a fix for choosing the wrong restart; they are a constraint on
the reconstruction search space itself.

This points to a second issue beyond tensor weighting. The objective can prefer
feature vectors that are good gradient matches but poor transaction records.

## Balance Residual Diagnostic

Before adding balance-difference penalties, the residuals were measured on the
real dataset:

```bash
python -m experiments.analyze_paysim_balance_residuals
```

| Class | Residual | n | Mean abs | Median abs | Max abs | Near-zero rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Non-fraud | `balance_diff_orig` | 6354407 | 0.0 | 0.0 | 0.0 | 1.0 |
| Non-fraud | `balance_diff_dest` | 6354407 | 0.0 | 0.0 | 0.0 | 1.0 |
| Fraud | `balance_diff_orig` | 8213 | 0.0 | 0.0 | 0.0 | 1.0 |
| Fraud | `balance_diff_dest` | 8213 | 0.0 | 0.0 | 0.0 | 1.0 |

In this codebase, these residuals are exactly zero because the two features are
engineered directly from the balance columns. This is stronger than a statistical
prior: it is a deterministic preprocessing identity. The next diagnostic
therefore enforces the two balance-difference features by reparameterization,
not by a tunable penalty.

## Plausible Initialization Diagnostic

The next cheap test changed only the initialization. Numeric dummy features were
sampled uniformly between the global 1st and 99th scaled percentiles, and type
logits were sampled from the global transaction-type prior. The objective,
attack learning rate, iterations, and groups were unchanged:

```bash
python -m experiments.run_phase4_plausible_init_for_misselected \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z
```

| Group | Objective-best restart | Fraud-MSE-best restart | Objective-best fraud MSE | Fraud-best fraud MSE | Extra fraud MSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4 | 6 | 1100.9744 | 264.3344 | 836.6399 |
| 2 | 7 | 4 | 29061.5536 | 28286.9310 | 774.6225 |
| 6 | 2 | 6 | 6205.5602 | 4769.2766 | 1436.2836 |
| 9 | 8 | 0 | 665.5835 | 83.1302 | 582.4533 |

Plausible initialization improved some raw feature plausibility but did not fix
candidate selection. Objective-best still differs from fraud-MSE-best in all
four difficult groups.

| Group | Candidate | Amount negative | Amount | Type max prob | Dest-diff residual |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | Objective-best | No | 212566.02 | 0.3124 | -199476.76 |
| 0 | Fraud-MSE-best | Yes | -612590.05 | 0.8464 | -6464260.26 |
| 2 | Objective-best | No | 193958.71 | 0.8315 | -1371333.89 |
| 2 | Fraud-MSE-best | No | 643001.18 | 0.9667 | 2704764.84 |
| 6 | Objective-best | No | 78240.87 | 0.5103 | 1111797.75 |
| 6 | Fraud-MSE-best | No | 814739.69 | 0.9746 | 448195.20 |
| 9 | Objective-best | Yes | -183582.42 | 0.9804 | -364531.19 |
| 9 | Fraud-MSE-best | No | 24745.09 | 0.9741 | -1708423.24 |

Compared with random-normal initialization, the objective-best candidate no
longer has a negative amount in groups 0, 2, and 6. However, the official
objective still selects the wrong restart in all four groups, and the
reconstructions still violate feature consistency. Initialization helps the
optimization trajectory, but it is not enough. The next development run should
add validity terms inside the optimized loss, not only as post-hoc filtering.

## Hard Balance-Diff Reparameterization

The balance residual diagnostic shows that `balance_diff_orig` and
`balance_diff_dest` should not be optimized as free variables. A new diagnostic
runner changes only this parameterization. The optimized latent vector contains
the six base numeric features and the transaction-type logits. During every
attack step, the two derived features are reconstructed as:

```text
balance_diff_orig = oldbalanceOrg - newbalanceOrig
balance_diff_dest = newbalanceDest - oldbalanceDest
```

The gradient-matching objective, attack learning rate, iterations, and restart
count are unchanged:

```bash
python -m experiments.run_phase4_harddiff_reparam_for_misselected \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --init-mode standard --restarts 10
```

| Group | Objective-best restart | Fraud-MSE-best restart | Objective-best fraud MSE | Fraud-best fraud MSE | Extra fraud MSE | Prior delta | Zero delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 2 | 195.6457 | 8.7446 | 186.9011 | 151.5664 | -505.1292 |
| 2 | 5 | 5 | 13734.9338 | 13734.9338 | 0.0000 | -14012.7566 | -7765.2462 |
| 6 | 4 | 4 | 527.7847 | 527.7847 | 0.0000 | -4808.0087 | -5036.9549 |
| 9 | 7 | 5 | 31.2188 | 14.2197 | 16.9991 | 7.4439 | -48.6554 |

| Variant | Prior wins | Prior p | Zero wins | Zero p | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Extra restarts, original parameterization | 2/4 | 0.6875 | 1/4 | 0.9375 | FAIL |
| Hard balance-diff reparameterization | 2/4 | 0.6875 | 4/4 | 0.0625 | FAIL |

Hard reparameterization fixes two of the four earlier misselections: groups 2
and 6 now have the same objective-best and fraud-MSE-best restart. It also makes
the attack beat zero-update in all four difficult groups. It does not solve the
prior control: groups 0 and 9 still lose to the prior under objective-based
selection.

The residual is effectively removed by construction. The largest raw residual
reported across these groups is below 1.0 transaction unit, caused by float32
scaling/roundoff rather than an independent inconsistent feature.

## Non-Negative Validity Penalty

The remaining impossible values involve raw `amount` and balance columns, which
are non-negative in the PaySim records. Unlike the balance-diff identities, this
is enforced as a soft penalty so it does not replace the gradient-matching
signal. The penalty is applied during optimization, not only when selecting a
final candidate:

```text
loss = balanced_tensor_objective
       + lambda * mean(ReLU(-raw_amount_or_balance / feature_scale)^2)
```

Two small sensitivity points were run on the same four difficult groups:

```bash
python -m experiments.run_phase4_harddiff_reparam_for_misselected \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --init-mode standard --nonnegative-lambda 0.01 --restarts 10

python -m experiments.run_phase4_harddiff_reparam_for_misselected \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --init-mode standard --nonnegative-lambda 0.1 --restarts 10
```

| Variant | Prior wins | Prior mean delta | Prior median delta | Prior p | Zero wins | Zero mean delta | Zero median delta | Zero p | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Hard-diff only | 2/4 | -4665.4388 | -2400.2824 | 0.6875 | 4/4 | -3338.9964 | -2771.0420 | 0.0625 | FAIL |
| Hard-diff + nonneg `lambda=0.01` | 3/4 | -3371.3991 | -2204.5417 | 0.3125 | 4/4 | -2006.2310 | -1734.6536 | 0.0625 | FAIL |
| Hard-diff + nonneg `lambda=0.1` | 3/4 | -2284.2813 | -1841.1569 | 0.3125 | 4/4 | -1606.1996 | -1436.5019 | 0.0625 | FAIL |

At `lambda=0.01`, the objective beats prior in three of four hard groups but
still misselects groups 0, 2, and 9. At `lambda=0.1`, group 9 is resolved and
group 6 remains resolved, but group 0 is only marginally worse than the prior
and group 2 still has a large objective-selection gap. Stronger validity terms
help on these known hard groups, but these runs are diagnostic only. They were
run on groups selected after inspecting the original evaluation set, so their
lambda values cannot be used as official tuning evidence.

## Development-Separated Lambda Selection

To avoid tuning on the evaluation groups, the hard-diff/non-negative attacker
was rerun on `development_targets.pt`. This file contains 4 groups and 16 source
rows, with one fraud record per group. It has no source-row overlap with
`evaluation_targets.pt`.

```text
development_targets.pt: 4 groups, 16 records, overlap with evaluation = 0
evaluation_targets.pt: 12 groups, 48 records
```

The lambda sweep used the same attack budget as above: 10 restarts, 600
iterations, attack learning rate 0.1, standard initialization, hard balance-diff
reparameterization, and objective-based candidate selection. The selection rule
minimizes the worse of the two mean deltas against prior and zero-update; lower
is better. The gate itself still requires both controls.

| Lambda | Prior wins | Prior mean delta | Prior p | Zero wins | Zero mean delta | Zero p | Selection score | Dev gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0 | 2/4 | -524.5452 | 0.6875 | 2/4 | -327.1478 | 0.6875 | -327.1478 | FAIL |
| 0.001 | 3/4 | -621.9747 | 0.3125 | 4/4 | -621.0211 | 0.0625 | -621.0211 | FAIL |
| 0.01 | 2/4 | -543.4585 | 0.6875 | 3/4 | -669.3634 | 0.3125 | -543.4585 | FAIL |
| 0.1 | 4/4 | -435.1611 | 0.0625 | 3/4 | -234.7315 | 0.3125 | -234.7315 | FAIL |
| 1.0 | 3/4 | -412.4316 | 0.3125 | 4/4 | -459.1364 | 0.0625 | -412.4316 | FAIL |
| 10.0 | 3/4 | -359.7623 | 0.3125 | 2/4 | -567.9934 | 0.6875 | -359.7623 | FAIL |

No lambda passes the development gate with only 4 groups. The least bad
development candidate is `lambda=0.001`, because it keeps both prior and
zero-update mean deltas strongly negative and has the lowest selection score.
This lambda was locked before rerunning the 12-group evaluation.

## Revalidated 12-Group Evaluation

The locked configuration was then run once on all 12 groups in
`evaluation_targets.pt`:

```bash
python -m experiments.run_phase4_harddiff_reparam_for_misselected \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --target-file evaluation_targets.pt \
  --init-mode standard \
  --nonnegative-lambda 0.001 \
  --restarts 10
```

| Control | Groups | Wins | Mean delta MSE | Median delta MSE | One-sided p | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prior | 12 | 10/12 | -1789.3559 | -280.6081 | 0.0193 | PASS |
| Zero-update | 12 | 10/12 | -1278.9338 | -305.5484 | 0.0193 | PASS |

This is the first no-oracle fraud-focused baseline gate that passes both
controls for the bounded Phase 4 scope. However, this evaluation set had already
been inspected during the misselection diagnostics. The result is useful as a
revalidation signal, but not clean enough to close Gate A by itself.

The result is also scoped narrowly:
4 records per group, 1 local Adam step, known labels/order/RNG inherited from
Phase 3, objective-based selection, and no class-decomposed oracle update.

## Fresh Single-Shot Final Gate

A fresh target file was created after the attack rule was locked:

```bash
python -m experiments.create_phase4_fresh_targets \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --output-name fresh_final_targets.pt \
  --groups 12 --records-per-group 4 --fraud-per-group 1
```

The fresh set contains 12 groups, 48 total records, and one fraud record per
group. It has zero source-row overlap with both the Phase 4 development and
earlier evaluation targets:

```text
fresh_final_targets.pt: 12 groups, 48 records
overlap with development_targets.pt = 0
overlap with evaluation_targets.pt = 0
fresh target run seed = 685932499
```

The locked attacker was then run once on this fresh set. No parameter was changed
after seeing the result:

```bash
python -m experiments.run_phase4_harddiff_reparam_for_misselected \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --target-file fresh_final_targets.pt \
  --init-mode standard \
  --nonnegative-lambda 0.001 \
  --restarts 10
```

| Control | Groups | Wins | Mean delta MSE | Median delta MSE | One-sided p | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prior | 12 | 10/12 | -158.2181 | -21.4875 | 0.0193 | PASS |
| Zero-update | 12 | 12/12 | -196.7326 | -62.7579 | 0.0002 | PASS |

This fresh single-shot run passes both controls. It confirms that the hard
balance-diff parameterization plus non-negative penalty `lambda=0.001` is not
only fitting the previously inspected evaluation groups.

## Gate B Integrity Check

Gate B was verified on the fresh final run:

```bash
python -m experiments.verify_phase4_gate_b \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --attack-dir harddiff_reparam_fresh_final_standard_nonneg_0p001 \
  --target-file fresh_final_targets.pt
```

| Check | Result |
| --- | ---: |
| Expected method artifacts | 240 |
| Checked method artifacts | 240 |
| Paired baseline/zero initialization checks | 120 |
| Reload objective checks | 240 |
| Max reload objective absolute difference | 0.0 |
| Failed jobs | 0 |

The runner's `capture()` path also asserts that differentiable replay matches
native Adam for every checked group. The integrity report writes the derived
local and initialization seeds for each checked artifact.

## DNA Transform Specification

The DNA Transform implementation was inspected and documented before running any
paired DNA evaluation:

```text
phase4_dna_transform_spec.md
```

The transform requires a caller-provided seed, splits floating updates into
blocks, derives a DNA-dependent block seed from the raw update block, permutes
the block, attenuates low-energy elements, and mixes the result back into the
original update. Non-floating state entries are copied unchanged.

The key mechanics result is that the actual forward transform is input-dependent
and therefore not a single fixed global linear map. If the exact block
realization is known, however, each block becomes a linear map:

```text
y = ((1 - mix_ratio) I + mix_ratio D P) x
```

where `P` is the permutation and `D` is the attenuation diagonal.

A mechanics check on fresh group 0 with the conservative configuration produced:

| Quantity | Value |
| --- | ---: |
| Fixed-realization blocks | 64 |
| Full-rank blocks | 64/64 |
| Max condition number | 1.1905 |
| Median condition number | 1.1905 |
| Mean relative L2 transform delta | 0.1049 |
| Mean cosine similarity | 0.9966 |

This means direct inversion must be considered if the exact transform
realization is exposed. Knowing only the algorithm and base seed is weaker than
knowing the realization because the realization is derived from the unknown raw
update block.

## DNA Level 2 Direct Inversion

The first DNA evaluation was run in the strongest bounded knowledge setting:
the attacker is given the exact per-block transform realization. With that
realization fixed, each block is solved directly as a linear system before
running the existing data-inversion selection check:

```bash
python -m experiments.run_phase4_dna_level2_direct_inversion \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --target-file fresh_final_targets.pt \
  --baseline-attack-dir harddiff_reparam_fresh_final_standard_nonneg_0p001 \
  --mix-ratio 0.08 \
  --keep-ratio 0.88 \
  --shrink-factor 0.45 \
  --block-size 256
```

This is an upper-bound attack condition. It is stronger than merely knowing the
public algorithm and seed, because the current transform derives each block
realization from the unknown raw update block.

| Quantity | Value |
| --- | ---: |
| DNA run seed | 1776371525 |
| Target groups | 12 |
| Floating tensors recovered | 240 |
| Blocks inverted | 768 |
| Full-rank blocks | 768/768 |
| Max condition number | 1.1905 |
| Median condition number | 1.1905 |
| Mean relative L2 recovery error | 3.3163e-08 |
| Max relative L2 recovery error | 8.3623e-08 |
| Max absolute recovery error | 3.8147e-06 |

The recovered update is numerically the same as the raw observed update for
this scope. After replacing the transformed signal with the directly recovered
raw signal, the same locked baseline candidate-selection check remains open:

| Control | Groups | Wins | Mean delta MSE | Median delta MSE | One-sided p | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prior | 12 | 10/12 | -158.2181 | -21.4875 | 0.0193 | PASS |
| Zero-update | 12 | 12/12 | -196.7326 | -62.7579 | 0.0002 | PASS |

The result is straightforward: if the exact DNA Transform realization is
available, the conservative transform does not add measurable protection in
this bounded setting. The transform can be inverted first, and the attacker is
back to the validated baseline update. This does not answer the harder
Level 1 case, where the attacker knows the algorithm and configuration but not
the exact realization.

## DNA Level 1 Seed-Space Diagnostic

The next cheap check inspected whether Level 1 can be reduced to brute-force or
structured seed search before building a new optimization attack. The relevant
code paths are:

```text
privacy/seed_manager.py
dna_encoder/transform_defense.py
experiments/run_fraud_fl_dna_transform.py
```

The run seed is generated from `secrets.randbelow(2**31 - 1) + 1` unless it is
explicitly supplied for reproduction. Client-round seeds are derived with
BLAKE2b. The per-block seed then adds tensor/block offsets plus DNA symbol
counts and a rolling hash over the first 512 DNA symbols of the raw update
block. Thus, even when the public algorithm and hyperparameters are known, the
realization is not determined without either the logged seed and the raw-block
dependent terms, or the realization itself.

A surrogate-realization diagnostic was run with random candidate realizations.
Two earlier rows are kept only as diagnostics:

- oracle realization selection chooses the best surrogate by raw-update error;
- inverse-pool selection compares candidate updates to `M_r^{-1}` decoded
  signals and uses the decoded signal as its own reference, so it is
  self-referential and deprecated.

The corrected attacker-visible diagnostic applies the surrogate transform
forward to each saved candidate update and compares it directly to the
transmitted DNA update.

```bash
python -m experiments.analyze_phase4_dna_level1_surrogate_inversion \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --target-file fresh_final_targets.pt \
  --baseline-attack-dir harddiff_reparam_fresh_final_standard_nonneg_0p001 \
  --candidates 8 \
  --mix-ratio 0.08 \
  --keep-ratio 0.88 \
  --shrink-factor 0.45 \
  --block-size 256 \
  --dna-run-seed 477885591
```

Development diagnostic with 8 candidates/group:

| Selection rule | Groups | Mean relative L2 error | Median relative L2 error | Mean cosine to raw update |
| --- | ---: | ---: | ---: | ---: |
| All surrogate candidates | 32 | 0.0849 | 0.0811 | N/A |
| Oracle realization selection, diagnostic only | 4 | 0.0787 | 0.0748 | 0.9969 |
| Deprecated inverse-pool selection | 4 | 0.0826 | 0.0786 | 0.9966 |
| Forward-transmitted selection | 4 | 0.0855 | 0.0821 | 0.9963 |

Fresh diagnostic with 8 candidates/group:

| Selection rule | Groups | Mean relative L2 error | Median relative L2 error | Mean cosine to raw update |
| --- | ---: | ---: | ---: | ---: |
| All surrogate candidates | 96 | 0.0946 | 0.0915 | N/A |
| Oracle realization selection, diagnostic only | 12 | 0.0883 | 0.0870 | 0.9961 |
| Deprecated inverse-pool selection | 12 | 0.0967 | 0.0957 | 0.9953 |
| Forward-transmitted selection | 12 | 0.0965 | 0.0926 | 0.9954 |

The oracle row is useful only as a ceiling for this surrogate family: it uses
raw-update error to choose the best realization. The corrected attacker-visible
selection rule used here is:

```text
r*, c* = argmin_r,c L_balanced(M_r delta(c), transmitted_update;
                               reference = transmitted_update)
```

Here `r` is a surrogate realization, `M_r` is its fixed block matrix, and `c`
is a saved candidate reconstruction from the existing hard-diff candidate pool.
The rule uses the transmitted update as the external observed signal. It does
not use raw-update recovery error and does not compare a decoded signal to
itself. The pool itself was still not generated by a fresh DNA-aware
optimization run, so this remains a selection-rule diagnostic rather than a full
Level 1 attack.

Candidate-count sensitivity was then run with 32 surrogate realizations per
group. These runs are diagnostic only; the fresh set has already been inspected.

| Target split | Candidates/group | Oracle mean rel L2 | Forward-visible mean rel L2 | Oracle mean cosine | Forward-visible mean cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development | 8 | 0.0787 | 0.0855 | 0.9969 | 0.9963 |
| Development | 32 | 0.0751 | 0.0864 | 0.9972 | 0.9963 |
| Fresh | 8 | 0.0883 | 0.0965 | 0.9961 | 0.9954 |
| Fresh | 32 | 0.0841 | 0.0945 | 0.9964 | 0.9955 |

The corrected diagnostic shows two useful facts. First, wrong/random
realizations do not reproduce the Level 2 result: the error rises from about
`3.3e-08` to roughly `8e-02`-`1e-01`. Second, the conservative transform is mild
enough that even wrong-realization inversions remain highly collinear with the
raw update. Third, increasing the surrogate pool improves the oracle row more
than the forward-visible row. This confirms that surrogate selection remains
the unresolved Level 1 problem. The next Level 1 implementation must optimize or
select against the transmitted update from the start; the oracle and deprecated
inverse-pool numbers must not be used as Level 1 evidence.

## DNA Level 1 Fresh-Optimized Forward Attack

A fresh optimization runner was then added for the Level 1 surrogate-forward
attack. Unlike the pool-selection diagnostic, this runner optimizes dummy data
directly against the DNA-transmitted update:

```text
r*, c* = argmin_r,c L_balanced(M_r delta(c), transmitted_update;
                               reference = transmitted_update)
```

The runner keeps the hard balance-diff parameterization and non-negative
validity penalty from the validated baseline attack. It does not use raw-update
error for candidate selection, and it does not reuse the baseline candidate pool
for optimization. Each surrogate realization is still only a candidate
realization; it is not the true raw-block-derived Level 2 realization.

First, a one-group smoke test checked that the new differentiable path runs:

```bash
python -m experiments.run_phase4_dna_level1_forward_attack \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --target-file development_targets.pt \
  --groups 0 \
  --candidates 1 \
  --restarts 1 \
  --iterations 50 \
  --attack-lr 0.1 \
  --nonnegative-lambda 0.001 \
  --dna-run-seed 477885591
```

The smoke test completed. On that single group, the attack beat both controls,
but `n=1` cannot satisfy a statistical gate.

The development run then used four development groups with four surrogate
realizations, two restarts per realization, and the locked 600-iteration attack
budget:

```bash
python -m experiments.run_phase4_dna_level1_forward_attack \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --target-file development_targets.pt \
  --candidates 4 \
  --restarts 2 \
  --iterations 600 \
  --attack-lr 0.1 \
  --nonnegative-lambda 0.001 \
  --dna-run-seed 477885591
```

| Control | Groups | Wins | Mean delta MSE | Median delta MSE | One-sided p | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prior | 4 | 2/4 | -582.5436 | -17.5868 | 0.6875 | FAIL |
| Zero-update | 4 | 3/4 | -394.3355 | -46.3814 | 0.3125 | FAIL |

This result did not separate a weak Level 1 objective from insufficient search
budget. The baseline attack needed more than two restarts in several hard
groups, while the Level 1 objective must search over both dummy records and a
surrogate realization. A targeted budget diagnostic was therefore run on the
same four development groups only. The objective, DNA seed, learning rate,
iteration count, surrogate count, and penalties were unchanged; only restarts
per surrogate realization were increased from 2 to 8:

```bash
python -m experiments.run_phase4_dna_level1_forward_attack \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --target-file development_targets.pt \
  --candidates 4 \
  --restarts 8 \
  --iterations 600 \
  --attack-lr 0.1 \
  --nonnegative-lambda 0.001 \
  --dna-run-seed 477885591
```

| Control | Groups | Wins | Mean delta MSE | Median delta MSE | One-sided p | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prior | 4 | 4/4 | -475.1981 | -21.0412 | 0.0625 | FAIL |
| Zero-update | 4 | 4/4 | -294.5972 | -129.8088 | 0.0625 | FAIL |

At four groups, even a clean 4/4 result cannot satisfy the locked `p < 0.05`
sign-test threshold; the best possible one-sided p-value is `0.0625`. This run
is therefore a budget diagnostic, not a formal gate-opening result. It does
show that the earlier two-restart Level 1 run was under-budgeted: increasing
only the restart budget changed the development result from 2/4 prior and 3/4
zero-update wins to 4/4 against both controls.

| Group | Best fraud MSE | Prior MSE | Zero-update MSE | Delta vs prior | Delta vs zero | Realization | Restart |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 31.7529 | 64.1741 | 80.3544 | -32.4212 | -48.6015 | 1 | 5 |
| 1 | 45.2317 | 54.8930 | 182.5993 | -9.6613 | -137.3675 | 3 | 1 |
| 2 | 516.0605 | 2367.5744 | 1386.2299 | -1851.5140 | -870.1694 | 0 | 7 |
| 3 | 35.0719 | 42.2678 | 157.3221 | -7.1958 | -122.2501 | 3 | 5 |

The groups that previously failed are no longer failing under the larger search
budget. Groups 0, 2, and 3 selected restarts 5, 7, and 5 respectively, so the
two-restart run could not have found those candidates. The Level 1
surrogate-forward attacker is therefore implementation-complete for this
development scope, but the statistical gate remains formally closed because the
development set has only four groups. Fresh evaluation is still not opened from
this diagnostic alone.

The development gate was then enlarged to eight source-disjoint groups. The
generic target sampler excludes the fixed 500k reference split and every
existing `*targets.pt` artifact before sampling:

```bash
python -m experiments.create_phase4_source_disjoint_targets \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --output-name development_gate_targets.pt \
  --groups 8 \
  --records-per-group 4 \
  --fraud-per-group 1 \
  --seed 196482731 \
  --purpose "Source-disjoint Phase 4 Level 1 development gate for locked C4/R8/I600 surrogate-forward attack budget."
```

The resulting file contains 8 groups, 32 records, one fraud record per group,
and zero overlap with all existing target artifacts. The locked C4/R8/I600
attacker was then run without further changes:

```bash
python -m experiments.run_phase4_dna_level1_forward_attack \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --target-file development_gate_targets.pt \
  --candidates 4 \
  --restarts 8 \
  --iterations 600 \
  --attack-lr 0.1 \
  --nonnegative-lambda 0.001 \
  --dna-run-seed 477885591
```

| Control | Groups | Wins | Mean delta MSE | Median delta MSE | One-sided p | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prior | 8 | 7/8 | -3613.8047 | -59.0925 | 0.0352 | PASS |
| Zero-update | 8 | 7/8 | -4731.7820 | -51.4396 | 0.0352 | PASS |

This larger development gate passes both controls. The locked Level 1
configuration is therefore:

```text
4 surrogate realizations / 8 restarts per realization / 600 iterations /
attack lr 0.1 / nonnegative lambda 0.001 / DNA run seed 477885591
```

After the development gate passed, a new fresh final Level 1 target set was
created. The earlier `fresh_final_targets.pt` is no longer used as the final
Level 1 set because it had already appeared in surrogate diagnostics. The new
file is source-disjoint from every prior target artifact:

```bash
python -m experiments.create_phase4_source_disjoint_targets \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --output-name level1_fresh_final_targets.pt \
  --groups 12 \
  --records-per-group 4 \
  --fraud-per-group 1 \
  --seed 739284615 \
  --purpose "Fresh source-disjoint single-shot Phase 4 Level 1 DNA evaluation for locked C4/R8/I600 surrogate-forward attack after development gate pass."
```

The new fresh file contains 12 groups, 48 records, one fraud record per group,
and zero overlap with all existing targets. The locked attacker was then run
once, with no tuning after seeing the result:

```bash
python -m experiments.run_phase4_dna_level1_forward_attack \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --target-file level1_fresh_final_targets.pt \
  --candidates 4 \
  --restarts 8 \
  --iterations 600 \
  --attack-lr 0.1 \
  --nonnegative-lambda 0.001 \
  --dna-run-seed 477885591
```

| Control | Groups | Wins | Mean delta MSE | Median delta MSE | One-sided p | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prior | 12 | 10/12 | -5291.2089 | -568.5679 | 0.0193 | PASS |
| Zero-update | 12 | 11/12 | -5409.7163 | -441.6595 | 0.0032 | PASS |

The fresh single-shot Level 1 result passes both controls. This means the
surrogate-forward attacker reconstructs the fraud record more closely than both
allowed no-update controls in this bounded setting. It does not yet quantify how
much harder DNA is than the raw no-DNA update on the same fresh targets; that
requires a paired raw-update baseline run under the same target set and budget.

That paired run was then executed on the same `level1_fresh_final_targets.pt`
file. The raw-update branch used the same hard balance-diff parameterization,
standard initialization, 600 iterations, attack learning rate `0.1`,
non-negative lambda `0.001`, and 8 restarts. It attacks the raw no-DNA update
directly:

```bash
python -m experiments.run_phase4_harddiff_reparam_for_misselected \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --target-file level1_fresh_final_targets.pt \
  --restarts 8 \
  --init-mode standard \
  --nonnegative-lambda 0.001
```

The raw-update branch also passes the same two controls:

| Raw control | Groups | Wins | Mean delta MSE | Median delta MSE | One-sided p | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prior | 12 | 10/12 | -5512.5247 | -508.7808 | 0.0193 | PASS |
| Zero-update | 12 | 12/12 | -4264.2244 | -667.8916 | 0.0002 | PASS |

The paired DNA-vs-raw comparison then uses:

```text
delta = fraud MSE(DNA-transformed Level 1 attack)
      - fraud MSE(raw-update attack)
```

Positive values mean the DNA-transformed update was harder to invert than the
raw update. The paired result is:

| Comparison | Groups | DNA harder wins | Mean delta MSE | Median delta MSE | One-sided p | DNA-harder gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DNA minus raw | 12 | 6/12 | 245.7226 | 1.4513 | 0.6128 | FAIL |

The sign-test result does not support a claim that DNA Transform conservative
adds measurable Level 1 protection over the raw update in this bounded scope.
Both raw and DNA-transformed updates leak enough information for the attacker to
beat no-update controls, and the paired difference is essentially split across
groups. The positive mean is driven by a few large-MSE groups; the median is
near zero and the sign count is exactly balanced.

An outlier check was then run on the paired result because the mean delta is much
larger than the median. The largest negative delta is group 1
(`DNA - raw = -2809.8713`), where DNA reconstructs better than raw. The largest
positive delta is group 2 (`DNA - raw = 5577.0186`), where DNA reconstructs
worse than raw. The diagnostic reloads the selected artifacts, checks the stored
optimization histories, and inspects the selected restarts. It does not show a
missing artifact, replay crash, or final-step selection error. The selected
histories are non-monotone under Adam, but candidates are selected by the stored
best objective, not by the last optimization step. The group-2 DNA top-10
candidate set also contains a better fraud-MSE candidate (`62716.4847`) than the
objective-selected one (`68601.4055`), but this is the same objective-vs-MSE
selection limitation already documented in earlier Phase 4 diagnostics; it is
not enough to justify deleting the outlier. The official paired statistic is
therefore kept unchanged.

Finally, one additional DNA configuration was tested as the allowed second DNA
candidate configuration under the tuning/evaluation rule. The `stronger`
configuration uses:

```text
mix_ratio = 0.12, keep_ratio = 0.82, shrink_factor = 0.35
```

The same fresh target set, DNA run seed, C4/R8/I600 budget, learning rate, and
validity penalty were used:

```bash
python -m experiments.run_phase4_dna_level1_forward_attack \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --target-file level1_fresh_final_targets.pt \
  --candidates 4 \
  --restarts 8 \
  --iterations 600 \
  --attack-lr 0.1 \
  --nonnegative-lambda 0.001 \
  --dna-run-seed 477885591 \
  --mix-ratio 0.12 \
  --keep-ratio 0.82 \
  --shrink-factor 0.35
```

The stronger DNA branch passes both no-update controls:

| Stronger DNA control | Groups | Wins | Mean delta MSE | Median delta MSE | One-sided p | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prior | 12 | 12/12 | -5194.1145 | -567.5200 | 0.0002 | PASS |
| Zero-update | 12 | 12/12 | -2433.6148 | -753.4752 | 0.0002 | PASS |

However, the paired stronger-DNA-vs-raw comparison also fails to show added
protection:

| DNA configuration | Groups | DNA harder wins | Mean delta MSE | Median delta MSE | One-sided p | DNA-harder gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Conservative | 12 | 6/12 | 245.7226 | 1.4513 | 0.6128 | FAIL |
| Stronger | 12 | 4/12 | 342.8169 | -9.3844 | 0.9270 | FAIL |

The stronger configuration does not rescue the privacy claim in this bounded
Level 1 setting. It still leaks enough for the attacker to beat controls, and it
is not harder than raw by paired sign-test evidence.

## Simple Defense Controls

The protocol also requires checking whether simple update defenses match or
exceed DNA under comparable conditions. Three simple controls were run on the
same `level1_fresh_final_targets.pt` set and the same raw-update baseline. Each
uses the hard balance-diff parameterization, standard initialization, 8 restarts,
600 iterations, learning rate `0.1`, and non-negative lambda `0.001`.

The target perturbation scale was set to relative L2 delta `0.10`, close to the
measured conservative-DNA distortion scale. The simple defenses were:

- `random_retention`: randomly keep about `99%` of update entries and set the
  rest to zero; the mask is attacker-visible.
- `topk_retention`: keep entries carrying about `99%` of squared update energy;
  in this run that retained about `0.66%` of entries, and the mask is
  attacker-visible.
- `clipping_noise`: multiply the candidate update by the attacker-visible clip
  factor `0.95`, then compare against the observed update after clipping plus
  Gaussian noise. The noise realization is not attacker-visible and is not added
  to candidate updates.

```bash
python -m experiments.run_phase4_simple_defense_attack \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --target-file level1_fresh_final_targets.pt \
  --defense <random_retention|topk_retention|clipping_noise> \
  --restarts 8 \
  --iterations 600 \
  --attack-lr 0.1 \
  --nonnegative-lambda 0.001 \
  --target-rel-l2 0.10 \
  --clip-factor 0.95 \
  --defense-seed 314159265
```

Control gates for the simple defenses:

| Method | Mean relative L2 delta | Keep ratio | Prior gate | Zero gate | Open for paired interpretation |
| --- | ---: | ---: | --- | --- | --- |
| Random retention | 0.1001 | 0.9900 | PASS, 11/12, p=0.0032 | PASS, 12/12, p=0.0002 | Yes |
| Top-k retention | 0.0987 | 0.0066 | FAIL, 7/12, p=0.3872 | FAIL, 8/12, p=0.1938 | No |
| Clipping + noise | 0.1000 | 1.0000 | FAIL, 9/12, p=0.0730 | PASS, 10/12, p=0.0193 | No |

Paired method-vs-raw results:

| Method | Groups | Method harder wins | Mean delta MSE | Median delta MSE | One-sided p | Harder-than-raw gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DNA conservative | 12 | 6/12 | 245.7226 | 1.4513 | 0.6128 | FAIL |
| DNA stronger | 12 | 4/12 | 342.8169 | -9.3844 | 0.9270 | FAIL |
| Random retention | 12 | 7/12 | 450.0530 | 7.2462 | 0.3872 | FAIL |
| Top-k retention | 12 | 9/12 | 8414.8267 | 199.9505 | 0.0730 | Not opened; controls fail |
| Clipping + noise | 12 | 9/12 | 3545.1516 | 92.4057 | 0.0730 | Not opened; controls fail |

Random retention is the only opened simple-defense comparison here, because its
same-scope attack beats both controls. It is not significantly harder than raw
under the paired sign test. Top-k retention and clipping/noise both look harder
by raw paired MSE, but their same-scope attacks do not beat both controls. Those
two rows are therefore inconclusive attack results, not evidence that either
method protects better than raw.

These controls answer the bounded-scope version of RQ2. In this scope, DNA does
not outperform the opened simple method, and none of the opened method
comparisons shows statistically supported added protection over raw.

## Simple Defense Budget Diagnostic

Top-k retention and clipping/noise both missed their same-scope control gates on
the already-spent `level1_fresh_final_targets.pt` set. They therefore could not
be interpreted as protection evidence. To check whether this was only a restart
budget issue, both methods were rerun on the separate `development_gate_targets.pt`
set, increasing restarts from 8 to 16 while keeping the target perturbation
scale, objective, initialization, learning rate, non-negative lambda, and
iteration count fixed.

```bash
python -m experiments.run_phase4_simple_defense_attack \
  artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z \
  --target-file development_gate_targets.pt \
  --defense <topk_retention|clipping_noise> \
  --restarts 16 \
  --iterations 600 \
  --attack-lr 0.1 \
  --nonnegative-lambda 0.001 \
  --target-rel-l2 0.10 \
  --clip-factor 0.95 \
  --defense-seed 314159265
```

Development-set budget diagnostic:

| Method | Groups | Prior gate | Zero gate | Mean relative L2 delta | Keep ratio | Decision |
| --- | ---: | --- | --- | ---: | ---: | --- |
| Top-k retention, R16 | 8 | FAIL, 6/8, p=0.1445 | PASS, 7/8, p=0.0352 | 0.0979 | 0.0063 | Do not open fresh evaluation |
| Clipping + noise, R16 | 8 | FAIL, 6/8, p=0.1445 | FAIL, 6/8, p=0.1445 | 0.1000 | 1.0000 | Do not open fresh evaluation |

Doubling restart count did not open either method. This weakens the simple
"not enough restarts" explanation. Top-k may need a method-specific attack for
its hard thresholding behavior, and clipping/noise may need a noise-aware attack
or a different evaluation design. In this phase, both remain inconclusive rather
than positive protection evidence.

## Clipping/Noise Monte Carlo Diagnostic

The direct clipping/noise attack above tried to match one fixed noisy transmitted
update without modeling the unknown noise distribution. A second diagnostic used
Monte Carlo surrogate noise instead. For each candidate update, the objective
uses the attacker-visible clip factor and the mean of independently sampled
surrogate Gaussian noise:

```text
argmin_c L_balanced(clip(delta(c)) + mean(z_1, ..., z_N),
                    transmitted_clipped_noisy_update)
```

The realized noise added to the transmitted update is still not revealed to the
attacker. The Monte Carlo samples are only surrogate samples from the known noise
distribution. For an additive zero-mean Gaussian with an MSE-like objective,
this mainly tests whether finite-sample expectation modeling changes the
optimization behavior relative to the zero-mean objective.

The N sweep was run on `development_gate_targets.pt` only:

| MC samples | Groups | Prior gate | Zero gate | Mean relative L2 delta | Decision |
| ---: | ---: | --- | --- | ---: | --- |
| 10 | 8 | PASS, 7/8, p=0.0352 | PASS, 7/8, p=0.0352 | 0.1000 | Development pass |
| 50 | 8 | PASS, 7/8, p=0.0352 | PASS, 7/8, p=0.0352 | 0.1000 | Development pass |
| 100 | 8 | PASS, 7/8, p=0.0352 | PASS, 7/8, p=0.0352 | 0.1000 | Selected before fresh run |

N=100 was selected before fresh evaluation because it is the largest expectation
sample count in the predeclared sweep. A new source-disjoint fresh set was then
created:

```text
clipping_noise_mc_fresh_targets.pt
seed = 19940717
groups = 12
records/group = 4
fraud/group = 1
max overlap with existing target artifacts = 0
```

On this new fresh set, clipping/noise MC passes its own controls:

| Fresh method gate | Groups | Wins | Mean delta MSE | Median delta MSE | One-sided p | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prior | 12 | 11/12 | -8267.4639 | -1999.5842 | 0.0032 | PASS |
| Zero-update | 12 | 12/12 | -7971.0412 | -1784.0756 | 0.0002 | PASS |

However, the raw no-defense baseline on the same new fresh set does not pass its
own objective-selection gate:

| Raw fresh gate | Groups | Wins | Mean delta MSE | Median delta MSE | One-sided p | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prior | 12 | 9/12 | -5229.8026 | -2698.6614 | 0.0730 | FAIL |
| Zero-update | 12 | 9/12 | -4344.5087 | -1403.8533 | 0.0730 | FAIL |

The paired clipping/noise-MC-vs-raw row is therefore diagnostic only:

| Comparison | Groups | Method harder wins | Mean delta MSE | Median delta MSE | One-sided p | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Clipping/noise MC minus raw | 12 | 3/12 | -3037.6613 | -177.1450 | 0.9807 | Not opened; raw gate fails |

This result is useful but narrower than a final RQ2 answer. Monte Carlo
expectation makes the clipping/noise attack itself pass controls, so the previous
failure was partly an attack-strategy issue. But because the paired raw baseline
fails on the same newly sampled fresh set, this run cannot establish whether
clipping/noise is better or worse than raw in a valid paired comparison.

The raw failure was then checked rather than ignored. A target diagnostic compared
the 12 fraud records in `clipping_noise_mc_fresh_targets.pt` with the two earlier
fresh sets. The target file has zero overlap with prior targets and the sampler
provenance/checksum match the expected source-disjoint path, so no target-creation
bug was found. The fraud transactions are heavier than the first fresh set and
similar in scale to the Level 1 fresh set:

| Target set | Fraud amount mean | Fraud amount median | Max amount | Raw objective gate |
| --- | ---: | ---: | ---: | --- |
| `fresh_final_targets.pt` | 262778.1 | 25933.6 | 1398946.0 | PASS, 10/12 prior, 12/12 zero |
| `level1_fresh_final_targets.pt` | 2896799.0 | 862832.9 | 10000000.0 | PASS, 10/12 prior, 12/12 zero |
| `clipping_noise_mc_fresh_targets.pt` | 3571185.0 | 2087141.0 | 10000000.0 | FAIL, 9/12 prior, 9/12 zero |

On `clipping_noise_mc_fresh_targets.pt`, the raw oracle-candidate diagnostic
still passes both controls (11/12, p=0.0032), while objective selection fails
both controls (9/12, p=0.0730). The issue is therefore not that these records are
impossible to reconstruct; the selected raw candidate is less stable on this
fresh draw.

To test whether this was a one-off draw effect, one additional source-disjoint
fresh target set was created and used once:

```text
clipping_noise_mc_fresh2_targets.pt
seed = 19940718
groups = 12
records/group = 4
fraud/group = 1
max overlap with existing target artifacts = 0
purpose = second robustness check after the raw gate failed on the first
          clipping/noise MC fresh draw; not repeated sampling until success
```

On this second fresh set, the raw no-defense baseline passes:

| Raw fresh2 gate | Groups | Wins | Mean delta MSE | Median delta MSE | One-sided p | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prior | 12 | 11/12 | -3128.7075 | -918.5752 | 0.0032 | PASS |
| Zero-update | 12 | 10/12 | -2371.0704 | -1230.8724 | 0.0193 | PASS |

The clipping/noise MC attack also passes on the same `clipping_noise_mc_fresh2`
targets:

| Clipping/noise MC fresh2 gate | Groups | Wins | Mean delta MSE | Median delta MSE | One-sided p | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prior | 12 | 11/12 | -3014.9921 | -186.6637 | 0.0032 | PASS |
| Zero-update | 12 | 11/12 | -3358.5099 | -300.0252 | 0.0032 | PASS |

With both gates open, the paired comparison is valid for this fresh2 robustness
check:

| Comparison | Groups | Method harder wins | Mean delta MSE | Median delta MSE | One-sided p | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Clipping/noise MC minus raw | 12 | 4/12 | +113.7154 | -20.5826 | 0.9270 | No harder-than-raw evidence |

The positive mean is driven by a small number of large positive deltas; the
median is negative and only 4 of 12 groups are harder than raw. In the bounded
scope tested here, the method-specific Monte Carlo clipping/noise attacker is
valid, but clipping/noise does not show a statistically supported increase in
fraud reconstruction error relative to the raw update.

## Soft Top-k Diagnostic

A soft top-k variant was also tested on `development_gate_targets.pt`. The first
smoke test showed that applying a sigmoid mask over every parameter entry creates
large artificial penalties at positions where the transmitted top-k update is
exactly zero. The implemented diagnostic therefore keeps the attacker-visible
top-k support fixed and applies a sigmoid relaxation only inside that support:

```text
candidate = delta(c) * transmitted_topk_mask *
            sigmoid((abs(delta(c)) - threshold) / temperature)
```

With temperature multiplier `0.05`, the development result is:

| Method | Groups | Prior gate | Zero gate | Mean relative L2 delta | Keep ratio | Decision |
| --- | ---: | --- | --- | ---: | ---: | --- |
| Soft top-k, R8, temp=0.05 | 8 | FAIL, 5/8, p=0.3633 | PASS, 7/8, p=0.0352 | 0.0979 | 0.0063 | Do not open fresh evaluation |

Because the R8 run beat the zero-update control but not the prior control, the
same soft relaxation was rerun with a larger restart budget. This mirrors the
Level 1 DNA debugging step where insufficient restarts initially hid a working
attacker:

| Method | Groups | Prior gate | Zero gate | Mean relative L2 delta | Keep ratio | Decision |
| --- | ---: | --- | --- | ---: | ---: | --- |
| Soft top-k, R16, temp=0.05 | 8 | PASS, 7/8, p=0.0352 | PASS, 7/8, p=0.0352 | 0.0979 | 0.0063 | Development pass |

Since R16/temp=0.05 opened the development gate, no temperature sweep was run.
The configuration was fixed and a new source-disjoint fresh target set was
created:

```text
topk_fresh_targets.pt
seed = 27182818
groups = 12
records/group = 4
fraud/group = 1
max overlap with existing target artifacts = 0
```

The raw no-defense baseline passes on this fresh set:

| Raw top-k fresh gate | Groups | Wins | Mean delta MSE | Median delta MSE | One-sided p | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prior | 12 | 10/12 | -3426.7534 | -302.9970 | 0.0193 | PASS |
| Zero-update | 12 | 12/12 | -4847.4623 | -375.0089 | 0.0002 | PASS |

The soft top-k attack itself does not pass on the same fresh set:

| Soft top-k fresh gate | Groups | Wins | Mean delta MSE | Median delta MSE | One-sided p | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prior | 12 | 7/12 | +309.2945 | -21.4268 | 0.3872 | FAIL |
| Zero-update | 12 | 7/12 | +82.1729 | -56.0976 | 0.3872 | FAIL |

The paired comparison is therefore not opened. This is not evidence that top-k
protects better than raw; it means the current soft-top-k attacker is not stable
enough on a held-out fresh draw to serve as a defense-evaluation instrument.
Top-k remains unresolved. The useful finding is narrower: a support-limited
soft relaxation plus R16 can pass development, but that success did not
generalize to the fresh set under the locked single-shot protocol.

## Level B / Level B' Weaker-Knowledge Threat Model

All prior Phase 4 attackers assume the attacker knows the exact model
architecture (layer widths, BatchNorm/Dropout placement) and observes the
transmitted update directly on the real parameter tensors. Before extending
any conclusion, a narrower question was checked: does DNA Transform look more
protective once the attacker is weakened to not know the exact architecture?

A naive way to weaken the attacker — showing it a `state_dict` from an
architecture it must guess — leaks the architecture back through tensor names
and shapes, so the observation channel was redesigned first:

- `public_projection_matrix`: every candidate architecture's flattened update is
  projected through a fixed `signed_hadamard_jl` matrix (input dim 131072,
  output dim 4096, disclosed seed) into one common comparison space, after
  padding/truncating to the fixed input dimension and L2-normalizing.
- `surrogate_checkpoint`: each candidate architecture is initialized
  independently, not copied from the true checkpoint, so weight values cannot
  leak the true architecture either.
- A projection sanity check confirmed the switch was necessary: an earlier
  `signed_sparse_jl` projection leaked architecture through sparsity/scale
  patterns (`near_zero_fraction` differed sharply by architecture); after
  switching to `signed_hadamard_jl` plus L2-normalization, the projected
  statistics were architecture-agnostic (`near_zero_fraction=0`,
  max/min normalized-magnitude ratio ~1.03 across candidates).
- The comparison objective is complexity-adjusted so that a larger candidate
  architecture is not automatically favored just because it has more free
  parameters to fit the projected signal:
  `complexity_adjusted_objective = raw_objective * (1 + 0.05 * log(param_count / reference_param_count))`,
  with `reference_param_count` fixed to the grid median, not the true
  architecture, to avoid leaking the answer through the reference point.

**Level B (wide architecture grid).** Six candidate architectures (`B1`-`B6`,
3009-45697 parameters) were crossed with 4 local-learning-rate values and 4
loss configurations (96 cells), scored by whether the true architecture `B3`
lands in the visible top-3 by complexity-adjusted objective. This is a
coarse-screening sanity check, not a fresh evaluation, so only a subset of
cells was run before deciding whether to continue:

| Run | Budget | `B3` rank | In top-3 | Gate |
| --- | --- | ---: | --- | --- |
| First pass | R2 / I200 | 5 / 6 | No | FAIL |
| Budget increase (one-time, per protocol) | R4 / I300 | 4 / 6 | No | FAIL |

Per the protocol's stop rule (one budget increase allowed, no further tuning
after that), the Level B branch stopped here. The full 96-cell grid was not
run.

**Level B' (narrow width grid around the true architecture).** A second,
narrower check crossed layer widths within ±20% of the true `(128, 64, 32)`
around the true optimizer/loss (27 width tuples total), to see whether a
small, realistic uncertainty band still hides the true architecture:

| Metric | Value |
| --- | --- |
| True width tuple | `(128, 64, 32)` |
| True architecture raw-objective rank | 23 / 27 |
| True architecture complexity-adjusted rank | 23 / 27 |
| Best-ranked width tuple | `(102, 51, 26)`, the smallest candidate in the grid |

A direct diagnostic ruled out the complexity penalty as the cause: the true
architecture's `raw_rank` and `adjusted_rank` are identical (23/27 either way),
and its complexity factor differs from a bare `1.0` by only about `0.12%`
(`level_bprime_complexity_penalty_diagnostic.json`). The true, correctly-sized
architecture ranks near the bottom of the grid regardless of whether size is
penalized.

**Interpretation.** This is not a defect specific to DNA Transform. It is a
limitation of the raw gradient/update-matching objective itself once
architecture is also a free variable: smaller candidate architectures have
fewer degrees of freedom, so they can fit almost any observed signal —
including the wrong one — at a lower loss than the true, correctly-sized
architecture achieves on the true signal. Because the coarse-screening sanity
check already fails before any defense is even introduced, Level B/B' cannot
be used to rank DNA Transform, raw updates, or any other defense. No
development or fresh evaluation was run past this point for the weaker threat
model; RQ1/RQ2 in this report are therefore answered only under the
architecture-known threat model (Level 0/1/2 above).

## Decision

**Decision: LEVEL_2_REALIZATION_KNOWN_DNA_TRANSPARENT_IN_BOUNDED_SCOPE; LEVEL_1_SURROGATE_FORWARD_GATE_OPEN_IN_BOUNDED_SCOPE; PAIRED_LEVEL_1_DNA_ADDED_PROTECTION_NOT_DETECTED; LEVEL_B_BPRIME_ARCHITECTURE_UNCERTAINTY_NOT_EVALUABLE; STOP_OR_REDESIGN_DNA_FOR_THIS_SCOPE.**

Gate B also passes for the baseline fresh-final run, and the DNA Transform
specification is now documented. The exact-realization Level 2 check has also
completed for this exact scope only:

```text
bounded Adam local update / 4 records / 1 local step / fraud gate / no class-decomposed oracle update
```

The paired Level 1 comparison has now been run in the same bounded scope. The
current result shows that a realization-unknown surrogate-forward attacker can
beat controls on the DNA-transmitted update, but DNA does not show a statistically
supported increase in fraud reconstruction error relative to the raw no-DNA
update under the paired protocol. The second DNA configuration checked in this
phase (`stronger`) also fails the DNA-harder paired gate. Random retention and
clipping/noise were tested at comparable relative L2 scale. Random retention
opens for paired interpretation but fails to show significant added protection
over raw. Top-k retention and clipping/noise are not interpretable as protection
evidence because their same-scope attacks do not pass both controls. Under the
Phase 4 decision rules, the correct bounded-scope decision is to stop or redesign
the DNA Transform rather than proceed to larger-scope claims.

The follow-up R16 budget diagnostic on `development_gate_targets.pt` did not
open top-k retention or the direct clipping/noise strategy, so no additional
fresh evaluation was spent on those direct strategies.

Two further method-specific diagnostics were then run. Monte Carlo expectation
for clipping/noise opened the clipping/noise method gate on a new source-disjoint
fresh set (`clipping_noise_mc_fresh_targets.pt`), but the paired raw baseline
failed its own objective-selection gate on that same fresh set (9/12 both
controls, p=0.0730), so that first method-vs-raw comparison was not opened. The
raw failure was checked for a target-creation bug and none was found (checksum
and zero-overlap provenance both correct); a raw oracle-candidate diagnostic on
the same fresh set still passed both controls, so the instability was in
objective-based candidate selection on that particular draw, not in the data. A
second, source-disjoint fresh set (`clipping_noise_mc_fresh2_targets.pt`, one
robustness check, not repeated sampling until the raw gate happened to pass)
opened cleanly for both the raw baseline (11/12 prior, 10/12 zero) and the
clipping/noise Monte Carlo attacker (11/12 both controls). On this valid paired
draw, clipping/noise is harder than raw in only 4/12 groups, mean delta
+113.72, median delta -20.58, p=0.9270 — no statistically supported
harder-than-raw evidence. A support-limited soft top-k relaxation passed
development at R16 but failed both controls on its own fresh set (7/12, p=0.39
for both prior and zero-update), so top-k remains unresolved rather than
resolved in either direction. These results do not change the DNA decision;
they clarify that simple stochastic/discrete defenses require their own
validated, fresh-generalizing attackers before they can be ranked against DNA,
and that raw-baseline candidate-selection instability can appear on some fresh
draws even after the Gate A fix, which is why a second fresh set was used
before trusting a paired comparison.

1. keep hard balance-diff reparameterization and non-negative lambda `0.001`;
2. use at least 8 restarts per surrogate realization, 600 iterations, attack
   learning rate 0.1, four surrogate realizations, and standard initialization
   for this bounded scope;
3. do not tune further on the passed development gate or fresh final set;
4. treat the paired Level 1 result as bounded-scope evidence only;
5. do not weaken the controls after observing fresh results;
6. report fraud and non-fraud separately;
7. do not extrapolate this result to full-client FedAvg updates.

The Level 2 result does not establish a full-client or full-FedAvg attack, and
it does not by itself prove or disprove any defense property of DNA Transform
outside this bounded exact-realization setting.

## Conclusion

This closes Phase 4 for the validated scope: **4 records per group, 1 fraud
record per group, 1 local Adam step, known architecture/preprocessing/loss,
no oracle class-decomposed labels in the official attacker.** Nothing here is
claimed outside that scope.

**RQ1 — Does DNA Transform reduce reconstruction risk?**
No added protection was found in this scope, under either attacker knowledge
level tested:

- Level 2 (attacker knows the exact random realization): DNA Transform is
  almost fully invertible. Recovering the raw update from the transformed one
  gives a mean relative L2 error of about `3.3e-8` (max `8.4e-8`), i.e.
  reconstruction quality after "undoing" DNA is essentially the same as
  attacking the raw update directly.
- Level 1 (attacker knows the algorithm but not the seed/realization, and
  optimizes a realization-unknown surrogate-forward objective): this attacker
  does beat both controls on the DNA-transmitted update (10/12 vs prior,
  11/12 vs zero-update), so DNA is not "safe by attacker confusion" either.
  Paired against the same attacker run on the raw update on the same 12
  fresh targets, DNA is harder to reconstruct in only 6/12 groups (mean delta
  +245.72, median +1.45, p=0.6128) — not a statistically supported
  improvement. The `stronger` DNA configuration (heavier mixing/attenuation)
  does worse, not better: 4/12, p=0.9270.

**RQ2 — If there is a benefit, does it come from the DNA mechanism itself
rather than from simpler update distortion?**
Comparable-strength simple controls were checked on the same protocol and, in
most cases, the same target draws:

| Defense | Same-scope attack passes its own controls? | Harder-than-raw evidence? |
| --- | --- | --- |
| DNA Transform (conservative) | Yes | No (6/12, p=0.61) |
| DNA Transform (stronger) | Yes | No (4/12, p=0.93) |
| Random retention (~0.10 relative L2) | Yes | No (7/12, p=0.39) |
| Clipping + Gaussian noise (direct) | No (fails own gate at R8 and R16) | Not interpretable |
| Clipping + Gaussian noise (Monte Carlo attacker, valid fresh2 draw) | Yes | No (4/12, p=0.93) |
| Top-k retention (hard, then soft-relaxed) | No (fails own gate on the fresh set after passing development) | Not interpretable |

None of the five methods that reached an interpretable paired comparison
(DNA conservative, DNA stronger, random retention, clipping/noise MC) showed
statistically supported harder-than-raw reconstruction difficulty in this
scope. Top-k retention could not be evaluated either way: its attacker passed
a development gate twice but did not generalize to either of its held-out
fresh target sets, so no defense claim can be made about it. The practical
reading is that DNA Transform, at the strengths tested, behaves like the other
"mild update distortion" methods rather than showing a distinct DNA-specific
protection mechanism in this scope.

**RQ3 — Is any benefit stable and practical across broader FL conditions?**
Not assessed in Phase 4. Phase 5 (more clients, non-IID variation, a second
dataset, full-client and full-round updates) was never started, because the
Phase 4 decision rules call for stopping or redesigning the defense before
extrapolating to a larger protocol when no benefit is detected at the smallest
validated scope. Extending scope now would not answer RQ1/RQ2 more
convincingly; it would only make the same untested claim in a larger,
more expensive setting.

**Threat-model boundary.** All of the above assumes the attacker knows the
model architecture exactly. The one attempt to weaken that assumption (Level
B/B', architecture-uncertain attacker) could not produce an interpretable
result: even a narrow, realistic band of candidate architectures caused the
true architecture to rank 23rd of 27 by the same gradient-matching objective,
for reasons unrelated to DNA Transform (see `## Level B / Level B'` above).
This is reported as a limitation of the current attack methodology, not as
evidence that a weaker attacker changes the RQ1/RQ2 answer.

**Bottom line.** In the most rigorously checked scope available
(bounded, fraud-focused, single Adam step, architecture-known attacker), DNA
Transform does not show a defensible reconstruction-resistance advantage over
sending the raw update, and it does not clearly outperform simpler
alternatives tested at comparable update-distortion strength. The next
research step is not to keep tuning attack budget or DNA parameters inside
this same scope; it is either to redesign the DNA Transform mechanism itself,
or to treat this as a negative result and move future defense evaluation to a
larger Phase 5 protocol only if a redesigned method first clears this same
bounded gate.

## Files Written

- `experiments/run_pre_phase4_fraud_baseline_gate.py`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/protocol_lock.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/development_summary.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/development_summary.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/frozen.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/evaluation_selected_by_class.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/evaluation_records.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/baseline_gate.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/class_gradient_contribution.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/class_gradient_contribution_summary.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/candidate_alignment_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/evaluation_candidate_alignment_selection.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/evaluation_candidate_alignment_correlations.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/misselection_pattern_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/misselection_pattern_rows.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/extra_restarts_misselected/extra_restart_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/extra_restarts_misselected/extra_restart_selected_wrong_groups.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/group_2_candidate_plausibility.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/group_2_candidate_plausibility_constraints.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/group_2_candidate_plausibility_features.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/group_0_candidate_plausibility.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/group_6_candidate_plausibility.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/group_9_candidate_plausibility.json`
- `experiments/run_phase4_harddiff_reparam_for_misselected.py`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_standard/harddiff_reparam_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_standard/harddiff_reparam_group_summary.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_standard_nonneg_0p01/harddiff_reparam_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_standard_nonneg_0p01/harddiff_reparam_group_summary.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_standard_nonneg_0p1/harddiff_reparam_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_standard_nonneg_0p1/harddiff_reparam_group_summary.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_nonnegative_development_selection.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_nonnegative_development_selection.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_development_standard_nonneg_0/harddiff_reparam_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_development_standard_nonneg_0p001/harddiff_reparam_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_development_standard_nonneg_0p01/harddiff_reparam_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_development_standard_nonneg_0p1/harddiff_reparam_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_development_standard_nonneg_1/harddiff_reparam_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_development_standard_nonneg_10/harddiff_reparam_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_evaluation_standard_nonneg_0p001/harddiff_reparam_report.json`
- `experiments/create_phase4_fresh_targets.py`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/fresh_final_targets.pt`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/fresh_final_targets.provenance.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_fresh_final_standard_nonneg_0p001/harddiff_reparam_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_fresh_final_standard_nonneg_0p001/harddiff_reparam_group_summary.csv`
- `experiments/verify_phase4_gate_b.py`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_fresh_final_standard_nonneg_0p001/gate_b_integrity_report.json`
- `experiments/analyze_dna_transform_mechanics.py`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_transform_mechanics_fresh_group0.json`
- `phase4_dna_transform_spec.md`
- `experiments/run_phase4_dna_level2_direct_inversion.py`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level2_realization_known_direct/dna_level2_direct_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level2_realization_known_direct/dna_level2_direct_selected.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level2_realization_known_direct/dna_level2_direct_transform_stats.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level2_realization_known_direct/dna_level2_direct_block_stats.csv`
- `experiments/analyze_phase4_dna_level1_surrogate_inversion.py`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_surrogate_realization_development/dna_level1_surrogate_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_surrogate_realization_development/dna_level1_surrogate_rows.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_surrogate_realization_development/dna_level1_surrogate_oracle_best_by_group.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_surrogate_realization_development/dna_level1_surrogate_visible_selection_by_group.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_surrogate_realization_fresh_final/dna_level1_surrogate_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_surrogate_realization_fresh_final/dna_level1_surrogate_rows.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_surrogate_realization_fresh_final/dna_level1_surrogate_oracle_best_by_group.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_surrogate_realization_fresh_final/dna_level1_surrogate_visible_selection_by_group.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_surrogate_realization_development_c8/dna_level1_surrogate_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_surrogate_realization_development_c32/dna_level1_surrogate_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_surrogate_realization_fresh_final_c8/dna_level1_surrogate_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_surrogate_realization_fresh_final_c32/dna_level1_surrogate_report.json`
- `experiments/run_phase4_dna_level1_forward_attack.py`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_forward_attack_development_c1_r1_i50_lr0p1_nonneg0p001/dna_level1_forward_attack_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_forward_attack_development_c4_r2_i600_lr0p1_nonneg0p001/dna_level1_forward_attack_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_forward_attack_development_c4_r8_i600_lr0p1_nonneg0p001/dna_level1_forward_attack_report.json`
- `experiments/create_phase4_source_disjoint_targets.py`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/development_gate_targets.pt`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/development_gate_targets.provenance.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_forward_attack_development_gate_c4_r8_i600_lr0p1_nonneg0p001/dna_level1_forward_attack_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_forward_attack_development_gate_c4_r8_i600_lr0p1_nonneg0p001/dna_level1_forward_attack_group_summary.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_forward_attack_development_gate_c4_r8_i600_lr0p1_nonneg0p001/dna_level1_forward_attack_records.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level1_fresh_final_targets.pt`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level1_fresh_final_targets.provenance.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_forward_attack_level1_fresh_final_c4_r8_i600_lr0p1_nonneg0p001/dna_level1_forward_attack_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_forward_attack_level1_fresh_final_c4_r8_i600_lr0p1_nonneg0p001/dna_level1_forward_attack_group_summary.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_forward_attack_level1_fresh_final_c4_r8_i600_lr0p1_nonneg0p001/dna_level1_forward_attack_records.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_level1_fresh_final_standard_nonneg_0p001/harddiff_reparam_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_level1_fresh_final_standard_nonneg_0p001/harddiff_reparam_group_summary.csv`
- `experiments/compare_phase4_level1_dna_vs_raw.py`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level1_paired_dna_vs_raw/level1_paired_dna_vs_raw_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level1_paired_dna_vs_raw/level1_paired_dna_vs_raw.csv`
- `experiments/analyze_phase4_paired_outliers.py`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level1_paired_outlier_diagnostic/paired_outlier_diagnostic.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level1_paired_outlier_diagnostic/paired_outlier_diagnostic.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_forward_attack_level1_fresh_final_c4_r8_i600_lr0p1_nonneg0p001_mix0p12_keep0p82_shrink0p35/dna_level1_forward_attack_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_forward_attack_level1_fresh_final_c4_r8_i600_lr0p1_nonneg0p001_mix0p12_keep0p82_shrink0p35/dna_level1_forward_attack_group_summary.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_forward_attack_level1_fresh_final_c4_r8_i600_lr0p1_nonneg0p001_mix0p12_keep0p82_shrink0p35/dna_level1_forward_attack_records.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level1_paired_stronger_dna_vs_raw/level1_paired_dna_vs_raw_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level1_paired_stronger_dna_vs_raw/level1_paired_dna_vs_raw.csv`
- `experiments/run_phase4_simple_defense_attack.py`
- `experiments/compare_phase4_methods_vs_raw.py`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/simple_defense_attack_random_retention_level1_fresh_final_r8_i600_lr0p1_nonneg0p001_rel0p1/simple_defense_attack_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/simple_defense_attack_topk_retention_level1_fresh_final_r8_i600_lr0p1_nonneg0p001_rel0p1/simple_defense_attack_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/simple_defense_attack_clipping_noise_level1_fresh_final_r8_i600_lr0p1_nonneg0p001_rel0p1/simple_defense_attack_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level1_methods_vs_raw_v2/level1_methods_vs_raw_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level1_methods_vs_raw_v2/level1_methods_vs_raw_summary.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level1_methods_vs_raw_v2/level1_methods_vs_raw_rows.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/simple_defense_attack_topk_retention_development_gate_r16_i600_lr0p1_nonneg0p001_rel0p1/simple_defense_attack_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/simple_defense_attack_clipping_noise_development_gate_r16_i600_lr0p1_nonneg0p001_rel0p1/simple_defense_attack_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/simple_defense_attack_clipping_noise_mc_development_gate_r8_i600_lr0p1_nonneg0p001_rel0p1_mc10/simple_defense_attack_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/simple_defense_attack_clipping_noise_mc_development_gate_r8_i600_lr0p1_nonneg0p001_rel0p1_mc50/simple_defense_attack_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/simple_defense_attack_clipping_noise_mc_development_gate_r8_i600_lr0p1_nonneg0p001_rel0p1_mc100/simple_defense_attack_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/clipping_noise_mc_fresh_targets.pt`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/clipping_noise_mc_fresh_targets.provenance.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_clipping_noise_mc_fresh_standard_nonneg_0p001/harddiff_reparam_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/simple_defense_attack_clipping_noise_mc_clipping_noise_mc_fresh_r8_i600_lr0p1_nonneg0p001_rel0p1_mc100/simple_defense_attack_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/clipping_noise_mc_vs_raw_diagnostic/level1_methods_vs_raw_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/simple_defense_attack_soft_topk_retention_development_gate_r8_i600_lr0p1_nonneg0p001_rel0p1_temp0p05/simple_defense_attack_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/clipping_noise_mc_fresh_target_diagnostic/target_diagnostic_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/clipping_noise_mc_fresh_target_diagnostic/fraud_numeric_descriptive_stats.csv`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/clipping_noise_mc_fresh2_targets.pt`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/clipping_noise_mc_fresh2_targets.provenance.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_clipping_noise_mc_fresh2_standard_nonneg_0p001/harddiff_reparam_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/simple_defense_attack_clipping_noise_mc_clipping_noise_mc_fresh2_r8_i600_lr0p1_nonneg0p001_rel0p1_mc100/simple_defense_attack_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/clipping_noise_mc_fresh2_vs_raw/level1_methods_vs_raw_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/simple_defense_attack_soft_topk_retention_development_gate_r16_i600_lr0p1_nonneg0p001_rel0p1_temp0p05/simple_defense_attack_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/topk_fresh_targets.pt`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/topk_fresh_targets.provenance.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/harddiff_reparam_topk_fresh_standard_nonneg_0p001/harddiff_reparam_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/simple_defense_attack_soft_topk_retention_topk_fresh_r16_i600_lr0p1_nonneg0p001_rel0p1_temp0p05/simple_defense_attack_report.json`
- `experiments/run_phase4_level_b_runtime_probe.py`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level_b_runtime_probe_20260910T095410770388Z/level_b_runtime_probe_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level_b_runtime_probe_20260910T103541108694Z/level_b_runtime_probe_report.json`
- `experiments/analyze_phase4_level_b_projection_sanity.py`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level_b_projection_sanity_20260910T095040015130Z/level_b_projection_sanity_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level_b_projection_sanity_20260910T095138783853Z/level_b_projection_sanity_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level_b_projection_sanity_20260910T095229189755Z/level_b_projection_sanity_report.json`
- `experiments/run_phase4_level_b_coarse_sanity.py`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level_b_coarse_sanity_20260910T103726468049Z/level_b_coarse_sanity_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level_b_coarse_sanity_20260910T104656511127Z/level_b_coarse_sanity_report.json`
- `experiments/run_phase4_level_bprime_coarse_sanity.py`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level_bprime_coarse_sanity_20260910T131934620907Z/level_bprime_coarse_sanity_report.json`
- `artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/level_bprime_coarse_sanity_20260910T131934620907Z/level_bprime_complexity_penalty_diagnostic.json`
