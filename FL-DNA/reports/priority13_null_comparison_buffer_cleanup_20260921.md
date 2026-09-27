# Priority 13 — null comparison buffer-cleanup replay

Date: 2026-09-21  
Status: STEP 1 COMPLETE; STEP 2 NOT RUN

## Scope

This report cleans up the two remaining tabular null comparisons identified as
BatchNorm-buffer contaminated in the Priority 11 audit:

1. RQ1 Group 2 stronger v1 vs distortion-matched DP, reported as `25/44`,
   one-sided exact sign-test `p=0.2257`.
2. RQ1-v2 confirmatory DNA-v2 vs DP-v2, reported as `85/176`,
   one-sided exact sign-test `p=0.70106`.

This was Step 1 only:

- no attacker optimization was run;
- no target set was created;
- no frozen artifact was modified;
- saved candidate/target pairs were replay-scored after excluding BatchNorm
  buffers and keeping only trainable parameter keys.

## Important limitation

This replay is a descriptive sensitivity check, not a fresh corrected-vector
experiment. The candidates were originally optimized under the contaminated
objective, where BatchNorm running-stat buffers dominated the update vector.
Rescoring those same candidates on trainable parameters only does not prove what
would happen if the attacker were optimized from scratch on a clean vector. It
only checks whether the historical null comparison appears to reverse or become
positive under buffer-excluded scoring.

Because the historical primary tests used feature-MSE tie thresholds
(`0.0390625` for Group 2 stronger and `0.01953125` for RQ1-v2), applying those
same frozen thresholds to trainable-update MSE yields all ties. I therefore also
report a zero-threshold descriptive sign count, but the frozen-rule decision is
the primary replay result.

## Script and artifact

Replay script:

```text
experiments/priority13_null_comparison_buffer_cleanup.py
```

Replay output:

```text
artifacts/priority13_null_cleanup/step1_replay_20260921.json
```

Replay output SHA-256:

```text
e65e1ca942d3c2e2879f2d1f0a09d8d637bb9883d913c782904ddc6dc01547ea
```

## Commands run

```bash
PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python -m py_compile \
  FL-DNA/experiments/priority13_null_comparison_buffer_cleanup.py

PYTHONPATH=FL-DNA FL-DNA/.venv-phase1/bin/python \
  FL-DNA/experiments/priority13_null_comparison_buffer_cleanup.py \
  --output FL-DNA/artifacts/priority13_null_cleanup/step1_replay_20260921.json
```

## Data sources

| Test | Run artifact | Target source used |
|---|---|---|
| RQ1 Group 2 stronger v1 vs DP | `artifacts/rq1/group2_stronger_confirmatory_run_20260913` | `artifacts/rq1/group2_stronger_confirmatory_run_20260913/dna/group_0_run/rq1_stronger_confirmatory_targets.pt` |
| RQ1-v2 DNA-v2 vs DP-v2 | `artifacts/rq1_v2/confirmatory_run_20260916` | `artifacts/rq1_v2/confirmatory_freeze_20260916/rq1_v2_confirmatory_targets.pt` |

For the Group 2 stronger replay, the frozen standalone target path was not present
in this checkout, so the script used the target copy that was stored inside the
official run directory. No new target was generated.

## Step 1 replay results

### Frozen-threshold primary replay

| Test | Historical result | Replay metric | Frozen tie threshold | Wins | Losses | Ties | Non-tied n | p | Reject H0? |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| RQ1 Group 2 stronger v1 vs DP | 25/44, p=0.2257 | trainable-update MSE | 0.0390625 | 0 | 0 | 44 | 0 | 1.0 | No |
| RQ1-v2 DNA-v2 vs DP-v2 | 85/176, p=0.70106 | trainable-update MSE | 0.01953125 | 0 | 0 | 176 | 0 | 1.0 | No |

### Descriptive zero-threshold sign count

Because the frozen feature-MSE tie thresholds made every trainable-update pair a
tie, I also decomposed the raw zero-threshold signs into all three categories:
DNA win (`DNA-minus-DP > 0`), DNA loss / DP win (`DNA-minus-DP < 0`), and exact
tie (`DNA-minus-DP == 0`). This is a descriptive, post-hoc sensitivity analysis
added after observing that the frozen feature-space tie threshold was not
informative in trainable-update space. It is not a replacement for the
pre-registered confirmatory test.

| Test | Mean DNA-minus-DP trainable-update MSE | Median DNA-minus-DP trainable-update MSE | DNA wins | DNA losses / DP wins | Exact ties | DNA-advantage p | DP-advantage descriptive p |
|---|---:|---:|---:|---:|---:|---:|---:|
| RQ1 Group 2 stronger v1 vs DP | -0.0016171075 | -0.0016167462 | 0 | 44 | 0 | 1.0 | 5.684341886e-14 |
| RQ1-v2 DNA-v2 vs DP-v2 | -0.0111451912 | -0.0111362139 | 0 | 176 | 0 | 1.0 | 1.044048715e-53 |

Positive DNA-minus-DP would mean DNA had higher MSE than DP. Negative
DNA-minus-DP means the DP comparator had higher trainable-update MSE than DNA
for that pair. In both replayed null comparisons, the descriptive zero-threshold
direction is negative for every pair.

The exact-tie counts are zero in both tests, so this is not a case of duplicated
or identical candidate/target vectors producing `0 wins / 0 losses`. The
zero-threshold sign decomposition is:

- RQ1 Group 2 stronger: `0 DNA wins / 44 DNA losses / 0 exact ties`.
- RQ1-v2 confirmatory: `0 DNA wins / 176 DNA losses / 0 exact ties`.

The DP-advantage p-values above are descriptive one-sided exact sign tests over
the same zero-threshold signs, computed only because the DNA losses were nonzero
and systematic. They were not pre-registered and are included only to quantify
the direction of this replay sensitivity check.

### Mechanism split and permutation-invariant v1 check

I then separated the two mechanisms to check whether the zero-threshold result
could be explained only by position-wise unfairness for v1, whose transform
contains a permutation-like component. This check used the same replay artifact
and still did not run any attacker or create any target.

Mechanism split at zero threshold:

| Mechanism/test | Transform property relevant here | DNA wins | DNA losses / DP wins | Exact ties | DP-advantage descriptive p |
|---|---|---:|---:|---:|---:|
| v1 stronger, Group 2 n=44 | v1 has permutation/mixing in its transform family | 0 | 44 | 0 | 5.684341886e-14 |
| v2 ratio0.95/eta0.01, n=176 | v2 uses SRHT sketch/lift + quantization, not the same v1 element permutation | 0 | 176 | 0 | 1.044048715e-53 |

For v1 specifically, I also computed a permutation-invariant descriptive score by
sorting the concatenated trainable values before MSE:

```text
sorted_mse(candidate, target) = mean((sort(flatten(candidate_trainable))
                                  - sort(flatten(target_trainable)))^2)
```

This removes position-by-position matching while preserving the value
distribution. Result:

| v1 scoring variant | Mean DNA-minus-DP MSE | Median DNA-minus-DP MSE | DNA wins | DNA losses / DP wins | Exact ties | DP-advantage descriptive p |
|---|---:|---:|---:|---:|---:|---:|
| Position-wise trainable-update MSE | -0.0016171075 | -0.0016167462 | 0 | 44 | 0 | 5.684341886e-14 |
| Sorted-value trainable-update MSE | -0.0012922410 | -0.0012861396 | 0 | 44 | 0 | 5.684341886e-14 |

Thus the v1 zero-threshold pattern did not disappear under the sorted-value
permutation-invariant descriptive score. Separately, the v2 replay also shows
the same `0 DNA wins / 176 DNA losses / 0 exact ties` pattern even though the v2
mechanism is not the v1 permutation/mixing mechanism. This section is still only
descriptive and does not decide whether a fresh Step 2 rerun is required.

## Step 2 decision

Step 2 was not run.

Reason: Step 1 did not reveal a reversal or a newly positive DNA advantage. Under
the frozen tie-threshold rule, both tests remain non-rejecting with all pairs tied.
Under the zero-threshold descriptive count, both tests point uniformly away from a
DNA advantage.

Per supervisor instruction, this report does not decide whether Step 2 is needed.
If the supervisor treats the scale mismatch/all-tie frozen-threshold replay as
insufficiently informative, the correct next action is a separately timestamped
Step 2 amendment and fresh corrected-vector run. That was not started here.

## Checks

- No new target set was created.
- No attacker optimization was run.
- No frozen artifact was modified.
- `torch.set_num_threads(1)` was not changed.
- `py_compile` passed for the replay script.
