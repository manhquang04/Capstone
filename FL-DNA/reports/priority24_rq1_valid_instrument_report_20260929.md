# Priority 24 — valid-instrument RQ1 report

Date: 2026-09-29  
Status: COMPLETED for the pre-registered reduced Priority 24 scope  
Scope: fraud-domain T1 BN-statistics channel, distortion-matched DP only  
No edits were made under `Latex/`.

## 1. What was pre-registered

Protocol amendment:

- `protocols/amendments/2026-09-29_priority24_rq1_valid_instrument.md`
- SHA-256: `d981496ad9e03dbe49dbfb087ea84ba37f21f76c1d1562fabc01d06df79b0973`

The amendment retains the original RQ1 wording verbatim:

> RQ1. Does DNA-based update transformation reduce gradient-inversion reconstruction quality, measured via attack success under an exact one-sided sign test on reconstruction MSE, more than a distortion- or utility-matched DP baseline?

Because Priorities 21--23 invalidated the earlier RQ1 evidence chain, Priority 24 pre-registered a reduced but valid plan before execution. The full requested scope was estimated to exceed the approximately 3-day compute budget, so this run executed only the fraud-domain T1 instrument:

- T1: closed-form reconstruction of the 4-record batch mean from the transmitted first BatchNorm `running_mean` channel.
- Domain: PaySim fraud setting.
- Transforms: v1 conservative and v2 `compression_ratio=0.95`, `eta=0.01`.
- Comparator: distortion-matched Gaussian DP on the same attacked BN-statistics vector.
- Multiplicity: Holm correction across the two confirmatory DNA-vs-DP tests in this amendment.

Deferred by the amendment: T2/T3 upgraded tabular optimization attackers, utility-matched DP grid, and CIFAR I1 expansion.

## 2. Commands run

Initial target-generation attempts failed before any target file was written because the output directories did not yet exist. I then created the directories and reran with the same frozen seeds.

```bash
mkdir -p artifacts/priority24_valid_rq1/t1_dev_targets_20260929 \
  artifacts/priority24_valid_rq1/t1_pilot_targets_20260929 \
  artifacts/priority24_valid_rq1/t1_confirm_targets_20260929
```

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority24_valid_rq1/t1_dev_targets_20260929 \
  --output-name paysim_priority24_t1_dev_targets.pt \
  --groups 8 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092901 \
  --purpose 'Priority 24 T1 BN valid-instrument development n8'
```

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority24_valid_rq1/t1_pilot_targets_20260929 \
  --output-name paysim_priority24_t1_pilot_targets.pt \
  --groups 24 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092902 \
  --purpose 'Priority 24 T1 BN valid-instrument pilot n24'
```

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py \
  artifacts/priority24_valid_rq1/t1_confirm_targets_20260929 \
  --output-name paysim_priority24_t1_confirm_targets.pt \
  --groups 39 --records-per-group 4 --fraud-per-group 1 \
  --seed 2026092903 \
  --purpose 'Priority 24 T1 BN valid-instrument confirmatory n39'
```

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/priority24_t1_bn_valid_rq1.py \
  --dev-target artifacts/priority24_valid_rq1/t1_dev_targets_20260929/paysim_priority24_t1_dev_targets.pt \
  --pilot-target artifacts/priority24_valid_rq1/t1_pilot_targets_20260929/paysim_priority24_t1_pilot_targets.pt \
  --confirm-target artifacts/priority24_valid_rq1/t1_confirm_targets_20260929/paysim_priority24_t1_confirm_targets.pt \
  --output-dir artifacts/priority24_valid_rq1/t1_run_20260929 \
  --seed 2026092901
```

Derived branch-vs-control summary:

```bash
PYTHONPATH=. .venv-phase1/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
from scipy.stats import binomtest
out=Path('artifacts/priority24_valid_rq1/t1_run_20260929')
df=pd.read_csv(out/'confirm_t1_per_target.csv')
branches={
'none':'none_mse',
'v1_conservative':'v1_conservative_mse',
'v2_0p95_eta0p01':'v2_0p95_eta0p01_mse',
'dp_distortion_matched_for_v1_conservative':'dp_distortion_matched_for_v1_conservative_mse',
'dp_distortion_matched_for_v2_0p95_eta0p01':'dp_distortion_matched_for_v2_0p95_eta0p01_mse',
}
rows=[]
for branch,col in branches.items():
    decoy_col = 'none_decoy_mse' if branch=='none' else col.replace('_mse','_decoy_mse')
    wins_prior=int((df[col] < df['prior_mse']).sum())
    wins_decoy=int((df[col] < df[decoy_col]).sum())
    row={
        'branch': branch,
        'n': len(df),
        'mean_mse': df[col].mean(),
        'median_mse': df[col].median(),
        'beats_prior_wins': wins_prior,
        'beats_prior_p_greater': binomtest(wins_prior, len(df), 0.5, alternative='greater').pvalue,
        'beats_decoy_wins': wins_decoy,
        'beats_decoy_p_greater': binomtest(wins_decoy, len(df), 0.5, alternative='greater').pvalue,
    }
    if branch!='none':
        higher=int((df[col] > df['none_mse']).sum())
        row['higher_mse_than_none_wins']=higher
        row['higher_mse_than_none_p_greater']=binomtest(higher, len(df),0.5, alternative='greater').pvalue
    rows.append(row)
pd.DataFrame(rows).to_csv(out/'confirm_t1_branch_vs_controls.csv', index=False)
PY
```

## 3. Data firewall and artifacts

All three generated target provenance files reported `max_overlap_with_existing_targets: 0`. The confirmatory provenance also reported zero overlap with the Priority 24 development and pilot target sets.

Dataset SHA-256 recorded in target provenance:

- `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`

| Artifact | SHA-256 |
|---|---|
| `experiments/priority24_t1_bn_valid_rq1.py` | `8b210a14e8542795d5726b5a6842b67920e5fa9b0008f41dba7a71de4e7e8293` |
| `artifacts/priority24_valid_rq1/t1_dev_targets_20260929/paysim_priority24_t1_dev_targets.pt` | `a3fc345d2243c8ff83b307c559417e48e9563b5a3d5987cd0faeeb96e1db94cb` |
| `artifacts/priority24_valid_rq1/t1_dev_targets_20260929/paysim_priority24_t1_dev_targets.provenance.json` | `247d8864b5d1d349f119d00c22204ef6fdd6e4b25449f0de2c96dbf2d2477af8` |
| `artifacts/priority24_valid_rq1/t1_pilot_targets_20260929/paysim_priority24_t1_pilot_targets.pt` | `26b3f2e5836ac5dda9c02aab03d55c7aa856aea0defdc86bc3828d211c206016` |
| `artifacts/priority24_valid_rq1/t1_pilot_targets_20260929/paysim_priority24_t1_pilot_targets.provenance.json` | `56712bd6af113f4a0005ca1a6589765e639e4bf3570848d1ca005112af06237a` |
| `artifacts/priority24_valid_rq1/t1_confirm_targets_20260929/paysim_priority24_t1_confirm_targets.pt` | `df3f5e24ea53f2fb3510a8cfd37ef1fb98080d08336873a37df3f28bcd6b962b` |
| `artifacts/priority24_valid_rq1/t1_confirm_targets_20260929/paysim_priority24_t1_confirm_targets.provenance.json` | `8d832cc095955efc3c6a9bd8db1c0d2426afd69656053b2ee5a1ecdbfa358921` |
| `artifacts/priority24_valid_rq1/t1_run_20260929/priority24_t1_summary.json` | `7b1635e846c15720c5cb2b7c444b2d25e5a1be385942b6ccc92c523186eeb3ca` |
| `artifacts/priority24_valid_rq1/t1_run_20260929/dev_t1_qualification.csv` | `3d0f51c9a0257d9de5c44d58dc37c79f5680b6e82caef5b71b095a48735cdb5e` |
| `artifacts/priority24_valid_rq1/t1_run_20260929/pilot_t1_qualification.csv` | `9deb3f0c695710e211cd28b18be9b98466865accf3c1b1b63e05953b95261d60` |
| `artifacts/priority24_valid_rq1/t1_run_20260929/confirm_t1_per_target.csv` | `1c1278c278eed57ac0f7af6fea8fbb3716a98eb39003bd2e73d088dea480e2d7` |
| `artifacts/priority24_valid_rq1/t1_run_20260929/confirm_t1_examples.csv` | `790efcd74286f3e2baccd0299e015820b98a0976f2a0e3df766a268a9b7d17e8` |
| `artifacts/priority24_valid_rq1/t1_run_20260929/confirm_t1_branch_vs_controls.csv` | `4bde8cf948fd3fef55515caaab37c60fa22b54c70b5bf4be5954aafae79120d2` |
| `artifacts/priority24_valid_rq1/t1_run_20260929/manifest.json` | `060b8d293172bb3ceddf1963b5342e904c9285ee6e890eac22c8c0ce74396218` |

## 4. Instrument qualification

T1 qualified at both required stages.

| Stage | n | Wins vs Prior | p vs Prior | Wins vs decoy | p vs decoy | Gate |
|---|---:|---:|---:|---:|---:|---|
| Development | 8 | 8/8 | 0.00390625 | 8/8 | 0.00390625 | PASS |
| Pilot | 24 | 24/24 | 5.960464477539063e-08 | 24/24 | 5.960464477539063e-08 | PASS |

Interpretation under the amendment: T1 is a qualified fraud-domain instrument for this reduced Priority 24 RQ1 test.

## 5. Distortion-matched DP calibration on the attacked vector

Attacked vector: first-BN `network.1.running_mean` update, dimension 128.  
Clip norm rule: at or above the 95th percentile of development pre-clip norms. The implementation used `1.01 * max_bn_norm` on the development captures.

| Transform | Clip norm | Noise multiplier | Median BN norm | Max BN norm | Median DNA L2 distortion | ε, one release, δ=1e-5 | ε, 50 releases, δ=1e-5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 conservative | 14.334709027708206 | 0.002467816163589537 | 3.673275648117977 | 14.192781215552678 | 0.3941954292430625 | 84072.52899172972 | 4118761.042101533 |
| v2 0.95 / 0.01 | 14.334709027708206 | 0.003297124505600532 | 3.673275648117977 | 14.192781215552678 | 0.5416803490732032 | 47489.32416257324 | 2309981.097289703 |

These epsilons remain very large. The DP comparators are distortion-matched on the attacked BN-statistics vector, not strong formal-DP baselines.

## 6. Confirmatory DNA-vs-DP tests

Tie band from development replay: `0.0`.

Primary direction: DNA wins if DNA reconstruction MSE is larger than DP reconstruction MSE. Holm correction is across the two DNA-greater tests in this amendment.

| Cell | DNA wins | DP wins | Ties | Non-tied n | p, DNA > DP | Holm p, DNA > DP | p, DP > DNA | Holm p, DP > DNA | Median MSE ratio DNA/DP | Mean MSE ratio DNA/DP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 conservative vs DP | 16 | 23 | 0 | 39 | 0.9002045665401965 | 0.9002045665401965 | 0.16839181759496574 | 0.3367836351899315 | 0.34926089419925144 | 3.5425219035662767 |
| v2 0.95/0.01 vs DP | 24 | 15 | 0 | 39 | 0.09979543345980349 | 0.19959086691960698 | 0.9459354892969714 | 0.9459354892969714 | 1.278198187849433 | 3.3790152288599935 |

Result: neither DNA transform significantly beats its distortion-matched DP comparator under the pre-registered Holm-corrected primary test. DP also does not significantly beat either DNA transform under the opposite-direction descriptive test.

## 7. Branches vs none, Prior, and decoy

This table reports the confirmatory branch-level input-space standardized MSE behavior. `Beats Prior` and `Beats decoy` mean lower reconstruction MSE than the control. `Higher MSE than none` means the defense branch reconstructs worse than the undefended branch.

| Branch | Mean MSE | Median MSE | Beats Prior | p | Beats decoy | p | Higher MSE than none | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 4.219221454748258e-12 | 5.692941231602893e-13 | 39/39 | 1.8189894035458565e-12 | 39/39 | 1.8189894035458565e-12 | n/a | n/a |
| v1 conservative | 5.040582289064264 | 0.1082833385583946 | 26/39 | 0.02662595704896376 | 29/39 | 0.0016889239559532143 | 39/39 | 1.8189894035458565e-12 |
| v2 0.95/0.01 | 9.585705227989543 | 0.6452844749258871 | 5/39 | 0.9999998323419277 | 25/39 | 0.05406451070302866 | 39/39 | 1.8189894035458565e-12 |
| DP for v1 conservative | 1.1755852430934954 | 0.4161463010362646 | 18/39 | 0.7388013097515794 | 30/39 | 0.00053250981727615 | 39/39 | 1.8189894035458565e-12 |
| DP for v2 0.95/0.01 | 1.7148022750139613 | 0.7629640362086518 | 12/39 | 0.9952623478602618 | 32/39 | 3.5127392038702965e-05 | 39/39 | 1.8189894035458565e-12 |

The undefended branch is reconstructed essentially exactly by T1. All defense branches increase reconstruction MSE over none on all 39 confirmatory targets.

## 8. RQ1 answer under the pre-registered reduced rule

Pre-registered result code in `priority24_t1_summary.json`:

- `NO_NOT_SHOWN`

Plain-language answer for the reduced Priority 24 scope:

T1 is a valid fraud-domain instrument, but DNA does not significantly beat its distortion-matched DP comparator for either v1 conservative or v2 after Holm correction. Therefore, under the pre-registered reduced Priority 24 answer rule, RQ1 is answered as "No / not shown" for this valid-instrument fraud-domain test.

This does not execute the deferred T2/T3/I1 or utility-matched DP portions; it only closes the reduced T1 distortion-matched fraud-domain chain authorized in the amendment.

