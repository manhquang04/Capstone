# Priority34C — individual-gradient record recovery

Verified complete; all scheduled outputs and fixed72 directional tests audited.

Honest-but-curious server: client-side noise before upload, not central DP. Whole-training local epsilon10 uses per-client update-level replace-one adjacency over50 rounds, delta1e-5; NOT record DP.

Batch1 is the easiest recovery setting. Public initial checkpoints/known labels and public population priors are assumed. Ratio uses full P32 FraudMLP/evalBN; native TabLeak uses P33B FC (IEEE60 selected features, BAF58, PaySim13). FedSGD single-gradient queries differ from multi-step Adam utility; epsilon transfer is conditional on the same clipped individual-query sensitivity, not a final-model inversion claim. Key-known v2 uses LSMR/sketch-space loss. V1 selection uses observable objectives only. Protected uploads exclude BN; private truth/noise bundles are audit-only, outside the DP release.

Commands: `.venv-phase1/bin/python -B -u experiments/priority34c_qualify.py --launch`; development and confirmatory drivers `--prepare` then `--launch`; `experiments/analyze_priority34c.py --verify`. Resume skips only validated outputs after verified infrastructure interruption.

## Qualification

|Cell|n8|n24|Verdict|
|---|---|---|---|
|baf/ratio_batch1|{"empirical_mean": {"losses": 0, "p": 0.00390625, "ties": 0, "wins": 8}, "mean_mode": {"losses": 0, "p": 0.00390625, "ties": 0, "wins": 8}}|{"empirical_mean": {"losses": 0, "p": 5.960464477539063e-08, "ties": 0, "wins": 24}, "mean_mode": {"losses": 0, "p": 5.960464477539063e-08, "ties": 0, "wins": 24}}|QUALIFIED|
|baf/tableak_batch1|{"empirical_mean": {"losses": 0, "p": 0.00390625, "ties": 0, "wins": 8}, "mean_mode": {"losses": 0, "p": 0.00390625, "ties": 0, "wins": 8}}|{"empirical_mean": {"losses": 0, "p": 5.960464477539063e-08, "ties": 0, "wins": 24}, "mean_mode": {"losses": 0, "p": 5.960464477539063e-08, "ties": 0, "wins": 24}}|QUALIFIED|
|baf/tableak_batch2|{"empirical_mean": {"losses": 0, "p": 0.00390625, "ties": 0, "wins": 8}, "mean_mode": {"losses": 0, "p": 0.00390625, "ties": 0, "wins": 8}}|{"empirical_mean": {"losses": 0, "p": 5.960464477539063e-08, "ties": 0, "wins": 24}, "mean_mode": {"losses": 0, "p": 5.960464477539063e-08, "ties": 0, "wins": 24}}|QUALIFIED|
|ieee_cis/ratio_batch1|{"empirical_mean": {"losses": 0, "p": 0.00390625, "ties": 0, "wins": 8}, "mean_mode": {"losses": 0, "p": 0.00390625, "ties": 0, "wins": 8}}|{"empirical_mean": {"losses": 0, "p": 5.960464477539063e-08, "ties": 0, "wins": 24}, "mean_mode": {"losses": 0, "p": 5.960464477539063e-08, "ties": 0, "wins": 24}}|QUALIFIED|
|ieee_cis/tableak_batch1|{"empirical_mean": {"losses": 0, "p": 0.00390625, "ties": 0, "wins": 8}, "mean_mode": {"losses": 0, "p": 0.00390625, "ties": 0, "wins": 8}}|{"empirical_mean": {"losses": 0, "p": 5.960464477539063e-08, "ties": 0, "wins": 24}, "mean_mode": {"losses": 0, "p": 5.960464477539063e-08, "ties": 0, "wins": 24}}|QUALIFIED|
|ieee_cis/tableak_batch2|{"empirical_mean": {"losses": 5, "p": 0.85546875, "ties": 0, "wins": 3}, "mean_mode": {"losses": 1, "p": 0.03515625, "ties": 0, "wins": 7}}|not eligible|NOT_ASSESSABLE|
|paysim/ratio_batch1|{"empirical_mean": {"losses": 0, "p": 0.00390625, "ties": 0, "wins": 8}, "mean_mode": {"losses": 0, "p": 0.0078125, "ties": 1, "wins": 7}}|{"empirical_mean": {"losses": 0, "p": 5.960464477539063e-08, "ties": 0, "wins": 24}, "mean_mode": {"losses": 0, "p": 2.384185791015625e-07, "ties": 2, "wins": 22}}|QUALIFIED|
|paysim/tableak_batch1|{"empirical_mean": {"losses": 0, "p": 0.00390625, "ties": 0, "wins": 8}, "mean_mode": {"losses": 0, "p": 0.0078125, "ties": 1, "wins": 7}}|{"empirical_mean": {"losses": 0, "p": 5.960464477539063e-08, "ties": 0, "wins": 24}, "mean_mode": {"losses": 0, "p": 2.384185791015625e-07, "ties": 2, "wins": 22}}|QUALIFIED|
|paysim/tableak_batch2|{"empirical_mean": {"losses": 0, "p": 0.00390625, "ties": 0, "wins": 8}, "mean_mode": {"losses": 0, "p": 0.00390625, "ties": 0, "wins": 8}}|{"empirical_mean": {"losses": 1, "p": 1.4901161193847656e-06, "ties": 0, "wins": 23}, "mean_mode": {"losses": 0, "p": 5.960464477539063e-08, "ties": 0, "wins": 24}}|QUALIFIED|

## Paired confirmatory results (feature accuracy %, DNA−DP)

|Dataset/cell|DNA|Comparator|DNA median|DP median|Median effect [ranks13,27]|Direction|raw p|Holm72 p|
|---|---|---|---:|---:|---|---|---:|---:|
|paysim/ratio_batch1|dna_v1_conservative|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_lower_recovery|1|1|
|paysim/ratio_batch1|dna_v1_conservative|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_higher_recovery|1|1|
|paysim/ratio_batch1|dna_v1_conservative|local_dp_eps10|100 [100.0, 100.0]|77.777778 [66.66666666666667, 77.77777777777779]|22.222222 [22.222222222222214, 33.33333333333333]|dna_lower_recovery|1|1|
|paysim/ratio_batch1|dna_v1_conservative|local_dp_eps10|100 [100.0, 100.0]|77.777778 [66.66666666666667, 77.77777777777779]|22.222222 [22.222222222222214, 33.33333333333333]|dna_higher_recovery|3.637978807e-12|2.182787284e-10|
|paysim/ratio_batch1|dna_v2_0p95|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_lower_recovery|1|1|
|paysim/ratio_batch1|dna_v2_0p95|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_higher_recovery|1|1|
|paysim/ratio_batch1|dna_v2_0p95|local_dp_eps10|100 [100.0, 100.0]|77.777778 [66.66666666666667, 77.77777777777779]|22.222222 [22.222222222222214, 33.33333333333333]|dna_lower_recovery|1|1|
|paysim/ratio_batch1|dna_v2_0p95|local_dp_eps10|100 [100.0, 100.0]|77.777778 [66.66666666666667, 77.77777777777779]|22.222222 [22.222222222222214, 33.33333333333333]|dna_higher_recovery|3.637978807e-12|2.182787284e-10|
|paysim/tableak_batch1|dna_v1_conservative|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_lower_recovery|1|1|
|paysim/tableak_batch1|dna_v1_conservative|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_higher_recovery|1|1|
|paysim/tableak_batch1|dna_v1_conservative|local_dp_eps10|100 [100.0, 100.0]|22.222222 [11.111111111111116, 33.333333333333336]|77.777778 [66.66666666666666, 88.88888888888889]|dna_lower_recovery|1|1|
|paysim/tableak_batch1|dna_v1_conservative|local_dp_eps10|100 [100.0, 100.0]|22.222222 [11.111111111111116, 33.333333333333336]|77.777778 [66.66666666666666, 88.88888888888889]|dna_higher_recovery|1.818989404e-12|1.309672371e-10|
|paysim/tableak_batch1|dna_v2_0p95|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_lower_recovery|1|1|
|paysim/tableak_batch1|dna_v2_0p95|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_higher_recovery|1|1|
|paysim/tableak_batch1|dna_v2_0p95|local_dp_eps10|100 [100.0, 100.0]|22.222222 [11.111111111111116, 33.333333333333336]|77.777778 [66.66666666666666, 88.88888888888889]|dna_lower_recovery|1|1|
|paysim/tableak_batch1|dna_v2_0p95|local_dp_eps10|100 [100.0, 100.0]|22.222222 [11.111111111111116, 33.333333333333336]|77.777778 [66.66666666666666, 88.88888888888889]|dna_higher_recovery|1.818989404e-12|1.309672371e-10|
|paysim/tableak_batch2|dna_v1_conservative|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_lower_recovery|1|1|
|paysim/tableak_batch2|dna_v1_conservative|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_higher_recovery|1|1|
|paysim/tableak_batch2|dna_v1_conservative|local_dp_eps10|100 [100.0, 100.0]|27.777778 [22.22222222222222, 33.33333333333333]|66.666667 [61.111111111111114, 77.77777777777777]|dna_lower_recovery|1|1|
|paysim/tableak_batch2|dna_v1_conservative|local_dp_eps10|100 [100.0, 100.0]|27.777778 [22.22222222222222, 33.33333333333333]|66.666667 [61.111111111111114, 77.77777777777777]|dna_higher_recovery|1.818989404e-12|1.309672371e-10|
|paysim/tableak_batch2|dna_v2_0p95|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_lower_recovery|1|1|
|paysim/tableak_batch2|dna_v2_0p95|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_higher_recovery|1|1|
|paysim/tableak_batch2|dna_v2_0p95|local_dp_eps10|100 [100.0, 100.0]|27.777778 [22.22222222222222, 33.33333333333333]|72.222222 [66.66666666666666, 77.77777777777777]|dna_lower_recovery|1|1|
|paysim/tableak_batch2|dna_v2_0p95|local_dp_eps10|100 [100.0, 100.0]|27.777778 [22.22222222222222, 33.33333333333333]|72.222222 [66.66666666666666, 77.77777777777777]|dna_higher_recovery|1.818989404e-12|1.309672371e-10|
|ieee_cis/ratio_batch1|dna_v1_conservative|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_lower_recovery|1|1|
|ieee_cis/ratio_batch1|dna_v1_conservative|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_higher_recovery|1|1|
|ieee_cis/ratio_batch1|dna_v1_conservative|local_dp_eps10|95 [83.33333333333334, 98.57142857142858]|60.714286 [59.285714285714285, 61.66666666666667]|31.904762 [25.0, 35.0]|dna_lower_recovery|1|1|
|ieee_cis/ratio_batch1|dna_v1_conservative|local_dp_eps10|95 [83.33333333333334, 98.57142857142858]|60.714286 [59.285714285714285, 61.66666666666667]|31.904762 [25.0, 35.0]|dna_higher_recovery|1.818989404e-12|1.309672371e-10|
|ieee_cis/ratio_batch1|dna_v2_0p95|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_lower_recovery|1|1|
|ieee_cis/ratio_batch1|dna_v2_0p95|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_higher_recovery|1|1|
|ieee_cis/ratio_batch1|dna_v2_0p95|local_dp_eps10|64.285714 [53.80952380952381, 71.9047619047619]|60.714286 [59.285714285714285, 61.66666666666667]|4.5238095 [-5.476190476190482, 10.238095238095234]|dna_lower_recovery|0.9459354893|1|
|ieee_cis/ratio_batch1|dna_v2_0p95|local_dp_eps10|64.285714 [53.80952380952381, 71.9047619047619]|60.714286 [59.285714285714285, 61.66666666666667]|4.5238095 [-5.476190476190482, 10.238095238095234]|dna_higher_recovery|0.09979543346|1|
|ieee_cis/tableak_batch1|dna_v1_conservative|distortion_matched|100 [100.0, 100.0]|100 [100.0, 100.0]|0 [0.0, 0.0]|dna_lower_recovery|1|1|
|ieee_cis/tableak_batch1|dna_v1_conservative|distortion_matched|100 [100.0, 100.0]|100 [100.0, 100.0]|0 [0.0, 0.0]|dna_higher_recovery|1|1|
|ieee_cis/tableak_batch1|dna_v1_conservative|local_dp_eps10|100 [100.0, 100.0]|35.185185 [33.333333333333336, 38.888888888888886]|64.814815 [61.111111111111114, 66.66666666666666]|dna_lower_recovery|1|1|
|ieee_cis/tableak_batch1|dna_v1_conservative|local_dp_eps10|100 [100.0, 100.0]|35.185185 [33.333333333333336, 38.888888888888886]|64.814815 [61.111111111111114, 66.66666666666666]|dna_higher_recovery|1.818989404e-12|1.309672371e-10|
|ieee_cis/tableak_batch1|dna_v2_0p95|distortion_matched|100 [100.0, 100.0]|100 [100.0, 100.0]|0 [0.0, 0.0]|dna_lower_recovery|1|1|
|ieee_cis/tableak_batch1|dna_v2_0p95|distortion_matched|100 [100.0, 100.0]|100 [100.0, 100.0]|0 [0.0, 0.0]|dna_higher_recovery|1|1|
|ieee_cis/tableak_batch1|dna_v2_0p95|local_dp_eps10|100 [100.0, 100.0]|35.185185 [33.333333333333336, 38.888888888888886]|64.814815 [61.111111111111114, 66.66666666666666]|dna_lower_recovery|1|1|
|ieee_cis/tableak_batch1|dna_v2_0p95|local_dp_eps10|100 [100.0, 100.0]|35.185185 [33.333333333333336, 38.888888888888886]|64.814815 [61.111111111111114, 66.66666666666666]|dna_higher_recovery|1.818989404e-12|1.309672371e-10|
|ieee_cis/tableak_batch2|dna_v1_conservative|distortion_matched|—|—|NOT_ASSESSABLE: unprotected qualification gate failed|dna_lower_recovery|1|1|
|ieee_cis/tableak_batch2|dna_v1_conservative|distortion_matched|—|—|NOT_ASSESSABLE: unprotected qualification gate failed|dna_higher_recovery|1|1|
|ieee_cis/tableak_batch2|dna_v1_conservative|local_dp_eps10|—|—|NOT_ASSESSABLE: unprotected qualification gate failed|dna_lower_recovery|1|1|
|ieee_cis/tableak_batch2|dna_v1_conservative|local_dp_eps10|—|—|NOT_ASSESSABLE: unprotected qualification gate failed|dna_higher_recovery|1|1|
|ieee_cis/tableak_batch2|dna_v2_0p95|distortion_matched|—|—|NOT_ASSESSABLE: unprotected qualification gate failed|dna_lower_recovery|1|1|
|ieee_cis/tableak_batch2|dna_v2_0p95|distortion_matched|—|—|NOT_ASSESSABLE: unprotected qualification gate failed|dna_higher_recovery|1|1|
|ieee_cis/tableak_batch2|dna_v2_0p95|local_dp_eps10|—|—|NOT_ASSESSABLE: unprotected qualification gate failed|dna_lower_recovery|1|1|
|ieee_cis/tableak_batch2|dna_v2_0p95|local_dp_eps10|—|—|NOT_ASSESSABLE: unprotected qualification gate failed|dna_higher_recovery|1|1|
|baf/ratio_batch1|dna_v1_conservative|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_lower_recovery|1|1|
|baf/ratio_batch1|dna_v1_conservative|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_higher_recovery|1|1|
|baf/ratio_batch1|dna_v1_conservative|local_dp_eps10|97.297297 [97.2972972972973, 100.0]|37.837838 [35.13513513513513, 43.24324324324324]|59.459459 [51.35135135135136, 62.16216216216216]|dna_lower_recovery|1|1|
|baf/ratio_batch1|dna_v1_conservative|local_dp_eps10|97.297297 [97.2972972972973, 100.0]|37.837838 [35.13513513513513, 43.24324324324324]|59.459459 [51.35135135135136, 62.16216216216216]|dna_higher_recovery|1.818989404e-12|1.309672371e-10|
|baf/ratio_batch1|dna_v2_0p95|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_lower_recovery|1|1|
|baf/ratio_batch1|dna_v2_0p95|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_higher_recovery|1|1|
|baf/ratio_batch1|dna_v2_0p95|local_dp_eps10|94.594595 [89.1891891891892, 97.2972972972973]|37.837838 [35.13513513513513, 43.24324324324324]|51.351351 [43.243243243243235, 59.45945945945946]|dna_lower_recovery|1|1|
|baf/ratio_batch1|dna_v2_0p95|local_dp_eps10|94.594595 [89.1891891891892, 97.2972972972973]|37.837838 [35.13513513513513, 43.24324324324324]|51.351351 [43.243243243243235, 59.45945945945946]|dna_higher_recovery|7.275957614e-11|4.220055416e-09|
|baf/tableak_batch1|dna_v1_conservative|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_lower_recovery|1|1|
|baf/tableak_batch1|dna_v1_conservative|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_higher_recovery|1|1|
|baf/tableak_batch1|dna_v1_conservative|local_dp_eps10|100 [100.0, 100.0]|35.135135 [32.432432432432435, 37.83783783783784]|64.864865 [62.16216216216216, 67.56756756756756]|dna_lower_recovery|1|1|
|baf/tableak_batch1|dna_v1_conservative|local_dp_eps10|100 [100.0, 100.0]|35.135135 [32.432432432432435, 37.83783783783784]|64.864865 [62.16216216216216, 67.56756756756756]|dna_higher_recovery|1.818989404e-12|1.309672371e-10|
|baf/tableak_batch1|dna_v2_0p95|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_lower_recovery|1|1|
|baf/tableak_batch1|dna_v2_0p95|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_higher_recovery|1|1|
|baf/tableak_batch1|dna_v2_0p95|local_dp_eps10|100 [100.0, 100.0]|35.135135 [32.432432432432435, 37.83783783783784]|64.864865 [62.16216216216216, 67.56756756756756]|dna_lower_recovery|1|1|
|baf/tableak_batch1|dna_v2_0p95|local_dp_eps10|100 [100.0, 100.0]|35.135135 [32.432432432432435, 37.83783783783784]|64.864865 [62.16216216216216, 67.56756756756756]|dna_higher_recovery|1.818989404e-12|1.309672371e-10|
|baf/tableak_batch2|dna_v1_conservative|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_lower_recovery|1|1|
|baf/tableak_batch2|dna_v1_conservative|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_higher_recovery|1|1|
|baf/tableak_batch2|dna_v1_conservative|local_dp_eps10|100 [100.0, 100.0]|36.486486 [33.78378378378378, 37.83783783783784]|63.513514 [60.810810810810814, 66.21621621621622]|dna_lower_recovery|1|1|
|baf/tableak_batch2|dna_v1_conservative|local_dp_eps10|100 [100.0, 100.0]|36.486486 [33.78378378378378, 37.83783783783784]|63.513514 [60.810810810810814, 66.21621621621622]|dna_higher_recovery|1.818989404e-12|1.309672371e-10|
|baf/tableak_batch2|dna_v2_0p95|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_lower_recovery|1|1|
|baf/tableak_batch2|dna_v2_0p95|distortion_matched|—|—|NOT_ASSESSABLE: development distortion matching gate failed|dna_higher_recovery|1|1|
|baf/tableak_batch2|dna_v2_0p95|local_dp_eps10|100 [100.0, 100.0]|36.486486 [33.78378378378378, 37.83783783783784]|63.513514 [62.16216216216216, 66.21621621621622]|dna_lower_recovery|1|1|
|baf/tableak_batch2|dna_v2_0p95|local_dp_eps10|100 [100.0, 100.0]|36.486486 [33.78378378378378, 37.83783783783784]|63.513514 [62.16216216216216, 66.21621621621622]|dna_higher_recovery|1.818989404e-12|1.309672371e-10|

## Distortion calibration

```json
{
  "baf/ratio_batch1": {
    "dna_v1_conservative": {
      "clip": 0.40554958077960634,
      "defense_distortions": [
        0.004363939185115438,
        0.00618135883673306,
        0.0012280390009763215,
        0.0016081393570007746,
        0.0037653674040825135,
        0.00014024594781884833,
        0.002897706374713573,
        0.005648230105977401,
        0.004878236958299581,
        0.001244547392100586,
        0.002786137708720208,
        0.0012552919522944275,
        0.0036198208318303934,
        0.004096432909671564,
        0.0013048931689247179,
        0.001230048363250326,
        0.001634501213521955,
        0.02664376679232203,
        0.001178299365327283,
        0.0046690152624255556,
        0.0010445632130573466,
        0.0,
        0.0011091149065576116,
        0.001309460977211597
      ],
      "matching_rule": "P33B median per-record relative error<=.05",
      "median_relative_error": 0.44420136124696263,
      "n": 24,
      "noise_sd": 1.2103371907457314e-05,
      "sigma": 2.9844370407658793e-05,
      "status": "NOT_ASSESSABLE",
      "unit_gaussian_norms": [
        134.705737878875,
        132.2199867438067,
        134.97093575944223,
        134.7799620645291,
        133.03366050277657,
        134.81864396887843,
        134.29822416679784,
        133.48707949399636,
        134.61423223611544,
        133.24908208862746,
        133.94502551928937,
        134.04795225716464,
        134.2811269732099,
        133.3516088039967,
        134.63323810929998,
        132.724087828756,
        133.46283184647186,
        133.656211307592,
        134.0349159598768,
        132.8149198473375,
        134.25571546449493,
        133.55809773933595,
        132.81389914326624,
        133.96713939773056
      ]
    },
    "dna_v2_0p95": {
      "clip": 0.40554958077960634,
      "defense_distortions": [
        0.008669090093845097,
        0.012416634647992964,
        0.002020108119386251,
        0.0029952513427337363,
        0.007538743021264496,
        0.000291482843656982,
        0.006113517282446459,
        0.011775036933747898,
        0.009855063571466736,
        0.0020688463981157904,
        0.005902805469694118,
        0.002438622079005688,
        0.007400305254087118,
        0.008728370334174075,
        0.0024237306124041706,
        0.0023076019281611385,
        0.0032084368569059653,
        0.04381148497191504,
        0.0019986721181834456,
        0.009570942354608635,
        0.0015916757639474087,
        0.0,
        0.0018579368428828136,
        0.0024754970490409134
      ],
      "matching_rule": "P33B median per-record relative error<=.05",
      "median_relative_error": 0.5564019677536711,
      "n": 24,
      "noise_sd": 2.3090630817607854e-05,
      "sigma": 5.6936640825074176e-05,
      "status": "NOT_ASSESSABLE",
      "unit_gaussian_norms": [
        135.2069041258028,
        134.36112406936297,
        133.57262605752834,
        134.55992797879097,
        134.41235566497215,
        134.0850323173305,
        134.72847854086234,
        133.8020949805618,
        133.6798668053839,
        134.666371350482,
        135.6175356428702,
        134.2722072101764,
        133.97915838856474,
        134.36135352956143,
        134.3399828472494,
        133.24517861764323,
        134.32691710201726,
        136.46977972164703,
        132.50680896587883,
        135.7170537763323,
        133.0500862139633,
        133.67474522417825,
        134.82204538695746,
        133.53282737698044
      ]
    }
  },
  "baf/tableak_batch1": {
    "dna_v1_conservative": {
      "clip": 3.0863009146032248,
      "defense_distortions": [
        0.2077332867059067,
        0.18088704825968013,
        0.24246482346688222,
        0.2494855347630904,
        0.19920084731834614,
        0.22848270893585015,
        0.334222923021884,
        0.15341165540159898,
        0.18398228425576715,
        0.21539054748338302,
        0.21925631362329925,
        0.15335314754835463,
        0.18767535719413078,
        0.27053373206954345,
        0.23617699774831888,
        0.2488610954836362,
        0.19639839058817804,
        0.1654339136973002,
        0.22515485391926543,
        0.25650420810059843,
        0.17180391149335647,
        0.23819968674634068,
        0.1770153211175951,
        0.23026099397276906
      ],
      "matching_rule": "P33B median per-record relative error<=.05",
      "median_relative_error": 0.1246116863055613,
      "n": 24,
      "noise_sd": 0.0017073303738208588,
      "sigma": 0.000553196341206526,
      "status": "NOT_ASSESSABLE",
      "unit_gaussian_norms": [
        127.28785282662159,
        128.1412250400619,
        126.77878395912181,
        127.43784833069081,
        126.24088622610493,
        125.79750547322367,
        128.02257549715452,
        126.71987120535614,
        126.7032004149808,
        127.769193671331,
        127.28903971350228,
        127.73656472468953,
        128.34796612828853,
        126.74703072738993,
        127.22752362626217,
        128.0749540338008,
        127.17391325581806,
        128.44283284944444,
        127.4597046407048,
        127.62493909259615,
        127.65262557315927,
        126.83874099322969,
        126.01132530931892,
        127.26032594814251
      ]
    },
    "dna_v2_0p95": {
      "clip": 3.0863009146032248,
      "defense_distortions": [
        0.3614219770936488,
        0.28045102157500795,
        0.3858351261911114,
        0.3906225041486013,
        0.36923443619836394,
        0.3161118310535001,
        0.60324794728541,
        0.28020876686309404,
        0.31899886273038525,
        0.39092357284037116,
        0.3090074652199809,
        0.26400289007302,
        0.29588670341538903,
        0.4328122082949413,
        0.44278965591174524,
        0.35877192996543594,
        0.3134350294480393,
        0.26570296054760795,
        0.32000987652551643,
        0.4085587155415255,
        0.30376048721242294,
        0.34236840578862643,
        0.3065699940358191,
        0.32151083162579774
      ],
      "matching_rule": "P33B median per-record relative error<=.05",
      "median_relative_error": 0.1301667882363297,
      "n": 24,
      "noise_sd": 0.002519263751492521,
      "sigma": 0.0008162728849841908,
      "status": "NOT_ASSESSABLE",
      "unit_gaussian_norms": [
        125.80588508503904,
        126.90464879881374,
        127.4157411901577,
        127.02481313193574,
        126.44791250098261,
        129.08718980161132,
        127.21665486043884,
        127.98835207275218,
        128.11605018298732,
        126.2507705702525,
        126.77119218985098,
        128.35190908239903,
        128.12952133318277,
        127.49645717641502,
        128.64413954453485,
        126.55010730430175,
        126.94725922606233,
        127.33241121932856,
        128.14056382818328,
        128.35170544717857,
        126.26642209587821,
        127.31369630762232,
        126.98455619559809,
        127.71644032972353
      ]
    }
  },
  "baf/tableak_batch2": {
    "dna_v1_conservative": {
      "clip": 2.8729732627128457,
      "defense_distortions": [
        0.21215018804418964,
        0.14941234156723385,
        0.1906413277299464,
        0.17651477810783014,
        0.1997471175857946,
        0.11375789786555598,
        0.17129200428559194,
        0.32050361215652406,
        0.15466541534256553,
        0.20478509568825187,
        0.17482523043598291,
        0.1522510793982569,
        0.19336081187376514,
        0.17990569494008218,
        0.15671858389237342,
        0.19504471778781973,
        0.222269700192449,
        0.12082110963650432,
        0.1918481257442867,
        0.1832915981232637,
        0.18084680429428546,
        0.20204840092896223,
        0.17942659937346644,
        0.15639115996490408
      ],
      "matching_rule": "P33B median per-record relative error<=.05",
      "median_relative_error": 0.10756224687470212,
      "n": 24,
      "noise_sd": 0.001413657331483678,
      "sigma": 0.0004920537722473658,
      "status": "NOT_ASSESSABLE",
      "unit_gaussian_norms": [
        127.63206106423,
        127.16504971586762,
        128.0639487684757,
        127.55884779325565,
        127.06782077963173,
        127.89809114670928,
        128.40234157782314,
        127.89248502277027,
        127.84675528949523,
        127.07964274670094,
        126.90606764483708,
        128.55480195450642,
        127.45127715745193,
        127.81094478728161,
        128.3152856775276,
        127.08429674111106,
        127.68271997411387,
        127.05739698846129,
        126.88307276001721,
        126.44332204572925,
        127.67623163192359,
        126.57336612578231,
        127.08776841216743,
        127.98994073999793
      ]
    },
    "dna_v2_0p95": {
      "clip": 2.8729732627128457,
      "defense_distortions": [
        0.3670291828044602,
        0.22834322057372639,
        0.28724197762501014,
        0.33547934872159774,
        0.2943850230297663,
        0.19176700569181712,
        0.2986362375069779,
        0.5157558848386211,
        0.26773255900597825,
        0.3088546954243362,
        0.2678147744813124,
        0.2777410313949515,
        0.3510278025637253,
        0.21992777050525708,
        0.25015613757733723,
        0.30381327290712234,
        0.42731849671903555,
        0.20357197172755465,
        0.2730504287528517,
        0.24212660342325115,
        0.3467623985666687,
        0.27512749084397164,
        0.32369411887001703,
        0.2452816901754755
      ],
      "matching_rule": "P33B median per-record relative error<=.05",
      "median_relative_error": 0.14792117585322603,
      "n": 24,
      "noise_sd": 0.0022156597222436655,
      "sigma": 0.0007712079158549138,
      "status": "NOT_ASSESSABLE",
      "unit_gaussian_norms": [
        129.08086281341258,
        128.68309843329203,
        128.39845152171307,
        127.70199715001887,
        127.94977384561562,
        126.6071539246096,
        127.96190090152511,
        127.12955814394688,
        126.03686475496359,
        126.99756932629704,
        126.27237390419651,
        126.67594129293835,
        127.73469942935466,
        128.80771833394425,
        129.17042994303372,
        127.49064883728501,
        127.17949638443417,
        127.23686690114806,
        127.39493353833902,
        127.44254276087747,
        127.39722720715368,
        127.52434668233127,
        128.19039409039493,
        127.50473847238602
      ]
    }
  },
  "ieee_cis/ratio_batch1": {
    "dna_v1_conservative": {
      "clip": 2738.3768019027116,
      "defense_distortions": [
        0.014968985816767767,
        0.012973333046894043,
        0.01147130727168876,
        17.715104075094764,
        22.620462332519537,
        0.0005154624139537737,
        0.033763594098088504,
        0.09046045992877409,
        0.015223792316123926,
        0.015030597878542683,
        0.0434805401119687,
        0.42164844761506576,
        0.004048211053851119,
        2.1797256797684876,
        0.010909124222285542,
        0.01267299740937691,
        294.2278655265851,
        0.032098341555520486,
        0.005543418788241768,
        16.001953434788323,
        59.26346859035479,
        0.05762528674509011,
        251.1743542797206,
        0.12222472866380887
      ],
      "matching_rule": "P33B median per-record relative error<=.05",
      "median_relative_error": 0.9995969866350058,
      "n": 24,
      "noise_sd": 0.0001444449577753282,
      "sigma": 5.274838644373676e-08,
      "status": "NOT_ASSESSABLE",
      "unit_gaussian_norms": [
        267.36702531728497,
        267.68297981725766,
        267.3807217548008,
        268.01573118534145,
        266.78352634860113,
        266.6350771782987,
        266.91158163731524,
        266.0207673246632,
        266.94296950647043,
        267.43229531389494,
        267.5061800266576,
        267.7623375030628,
        268.09481949028105,
        267.38446077378467,
        267.954273035566,
        267.14777365964557,
        268.145166456597,
        266.8388445265719,
        266.23554193810855,
        266.3825650520677,
        267.5438998540907,
        267.03474291887846,
        267.672135805054,
        267.7536555972047
      ]
    },
    "dna_v2_0p95": {
      "clip": 2738.3768019027116,
      "defense_distortions": [
        0.03063908144184315,
        0.02588709647254326,
        0.023362725879859134,
        35.69635879373564,
        45.38865974328387,
        0.001041506366467954,
        0.06753470947401018,
        0.17943805524654338,
        0.03176321199504208,
        0.03028916955734396,
        0.085280331004452,
        0.8485590842627103,
        0.007973188824178143,
        4.395862729361682,
        0.023176532627091698,
        0.025675054047258207,
        608.3786168426354,
        0.06559209918649922,
        0.01137164674867016,
        32.08227153572804,
        119.10373988186275,
        0.12129814326467434,
        506.35225305203073,
        0.24860862048780533
      ],
      "matching_rule": "P33B median per-record relative error<=.05",
      "median_relative_error": 0.9996025294780634,
      "n": 24,
      "noise_sd": 0.0002861794121209976,
      "sigma": 1.045069516810656e-07,
      "status": "NOT_ASSESSABLE",
      "unit_gaussian_norms": [
        267.258480609148,
        266.24038078878414,
        266.0988742315062,
        268.0974499990952,
        266.5400609641361,
        268.0958067249556,
        269.01486810775697,
        266.7089359648066,
        266.5803574877947,
        266.6994792821863,
        266.1510702651858,
        266.3147351711526,
        267.996822083953,
        266.47186973609786,
        267.52414354057964,
        268.67868004111136,
        266.9769432310444,
        267.00640429869145,
        267.12641758040854,
        267.85174322544816,
        268.3411540826996,
        268.31849698139035,
        265.71746254029495,
        266.7831212319273
      ]
    }
  },
  "ieee_cis/tableak_batch1": {
    "dna_v1_conservative": {
      "clip": 4.254522985239855,
      "defense_distortions": [
        0.22128999364842442,
        0.26106154994650044,
        0.07174700882546463,
        0.07174700882546463,
        0.07174700882546463,
        0.07174700882546463,
        0.4551916800535672,
        0.07174700882546463,
        0.07174700882546463,
        0.07174700882546463,
        0.11542803802389057,
        0.07174700882546463,
        0.07174700882546463,
        0.13789845283419538,
        0.07174700882546463,
        0.07174700882546463,
        0.42308725521498514,
        0.2088531556357867,
        0.07174700882546463,
        0.4550572941724072,
        0.07174700882546463,
        0.07174700882546463,
        0.07174700882546463,
        0.07174700882546463
      ],
      "matching_rule": "P33B median per-record relative error<=.05",
      "median_relative_error": 0.007250055766508282,
      "n": 24,
      "noise_sd": 0.0005606182744943705,
      "sigma": 0.00013176994846174627,
      "status": "MATCHED",
      "unit_gaussian_norms": [
        127.6715984749082,
        128.72313391964525,
        127.82687660401915,
        127.40358228089423,
        128.25832458620067,
        127.57484494353908,
        127.64368887897066,
        128.13139598077262,
        129.0822235205549,
        128.18322446436514,
        127.8725260614843,
        128.73063941411783,
        126.82239468607403,
        128.6535414974694,
        128.66971923895784,
        128.0841991814602,
        128.17564649261942,
        127.06404310223652,
        128.91740297765952,
        126.54009135241533,
        127.02567720121013,
        128.13105776009792,
        127.06170244583534,
        127.52436169726826
      ]
    },
    "dna_v2_0p95": {
      "clip": 4.254522985239855,
      "defense_distortions": [
        0.30615331737258533,
        0.5385698480139663,
        0.12954568121280427,
        0.12954568121280427,
        0.12954568121280427,
        0.12954568121280427,
        0.8580699327287907,
        0.12954568121280427,
        0.12954568121280427,
        0.12954568121280427,
        0.18799419300999437,
        0.12954568121280427,
        0.12954568121280427,
        0.12782440903443745,
        0.12954568121280427,
        0.12954568121280427,
        0.7827904058373191,
        0.31000342898732935,
        0.12954568121280427,
        0.8163247375591717,
        0.12954568121280427,
        0.12954568121280427,
        0.12954568121280427,
        0.12954568121280427
      ],
      "matching_rule": "P33B median per-record relative error<=.05",
      "median_relative_error": 0.006969670997119405,
      "n": 24,
      "noise_sd": 0.0010123274834479069,
      "sigma": 0.0002379414770962474,
      "status": "MATCHED",
      "unit_gaussian_norms": [
        128.3155178834615,
        129.24055006778562,
        127.10480767987123,
        127.73339346778194,
        128.3673047368322,
        127.73183587978657,
        128.61826084667777,
        127.89809339543962,
        128.3209232903235,
        128.94803009303808,
        125.91696425233614,
        128.88859956918313,
        126.82820235084681,
        128.66424634923357,
        127.62734135249715,
        127.21625115659327,
        128.19675473567938,
        127.2343266286337,
        127.84693905481632,
        127.50746275330634,
        128.2891726761156,
        128.03821838300183,
        127.3252058130871,
        129.0990870314376
      ]
    }
  },
  "paysim/ratio_batch1": {
    "dna_v1_conservative": {
      "clip": 0.09510774940427565,
      "defense_distortions": [
        0.0071291184955414415,
        0.008075049976907243,
        0.0012746146266553674,
        0.009790018751708141,
        0.0011630189361969078,
        0.0011786685448542276,
        0.0009480890930233804,
        0.001273160219404958,
        0.00452976513129518,
        0.0094613140797529,
        0.0008429534829596618,
        0.003408063337801278,
        0.0023460229954964324,
        0.0009440165070576395,
        0.004253196802124867,
        0.0012672360498073025,
        0.001147101533072829,
        0.0010010496337466957,
        0.001038621612690449,
        0.000828615213769906,
        0.0008673664777066372,
        0.0018364964986941066,
        0.0028912037807462777,
        0.0010317226594267624
      ],
      "matching_rule": "P33B median per-record relative error<=.05",
      "median_relative_error": 0.39944882115066704,
      "n": 24,
      "noise_sd": 1.1515533285066237e-05,
      "sigma": 0.00012107881173927291,
      "status": "NOT_ASSESSABLE",
      "unit_gaussian_norms": [
        109.43958655286905,
        110.06536442965827,
        109.96444919209453,
        110.36315461316642,
        109.10400003181701,
        109.91938457466547,
        110.76749861454724,
        109.35627681472427,
        110.40775737012896,
        110.33304441288581,
        110.29308489602975,
        110.3380475308759,
        111.33455635721576,
        110.106838568822,
        108.52366817451527,
        109.03449853141261,
        110.93659057326738,
        110.54660423335315,
        110.3129614176689,
        109.47596823602099,
        110.41694487494524,
        110.67761594558148,
        110.41537943997288,
        110.01046352845916
      ]
    },
    "dna_v2_0p95": {
      "clip": 0.09510774940427565,
      "defense_distortions": [
        0.01590973537556451,
        0.01812439667114644,
        0.0028161235781124705,
        0.021816663080387717,
        0.002074152012146911,
        0.002470195561864069,
        0.001574042750516273,
        0.002590877628088078,
        0.00984927581661171,
        0.020761045549115044,
        0.001624839615325863,
        0.006971089487937741,
        0.0055861899323456895,
        0.001660871101643416,
        0.00949222437588961,
        0.002684684151585399,
        0.0022257748376583895,
        0.001913465251850632,
        0.0019955495008302598,
        0.0014018576830179704,
        0.0016356287723713649,
        0.003916833642123223,
        0.0057657497550255635,
        0.0017272189920834091
      ],
      "matching_rule": "P33B median per-record relative error<=.05",
      "median_relative_error": 0.5693515772519866,
      "n": 24,
      "noise_sd": 2.395297710744335e-05,
      "sigma": 0.00025185095071092624,
      "status": "NOT_ASSESSABLE",
      "unit_gaussian_norms": [
        109.37195034379567,
        109.61649296906154,
        109.82444842324959,
        110.48667215002467,
        110.33342062341256,
        109.78289099665541,
        111.06971538197344,
        109.98308475712199,
        109.71947684304098,
        110.08841191558155,
        110.53714183083626,
        110.11120526611604,
        110.87546219074476,
        110.28771942832708,
        110.13539522990763,
        110.75863774484408,
        110.56601087295171,
        110.36813123794535,
        109.70549043882899,
        109.44835754961198,
        110.05033405795106,
        110.50996197428975,
        108.76759267997151,
        111.59760087875728
      ]
    }
  },
  "paysim/tableak_batch1": {
    "dna_v1_conservative": {
      "clip": 4.105509804267656,
      "defense_distortions": [
        0.19300441501831958,
        0.13169177941080742,
        0.128935423700123,
        0.1323203801953114,
        0.1751517814478369,
        0.19544197824719814,
        0.17044982742278228,
        0.16877442990961591,
        0.44423652637787187,
        0.18566965237330565,
        0.11501300214203064,
        0.16191534635994898,
        0.11951730049277866,
        0.138514330801935,
        0.24507050719804932,
        0.1731377548069311,
        0.18540275064845882,
        0.15730830734641815,
        0.20185251119223768,
        0.12542096925425014,
        0.17022205089535683,
        0.2216827719624784,
        0.13174668630659211,
        0.1964545754754129
      ],
      "matching_rule": "P33B median per-record relative error<=.05",
      "median_relative_error": 0.14213239018724833,
      "n": 24,
      "noise_sd": 0.001574941930495729,
      "sigma": 0.00038361665312760553,
      "status": "NOT_ASSESSABLE",
      "unit_gaussian_norms": [
        107.1681792406168,
        107.77837749351008,
        108.40114288185191,
        108.14331904679183,
        109.27566700124011,
        108.25581409165986,
        108.7668631626791,
        107.4966738666344,
        108.96734123456297,
        107.79205260951859,
        107.45952477397975,
        109.25122330599419,
        108.39940064109442,
        108.31656041366205,
        106.92188442550761,
        107.98998778134685,
        105.78981718654717,
        108.99806157299044,
        108.92068131092144,
        108.11203249652031,
        108.21053423621528,
        108.16426139264013,
        107.73124359365839,
        108.00887461723296
      ]
    },
    "dna_v2_0p95": {
      "clip": 4.105509804267656,
      "defense_distortions": [
        0.24954254078776805,
        0.1953722116883643,
        0.2510324285478537,
        0.2117495512477148,
        0.21042428554937817,
        0.3332527628652706,
        0.21105812184779485,
        0.19926386618648137,
        0.6399127753574103,
        0.25668643661412177,
        0.20610595053544498,
        0.24477413296473655,
        0.22170978601565516,
        0.23299159521204635,
        0.41006986004691354,
        0.19629437012757212,
        0.25460247797205604,
        0.2044828570484941,
        0.28419327713983666,
        0.19651881049007683,
        0.19870844892323217,
        0.3778107398499393,
        0.20364362956132576,
        0.3212339224077319
      ],
      "matching_rule": "P33B median per-record relative error<=.05",
      "median_relative_error": 0.12205013126828251,
      "n": 24,
      "noise_sd": 0.002102736145968679,
      "sigma": 0.000512174186938464,
      "status": "NOT_ASSESSABLE",
      "unit_gaussian_norms": [
        106.9777650588482,
        107.58929280984357,
        108.65920910581876,
        108.24402116075234,
        107.58881554754626,
        108.77962960063299,
        108.14707454408936,
        108.41830979179417,
        106.07800674716138,
        107.3137794667089,
        107.5910000954557,
        108.60097772277904,
        107.81855291222361,
        108.4828550651125,
        108.09564430556061,
        108.7035907170588,
        108.90548190889618,
        106.63061752965875,
        108.50538305943775,
        108.04739523841496,
        109.09165191000741,
        107.37987363305453,
        108.77836248338636,
        107.90261145625159
      ]
    }
  },
  "paysim/tableak_batch2": {
    "dna_v1_conservative": {
      "clip": 2.356205110459228,
      "defense_distortions": [
        0.12454340691219094,
        0.11996313132829482,
        0.142796794965069,
        0.16420746310729128,
        0.1374078764790727,
        0.168890710253582,
        0.13608836373964767,
        0.2662689162978193,
        0.12936624521430826,
        0.15414715896625344,
        0.1453002548133871,
        0.12429142182176002,
        0.17034188101273154,
        0.16072312599641453,
        0.12415324284885647,
        0.15272789076105203,
        0.15461343111166384,
        0.14881702477548234,
        0.16916357988538336,
        0.15241573514360068,
        0.18014036583786341,
        0.10509901594344301,
        0.1234516804060686,
        0.16156582129512015
      ],
      "matching_rule": "P33B median per-record relative error<=.05",
      "median_relative_error": 0.10366599354039466,
      "n": 24,
      "noise_sd": 0.0013971747143614217,
      "sigma": 0.0005929766929709741,
      "status": "NOT_ASSESSABLE",
      "unit_gaussian_norms": [
        107.96852445509732,
        108.51510731296186,
        109.50713837457872,
        107.7440513876467,
        108.47387382704787,
        108.41672896655241,
        108.3912279799792,
        107.83440521753832,
        107.15784208503499,
        107.38578425745263,
        107.73876588416238,
        108.85517476846951,
        108.28464556091731,
        107.61874891470444,
        107.76694858792428,
        107.41322829278158,
        107.70983183602846,
        107.73074103551359,
        108.4562220568992,
        107.4418705261114,
        107.99687960540204,
        108.73313127065302,
        106.53417160346342,
        107.0184194051713
      ]
    },
    "dna_v2_0p95": {
      "clip": 2.356205110459228,
      "defense_distortions": [
        0.1961169454315456,
        0.23083049408925954,
        0.21647164939959493,
        0.19230953172092938,
        0.23353350404324388,
        0.1939584481979818,
        0.2748057169728768,
        0.38450971387689015,
        0.20936294843730122,
        0.2095767920847655,
        0.17085738686321322,
        0.19236512776176384,
        0.19496858872241904,
        0.21594805853048538,
        0.23488591966127503,
        0.16819070545879158,
        0.19840946166150195,
        0.17307101568112648,
        0.20148269524419105,
        0.167449876927898,
        0.28333105161585564,
        0.17297737170503555,
        0.19198189613638492,
        0.2412788487358876
      ],
      "matching_rule": "P33B median per-record relative error<=.05",
      "median_relative_error": 0.10352479002862224,
      "n": 24,
      "noise_sd": 0.0018493922341195426,
      "sigma": 0.0007849029042123982,
      "status": "NOT_ASSESSABLE",
      "unit_gaussian_norms": [
        108.30631632709995,
        108.76638242466606,
        108.68300039585925,
        107.37605210354165,
        106.89220485790767,
        108.59073830143467,
        107.10503979536112,
        107.27192180697409,
        109.07270199286533,
        108.35306279759901,
        108.61485940399582,
        107.34994264234926,
        108.15023520862907,
        107.6036751030236,
        107.82424935972095,
        108.71897930464388,
        107.5891511660045,
        109.37722477750869,
        108.0787234341401,
        109.2658642058868,
        107.4997544098149,
        108.76591709720046,
        107.62219421399594,
        107.73649366439862
      ]
    }
  }
}
```

Historical metadata preflight failures were preserved and amended before science. No outcome-based tuning, clamping, seed exclusion or central aggregate comparator. Failed qualification/matching is NOT_ASSESSABLE, not evidence of privacy. No optional secure-aggregate secondary arm was run.

Independent reload, gates, accounting, paired sign/Holm72: PASS. SHA256 manifest includes original and additive freezes, receipts and attempt artifacts; earlier evidence is preserved.
