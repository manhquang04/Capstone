# Priority33c — CIFAR recovery coverage and per-tensor DP

Scientific stages resolved; final checks and supervisor receipt determine completion.

Earlier artifacts unchanged. CPU training/thread1; native image loops4800iterations/TV.01. No nonfinite/negative BN clamp or seed exclusion. Failed gates are NOT_ASSESSABLE, not privacy evidence.

Protocols: 2026-10-03_priority33c_image_coverage.md; 2026-10-03_priority33c_execution_annex.md. Commands, launches, errors and resume skips: artifacts/priority33c/runs.jsonl and supervisor.jsonl.

P31 checkpoint replay exactly reproduced validation46.1667% and test46.29%, seed51016; no checkpoint or learning-rate selection on test. C2 is two separate sensitivities, not a joint factorial.

Per-tensor Gaussian std=sigma*C_l, joint update-level whitened sensitivity sqrt(L). This differs from P27 isotropic noise; epsilon is one release/delta1e-5, not record-level accounting.

## Calibration

```json
{
  "matches": {
    "dna_v1_conservative::single": {
      "status": "matched",
      "sigma": 0.001,
      "bracketed": true,
      "target_delta": -0.005979166666666678,
      "threshold": -0.010979166666666679,
      "next_sigma": 0.003,
      "grid_means": {
        "0.0001": -0.00035416666666667623,
        "0.0003": -0.0007291666666666766,
        "0.001": -0.006916666666666672,
        "0.003": -0.050104166666666686,
        "0.01": -0.21150000000000002,
        "0.03": -0.35852083333333334,
        "0.1": -0.35860416666666667,
        "0.3": -0.3587916666666667
      },
      "epsilon": 504798.5261385355,
      "sensitivity_ratio": 1.0
    },
    "dna_v1_conservative::per_tensor": {
      "status": "matched",
      "sigma": 0.001,
      "bracketed": true,
      "target_delta": -0.005979166666666678,
      "threshold": -0.010979166666666679,
      "next_sigma": 0.003,
      "grid_means": {
        "0.0001": -8.333333333334497e-05,
        "0.0003": -0.0010416666666666803,
        "0.001": -0.00289583333333333,
        "0.003": -0.020520833333333342,
        "0.01": -0.09320833333333334,
        "0.03": -0.23043750000000002,
        "0.1": -0.3530833333333333,
        "0.3": -0.3597708333333333
      },
      "epsilon": 4013572.309097043,
      "sensitivity_ratio": 2.8284271247461903
    },
    "dna_v1_medium::single": {
      "status": "matched",
      "sigma": 0.001,
      "bracketed": true,
      "target_delta": -0.006729166666666682,
      "threshold": -0.011729166666666683,
      "next_sigma": 0.003,
      "grid_means": {
        "0.0001": -0.00035416666666667623,
        "0.0003": -0.0007291666666666766,
        "0.001": -0.006916666666666672,
        "0.003": -0.050104166666666686,
        "0.01": -0.21150000000000002,
        "0.03": -0.35852083333333334,
        "0.1": -0.35860416666666667,
        "0.3": -0.3587916666666667
      },
      "epsilon": 504798.5261385355,
      "sensitivity_ratio": 1.0
    },
    "dna_v1_medium::per_tensor": {
      "status": "matched",
      "sigma": 0.001,
      "bracketed": true,
      "target_delta": -0.006729166666666682,
      "threshold": -0.011729166666666683,
      "next_sigma": 0.003,
      "grid_means": {
        "0.0001": -8.333333333334497e-05,
        "0.0003": -0.0010416666666666803,
        "0.001": -0.00289583333333333,
        "0.003": -0.020520833333333342,
        "0.01": -0.09320833333333334,
        "0.03": -0.23043750000000002,
        "0.1": -0.3530833333333333,
        "0.3": -0.3597708333333333
      },
      "epsilon": 4013572.309097043,
      "sensitivity_ratio": 2.8284271247461903
    },
    "dna_v1_stronger::single": {
      "status": "matched",
      "sigma": 0.001,
      "bracketed": true,
      "target_delta": -0.005500000000000008,
      "threshold": -0.01050000000000001,
      "next_sigma": 0.003,
      "grid_means": {
        "0.0001": -0.00035416666666667623,
        "0.0003": -0.0007291666666666766,
        "0.001": -0.006916666666666672,
        "0.003": -0.050104166666666686,
        "0.01": -0.21150000000000002,
        "0.03": -0.35852083333333334,
        "0.1": -0.35860416666666667,
        "0.3": -0.3587916666666667
      },
      "epsilon": 504798.5261385355,
      "sensitivity_ratio": 1.0
    },
    "dna_v1_stronger::per_tensor": {
      "status": "matched",
      "sigma": 0.001,
      "bracketed": true,
      "target_delta": -0.005500000000000008,
      "threshold": -0.01050000000000001,
      "next_sigma": 0.003,
      "grid_means": {
        "0.0001": -8.333333333334497e-05,
        "0.0003": -0.0010416666666666803,
        "0.001": -0.00289583333333333,
        "0.003": -0.020520833333333342,
        "0.01": -0.09320833333333334,
        "0.03": -0.23043750000000002,
        "0.1": -0.3530833333333333,
        "0.3": -0.3597708333333333
      },
      "epsilon": 4013572.309097043,
      "sensitivity_ratio": 2.8284271247461903
    },
    "dna_v2_0p95::single": {
      "status": "matched",
      "sigma": 0.0003,
      "bracketed": true,
      "target_delta": -0.0008125000000000111,
      "threshold": -0.005812500000000011,
      "next_sigma": 0.001,
      "grid_means": {
        "0.0001": -0.00035416666666667623,
        "0.0003": -0.0007291666666666766,
        "0.001": -0.006916666666666672,
        "0.003": -0.050104166666666686,
        "0.01": -0.21150000000000002,
        "0.03": -0.35852083333333334,
        "0.1": -0.35860416666666667,
        "0.3": -0.3587916666666667
      },
      "epsilon": 5571550.642684008,
      "sensitivity_ratio": 1.0
    },
    "dna_v2_0p95::per_tensor": {
      "status": "matched",
      "sigma": 0.001,
      "bracketed": true,
      "target_delta": -0.0008125000000000111,
      "threshold": -0.005812500000000011,
      "next_sigma": 0.003,
      "grid_means": {
        "0.0001": -8.333333333334497e-05,
        "0.0003": -0.0010416666666666803,
        "0.001": -0.00289583333333333,
        "0.003": -0.020520833333333342,
        "0.01": -0.09320833333333334,
        "0.03": -0.23043750000000002,
        "0.1": -0.3530833333333333,
        "0.3": -0.3597708333333333
      },
      "epsilon": 4013572.309097043,
      "sensitivity_ratio": 2.8284271247461903
    }
  },
  "grid": [
    0.0001,
    0.0003,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3
  ],
  "baseline_mean": 0.4593333333333333,
  "baseline_passed": true
}
```

PaySim full-state CPU utility (validation-F1 only):

```json
{
  "matches": {
    "dna_v1_conservative": {
      "status": "NOT_ASSESSABLE",
      "bracketed": false,
      "reason": "full-state CPU utility BN/numerical gate failed"
    },
    "dna_v2_0p95": {
      "status": "NOT_ASSESSABLE",
      "bracketed": false,
      "reason": "full-state CPU utility BN/numerical gate failed"
    }
  },
  "seeds": [
    270201,
    270202,
    270203,
    270204,
    270205,
    270206,
    270207,
    270208,
    270209,
    270210,
    270211,
    270212,
    270213,
    270214,
    270215,
    270216
  ],
  "baseline_valid": true,
  "grid": [
    1e-06,
    3e-06,
    1e-05,
    3e-05,
    0.0001,
    0.0003,
    0.001,
    0.003
  ],
  "own_validation_F1_target": true,
  "full_state": true,
  "CPU": true
}
```

## Paired primary tests

PSNR DNA−comparator <0 favors DNA protection; BN-MSE difference >0 favors DNA. Intervals are one-based order ranks13/27 of39, not differences of marginal medians. NA hypotheses reserve p1 in fixed families20/135, no equivalence interpretation.

| hypothesis_id | status | wins | losses | ties | p_raw | holm_p33c | holm_combined135 | median_dna | median_comparator | median_dna_ssim | median_comparator_ssim | median_paired_difference | rank13 | rank27 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1::dna_v1_medium::unprotected::DNA_stronger | VALID | 34 | 5 | 0 | 1.21495395e-06 | 1.94392633e-05 | 0.00011056081 | 23.5034695 | 23.725499 | 0.872359037 | 0.879140675 | -0.193276557 | -0.324913237 | -0.0857166934 |
| C1::dna_v1_medium::unprotected::comparator_stronger | VALID | 5 | 34 | 0 | 0.999999832 | 1 | 1 | 23.5034695 | 23.725499 | 0.872359037 | 0.879140675 | -0.193276557 | -0.324913237 | -0.0857166934 |
| C1::dna_v1_medium::distortion::DNA_stronger | VALID | 17 | 22 | 0 | 0.831608182 | 1 | 1 | 23.5034695 | 23.5737042 | 0.872359037 | 0.865667284 | 0.0817532955 | -0.0582612537 | 0.154349309 |
| C1::dna_v1_medium::distortion::comparator_stronger | VALID | 22 | 17 | 0 | 0.26119869 | 1 | 1 | 23.5034695 | 23.5737042 | 0.872359037 | 0.865667284 | 0.0817532955 | -0.0582612537 | 0.154349309 |
| C1::dna_v1_medium::single::DNA_stronger | VALID | 4 | 35 | 0 | 0.999999982 | 1 | 1 | 23.5034695 | 23.2844202 | 0.872359037 | 0.861492634 | 0.30833209 | 0.189076705 | 0.427715449 |
| C1::dna_v1_medium::single::comparator_stronger | VALID | 35 | 4 | 0 | 1.67658072e-07 | 3.0178453e-06 | 1.60951749e-05 | 23.5034695 | 23.2844202 | 0.872359037 | 0.861492634 | 0.30833209 | 0.189076705 | 0.427715449 |
| C1::dna_v1_stronger::unprotected::DNA_stronger | VALID | 35 | 4 | 0 | 1.67658072e-07 | 3.0178453e-06 | 1.60951749e-05 | 23.3488725 | 23.725499 | 0.868665397 | 0.879140675 | -0.264525672 | -0.437552873 | -0.162145877 |
| C1::dna_v1_stronger::unprotected::comparator_stronger | VALID | 4 | 35 | 0 | 0.999999982 | 1 | 1 | 23.3488725 | 23.725499 | 0.868665397 | 0.879140675 | -0.264525672 | -0.437552873 | -0.162145877 |
| C1::dna_v1_stronger::distortion::DNA_stronger | VALID | 14 | 25 | 0 | 0.973374043 | 1 | 1 | 23.3488725 | 23.2294313 | 0.868665397 | 0.862981558 | 0.137718274 | -0.0636935632 | 0.252035195 |
| C1::dna_v1_stronger::distortion::comparator_stronger | VALID | 25 | 14 | 0 | 0.0540645107 | 0.75690315 | 1 | 23.3488725 | 23.2294313 | 0.868665397 | 0.862981558 | 0.137718274 | -0.0636935632 | 0.252035195 |
| C1::dna_v1_stronger::single::DNA_stronger | VALID | 12 | 27 | 0 | 0.995262348 | 1 | 1 | 23.3488725 | 23.3496028 | 0.868665397 | 0.862890482 | 0.13495846 | 0.0189705801 | 0.242783556 |
| C1::dna_v1_stronger::single::comparator_stronger | VALID | 27 | 12 | 0 | 0.0118513512 | 0.177770269 | 0.948108099 | 23.3488725 | 23.3496028 | 0.868665397 | 0.862890482 | 0.13495846 | 0.0189705801 | 0.242783556 |
| D::image::dna_v1_conservative::per_tensor::DNA_stronger | VALID | 0 | 39 | 0 | 1 | 1 | 1 | 23.6334928 | 18.55081 | 0.875396073 | 0.63682133 | 4.93793133 | 4.36602628 | 6.6218333 |
| D::image::dna_v1_conservative::per_tensor::comparator_stronger | VALID | 39 | 0 | 0 | 1.8189894e-12 | 3.63797881e-11 | 2.45563569e-10 | 23.6334928 | 18.55081 | 0.875396073 | 0.63682133 | 4.93793133 | 4.36602628 | 6.6218333 |
| D::PaySim::dna_v1_conservative::per_tensor::DNA_stronger | NOT_ASSESSABLE | — | — | — | 1 | 1 | 1 | — | — | — | — | — | — | — |
| D::PaySim::dna_v1_conservative::per_tensor::comparator_stronger | NOT_ASSESSABLE | — | — | — | 1 | 1 | 1 | — | — | — | — | — | — | — |
| D::image::dna_v2_0p95::per_tensor::DNA_stronger | VALID | 0 | 39 | 0 | 1 | 1 | 1 | 23.7319918 | 18.5284483 | 0.877713025 | 0.632530928 | 5.09783781 | 4.51710596 | 6.45825864 |
| D::image::dna_v2_0p95::per_tensor::comparator_stronger | VALID | 39 | 0 | 0 | 1.8189894e-12 | 3.63797881e-11 | 2.45563569e-10 | 23.7319918 | 18.5284483 | 0.877713025 | 0.632530928 | 5.09783781 | 4.51710596 | 6.45825864 |
| D::PaySim::dna_v2_0p95::per_tensor::DNA_stronger | NOT_ASSESSABLE | — | — | — | 1 | 1 | 1 | — | — | — | — | — | — | — |
| D::PaySim::dna_v2_0p95::per_tensor::comparator_stronger | NOT_ASSESSABLE | — | — | — | 1 | 1 | 1 | — | — | — | — | — | — | — |

## C2 descriptive sensitivity

| setting | arm | n | median_psnr | median_ssim | psnr_min | psnr_max |
| --- | --- | --- | --- | --- | --- | --- |
| C2_batch4 | dp_single_for_dna_v2_0p95 | 24 | 18.8959618 | 0.631494153 | 15.5069775 | 23.1073612 |
| C2_batch4 | unprotected | 24 | 19.170486 | 0.634588659 | 15.715251 | 21.9088901 |
| C2_batch4 | dna_v1_conservative | 24 | 19.1081485 | 0.625145912 | 14.8154785 | 23.2142125 |
| C2_batch4 | dp_single_for_dna_v1_conservative | 24 | 17.8452998 | 0.553276764 | 14.1708489 | 20.8066553 |
| C2_batch4 | dna_v2_0p95 | 24 | 19.3470454 | 0.636472896 | 14.2472126 | 22.7812819 |
| C2_trained | dp_single_for_dna_v2_0p95 | 24 | 21.7488972 | 0.826420128 | 13.8812827 | 29.0146115 |
| C2_trained | unprotected | 24 | 22.1328378 | 0.823707849 | 14.8843821 | 28.4668245 |
| C2_trained | dna_v1_conservative | 24 | 21.9103962 | 0.801188499 | 14.3606392 | 27.0085891 |
| C2_trained | dp_single_for_dna_v1_conservative | 24 | 20.025927 | 0.774957001 | 13.4424887 | 27.4795117 |
| C2_trained | dna_v2_0p95 | 24 | 21.8410329 | 0.807107329 | 15.0161944 | 28.4856737 |

No confirmatory p-values or n39 intervals are attached to n24 sensitivity.

## All Priority33 evidence

| priority | dataset | method | comparator | status | metric | median_dna | median_comparator | delta | interval | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33a | ieee_cis | dna_v1_conservative | distortion | NOT_ASSESSABLE | batch_mean_standardized_MSE | None | None | None | None | Unprotected batch-mean qualification n8 failed |
| 33a | ieee_cis | dna_v1_conservative | utility | NOT_ASSESSABLE | batch_mean_standardized_MSE | None | None | None | None | Unprotected batch-mean qualification n8 failed |
| 33a | ieee_cis | dna_v2_0p95 | distortion | NOT_ASSESSABLE | batch_mean_standardized_MSE | None | None | None | None | Unprotected batch-mean qualification n8 failed |
| 33a | ieee_cis | dna_v2_0p95 | utility | NOT_ASSESSABLE | batch_mean_standardized_MSE | None | None | None | None | Unprotected batch-mean qualification n8 failed |
| 33a | baf | dna_v1_conservative | distortion | VALID | batch_mean_standardized_MSE | 18.471632 | 36.6375621 | -23.8502971 | [-33.030458596365996, -13.017699350634839] | None |
| 33a | baf | dna_v1_conservative | utility | NOT_ASSESSABLE | batch_mean_standardized_MSE | None | None | None | None | Required full-state CPU transform violates BN/finite training gate |
| 33a | baf | dna_v2_0p95 | distortion | VALID | batch_mean_standardized_MSE | 0.0449902835 | 152.964377 | -152.903744 | [-182.61600155519366, -104.01579599771509] | None |
| 33a | baf | dna_v2_0p95 | utility | NOT_ASSESSABLE | batch_mean_standardized_MSE | None | None | None | None | Required full-state CPU transform violates BN/finite training gate |
| 33b | ieee_cis | dna_v1_conservative | distortion | NOT_ASSESSABLE | record_feature_accuracy | — | — | — | — | qualification failed |
| 33b | ieee_cis | dna_v1_conservative | utility | NOT_ASSESSABLE | record_feature_accuracy | — | — | — | — | qualification failed |
| 33b | ieee_cis | dna_v2_0p95 | distortion | NOT_ASSESSABLE | record_feature_accuracy | — | — | — | — | qualification failed |
| 33b | ieee_cis | dna_v2_0p95 | utility | NOT_ASSESSABLE | record_feature_accuracy | — | — | — | — | qualification failed |
| 33b | baf | dna_v1_conservative | distortion | NOT_ASSESSABLE | record_feature_accuracy | — | — | — | — | qualification failed |
| 33b | baf | dna_v1_conservative | utility | NOT_ASSESSABLE | record_feature_accuracy | — | — | — | — | qualification failed |
| 33b | baf | dna_v2_0p95 | distortion | NOT_ASSESSABLE | record_feature_accuracy | — | — | — | — | qualification failed |
| 33b | baf | dna_v2_0p95 | utility | NOT_ASSESSABLE | record_feature_accuracy | — | — | — | — | qualification failed |
| 33c | CIFAR10 | C1::dna_v1_medium::unprotected::DNA_stronger | unprotected | VALID | PSNR | 23.5034695 | 23.725499 | -0.193276557 | [-0.32491323747662193, -0.08571669343453792] | None |
| 33c | CIFAR10 | C1::dna_v1_medium::distortion::DNA_stronger | distortion | VALID | PSNR | 23.5034695 | 23.5737042 | 0.0817532955 | [-0.05826125369505064, 0.15434930903880684] | None |
| 33c | CIFAR10 | C1::dna_v1_medium::single::DNA_stronger | single | VALID | PSNR | 23.5034695 | 23.2844202 | 0.30833209 | [0.18907670548800226, 0.42771544920377025] | None |
| 33c | CIFAR10 | C1::dna_v1_stronger::unprotected::DNA_stronger | unprotected | VALID | PSNR | 23.3488725 | 23.725499 | -0.264525672 | [-0.4375528727790936, -0.16214587734307173] | None |
| 33c | CIFAR10 | C1::dna_v1_stronger::distortion::DNA_stronger | distortion | VALID | PSNR | 23.3488725 | 23.2294313 | 0.137718274 | [-0.06369356318401742, 0.25203519461067003] | None |
| 33c | CIFAR10 | C1::dna_v1_stronger::single::DNA_stronger | single | VALID | PSNR | 23.3488725 | 23.3496028 | 0.13495846 | [0.018970580057061426, 0.24278355633781246] | None |
| 33c | CIFAR10 | D::image::dna_v1_conservative::per_tensor::DNA_stronger | per_tensor | VALID | PSNR | 23.6334928 | 18.55081 | 4.93793133 | [4.366026281321062, 6.62183329708083] | None |
| 33c | PaySim | D::PaySim::dna_v1_conservative::per_tensor::DNA_stronger | per_tensor | NOT_ASSESSABLE | None | None | None | None | [None, None] | no BN-valid full-state CPU utility target and bracketed comparator |
| 33c | CIFAR10 | D::image::dna_v2_0p95::per_tensor::DNA_stronger | per_tensor | VALID | PSNR | 23.7319918 | 18.5284483 | 5.09783781 | [4.5171059635766895, 6.45825863812691] | None |
| 33c | PaySim | D::PaySim::dna_v2_0p95::per_tensor::DNA_stronger | per_tensor | NOT_ASSESSABLE | None | None | None | None | [None, None] | no BN-valid full-state CPU utility target and bracketed comparator |

BN recovery is a four-record batch mean, not individual records. TabLeak33b is single-gradient batch8, not multi-step Adam. Its failed two-reference qualification leaves BAF/IEEE record cells not assessable. Adult has no new P33 arm; historical results remain in existing115.

## Independent checks and disclosures

SciPy sign tests and independent vectorized Holm agree; 669 saved reconstruction arrays reloaded. Native payload identity/unknown-mode/identity-loss and recovery tests passed before attacks. P31 stored no checkpoint, so a separately hashed exact baseline replay was required.

See per-target CSVs, image_gated_cells.json, calibration job failures and numerical logs. No failed target is dropped from an inferential comparison. Raw/checkpoint/code/output hashes are in sha256_manifest.csv; final_checks.json and COMPLETE.json are required for final completion.

