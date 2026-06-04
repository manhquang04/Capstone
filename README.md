# FL-DNA PaySim Prototype

Prototype Federated Learning + DNA Encoder cho fraud detection trên PaySim. File
code chính nằm trong `FL-DNA/`. Các lệnh chạy bên dưới giả định bạn đang đứng
trong thư mục `FL-DNA/`. File dataset đang đặt tại
`FL-DNA/datasets/creditcard.csv`, nhưng schema được xử lý theo
PaySim:

```text
step, type, amount, oldbalanceOrg, newbalanceOrig,
oldbalanceDest, newbalanceDest, isFraud
```

Các cột `nameOrig`, `nameDest` bị bỏ vì là ID định danh có cardinality rất cao
và không phù hợp với prototype MLP đơn giản. `isFlaggedFraud` cũng không dùng để
tránh phụ thuộc vào rule có sẵn của simulator.

## Architecture

```text
PaySim CSV
  -> feature engineering
  -> RobustScaler numeric features
  -> one-hot encode type
  -> stratified train/validation/test split
  -> mild non-IID client partition
  -> local MLP training
  -> FedAvg
```

DNA communication path:

```text
local model update
  -> float32
  -> binary
  -> DNA nucleotide mapping
  -> AES-256-GCM encrypted payload
  -> decrypt
  -> DNA decode
  -> float32 restored update
  -> FedAvg aggregation
```

DNA Encoder chỉ được áp dụng cho model updates sau local training. Raw input
features không bị encode.

DP update path:

```text
local model update
  -> full-update L2 clipping
  -> Gaussian noise
  -> FedAvg aggregation
```

Hybrid path:

```text
local model update
  -> DP clipping + Gaussian noise
  -> DNA encode/decode protected update
  -> FedAvg aggregation
```

Secure Aggregation path:

```text
local model update
  -> pairwise random masks across clients
  -> server sums masked weighted updates
  -> masks cancel in aggregate
  -> server applies only aggregate update
```

DNA Transform Defense path:

```text
local model update
  -> flatten into fixed-size blocks
  -> map each block to binary and DNA sequence
  -> derive DNA-sequence seed per block
  -> sequence-seeded permutation + selective low-energy attenuation
  -> residual mixing back into numeric update
  -> FedAvg aggregation
```

## Structure

```text
FL-DNA/
├── data/
│   └── load_creditcard.py
├── models/
│   └── fraud_mlp.py
├── dna_encoder/
│   ├── aes_crypto.py
│   ├── binary_mapper.py
│   ├── dna_mapper.py
│   ├── transform_defense.py
│   └── encoder.py
├── privacy/
│   ├── dp_config.py
│   ├── dp_engine.py
│   └── secure_agg.py
├── attacks/
│   ├── gradient_inversion.py
│   ├── inversion_metrics.py
│   ├── pseudo_image.py
│   └── attack_runner.py
├── experiments/
│   ├── fraud_fl_common.py
│   ├── run_fraud_centralized.py
│   ├── run_fraud_fl_baseline.py
│   ├── run_fraud_fl_dna.py
│   ├── run_fraud_fl_dp.py
│   ├── run_fraud_fl_dna_dp.py
│   ├── run_fraud_fl_secureagg.py
│   ├── run_fraud_fl_dna_secureagg.py
│   ├── run_fraud_fl_dna_transform.py
│   ├── run_fraud_fl_dna_transform_secureagg.py
│   ├── compare_fraud_results.py
│   ├── run_attack_sweep.py
│   ├── compare_attack_sweeps.py
│   └── run_all_quick.py
└── results/fraud/
└── artifacts/gradient_inversion/
```

## Model

MLP tabular classifier:

```text
input
-> Linear 128 + BatchNorm + ReLU + Dropout
-> Linear 64  + BatchNorm + ReLU + Dropout
-> Linear 32  + BatchNorm + ReLU + Dropout
-> Linear 1 logits
```

Training mặc định dùng binary focal loss (`alpha=0.95`, `gamma=2.0`) để xử lý
imbalance. Có thể đổi về weighted BCE bằng `LOSS_TYPE=weighted_bce`. Evaluation
dùng `sigmoid(logits)` và threshold được tune trên validation split để tối ưu F1.

## Run

Use the project venv:

```bash
cd /Users/manhquang/Documents/UNIVERSITY/Capstone/FL-DNA
../venv/bin/python experiments/run_all_quick.py --quick
```

Quick mode tự set:

```text
QUICK=1
MAX_ROWS=500000
NUM_ROUNDS=3
LOCAL_EPOCHS=1
```

Run fraud experiments individually:

```bash
NUM_ROUNDS=15 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_centralized.py
NUM_ROUNDS=15 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_fl_baseline.py
NUM_ROUNDS=15 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_fl_dna.py
NUM_ROUNDS=15 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal DP_CLIP_NORM=100 DP_NOISE_PRESET=utility DP_OUTPUT_PATH=results/fraud/dp_utility_metrics.json ../venv/bin/python experiments/run_fraud_fl_dp.py
NUM_ROUNDS=15 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal DP_CLIP_NORM=100 DP_NOISE_PRESET=medium DP_OUTPUT_PATH=results/fraud/dp_metrics.json ../venv/bin/python experiments/run_fraud_fl_dp.py
NUM_ROUNDS=15 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal DP_CLIP_NORM=100 DP_NOISE_PRESET=medium ../venv/bin/python experiments/run_fraud_fl_dna_dp.py
NUM_ROUNDS=15 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_fl_secureagg.py
NUM_ROUNDS=15 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_fl_dna_secureagg.py
NUM_ROUNDS=15 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal DNA_TRANSFORM_MIX=0.05 DNA_TRANSFORM_KEEP=0.90 DNA_TRANSFORM_SHRINK=0.50 ../venv/bin/python experiments/run_fraud_fl_dna_transform.py
NUM_ROUNDS=15 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal DNA_TRANSFORM_MIX=0.05 DNA_TRANSFORM_KEEP=0.90 DNA_TRANSFORM_SHRINK=0.50 ../venv/bin/python experiments/run_fraud_fl_dna_transform_secureagg.py
../venv/bin/python experiments/compare_fraud_results.py
```

For publication-style results, use the official round count and keep
`NUM_ROUNDS` identical across every FL variant before using the result in the
main comparison table:

```bash
NUM_ROUNDS=50 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_centralized.py
NUM_ROUNDS=50 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_fl_baseline.py
NUM_ROUNDS=50 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_fl_dna.py
NUM_ROUNDS=50 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal DP_CLIP_NORM=100 DP_NOISE_PRESET=utility DP_OUTPUT_PATH=results/fraud/dp_utility_metrics.json ../venv/bin/python experiments/run_fraud_fl_dp.py
NUM_ROUNDS=50 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal DP_CLIP_NORM=100 DP_NOISE_PRESET=mild DP_OUTPUT_PATH=results/fraud/dp_mild_metrics.json ../venv/bin/python experiments/run_fraud_fl_dp.py
NUM_ROUNDS=50 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal DP_CLIP_NORM=100 DP_NOISE_PRESET=medium DP_OUTPUT_PATH=results/fraud/dp_metrics.json ../venv/bin/python experiments/run_fraud_fl_dp.py
NUM_ROUNDS=50 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_fl_secureagg.py
NUM_ROUNDS=50 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_fl_dna_secureagg.py
NUM_ROUNDS=50 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal DNA_TRANSFORM_MIX=0.05 DNA_TRANSFORM_KEEP=0.90 DNA_TRANSFORM_SHRINK=0.50 DNA_TRANSFORM_OUTPUT_PATH=results/fraud/dna_transform_metrics.json ../venv/bin/python experiments/run_fraud_fl_dna_transform.py
NUM_ROUNDS=50 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal DNA_TRANSFORM_MIX=0.08 DNA_TRANSFORM_KEEP=0.88 DNA_TRANSFORM_SHRINK=0.45 DNA_TRANSFORM_OUTPUT_PATH=results/fraud/dna_transform_conservative_metrics.json ../venv/bin/python experiments/run_fraud_fl_dna_transform.py
NUM_ROUNDS=50 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal DNA_TRANSFORM_MIX=0.10 DNA_TRANSFORM_KEEP=0.85 DNA_TRANSFORM_SHRINK=0.40 DNA_TRANSFORM_OUTPUT_PATH=results/fraud/dna_transform_medium_metrics.json ../venv/bin/python experiments/run_fraud_fl_dna_transform.py
NUM_ROUNDS=50 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal DNA_TRANSFORM_MIX=0.12 DNA_TRANSFORM_KEEP=0.82 DNA_TRANSFORM_SHRINK=0.35 DNA_TRANSFORM_OUTPUT_PATH=results/fraud/dna_transform_stronger_metrics.json ../venv/bin/python experiments/run_fraud_fl_dna_transform.py
NUM_ROUNDS=50 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal DNA_TRANSFORM_MIX=0.05 DNA_TRANSFORM_KEEP=0.90 DNA_TRANSFORM_SHRINK=0.50 DNA_TRANSFORM_SECUREAGG_OUTPUT_PATH=results/fraud/dna_transform_secureagg_metrics.json ../venv/bin/python experiments/run_fraud_fl_dna_transform_secureagg.py
../venv/bin/python experiments/compare_fraud_results.py
```

Omit `MAX_ROWS` to use the full CSV.

Run gradient inversion privacy evaluation:

```bash
ATTACK_NUM_SAMPLES=10 ATTACK_ITERATIONS=300 ATTACK_WARMUP_ROUNDS=3 MAX_ROWS=500000 ../venv/bin/python attacks/attack_runner.py
```

Outputs:

```text
artifacts/gradient_inversion/metrics_summary.csv
artifacts/gradient_inversion/metrics_summary.json
artifacts/gradient_inversion/metrics_details.json
artifacts/gradient_inversion/*_original.png
artifacts/gradient_inversion/*_reconstructed.png
artifacts/gradient_inversion/*_loss_curve.png
```

## Current 500k / 50-Round Focal-Loss Results

Final round from `results/fraud/comparison_summary.json`:

```text
Method                         Loss       F1       ROC-AUC    PR-AUC    Precision  Recall
Centralized                    0.000314  0.730159  0.994363  0.773102  0.747967   0.713178
FL_Baseline                    0.000389  0.728889  0.991189  0.714399  0.854167   0.635659
FL_DNA                         0.000405  0.718367  0.990692  0.717207  0.758621   0.682171
FL_DP_utility                  0.000417  0.692308  0.989669  0.692074  0.771429   0.627907
FL_DP_mild                     0.001628  0.275556  0.920598  0.209120  0.322917   0.240310
FL_DP_medium                   0.033034  0.008321  0.789137  0.003731  0.004183   0.775194
FL_SecureAgg                   0.000394  0.711864  0.990631  0.711759  0.785047   0.651163
FL_DNA_SecureAgg_lossless      0.000394  0.711864  0.990631  0.711759  0.785047   0.651163
FL_DNA_TD_current              0.000394  0.705882  0.991183  0.713083  0.770642   0.651163
FL_DNA_TD_conservative         0.000395  0.708861  0.990047  0.716046  0.777778   0.651163
FL_DNA_TD_medium               0.000426  0.685484  0.991031  0.706114  0.714286   0.658915
FL_DNA_TD_stronger             0.000412  0.696721  0.990390  0.711561  0.739130   0.658915
FL_DNA_TD_SecureAgg            0.000407  0.712446  0.990742  0.711959  0.798077   0.643411
```

Confusion matrix values are saved as `tn`, `fp`, `fn`, `tp` per round.

Final confusion values:

```text
Centralized:       tn=99841, fp=31,    fn=37, tp=92
FL_Baseline:       tn=99858, fp=14,    fn=47, tp=82
FL_DNA:            tn=99844, fp=28,    fn=41, tp=88
FL_DP_utility:     tn=99848, fp=24,    fn=48, tp=81
FL_DP_mild:        tn=99807, fp=65,    fn=98, tp=31
FL_DP_medium:      tn=76066, fp=23806, fn=29, tp=100
FL_SecureAgg:      tn=99849, fp=23,    fn=45, tp=84
FL_DNA_SecureAgg:  tn=99849, fp=23,    fn=45, tp=84
FL_DNA_TD_cons.:   tn=99848, fp=24,    fn=45, tp=84
```

The current comparison artifact uses `target_rounds=50`. All rows in the main
table above are 50-round runs. Older mismatched artifacts are kept only as
diagnostic/appendix evidence and are listed under
`excluded_due_round_mismatch` in `comparison_summary.json`.

DP uses full client-update clipping plus Gaussian noise. The current official
comparison includes a utility-friendly point (`utility`, noise multiplier
`0.0001`), a middle stress point (`mild`, `0.001`), and a high-noise stress
point (`medium`, `0.005`). There is no privacy accountant in this prototype, so
report these as clipping/noise defenses, not as formal `(epsilon, delta)` DP.

DNA Transform diagnostics from the 50-round artifacts:

```text
FL_DNA_TransformDefense current:
  relative_l2_delta = 0.065085
  cosine_similarity = 0.998725

FL_DNA_TransformDefense_SecureAgg current:
  relative_l2_delta = 0.064462
  cosine_similarity = 0.998737
  server_sees_individual_raw_updates = false
```

`FL_DNA_SecureAgg` is numerically identical to `FL_SecureAgg` on the core
utility metrics in the saved results. Treat it as lossless DNA transport /
overhead analysis after SecureAgg, not as a separate privacy defense.

## Gradient Inversion Attack Results

Attack protocol:

```text
same PaySim preprocessing
same FraudMLP backbone
same selected validation fraud samples
known-label gradient matching attack
300 Adam iterations
batch size = 1
```

PaySim is tabular, so PSNR and SSIM are computed on deterministic pseudo-images
created by reshaping the normalized feature vector. These metrics are useful for
visual comparison only. For tabular privacy, prefer feature MSE, cosine
similarity, Pearson correlation, and sign-match ratio.

Current attack summary from `artifacts/gradient_inversion/metrics_summary.csv`:

The CSV/JSON summaries report mean and standard deviation for `mse`,
`feature_mse`, `psnr`, `ssim`, `cosine_similarity`, `pearson_correlation`, and
`sign_match_ratio`. `feature_mse` is an explicit tabular alias of vector MSE so
the report does not rely only on pseudo-image metrics.

```text
Method                         Threat model                         MSE mean   Cosine mean  Pearson mean  Sign match  PSNR mean  SSIM mean
FL_Baseline                    raw_visible_update                   6305.8605      0.5791        0.6417      0.5692    15.3504     0.3474
FL_DNA                         raw_visible_update                   6305.8605      0.5791        0.6417      0.5692    15.3504     0.3474
FL_DP                          raw_visible_update                   6299.2578      0.3878        0.4373      0.4769    11.8857     0.1930
FL_DNA_TransformDefense        raw_visible_update                   6305.8838      0.5646        0.6301      0.5462    14.9589     0.3353
FL_SecureAgg                   secure_aggregation_server_side       not directly applicable
FL_DNA_SecureAgg               secure_aggregation_server_side       not directly applicable
FL_DNA_TD_SecureAgg            secure_aggregation_server_side       not directly applicable
FL_PreAggregationLeakage       analysis_only_pre_aggregation        6323.5538      0.5942        0.6635      0.5231    15.8328     0.3702
FL_DNA_PreAggregationLeakage   analysis_only_pre_aggregation        6323.5538      0.5942        0.6635      0.5231    15.8328     0.3702
FL_DNA_TD_PreAggregationLeak   analysis_only_pre_aggregation        6323.5911      0.5715        0.6397      0.5154    15.3553     0.3434
```

Interpretation:

- DNA encode/decode thuần không làm giảm inversion trong raw-visible setting;
  kết quả trùng baseline vì update được restore gần bit-exact.
- DNA Transform làm reconstruction khó hơn một chút trong run này: cosine,
  Pearson, sign-match, PSNR và SSIM đều thấp hơn baseline, nhưng mức giảm còn
  nhỏ.
- DP utility keeps utility reasonably close to FL Baseline in the 50-round run
  (`F1=0.692308` vs baseline `0.728889`) but offers little attack degradation.
  DP medium reduces reconstruction similarity much more, but utility collapses
  (`F1=0.008321`). This is the intended privacy-utility tradeoff evidence.
- Secure Aggregation thay đổi threat model: server-side attacker không thấy
  individual raw update, nên sample-level individual-update inversion không áp
  dụng trực tiếp. Các dòng `PreAggregationLeakage` chỉ là upper-bound analysis
  nếu update cá nhân bị lộ trước aggregation.
- Story mạnh nhất hiện tại là `DNA Transform + SecureAgg`: DNA biến đổi update
  trước khi gửi, SecureAgg che individual update khỏi server. Đây vẫn chưa phải
  formal DP guarantee.

## Attack Sweeps

Run the full attack sweep:

```bash
../venv/bin/python experiments/run_attack_sweep.py --mode all --iterations 300 --max-rows 500000
../venv/bin/python experiments/compare_attack_sweeps.py
```

Saved sweep outputs:

```text
artifacts/gradient_inversion/metrics_by_strength.csv
artifacts/gradient_inversion/metrics_by_sample_count.csv
artifacts/gradient_inversion/metrics_by_round.csv
artifacts/gradient_inversion/metrics_by_round_group.csv
artifacts/gradient_inversion/metrics_by_dp_noise.csv
artifacts/gradient_inversion/privacy_utility_tradeoff.csv
artifacts/gradient_inversion/sweep_report.json
```

DNA Transform strength variants:

```text
current:      mix=0.05 keep=0.90 shrink=0.50
conservative: mix=0.08 keep=0.88 shrink=0.45
medium:       mix=0.10 keep=0.85 shrink=0.40
stronger:     mix=0.12 keep=0.82 shrink=0.35
```

Direct server-side attack, 10 samples, warmup round 3:

```text
Variant       Mean MSE    Mean PSNR  Mean SSIM  Mean cosine
current       6305.8838    14.9589     0.3353      0.5646
conservative  6305.9086    14.7204     0.3251      0.5543
medium        6305.9141    14.6456     0.3185      0.5503
stronger      6305.9195    14.5939     0.3152      0.5484
```

Sample-count stability for current DNA Transform:

```text
Samples  Baseline SSIM  DNA-TD SSIM  Baseline cosine  DNA-TD cosine
10          0.3474        0.3353         0.5791          0.5646
20          0.2683        0.2600         0.5758          0.5673
30          0.2579        0.2457         0.5671          0.5545
```

Round-group stability for current DNA Transform:

```text
Round group      Baseline SSIM  DNA-TD SSIM  Baseline cosine  DNA-TD cosine
early 1,2,3         0.3176        0.3008         0.5305          0.5094
mid 7,8             0.3412        0.3348         0.5660          0.5639
late 13,14,15       0.2770        0.2617         0.5211          0.5077
```

Utility sweep for 50-round FL DNA Transform on 500k rows:

```text
Variant       Final F1  ROC-AUC  PR-AUC   Peak F1
current        0.7059   0.9912   0.7131   0.7124
conservative   0.7089   0.9900   0.7160   0.7155
medium         0.6855   0.9910   0.7061   0.6975
stronger       0.6967   0.9904   0.7116   0.7039
```

Current recommendation: `conservative` is the best utility/privacy trade-off in
these runs. `stronger` reduces reconstruction metrics slightly more in the
attack sweep, but its utility is lower than conservative and it has more
zero-F1 diagnostic rounds, so it is likely over-perturbing the update path for
PaySim.

DP noise attack sweep, direct server-side `FL_DP`, 10 samples, warmup round 3:

```text
Preset  Noise   Mean cosine  Mean Pearson  Sign match  Mean PSNR  Mean SSIM
utility 0.0001     0.6036        0.6655       0.5846      15.6484    0.3676
weak    0.0005     0.5770        0.6349       0.5154      15.4314    0.3472
mild    0.0010     0.5743        0.6250       0.4923      15.3902    0.3506
medium  0.0050     0.3878        0.4373       0.4769      11.8857    0.1930
strong  0.0100     0.2468        0.2922       0.5000      11.0453    0.1061
```

`privacy_utility_tradeoff.csv` joins attack metrics with utility artifacts when
matching files exist. The current tradeoff report has matching 50-round utility
artifacts for `utility`, `mild`, and `medium`; `weak` and `strong` remain
attack-only unless their FL utility runs are generated separately. This exposes
missing evidence instead of inventing a privacy-utility conclusion.

Current non-IID diagnostics:

```text
client_sample_counts = [132143, 118991, 73866]
client_fraud_rates   = [0.001059, 0.001177, 0.001882]
```

The split is intentionally type-skewed but not label-collapsed: every client has
fraud samples, and fraud rates stay near the global fraud rate.

## Practical Notes

- PaySim is highly imbalanced, so raw accuracy is not useful.
- Always tune threshold on validation; default `0.5` often gives poor recall/F1.
- Use focal loss for the main prototype, then compare against
  `LOSS_TYPE=weighted_bce` as an ablation.
- Use at least `MAX_ROWS=500000`; tiny samples may contain too few fraud cases in
  validation/test and produce unstable F1.
- Keep `NUM_ROUNDS`, `LOCAL_EPOCHS`, split seed, preprocessing, model, loss, and
  threshold tuning fixed across FL variants before making a paper comparison.
  `compare_fraud_results.py` excludes saved variants whose round count differs
  from the target comparison round.
- DNA and Baseline should be nearly identical. If not, check update
  encode/decode bit-exactness.
- DP uses full client-update clipping plus Gaussian noise. Presets are
  `utility=0.0001`, `weak=0.0005`, `mild=0.001`, `medium=0.005`,
  `strong=0.01`; default is `medium`. There is no privacy accountant, so do not claim formal
  `(epsilon, delta)` DP. Lower clip norms or larger noise multipliers can
  collapse learning on PaySim.
- `FL_DNA_DP` applies DP first, then DNA encode/decode transports the protected
  update. This keeps the hybrid path interpretable: DP is privacy, DNA is
  communication protection.
- Secure Aggregation is a simulation, not production MPC. It models the privacy
  surface that the server sees only the aggregate masked sum, not individual
  client updates. It does not add noise, so utility should stay close to FL
  Baseline while hiding per-client updates from the server.
- In the current 50-round run, Secure Aggregation keeps much more utility than
  medium-noise DP (`F1=0.711864` vs `0.008321`) because it hides individual
  updates without perturbing the aggregate. This is communication/update-path
  privacy, not a formal record-level DP guarantee.
- DNA Transform Defense is the current DNA-as-core-defense prototype. Unlike
  DNA encode/decode, it changes the numeric update before aggregation using
  DNA-seeded block rules. The official comparison now uses 50 rounds for
  baseline, DNA, DP, SecureAgg, and TransformDefense.
