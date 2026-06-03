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
NUM_ROUNDS=15 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal DP_CLIP_NORM=100 DP_NOISE_MULTIPLIER=0.0005 ../venv/bin/python experiments/run_fraud_fl_dp.py
NUM_ROUNDS=15 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal DP_CLIP_NORM=100 DP_NOISE_MULTIPLIER=0.0005 ../venv/bin/python experiments/run_fraud_fl_dna_dp.py
NUM_ROUNDS=15 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_fl_secureagg.py
NUM_ROUNDS=15 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_fl_dna_secureagg.py
NUM_ROUNDS=15 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal DNA_TRANSFORM_MIX=0.05 DNA_TRANSFORM_KEEP=0.90 DNA_TRANSFORM_SHRINK=0.50 ../venv/bin/python experiments/run_fraud_fl_dna_transform.py
NUM_ROUNDS=15 LOCAL_EPOCHS=1 MAX_ROWS=500000 LOSS_TYPE=focal DNA_TRANSFORM_MIX=0.05 DNA_TRANSFORM_KEEP=0.90 DNA_TRANSFORM_SHRINK=0.50 ../venv/bin/python experiments/run_fraud_fl_dna_transform_secureagg.py
../venv/bin/python experiments/compare_fraud_results.py
```

For stronger preliminary results, use more rows and full rounds:

```bash
NUM_ROUNDS=20 LOCAL_EPOCHS=1 MAX_ROWS=1000000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_centralized.py
NUM_ROUNDS=20 LOCAL_EPOCHS=1 MAX_ROWS=1000000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_fl_baseline.py
NUM_ROUNDS=20 LOCAL_EPOCHS=1 MAX_ROWS=1000000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_fl_dna.py
NUM_ROUNDS=20 LOCAL_EPOCHS=1 MAX_ROWS=1000000 LOSS_TYPE=focal DP_CLIP_NORM=100 DP_NOISE_MULTIPLIER=0.0005 ../venv/bin/python experiments/run_fraud_fl_dp.py
NUM_ROUNDS=20 LOCAL_EPOCHS=1 MAX_ROWS=1000000 LOSS_TYPE=focal DP_CLIP_NORM=100 DP_NOISE_MULTIPLIER=0.0005 ../venv/bin/python experiments/run_fraud_fl_dna_dp.py
NUM_ROUNDS=20 LOCAL_EPOCHS=1 MAX_ROWS=1000000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_fl_secureagg.py
NUM_ROUNDS=20 LOCAL_EPOCHS=1 MAX_ROWS=1000000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_fl_dna_secureagg.py
NUM_ROUNDS=20 LOCAL_EPOCHS=1 MAX_ROWS=1000000 LOSS_TYPE=focal DNA_TRANSFORM_MIX=0.05 DNA_TRANSFORM_KEEP=0.90 DNA_TRANSFORM_SHRINK=0.50 ../venv/bin/python experiments/run_fraud_fl_dna_transform.py
NUM_ROUNDS=20 LOCAL_EPOCHS=1 MAX_ROWS=1000000 LOSS_TYPE=focal DNA_TRANSFORM_MIX=0.05 DNA_TRANSFORM_KEEP=0.90 DNA_TRANSFORM_SHRINK=0.50 ../venv/bin/python experiments/run_fraud_fl_dna_transform_secureagg.py
../venv/bin/python experiments/compare_fraud_results.py
```

Omit `MAX_ROWS` to use the full CSV.

Run gradient inversion privacy evaluation:

```bash
ATTACK_NUM_SAMPLES=3 ATTACK_ITERATIONS=300 ATTACK_WARMUP_ROUNDS=3 MAX_ROWS=500000 ../venv/bin/python attacks/attack_runner.py
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

## Current 500k / 15-Round Focal-Loss Results

Final round from `results/fraud/comparison_summary.json`:

```text
Method       Loss       F1       ROC-AUC    PR-AUC    Precision  Recall
Centralized  0.000372  0.704348  0.990417  0.702266  0.801980   0.627907
FL_Baseline  0.000450  0.642857  0.987949  0.653688  0.658537   0.627907
FL_DNA       0.000453  0.627273  0.987272  0.648983  0.758242   0.534884
FL_DP        0.000643  0.547368  0.979132  0.548452  0.852459   0.403101
FL_DNA_DP    0.000636  0.549020  0.978941  0.550767  0.746667   0.434109
FL_SecureAgg 0.000449  0.635945  0.987221  0.649952  0.784091   0.534884
FL_DNA_SA    0.000449  0.635945  0.987221  0.649952  0.784091   0.534884
FL_DNA_TD    0.000454  0.644068  0.987734  0.654060  0.710280   0.589147
FL_DNA_TD_SA 0.000453  0.644628  0.988291  0.655462  0.690265   0.604651
```

Confusion matrix values are saved as `tn`, `fp`, `fn`, `tp` per round.

Final confusion values:

```text
Centralized: tn=99852, fp=20, fn=48, tp=81
FL_Baseline: tn=99830, fp=42, fn=48, tp=81
FL_DNA:      tn=99850, fp=22, fn=60, tp=69
FL_DP:       tn=99863, fp=9,  fn=77, tp=52
FL_DNA_DP:   tn=99853, fp=19, fn=73, tp=56
FL_SecureAgg: tn=99853, fp=19, fn=60, tp=69
FL_DNA_SA:    tn=99853, fp=19, fn=60, tp=69
FL_DNA_TD:    tn=99841, fp=31, fn=53, tp=76
FL_DNA_TD_SA: tn=99837, fp=35, fn=51, tp=78
```

DNA transform diagnostics from final round:

```text
FL_DNA_TD:
  relative_l2_delta = 0.065085
  cosine_similarity = 0.998725

FL_DNA_TD_SA:
  relative_l2_delta = 0.064462
  cosine_similarity = 0.998737
  server_sees_individual_raw_updates = false
```

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

```text
Method                         Threat model                         MSE mean   PSNR mean  SSIM mean  Cosine mean
FL_Baseline                    raw_visible_update                    755.5571    16.6596     0.5627       0.6562
FL_DNA                         raw_visible_update                    755.5571    16.6596     0.5627       0.6562
FL_DP                          raw_visible_update                    756.0266    16.8356     0.5681       0.6699
FL_DNA_TransformDefense        raw_visible_update                    755.6067    16.3006     0.5484       0.6498
FL_SecureAgg                   secure_aggregation_server_side       not directly applicable
FL_DNA_SecureAgg               secure_aggregation_server_side       not directly applicable
FL_DNA_TD_SecureAgg            secure_aggregation_server_side       not directly applicable
FL_PreAggregationLeakage       analysis_only_pre_aggregation         747.5173    17.8280     0.5958       0.7451
FL_DNA_PreAggregationLeakage   analysis_only_pre_aggregation         747.5173    17.8280     0.5958       0.7451
FL_DNA_TD_PreAggregationLeak   analysis_only_pre_aggregation         747.5547    17.4811     0.5856       0.7412
```

Interpretation:

- DNA encode/decode thuần không làm giảm inversion trong raw-visible setting;
  kết quả trùng baseline vì update được restore gần bit-exact.
- DNA Transform làm reconstruction khó hơn một chút trong run này: PSNR, SSIM
  và cosine đều thấp hơn baseline, nhưng mức giảm còn nhỏ.
- DP với config utility-friendly hiện tại chưa làm attack metric giảm rõ. Muốn
  privacy mạnh hơn cần tăng noise hoặc giảm clip norm, nhưng utility PaySim có
  thể giảm mạnh.
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

Utility sweep for 20-round FL DNA Transform on 500k rows:

```text
Variant       Final F1  ROC-AUC  PR-AUC   Peak F1
current        0.6275   0.9883   0.6559   0.6723
conservative   0.6667   0.9886   0.6634   0.6667
medium         0.6480   0.9875   0.6566   0.6579
stronger       0.6307   0.9872   0.6576   0.6556
```

Current recommendation: `conservative` is the best utility/privacy trade-off in
these runs. `stronger` reduces reconstruction metrics slightly more, but the FL
run shows more unstable rounds with F1 collapsing to zero, so it is likely
over-perturbing the update path for PaySim.

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
- DNA and Baseline should be nearly identical. If not, check update
  encode/decode bit-exactness.
- DP uses full client-update clipping plus Gaussian noise. Start with
  `DP_CLIP_NORM=100` and `DP_NOISE_MULTIPLIER=0.0005`; lower clip norms or larger
  noise multipliers can collapse learning on PaySim.
- `FL_DNA_DP` applies DP first, then DNA encode/decode transports the protected
  update. This keeps the hybrid path interpretable: DP is privacy, DNA is
  communication protection.
- Secure Aggregation is a simulation, not production MPC. It models the privacy
  surface that the server sees only the aggregate masked sum, not individual
  client updates. It does not add noise, so utility should stay close to FL
  Baseline while hiding per-client updates from the server.
- In the current run, Secure Aggregation keeps much more utility than DP
  (`F1=0.635945` vs `0.547368`) because it hides individual updates without
  perturbing the aggregate. This is communication/update-path privacy, not a
  formal record-level DP guarantee.
- DNA Transform Defense is the current DNA-as-core-defense prototype. Unlike
  DNA encode/decode, it changes the numeric update before aggregation using
  DNA-seeded block rules. It keeps utility better than DP in this run
  (`F1=0.644068` vs `0.547368`) while reducing direct equivalence between the
  transmitted update and the raw client update.
