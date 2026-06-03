# FL-DNA PaySim Prototype

Prototype Federated Learning + DNA Encoder cho fraud detection trên PaySim. File
dataset đang đặt tại `datasets/creditcard.csv`, nhưng schema được xử lý theo
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
│   └── encoder.py
├── privacy/
│   └── dp_engine.py
├── experiments/
│   ├── fraud_fl_common.py
│   ├── run_fraud_centralized.py
│   ├── run_fraud_fl_baseline.py
│   ├── run_fraud_fl_dna.py
│   ├── run_fraud_fl_dp.py
│   ├── run_fraud_fl_dna_dp.py
│   ├── compare_fraud_results.py
│   └── run_all_quick.py
└── results/fraud/
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
../venv/bin/python experiments/compare_fraud_results.py
```

For stronger preliminary results, use more rows and full rounds:

```bash
NUM_ROUNDS=20 LOCAL_EPOCHS=1 MAX_ROWS=1000000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_centralized.py
NUM_ROUNDS=20 LOCAL_EPOCHS=1 MAX_ROWS=1000000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_fl_baseline.py
NUM_ROUNDS=20 LOCAL_EPOCHS=1 MAX_ROWS=1000000 LOSS_TYPE=focal ../venv/bin/python experiments/run_fraud_fl_dna.py
NUM_ROUNDS=20 LOCAL_EPOCHS=1 MAX_ROWS=1000000 LOSS_TYPE=focal DP_CLIP_NORM=100 DP_NOISE_MULTIPLIER=0.0005 ../venv/bin/python experiments/run_fraud_fl_dp.py
NUM_ROUNDS=20 LOCAL_EPOCHS=1 MAX_ROWS=1000000 LOSS_TYPE=focal DP_CLIP_NORM=100 DP_NOISE_MULTIPLIER=0.0005 ../venv/bin/python experiments/run_fraud_fl_dna_dp.py
../venv/bin/python experiments/compare_fraud_results.py
```

Omit `MAX_ROWS` to use the full CSV.

## Current 500k / 15-Round Focal-Loss Results

Final round from `results/fraud/comparison_summary.json`:

```text
Method       Loss       F1       ROC-AUC    PR-AUC    Precision  Recall
Centralized  0.000372  0.704348  0.990417  0.702266  0.801980   0.627907
FL_Baseline  0.000450  0.642857  0.987949  0.653688  0.658537   0.627907
FL_DNA       0.000453  0.627273  0.987272  0.648983  0.758242   0.534884
FL_DP        0.000643  0.547368  0.979132  0.548452  0.852459   0.403101
FL_DNA_DP    0.000636  0.549020  0.978941  0.550767  0.746667   0.434109
```

Confusion matrix values are saved as `tn`, `fp`, `fn`, `tp` per round.

Final confusion values:

```text
Centralized: tn=99852, fp=20, fn=48, tp=81
FL_Baseline: tn=99830, fp=42, fn=48, tp=81
FL_DNA:      tn=99850, fp=22, fn=60, tp=69
FL_DP:       tn=99863, fp=9,  fn=77, tp=52
FL_DNA_DP:   tn=99853, fp=19, fn=73, tp=56
```

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
