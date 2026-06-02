# FL-DNA Project Status

## Project Overview

Research project for protecting distributed AI training gradients with DNA
encoding and AES-256-GCM encryption before future Federated Learning
integration and Gradient Inversion Attack evaluation.

## Current Phase

Preliminary MNIST Gradient Inversion Attack evaluation completed: Raw Gradient
vs DNA-protected Gradient vs DP-protected Gradient.

## Completed Tasks

- Environment setup planned and Python virtual environment available.
- MNIST download planned and currently available under `datasets/MNIST/`.
- Created `AGENTS.md`.
- Created the independent DNA Encoder module.
- Created `dna_encoder/test_encoder.py`.
- Created `experiments/benchmark_dna_encoder.py`.
- Verified bit-exact float32 encode and decode round trips.
- Added simulated FL baseline on Credit Card Fraud Detection.
- Added simulated FL with DNA Encoder protection before FedAvg aggregation.
- Added final-round F1-score and AUC-ROC comparison.
- Added Differential Privacy baseline with local-update clipping and Gaussian
  noise.
- Updated final metrics comparison for Baseline vs DNA vs DP.
- Added MNIST CNN training pipeline with local dataset loading only.
- Added raw, DNA-protected, and DP-protected MNIST gradient artifacts.
- Added lightweight iDLG-style Gradient Inversion Attack reconstruction.
- Added MSE, PSNR, and SSIM reconstruction evaluation and privacy summary.
- Improved MNIST Gradient Inversion Attack with `2,000` optimization
  iterations, true-label baseline matching, pixel-range constraints, and
  per-method progress logging.
- Added original MNIST image, original label metadata, and reconstruction
  comparison grid artifacts.
- Added `experiments/run_all_quick.py` to execute the complete preliminary
  results pipeline through subprocesses with realtime terminal output.
- Added timestamped pipeline logs under `logs/run_<timestamp>.log`.
- Added `--quick` mode with three fraud FL rounds and `500` inversion
  iterations while preserving five rounds and `2,000` iterations in full mode.
- Created a professional GitHub-style `README.md`.
- Completed project documentation with architecture, directory structure,
  datasets, implemented components, current results, setup instructions,
  execution guide, workflow diagram, and output artifact reference.

## In-Progress Tasks

- None.

## Next Tasks

Integrate Flower framework and repeat Baseline vs DNA vs DP in real Federated
Learning.

## Commands

Run the DNA Encoder test:

```bash
python dna_encoder/test_encoder.py
```

Run the DNA Encoder benchmark:

```bash
python experiments/benchmark_dna_encoder.py
```

Run the simulated Credit Card Fraud FL baseline:

```bash
python experiments/run_fraud_fl_baseline.py
```

Run simulated FL with DNA Encoder protection:

```bash
python experiments/run_fraud_fl_dna.py
```

Run simulated FL with the Differential Privacy baseline:

```bash
python experiments/run_fraud_fl_dp.py
```

Compare final-round fraud metrics:

```bash
python experiments/compare_fraud_results.py
```

Train the MNIST CNN:

```bash
python experiments/train_mnist_model.py
```

Generate MNIST gradient variants:

```bash
python attacks/gradient_extraction.py
python attacks/dna_gradient.py
python attacks/dp_gradient.py
```

Run reconstruction attack and privacy summary:

```bash
python attacks/gradient_inversion.py
python attacks/evaluate_reconstruction.py
python experiments/compare_privacy_results.py
```

Run the complete preliminary pipeline in quick mode:

```bash
python experiments/run_all_quick.py --quick
```

Run the complete preliminary pipeline in full mode:

```bash
python experiments/run_all_quick.py
```

## Result Files

- `README.md`
- `results/fraud/baseline_metrics.json`
- `results/fraud/dna_metrics.json`
- `results/fraud/dp_metrics.json`
- `results/fraud/comparison_summary.json`
- `results/mnist/base_model.pt`
- `results/mnist/train_metrics.json`
- `results/mnist/raw_gradient.pt`
- `results/mnist/dna_gradient.pt`
- `results/mnist/dp_gradient.pt`
- `results/mnist/reconstruction_metrics.json`
- `results/mnist/privacy_summary.json`
- `results/mnist/reconstructions/`
- `results/mnist/original_label.json`
- `results/mnist/reconstructions/original.png`
- `results/mnist/reconstructions/comparison_grid.png`
- `logs/run_<timestamp>.log`

## Latest Fraud Results

The current result files were refreshed by quick mode with three FL rounds.

| Method | F1-score | AUC-ROC | Accuracy | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.130882 | 0.981085 | 0.979249 | 0.070523 | 0.908163 |
| DNA | 0.130882 | 0.981085 | 0.979249 | 0.070523 | 0.908163 |
| DP | 0.108108 | 0.976029 | 0.974509 | 0.057516 | 0.897959 |

## Latest MNIST Privacy Results

- MNIST CNN test accuracy: `97.72%`.
- The current reconstruction files were refreshed by quick mode with `500`
  inversion iterations.

| Method | PSNR | SSIM | MSE |
| --- | ---: | ---: | ---: |
| RAW | 6.026153 | 0.106586 | 0.249681 |
| DNA | 6.026153 | 0.106586 | 0.249681 |
| DP | 5.877423 | 0.081636 | 0.258379 |

## Important Decisions

- Preserve NumPy float32 values at IEEE 754 bit level during serialization.
- Use DNA mapping: `00 -> A`, `01 -> T`, `10 -> G`, `11 -> C`.
- Use PyCryptodome AES-256-GCM with nonce, authentication tag, and ciphertext.
- Keep the preliminary FL simulation independent from Flower.
- Use same-process simulated FL with three clients, five communication rounds,
  one local epoch per client, and weighted FedAvg before introducing Flower.
- Standardize Credit Card Fraud `Amount` and `Time` using train-split
  statistics only.
- Use `BCEWithLogitsLoss(pos_weight)` for initial class imbalance handling.
- Use a preliminary DP baseline at local-update level: clip the complete
  floating update to L2 norm `1.0`, then add Gaussian noise with multiplier
  `0.05` before weighted FedAvg.
- Treat the current DP mechanism as a controlled comparison baseline, not as a
  formal privacy guarantee. Privacy accounting can be added in a later phase.
- Load MNIST from existing immutable files under `datasets/MNIST/` with
  `download=False`.
- Use one deterministic high-loss MNIST test sample for the preliminary
  inversion experiment so the extracted gradient contains a useful signal.
- Use a lightweight iDLG-style attack with classifier-bias label inference and
  dummy-image gradient matching. This is an executable baseline, not a
  state-of-the-art reconstruction attack.
- Use the known extracted label for the improved baseline inversion run while
  continuing to record inferred labels for inspection.
- Run the same `2,000`-iteration Adam gradient-matching attack for RAW, DNA,
  and DP gradients. Sigmoid image parameterization and bounded logits keep
  reconstructed pixels in `[0, 1]`.
- Use `QUICK=1` only for faster preliminary reruns: fraud FL uses three rounds
  and inversion uses `500` iterations. Full mode keeps five rounds and `2,000`
  iterations.
- Stream each complete-pipeline subprocess output to both the terminal and a
  timestamped log file without suppressing stderr.
- Keep `README.md` as the onboarding entry point for setup, architecture,
  experiment commands, and generated artifacts.
- Keep local datasets, virtual environments, Python caches, and timestamped
  runtime logs out of Git commits through repository ignore rules.

## Dataset Status

- MNIST: available under `datasets/MNIST/`.
- Credit Card Fraud Detection: local CSV currently available in `datasets/`;
  future experiments should organize it under `datasets/creditcard/` without
  editing the source data.
- Dataset files are immutable and must not be modified by agents.

## Environment Status

- Python virtual environment is available at `../venv/`.
- NumPy and PyCryptodome are installed.
- PyTorch and torchvision are installed.

## Known Issues

- Dataset directory naming currently uses `datasets/MNIST/`, while future
  conventions may use lowercase `datasets/mnist/`. Existing dataset files
  must not be renamed automatically.
- The improved lightweight inversion baseline reconstructs a recognizable RAW
  digit but retains background noise. Future work can add stronger attack
  variants before drawing formal privacy conclusions.
- Quick mode intentionally produces lower-quality reconstruction than full
  mode because it runs only `500` inversion iterations.

## Change Log

- 2026-06-02: Added initial project memory and agent guidance files.
- 2026-06-02: Implemented float32 binary mapping, DNA mapping, AES-256-GCM
  encryption, high-level DNA Encoder, direct smoke test, and basic benchmark.
- 2026-06-02: Added Credit Card Fraud loader, MLP model, same-process simulated
  FL baseline, FL with DNA-protected local states, and final metrics comparison.
- 2026-06-02: Added local-update Differential Privacy baseline, generated
  `results/fraud/dp_metrics.json`, and updated Baseline vs DNA vs DP summary.
- 2026-06-02: Added MNIST CNN, gradient extraction, DNA and DP gradient
  variants, lightweight inversion attack, reconstruction images, and
  PSNR/SSIM/MSE privacy summary.
- 2026-06-02: Improved gradient inversion to `2,000` iterations, saved original
  image and label artifacts, generated comparison grid, and refreshed
  PSNR/SSIM/MSE metrics.
- 2026-06-02: Added complete preliminary pipeline runner with realtime tee
  logging, timestamped logs, quick/full modes, and final summary output.
- 2026-06-02: Created comprehensive English `README.md` with project overview,
  setup instructions, execution guide, structure documentation, current
  results, workflow diagram, and artifact locations.
- 2026-06-02: Added repository hygiene rules before GitHub publication to keep
  local datasets, virtual environments, caches, and runtime logs out of Git.
