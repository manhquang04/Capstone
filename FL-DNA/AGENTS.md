# Agent Guide: FL-DNA

## Research Goal

This project studies how to protect gradients in distributed AI training by
combining Federated Learning (FL), DNA encoding, and AES-256-GCM encryption.
The protected gradients will later be evaluated against Gradient Inversion
Attack techniques.

## Planned Architecture

1. Train local models on distributed datasets.
2. Extract model gradients or updates.
3. Serialize float32 gradients without losing IEEE 754 bit information.
4. Map binary data to DNA symbols.
5. Encrypt the DNA representation with AES-256-GCM.
6. Transmit and decrypt protected updates.
7. Aggregate updates in a future FL phase.
8. Evaluate privacy, correctness, runtime, and bandwidth overhead.

## Tech Stack

- Python 3
- NumPy
- PyTorch and torchvision
- PyCryptodome for AES-256-GCM
- pytest or directly executable Python test scripts
- Flower and Opacus only in later phases

All Python code must run on Windows, macOS, and Linux.

## Datasets

- MNIST: image classification dataset, stored under `datasets/`.
- Credit Card Fraud Detection: tabular dataset, stored under
  `datasets/creditcard/` when organized for experiments.

Never delete, rename, rewrite, or edit data files inside `datasets/`.

## Directory Convention

- `datasets/`: immutable local datasets.
- `dna_encoder/`: independent DNA encoding and encryption module.
- `experiments/`: benchmarks and research experiments.
- `results/`: generated experiment outputs.
- `fl_baseline/`: future baseline FL implementation.
- `fl_dp/`: future differential privacy FL implementation.
- `fl_dna/`: future FL integration with DNA encryption.
- `attacks/`: future privacy attack evaluation code.
- `notebooks/`: exploratory analysis only.

## Coding Rules

- Before starting any task, read `project.md`.
- After completing any task, update `project.md`.
- Prefer clear, testable, and extensible Python modules.
- Preserve float32 bit-level data when serializing gradients.
- Validate external inputs and fail with clear error messages.
- Keep secrets out of source control.
- Avoid OS-specific paths and shell assumptions in Python code.
- Do not modify files inside `datasets/`.

## Implementation Priority

1. Finish and validate the independent DNA Encoder.
2. Integrate the encoder with real PyTorch MNIST gradients.
3. Build a baseline FL workflow.
4. Integrate DNA protection into FL.
5. Add differential privacy comparison.
6. Implement attack evaluation and collect benchmark results.

## Current Constraint

Do not implement Flower, Federated Learning, dashboards, web APIs, frontend
code, or Gradient Inversion Attack until the independent DNA Encoder is
complete and verified.
