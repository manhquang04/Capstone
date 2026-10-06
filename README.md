# Assessing the Privacy and Practicality of DNA-Inspired Update Encoding in Federated Fraud Detection

Code, protocols, analysis data and reports for a capstone project at FPT University
(Department of Information Assurance). The project asks whether two keyed
transforms applied to federated learning (FL) updates can replace differential
privacy (DP) for fraud detection.

**Short answer: no.** In the settings we tested, neither transform shows a
consistent privacy advantage over DP, both break batch normalization when
applied to every transmitted tensor, and neither fits our cost budget. DP at a
meaningful privacy level has a real utility cost too, so the evaluation does not
pick a free winner.

## Research questions

Quoted verbatim from the project proposal; "DNA encoding" refers to both transforms.

1. **RQ1.** Does DNA encoding protect gradients against gradient inversion more
   effectively than Differential Privacy, measured quantitatively (PSNR/SSIM of
   the reconstructed data)?
2. **RQ2.** How does applying DNA encoding affect the final accuracy of the
   global model (measured by F1-score and AUC-ROC)?
3. **RQ3.** Are the computational and bandwidth costs of DNA encoding within an
   acceptable range for real-world deployment?

## The two transforms

| | v1 | v2 |
|---|---|---|
| Idea | Seeded permutation, attenuation and mixing of each 256-value block; the block seed comes from a nucleotide encoding of the block's bytes | Subsampled randomized Hadamard sketch with stochastic quantization |
| Setting used | conservative `(m, rho, kappa) = (0.08, 0.88, 0.45)` (also medium and stronger) | `k/d = 0.95`, `eta = 0.01` |
| Server decodes? | No, it averages transformed blocks | Yes, and it needs the seed to do so |
| Code | `FL-DNA/dna_encoder/transform_defense.py` | `FL-DNA/dna_encoder/transform_defense_v2.py` |

Both are deterministic once the seed is fixed, so neither is a DP mechanism
(two short propositions in the paper make this precise).

## Main results

| Question | Finding | Scope |
|---|---|---|
| RQ1, images | Neither transform significantly outperforms utility-matched DP on any of four CIFAR-10 models; per-tensor DP is more private by 1 to 5 dB PSNR | Inverting Gradients, known labels, 39 targets per comparison, Holm over 211 tests |
| RQ1, fraud data | A server recovers most record features through either transform; client-side DP at a whole-training epsilon of 10 lowers recovery significantly in 15 of 16 comparisons. Batch-normalization (BN) statistics leak batch means in closed form | PaySim, IEEE-CIS, BAF; one or two records per gradient, known labels |
| RQ2 | Applied to every tensor, the transforms drive BN variances negative and break training. With BN kept on the clients, both pass F1 and AUC-ROC noninferiority on all three datasets | 21 paired replicates; margins -0.02 (F1) and -0.005 (AUC-ROC), fixed before any run |
| RQ3 | Payload grows by only 1.0x (v1) and about 1.3x (v2), but round overhead and memory fail the budget on every dataset | Single-threaded prototype, fixed budget over 27 cells |
| DP utility | Update-level client-side DP at a whole-training epsilon of 1 to 10 costs 0.16 to 0.31 in F1 | 11 paired replicates, descriptive |

An audit of five published defenses (Soteria, PRECODE, gradient pruning, ATS,
Count-Sketch) and both transforms under the same rules found that adaptive
attacks erase much of the apparent protection: only 3 of 13 defense-dataset
cells met our survival criterion, and one of those (image v1) later lost to
per-tensor DP.

## Repository layout

```text
FL-DNA/
  dna_encoder/          v1 and v2 transforms (plus the server-blind v2 variant)
  models/, data/        fraud MLP and data loaders
  attacks/              gradient-inversion attacks and metric adapters
  privacy/              seed derivation and DP helpers
  experiments/          one runner and analyzer per study (priority*.py)
  protocols/            research protocols and dated amendments, written before each run
  reports/              one report per study, with commands, results and verification
  artifacts/            analysis-level outputs (CSV/JSON summaries, receipts, manifests)
  results/              earlier RQ1-RQ3 result summaries
  tests/                unit tests
  external_defenses/    adapters for the defense audit; UPSTREAM.md lists the
                        original repositories and commits
  ARTIFACT_MANIFEST.tsv path, size and SHA-256 of heavy artifacts kept out of git
  requirements-lock.txt exact package versions used for the experiments
```

Large files (model checkpoints, tensors, prepared data; about 12.6 GB) are not in
the repository. Each one is listed with its SHA-256 in
`FL-DNA/ARTIFACT_MANIFEST.tsv`, so a regenerated file can be checked against the
original.

## Setup

The experiments ran on CPU with one thread per process, on macOS (ARM64).

```bash
cd FL-DNA
python3.9 -m venv .venv-phase1
.venv-phase1/bin/pip install -r requirements-lock.txt
```

`requirements-lock.txt` pins the versions actually used (Python 3.9, PyTorch 2.2.2,
NumPy 1.26.4, pandas 2.0.3, scikit-learn 1.3.2). The defense audit uses a
separate Python 3.11 environment described in
`FL-DNA/external_defenses/reference_environment.txt`, plus the upstream
repositories listed in `FL-DNA/external_defenses/UPSTREAM.md`.

## Data

Datasets are not redistributed. Download them and place them as follows.

| Dataset | Source | Expected path |
|---|---|---|
| PaySim | [Kaggle: ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1) | `FL-DNA/datasets/creditcard.csv` (the PaySim CSV, renamed) |
| IEEE-CIS Fraud Detection | [Kaggle competition](https://www.kaggle.com/c/ieee-fraud-detection) | `FL-DNA/datasets/ieee-fraud-detection/train_transaction.csv`, `train_identity.csv` |
| Bank Account Fraud (BAF) | [Kaggle: sgpjesus/bank-account-fraud-dataset-neurips-2022](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) | `FL-DNA/datasets/baf/Base.csv` |
| CIFAR-10 | [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html) | `FL-DNA/datasets/cifar10/` |
| UCI Adult | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult) | loaded by the TabLeak adapter |

The SHA-256 of each raw file used is recorded in
`FL-DNA/artifacts/priority32_multidataset/prepared/<dataset>/audit.json`.

## Reproducing results

Every study has a report in `FL-DNA/reports/` with the exact commands, the
protocol amendment it followed, and the hashes of its inputs and outputs. The
main entry points are:

| Result | Report | Runner |
|---|---|---|
| RQ1 on CIFAR-10 | `priority31_image_utility_dp_report.md`, `priority33c_report.md`, `priority34d_report.md` | `experiments/priority31_image_utility_dp.py`, `priority33c_image_*.py`, `priority34d_*.py` |
| RQ1, BN channel | `priority24_rq1_valid_instrument_report_20260929.md`, `priority25b_bracketed_utility_report_20260929.md`, `priority33a_report.md` | `experiments/priority33a_*.py` |
| RQ1, fraud records | `priority34c_report.md`, `priority34c_verified_summary.md` | `experiments/priority34c_*.py` |
| RQ2 and RQ3 | `priority32_repaired_trainable_rq2_rq3_report_20261002.md`, `priority34a_report.md`, `rq3_optimized_implementation_report_20260916.md` | `experiments/priority32_*.py`, `priority34a_local_bn.py`, `run_rq3_benchmark.py` |
| Client-side DP | `priority34b_local_dp_extension_report.md` | `experiments/priority34b_*.py` |
| Defense audit | `priority30_e3_validity_repair_report_20261001.md`, `priority30_native_audit_report_20260930.md` | `experiments/priority30_native_defenses/` |

For example, preparing the fraud datasets and running the local-BN utility study:

```bash
cd FL-DNA
PYTHONPATH=. .venv-phase1/bin/python -B experiments/priority32_multidataset.py --prepare-only
PYTHONPATH=. .venv-phase1/bin/python -B -u experiments/priority34a_local_bn.py --freeze
PYTHONPATH=. .venv-phase1/bin/python -B -u experiments/priority34a_local_bn.py --supervise
```

Run the unit tests with:

```bash
cd FL-DNA
PYTHONPATH=. .venv-phase1/bin/python -B -m unittest discover -s tests
```

### Reproduction smoke test

A fresh clone was checked against the recorded results
(`FL-DNA/reports/reproduction_smoke_test_20261006.md`). The committed source files
match the hashes recorded when the experiments ran, the prepared PaySim data was
rebuilt byte for byte, and three utility training jobs were retrained with
identical metrics, predicted probabilities and model weights. This is a sample,
not a full replication.

## How the evaluation was run

- Every confirmatory run follows a protocol amendment written before its data
  existed. Some amendments came after earlier results were known; they are dated
  and kept, so the process is protocol-driven but not pre-registered.
- An attack counts only after it beats data-free baselines on undefended updates.
- Each transform is compared with DP matched both to its distortion and to its
  utility, and the matching rule is stated for every comparison.
- Errors found along the way, including an attacker that could not invert
  undefended updates and a GPU backend that hid invalid BN states, are documented
  in the reports and were not removed from the record.

## Limitations

The fraud reconstruction results use one or two records per gradient and known
labels; recovery from multi-step Adam updates was not tested. The utility study
uses three clients, the cost study is a single-threaded prototype measured against
a budget we set, and the defense audit covers one port per defense.

## Authors

Ho Hai, Duong Viet Huy, Dao Manh Quang, Nguyen Thanh Nguyen,
Le Tran Gia Huy. Department of Information Assurance, FPT University, Ho Chi Minh
City, Vietnam.
