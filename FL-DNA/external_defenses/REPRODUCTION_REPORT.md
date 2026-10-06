# External Defense Reproduction Report

## Scope and environment

This directory contains five official checkouts and an isolated CPU environment at `.venv`; no project directory outside `external_defenses/` was changed. CPU work was capped to four threads with `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, and `OPENBLAS_NUM_THREADS=4`. Exact commands are in `commands.sh`; primary execution logs are `logs/clone_commits.log`, `logs/dlg_original_cpu.log`, `logs/cpu_qualitative_checks.log`, and `logs/pip_freeze.txt`.

`run_cpu_qualitative_checks.py` is a deliberately small, fixed-seed fallback, not a substitute for paper-scale reconstruction. It verifies source-level mechanism properties after the full CUDA-era workflow proved unavailable or too large for this host. Published results and reproductions below are only compared when they measure the same quantity; otherwise `Not assessable` is the scientifically valid result.

| Defense | Official commit | Published headline privacy result | This run | Agree within paper-stated variance? |
|---|---|---|---|---|
| Soteria | `23cf90e9e5cb41d5dc45e7540ef63a4a0ca0a8ca` | CVPR 2021, Fig. 6: at equal 80% pruning, Soteria MSE 1.49 versus gradient-compression MSE 0.71; text reports up to 160x MSE over baselines without accuracy loss. | No paper-scale reconstruction. CPU fallback ran the official percentile masking rule at 60% on 8 representation units: 5 masked units / 5 zeroed classifier-gradient columns. | Not assessable. No numerical variance is reported for Fig. 6, and the fallback is not an inversion MSE. |
| PRECODE | `c66adc4cdd62993139eafebc1b57fc25b0694874` | WACV 2022, Table 1, CIFAR-10 LeNet: baseline SSIM 0.55, ASR 42.45%; PRECODE SSIM 0.10, ASR 0%. | No 128-image/7,000-step IGA run: missing pinned attack submodule plus CUDA-era stack. Official bottleneck source executed twice at fixed weights; its gradient cosine was 0.792617 and the two gradients differed, as required by sampling. | Not assessable. The paper averages over three seeds but does not report a numerical variance; the metric differs. |
| Gradient pruning | `d21007fa1540ba2303ebc034976aa331814727c7` | NeurIPS 2019 §5.2: DLG is visually effective at 1-10% sparsity; images have artifacts at 20%; "When pruning ratio is larger, the recovered images are no longer visually recognizable." | The unmodified official DLG command completed 300 outer iterations but plateaued at gradient-match loss 352.5049 under PyTorch 2.5.1, so it is not a valid reproduction. Fallback exactly zeroed 100/143 (70%) lowest-magnitude synthetic-gradient coordinates. | Not assessable. The paper supplies no numerical variance and its outcome is visual recognizability. |
| ATS | `2496fc9571fe68bc65e839cbf6ccb6d58d27c93a` | CVPR 2021, Table 1, CIFAR-100/ResNet20: no transform PSNR 13.88 dB and accuracy 76.88%; hybrid ATS PSNR 7.64 dB and accuracy 77.92%. PSNR is against transformed images. | Full training/search is blocked by CUDA-only Lightning, a missing dependency, and syntax error in the supplied search script. The pinned `3-1-7` policy changed 2,784/3,072 synthetic-image channel values; raw-to-transformed PSNR was 5.459721 dB. | Not assessable: neither data/model nor PSNR target matches the paper inversion evaluation. |
| FetchSGD Count Sketch | `833ca44cc43a9b034515f55485524e4f1d0fad21` | No gradient-inversion/privacy evaluation is published. The paper evaluates test accuracy and compression only. | Official training requires undeclared `csvec` and unconditional CUDA/NCCL. Fixed-hash Count-Sketch linearity check: `max_abs(S(x+y)-S(x)-S(y))=2.861e-06` (float32); changing one hash seed gave discrepancy 19.929142. | Not applicable: there is no published privacy metric or stated variance. |

## C1. Noise / DP comparator verification

The quotations below are from the original papers. A paper's omission of clipping does not prove clipping was absent from every unpublished implementation; it does establish that the published comparison did not document a clip norm. None of the first four papers describes a calibrated DP-SGD comparator with a stated clip norm.

| Paper | Exact quotation and location | Finding |
|---|---|---|
| Soteria | §6.1, p.14: "In the experiments, we separately apply Gaussian and Laplacian noise to develop two DP baselines, i.e., DP-Gaussian and DP-Laplace." §6.1, p.15: "Regarding DP-Gaussian and DP-Laplace, we set the mean and variance of the noise distribution as 0 and σ, respectively." | Direct noise on uploaded gradients. Table 2 gives `σ` from `1e-4` to `1e-1`; no clipping or clip norm is specified in the quoted experimental protocol. |
| PRECODE | §5.2, p.6: "For noisy gradients, we add Gaussian noise with zero mean and standard deviation σ. We consider two noise levels: σ = 10−2 (NG-2) and σ = 10−3 (NG-3)." | Direct Gaussian perturbation. No clipping or clip norm is described for NG-2/NG-3. |
| DLG / pruning | §5.1, p.7: "One straightforward attempt to defense DLG is to add noise on gradients before sharing." The same paragraph reports Gaussian/Laplacian noise "with variance range from 10−1 to 10−4 and central 0." | Direct noise on gradients before sharing. No clipping or clip norm is specified. |
| ATS | §5.1, p.5: "Gaussian/Laplacian: using differential privacy to obfuscate the gradients with Gaussian or Laplacian noise. For instance, Gaussian(10−3) suggests a noise scale of N (0, 10−3)." | Direct Gaussian/Laplacian scales `1e-3`/`1e-2` in Table 3. No clipping or clip norm is described. |
| FetchSGD | §5, p.7: "We implement and compare FetchSGD, gradient sparsification (local top-k), and FedAvg using PyTorch." The paper then says: "For each method, we report the compression achieved relative to uncompressed SGD in terms of total bytes uploaded and downloaded." | No DP/noise baseline and no privacy/reconstruction metric; thus no clipping norm exists to verify. |

## C2. Count-sketch reconstruction literature and claims

### Search record

Searches were conducted on 2026-09-29 using these exact queries.

| Source | Query | Result |
|---|---|---|
| arXiv API | `all:"FetchSGD" AND (all:"gradient inversion" OR all:"gradient leakage" OR all:reconstruction)` | 0 results. |
| arXiv API | `(all:"Count Sketch" OR all:sketching) AND (all:"gradient inversion" OR all:"gradient leakage" OR all:"data reconstruction")` | 1 result: Song et al., arXiv:2210.08371. |
| Semantic Scholar Graph API | `FetchSGD gradient inversion` | Rate-limited (HTTP 429); no usable result returned. |
| Semantic Scholar Graph API | `Count Sketch gradient inversion` | Rate-limited (HTTP 429); no usable result returned. |
| OpenAlex API | `FetchSGD gradient inversion` | Rate-limited (HTTP 429); no usable result returned. |
| OpenAlex API | `Count Sketch gradient inversion` | Timed out after 30 seconds; no usable result returned. |

**Finding: FOUND for shared random-sketch aggregation, NOT VERIFIED for FetchSGD's exact `CSVec` Count-Sketch implementation.** Song et al. (ICML 2023, arXiv:2210.08371) study a random sketch/de-sketch FL algorithm, cite Rothchild et al. as an example of sketching in FL, and provide a defense-aware reconstruction objective. Their abstract (p.1) states: "However, such random sketching does not protect the privacy of local data directly. We show that the gradient leakage problem still exists after applying the sketching technique by presenting a specific gradient attack method." Their §7.2 (p.15-16) specifies that an attacker observes the sketched gradient, sketching matrix, and model parameters and minimizes `L_R(x)=||R(∇_wF(w,x))-R(g)||²`.

This is materially relevant to FetchSGD-style aggregation when the server knows the shared sketch map, but it is not an empirical attack evaluated on FetchSGD's unpinned external `CSVec` code. Because Semantic Scholar and OpenAlex searches were unavailable, the negative claim is limited: no direct FetchSGD-specific paper was verified by the available searches, rather than an assertion that none exists.

### Exact sketching privacy claims

FedSKETCH (Haddadpour et al., arXiv:2008.04975), abstract p.1: "The key idea is to compress the accumulation of local gradients using count sketch, therefore, the server does not have access to the gradients themselves which provides privacy." Its §4.2, p.6, additionally makes its assumptions explicit: "we suppose that for any input vector S with length |S| = l, each element si ∈ S is drawn i.i.d. from a Gaussian distribution: si ∼ N (0, σ2), and bounded by a large probability."

Privacy for Free (Li et al., arXiv:1911.00972), abstract p.1: "In particular, we prove that Count Sketch, a simple method for data stream summarization, has inherent differential privacy properties." But its footnote on p.1 says exactly: "We note that the definition of local differential privacy used in our work is weaker than the standard local privacy definition. In addition, we are aware of some issues with our current proof of the differential privacy properties of Count Sketch. We are currently working on a revision of this draft."

## Files

- `soteria/PORTING_NOTES.md`
- `precode/PORTING_NOTES.md`
- `dlg/PORTING_NOTES.md`
- `ats/PORTING_NOTES.md`
- `fetchsgd/PORTING_NOTES.md`
- `defenses.bib`
- `commands.sh`
- `run_cpu_qualitative_checks.py`
