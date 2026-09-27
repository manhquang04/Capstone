# Priority 7 — Literature review for gradient leakage and tabular FL

**Date:** 2026-09-16  
**Status:** `COMPLETE FOR CURRENT PRIORITY`  
**Purpose:** provide verified citations and scope comparison for RQ1/RQ4/RQ6
planning, especially against gradient inversion and tabular-data leakage work.

## 1. Đã làm gì

1. Downloaded and read primary PDFs for five relevant papers:
   DLG, iDLG, Inverting Gradients, TabLeak, and an ISSRE tabular FL privacy-risk
   paper.
2. Extracted text locally from each PDF with `pypdf`.
3. Verified publisher/metadata pages for DLG, Inverting Gradients, TabLeak, and
   IEEE DOI availability where possible.
4. Recorded title, authors, venue/year, data type, batch/client scope, and main
   result.
5. Compared the literature scope directly with the current FL-DNA scope:
   PaySim tabular, 4 records/group, 1 fraud/group, 1 local Adam step, and the
   planned IEEE-CIS feasibility track.

## 2. Source files and hashes

| Source PDF | SHA-256 |
| --- | --- |
| `artifacts/literature/priority7_20260916/dlg_1906.08935.pdf` | `029e092e921e032e0f8e036318759e14cc9cb26028cbbf1324d1121117bf271f` |
| `artifacts/literature/priority7_20260916/idlg_2001.02610.pdf` | `96336f96f5f0f202ae68177650fff007e63edd6549f8a1fe2cad45fdc99a5647` |
| `artifacts/literature/priority7_20260916/inverting_gradients_2003.14053.pdf` | `6509ccbc98c47010f588f0644d37f91d4d0f08d4bd1e9524e954fe4d9737862d` |
| `artifacts/literature/priority7_20260916/tableak_2210.01785.pdf` | `87195dc940fd16e58efe89bd14954cdb4ac567b2d216d6c723e2747f2fb5f3cd` |
| `artifacts/literature/priority7_20260916/tabular_fl_risk_2210.06856.pdf` | `d9a94e669eb74792d36091cfa7618261fe3cc670db91150418a7c3abda0d769a` |

Publisher / metadata URLs checked:

- DLG NeurIPS page:
  `https://papers.nips.cc/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html`
- Inverting Gradients NeurIPS page:
  `https://proceedings.neurips.cc/paper/2020/hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html`
- TabLeak PMLR page:
  `https://proceedings.mlr.press/v202/vero23a.html`
- ISSRE tabular FL risk DOI:
  `https://doi.org/10.1109/ISSRE55969.2022.00028`
- arXiv API records for `1906.08935`, `2001.02610`, `2003.14053`,
  `2210.01785`, and `2210.06856`.

## 3. Verified comparison table

| Work | Authors | Venue/year | Data type and datasets | Batch / client scope verified | Main result relevant to FL-DNA |
| --- | --- | --- | --- | --- | --- |
| Deep Leakage from Gradients (DLG) | Ligeng Zhu, Zhijian Liu, Song Han | NeurIPS 2019 | Images: MNIST, CIFAR-100, SVHN, LFW; also masked language modeling | Single sample; batched CIFAR experiment reported BS=1,2,4,8; authors state DLG works up to batch size 8 and 64x64 image resolution in their experiments | Shows gradient matching can recover training examples; larger batch size makes optimization harder. Provides the historical origin point, but mainly image/text and small batches. |
| iDLG: Improved Deep Leakage from Gradients | Bo Zhao, Konda Reddy Mopuri, Hakan Bilen | arXiv 2020 | Images: MNIST, CIFAR-100, LFW | Single-sample setup following DLG; 1000 runs with randomly initialized networks | Analytically recovers labels for cross-entropy classifiers and improves reconstruction fidelity over DLG; useful for label-leakage discussion but not a tabular batch attack. |
| Inverting Gradients — How easy is it to break privacy in federated learning? | Jonas Geiping, Hartmut Bauermeister, Hannah Dröge, Michael Moeller | NeurIPS 2020 | Images: CIFAR-10, CIFAR-100, ImageNet | Single image; multi-image/FedAvg settings including 4 and 8 local images, batch sizes 1/2/8, and a demonstrated batch of 100 averaged images | Stronger image-domain attack using magnitude-invariant / cosine-style matching and regularization; shows that multiple local steps and batches do not automatically protect privacy. |
| TabLeak: Tabular Data Leakage in Federated Learning | Mark Vero, Mislav Balunović, Dimitar I. Dimitrov, Martin Vechev | ICML 2023 | Tabular: Adult, German Credit, Lawschool Admissions, Health Heritage | FedSGD batch sizes 1,2,4,8,16,32,64,128; FedAvg local dataset size 32 across local batches/epochs | Most directly relevant tabular inversion work. Uses softmax relaxation and pooling; shows tabular FL can leak substantially even at large batch sizes. At Adult batch 8, TabLeak reports 95.2% with true labels and 86.7% with reconstructed labels; at batch 128, 71.4%. |
| Federated Learning for Tabular Data: Exploring Potential Risk to Privacy | Han Wu, Zilong Zhao, Lydia Y. Chen, Aad van Moorsel | IEEE ISSRE 2022 | Tabular finance-like datasets: Bank Loan and Income Type | FL with 2 clients for Bank Loan and 3 clients for Income Type; clients train 10 local epochs before upload; attack is over FL rounds | A GAN/property-inference style attack on tabular FL, not a direct gradient inversion attack. Useful for motivating tabular FL privacy risk and finance setting, but not directly comparable to RQ1 PSNR/SSIM reconstruction. |

## 4. Comparison to current FL-DNA scope

Current FL-DNA RQ1 confirmed scope:

```text
PaySim tabular fraud data
4 records/group
1 fraud record/group
1 local Adam step
bounded local update
feature-MSE primary; pseudo-image PSNR/SSIM secondary
```

Recent scope-boundary screening:

```text
records_per_group = 8
fraud_records_per_group = 1
local_steps = 1
raw branch failed prior/zero gates
```

Interpretation against the literature:

- Compared with DLG/iDLG/Inverting Gradients, FL-DNA is less image-centric and
  more constrained by mixed tabular/fraud structure.
- Compared with TabLeak, FL-DNA currently has a much narrower validated
  attacker scope. TabLeak explicitly attacks tabular batches up to 128 and
  FedAvg local datasets of size 32, while the current FL-DNA attacker is only
  validated at 4 PaySim rows and fails validity screening at 8 rows.
- The gap is scientifically useful: it suggests that the core novelty should not
  be overstated as a generally strong attacker.  The stronger framing is either:
  (a) pre-registered evaluation of DNA Transform under a bounded, validated
  attacker scope; or (b) a future attacker-redevelopment contribution if
  sequential deflation / tabular relaxation can pass the 8-row gate.
- For IEEE-CIS, feasibility must be established before claiming generalization:
  verify fraud rate, feature types, preprocessing burden, and whether a
  TabLeak-style mixed discrete/continuous objective is more appropriate than the
  current PaySim reparameterization.

## 5. Related-work positioning draft

Gradient inversion was first made concrete by DLG, which showed that shared
gradients can reconstruct private image/text examples rather than only leaking
aggregate properties. iDLG then showed that labels can be analytically recovered
for cross-entropy classifiers, reducing one key uncertainty in DLG-style
optimization. Inverting Gradients strengthened the image-domain threat model by
using cosine-style gradient matching and by demonstrating leakage under trained
networks, deeper architectures, multi-image batches, and FedAvg-like local
updates.

For tabular FL, TabLeak is the closest methodological baseline. It argues that
tabular inversion requires domain-specific treatment of mixed discrete and
continuous variables, introduces softmax relaxation plus pooled ensembling, and
reports successful reconstruction for batch sizes far beyond the current FL-DNA
validated PaySim scope. A separate ISSRE 2022 line of work studies tabular FL
privacy via GAN-based class-property inference; that work is useful motivation
for financial tabular privacy but should not be presented as equivalent to
record-level gradient inversion.

Therefore FL-DNA's contribution should be positioned carefully: the current
results do not establish a broadly superior gradient-inversion defense across
tabular FL. They do provide a rigorously gated, pre-registered comparison of a
DNA Transform against a distortion-matched clipping/noise comparator under a
validated bounded scope, with transparent negative results and scope-boundary
evidence.

## 6. Gaps / unresolved items

- I did not find a direct prior paper on DNA encoding as a defense against
  gradient inversion in FL during this pass.  That absence must still be checked
  again when writing Priority 6 novelty text, but no such citation should be
  invented.
- The ISSRE 2022 paper is property-inference / GAN-based, not record-level
  inversion; use it as motivation only.
- TabLeak's metric is tabular reconstruction accuracy, not PSNR/SSIM.  When
  comparing to FL-DNA, report metrics side-by-side rather than pretending they
  are identical.

## 7. Artifact/run ID

- Literature PDF/text artifacts:
  `artifacts/literature/priority7_20260916/`
- Citation CSV:
  `reports/priority7_literature_table.csv`
- BibTeX:
  `reports/priority7_references.bib`

## 8. Gate đạt/chưa đạt

| Gate | Status | Reason |
| --- | --- | --- |
| Primary PDFs obtained | PASS | Five PDFs downloaded and hashed |
| Direct reading performed | PASS | Text extracted locally and searched for batch/dataset/result fields |
| Tabular gradient-inversion baseline identified | PASS | TabLeak is directly relevant |
| Missing-source handling | PASS | Unverified DNA-gradient-inversion literature is explicitly marked absent, not fabricated |
| Ready for Priority 6 novelty draft | PASS | Positioning constraints are now clear |

## 9. Bước tiếp theo được phép

Priority 7 is complete enough to support:

1. Priority 6 novelty positioning text.
2. Priority 1 attacker-improvement framing, especially considering
   TabLeak-style mixed discrete/continuous relaxation.
3. Priority 5 IEEE-CIS feasibility planning.

The next supervisor-ordered item is Priority 3: formal DP accounting for the
existing clipping + Gaussian-noise configurations.
