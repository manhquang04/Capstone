# Priority34A — Local-BN utility without BN transmission

Status: COMPLETE — all189jobs, independent statistics, checkpoint replay, final commands and no-live-worker audit PASS.

New FedBN-style scientific variant, not replacement of original or raw-BN evidence. All189jobs (3datasets ×3methods ×21paired seeds) complete; no failed seed excluded.

## Frozen setup

Identical P32 prepared data, source rows, partitions and50round Adam(.001), focal(.95,2), batch1024, K3. CPU/thread1; final checkpoint. Seeds321000–321020. No selection or extra replicate. BN affine weights/biases and all buffers remain client-local and persist across rounds. Only non-BN trainable tensors upload/transform/sample-weighted average; payload assertions and per-step BN guards pass in every job.

Each client tunes validation-F1 threshold separately and evaluates the common test split with its own BN. Primary unit is the arithmetic mean of3client metrics per seed, not averaged predictions or63independent clients. Individual client metrics/thresholds in analysis/client_metrics.csv; probabilities and checkpoints saved per attempt.

## Paired noninferiority

Paired mean Student-t two-sided95%CI(df20), strict lower bound>-.02F1 and>-.005ROC-AUC; both required. PR-AUC descriptive. NOT_ESTABLISHED is not proof of inferiority or equivalence. No post-hoc n increase or multiplicity/sign-test family.

| Dataset | Method | Endpoint | Baseline mean | DNA mean | Mean delta | SD | Median delta | IQR | 95% CI | NI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paysim | dna_v1_conservative | f1 | 0.783971457 | 0.782375718 | -0.00159573911 | 0.0100505603 | -0.00258393577 | [-0.007210233898462981, 0.007821167568974108] | [-0.00617069932, 0.0029792211] | PASS |
| paysim | dna_v1_conservative | auc_roc | 0.997698124 | 0.997736879 | +3.8754312e-05 | 0.000176654844 | +2.23026723e-05 | [-6.791707038500405e-05, 0.00016923640281507613] | [-4.16580091e-05, 0.000119166633] | PASS |
| paysim | dna_v1_conservative | pr_auc | 0.806702073 | 0.807088485 | +0.000386411958 | 0.00772302022 | +0.00109393692 | [-0.004808327243707278, 0.0035234362516456397] | [-0.0031290647, 0.00390188862] | DESCRIPTIVE |
| paysim | dna_v2_0p95 | f1 | 0.783971457 | 0.783371974 | -0.000599482299 | 0.00869878996 | +0.0020751366 | [-0.0035573224279505933, 0.00512882754156474] | [-0.00455912402, 0.00336015942] | PASS |
| paysim | dna_v2_0p95 | auc_roc | 0.997698124 | 0.997769888 | +7.17635483e-05 | 0.00025966261 | +5.92495585e-05 | [-0.00013262586773121754, 0.0003156267968107418] | [-4.64334548e-05, 0.000189960551] | PASS |
| paysim | dna_v2_0p95 | pr_auc | 0.806702073 | 0.809769977 | +0.00306790327 | 0.00837109651 | +0.00308888093 | [-0.004145359964001782, 0.008757224200762526] | [-0.000742574177, 0.00687838071] | DESCRIPTIVE |
| ieee_cis | dna_v1_conservative | f1 | 0.346097579 | 0.342468529 | -0.00362904985 | 0.0164157466 | -0.00337856398 | [-0.011818623480375379, 0.009193346932418411] | [-0.0111014081, 0.00384330841] | PASS |
| ieee_cis | dna_v1_conservative | auc_roc | 0.837352883 | 0.836024767 | -0.00132811542 | 0.00335516252 | -0.00117804601 | [-0.0023620958571644035, 0.0005909446946499308] | [-0.00285536709, 0.000199136246] | PASS |
| ieee_cis | dna_v1_conservative | pr_auc | 0.318407695 | 0.313717521 | -0.00469017454 | 0.0152093314 | -0.00248680278 | [-0.01061105246438293, 0.0036096737785593636] | [-0.0116133792, 0.00223303009] | DESCRIPTIVE |
| ieee_cis | dna_v2_0p95 | f1 | 0.346097579 | 0.335642299 | -0.0104552806 | 0.0140536371 | -0.0116703506 | [-0.01765448347942511, -0.0045528761375704385] | [-0.0168524195, -0.00405814163] | PASS |
| ieee_cis | dna_v2_0p95 | auc_roc | 0.837352883 | 0.835895264 | -0.00145761852 | 0.00230191501 | -0.00187924087 | [-0.002917665091786259, -5.424424474254064e-05] | [-0.00250543767, -0.000409799366] | PASS |
| ieee_cis | dna_v2_0p95 | pr_auc | 0.318407695 | 0.312647109 | -0.00576058604 | 0.0103486078 | -0.00493421896 | [-0.012106409927738204, 0.0008319867855348106] | [-0.0104712158, -0.00104995626] | DESCRIPTIVE |
| baf | dna_v1_conservative | f1 | 0.225271825 | 0.225413618 | +0.000141792944 | 0.00342776786 | +0.000475667008 | [-0.0016685026521829904, 0.0015645549764224942] | [-0.00141850828, 0.00170209417] | PASS |
| baf | dna_v1_conservative | auc_roc | 0.871004246 | 0.87114985 | +0.000145603792 | 0.0017974237 | -0.000148557677 | [-0.0007634919380437033, 0.0010900845859671549] | [-0.000672573668, 0.000963781252] | PASS |
| baf | dna_v1_conservative | pr_auc | 0.158556718 | 0.158469528 | -8.7190291e-05 | 0.0010564238 | -2.61028902e-05 | [-0.0009257642570709712, 0.0007264718365019296] | [-0.00056806864, 0.000393688058] | DESCRIPTIVE |
| baf | dna_v2_0p95 | f1 | 0.225271825 | 0.226369519 | +0.00109769373 | 0.00384930595 | +0.00157249325 | [-0.000903469807798013, 0.004045918397025089] | [-0.000654489327, 0.00284987679] | PASS |
| baf | dna_v2_0p95 | auc_roc | 0.871004246 | 0.870908005 | -9.62412948e-05 | 0.00197577381 | -0.000199592291 | [-0.0012045832793740052, 0.0006877106995648585] | [-0.000995602753, 0.000803120163] | PASS |
| baf | dna_v2_0p95 | pr_auc | 0.158556718 | 0.15830588 | -0.000250838503 | 0.00152979275 | -0.000401565019 | [-0.0012582262192096239, 0.0010114898494917313] | [-0.000947191814, 0.000445514808] | DESCRIPTIVE |

Overall: {"baf/dna_v1_conservative": "PASS", "baf/dna_v2_0p95": "PASS", "ieee_cis/dna_v1_conservative": "PASS", "ieee_cis/dna_v2_0p95": "PASS", "paysim/dna_v1_conservative": "PASS", "paysim/dna_v2_0p95": "PASS"}

## P32 raw-BN comparison (descriptive)

P32 raw-BN transformed BN affine parameters and transmitted/averaged raw running buffers, with one global-model threshold/test metric. P34A sends no BN and reports a mean of locally thresholded client metrics. These cross-variant differences include training AND evaluation changes and are not isolated transform effects. Each NI claim is against its own baseline.

| Dataset | Method | Endpoint | P32 baseline mean | P32 raw-BN delta [95% CI] | P32 NI | P34A delta |
| --- | --- | --- | --- | --- | --- | --- |
| paysim | dna_v1_conservative | f1 | 0.786175079 | -0.00123853687 [-0.0076966660987846025, 0.005219592362342709] | PASS | -0.00159573911 |
| paysim | dna_v1_conservative | auc_roc | 0.997662213 | +2.77896323e-05 [-8.52610146316464e-05, 0.00014084027925102973] | PASS | +3.8754312e-05 |
| paysim | dna_v1_conservative | pr_auc | 0.80726339 | +0.0015280865 [-0.002585628308504387, 0.0056418013156345655] | DESCRIPTIVE | +0.000386411958 |
| paysim | dna_v2_0p95 | f1 | 0.786175079 | -0.00179521107 [-0.008926036630580055, 0.005335614488699554] | PASS | -0.000599482299 |
| paysim | dna_v2_0p95 | auc_roc | 0.997662213 | +3.7846903e-05 [-7.868231481406271e-05, 0.00015437612076873444] | PASS | +7.17635483e-05 |
| paysim | dna_v2_0p95 | pr_auc | 0.80726339 | +0.00308207531 [-0.0022321091864765585, 0.008396259801079206] | DESCRIPTIVE | +0.00306790327 |
| ieee_cis | dna_v1_conservative | f1 | 0.352168032 | -0.00672025543 [-0.01721685691501963, 0.003776346051157959] | PASS | -0.00362904985 |
| ieee_cis | dna_v1_conservative | auc_roc | 0.838572269 | -0.00132730978 [-0.002451927402199662, -0.00020269216055126952] | PASS | -0.00132811542 |
| ieee_cis | dna_v1_conservative | pr_auc | 0.326702814 | -0.00912300054 [-0.016778809407319493, -0.0014671916793585925] | DESCRIPTIVE | -0.00469017454 |
| ieee_cis | dna_v2_0p95 | f1 | 0.352168032 | -0.0127650766 [-0.020608218071844507, -0.0049219351275815205] | NOT_ESTABLISHED | -0.0104552806 |
| ieee_cis | dna_v2_0p95 | auc_roc | 0.838572269 | -0.000557864701 [-0.0013486153594022398, 0.00023288595787461974] | PASS | -0.00145761852 |
| ieee_cis | dna_v2_0p95 | pr_auc | 0.326702814 | -0.0078157807 [-0.01255956059124123, -0.00307200081822483] | DESCRIPTIVE | -0.00576058604 |
| baf | dna_v1_conservative | f1 | 0.225726006 | +0.000418350858 [-0.0010292463044095609, 0.001865948020503489] | PASS | +0.000141792944 |
| baf | dna_v1_conservative | auc_roc | 0.872388336 | +4.36483457e-05 [-0.0008676973256482109, 0.0009549940171482044] | PASS | +0.000145603792 |
| baf | dna_v1_conservative | pr_auc | 0.160628842 | -0.000387011307 [-0.0009363134380911736, 0.00016229082322363715] | DESCRIPTIVE | -8.7190291e-05 |
| baf | dna_v2_0p95 | f1 | 0.225726006 | -0.000373761332 [-0.002986084523303654, 0.0022385618586524833] | PASS | +0.00109769373 |
| baf | dna_v2_0p95 | auc_roc | 0.872388336 | -0.000323448364 [-0.0011755831602454844, 0.000528686432354581] | PASS | -9.62412948e-05 |
| baf | dna_v2_0p95 | pr_auc | 0.160628842 | -0.000547672485 [-0.0012501378407303232, 0.00015479287005191968] | DESCRIPTIVE | -0.000250838503 |

## Commands, integrity and disclosures

```text
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest discover -s tests -p test_priority34a_local_bn.py -v
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -u experiments/priority34a_local_bn.py --freeze
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -u experiments/priority34a_local_bn.py --supervise
```

Pre-run protocol: protocols/amendments/2026-10-03_priority34a_local_bn.md. Source/input hashes: execution_freeze.json; all input hashes revalidated. Job/attempt journals, launch receipt, progress logs and failure registry preserve interruptions. Tensor-local transform indices are the non-BN allowlist indices (removing BN changes this domain); P32 seed derivation is unchanged. Local BN is a scientific variant, not numerical clamping. No reconstruction/privacy/DP claim follows merely from no BN transmission. FedBN reference: [Li et al., ICLR2021](https://arxiv.org/abs/2102.07623).

Independent probability/metric/paired-CI recomputation, checkpoint BN validity and all150payload assertions per job verified; details in analysis/independent_recomputation.json. Sources, datasets, Latex/, external_defenses/ and all earlier evidence remain unchanged. Final checks/manifest and supervisor-exit receipt are in the new namespace. PROJECT note only after verified completion.

## Final administrative verification

The pre-exit report is preserved byte-identically in archive/priority34a_report_pre_final_audit.md. Completion addendum: priority34a_completion_addendum.md. Original scientific-manifest report checksum resolves to that archived snapshot; report_finalization.json records this administrative relocation. No scientific table, margin, seed or result changed.

## Per-client test metrics (mean over21seeds; individual seed/client values in client_metrics.csv)

| Dataset | Method | Client | Mean F1 | Mean ROC-AUC | Mean PR-AUC |
| --- | --- | --- | --- | --- | --- |
| paysim | baseline | 0 | 0.782948328 | 0.997499216 | 0.80519804 |
| paysim | baseline | 1 | 0.785905321 | 0.997805323 | 0.80838688 |
| paysim | baseline | 2 | 0.783060721 | 0.997789834 | 0.8065213 |
| paysim | dna_v1_conservative | 0 | 0.781877674 | 0.997502555 | 0.804523268 |
| paysim | dna_v1_conservative | 1 | 0.784789526 | 0.997864961 | 0.807879332 |
| paysim | dna_v1_conservative | 2 | 0.780459953 | 0.99784312 | 0.808862856 |
| paysim | dna_v2_0p95 | 0 | 0.784061786 | 0.997510972 | 0.806807066 |
| paysim | dna_v2_0p95 | 1 | 0.785322894 | 0.997895598 | 0.810073906 |
| paysim | dna_v2_0p95 | 2 | 0.780731243 | 0.997903094 | 0.812428958 |
| ieee_cis | baseline | 0 | 0.331087532 | 0.835132844 | 0.303643168 |
| ieee_cis | baseline | 1 | 0.357967889 | 0.838425866 | 0.326099106 |
| ieee_cis | baseline | 2 | 0.349237317 | 0.838499937 | 0.325480811 |
| ieee_cis | dna_v1_conservative | 0 | 0.330657034 | 0.834225839 | 0.305347606 |
| ieee_cis | dna_v1_conservative | 1 | 0.351880322 | 0.836951209 | 0.319883108 |
| ieee_cis | dna_v1_conservative | 2 | 0.344868232 | 0.836897254 | 0.315921848 |
| ieee_cis | dna_v2_0p95 | 0 | 0.325309666 | 0.833320028 | 0.302193143 |
| ieee_cis | dna_v2_0p95 | 1 | 0.344604105 | 0.837353563 | 0.318847869 |
| ieee_cis | dna_v2_0p95 | 2 | 0.337013126 | 0.837012202 | 0.316900315 |
| baf | baseline | 0 | 0.226398048 | 0.871754669 | 0.159208024 |
| baf | baseline | 1 | 0.220765439 | 0.870251247 | 0.157544446 |
| baf | baseline | 2 | 0.228651988 | 0.871006821 | 0.158917685 |
| baf | dna_v1_conservative | 0 | 0.225354252 | 0.872111954 | 0.159222912 |
| baf | dna_v1_conservative | 1 | 0.221621293 | 0.870372842 | 0.157419419 |
| baf | dna_v1_conservative | 2 | 0.229265309 | 0.870964752 | 0.158766253 |
| baf | dna_v2_0p95 | 0 | 0.226506582 | 0.872115109 | 0.159333591 |
| baf | dna_v2_0p95 | 1 | 0.221433812 | 0.869663831 | 0.156814231 |
| baf | dna_v2_0p95 | 2 | 0.231168162 | 0.870945074 | 0.158769817 |

Overall NI: {"baf/dna_v1_conservative": "PASS", "baf/dna_v2_0p95": "PASS", "ieee_cis/dna_v1_conservative": "PASS", "ieee_cis/dna_v2_0p95": "PASS", "paysim/dna_v1_conservative": "PASS", "paysim/dna_v2_0p95": "PASS"}

Commands, paired endpoint effects/CIs, comparison caveats and input provenance remain in the scientific report. Final verified receipt: artifacts/priority34a/final_verified_receipt.json; final manifest includes this addendum and administrative verifier.

## Runtime provenance

Apple M1 Pro; macOS-15.7.7-arm64-arm-64bit; CPU/thread1. Package versions: {"numpy": "1.26.4", "pandas": "2.0.3", "scikit-learn": "1.3.2", "scipy": "1.13.1", "torch": "2.2.2"}. Supplementary runtime capture timestamp: 2026-10-03T07:52:23.764322+00:00; this did not change the pre-run execution freeze.
