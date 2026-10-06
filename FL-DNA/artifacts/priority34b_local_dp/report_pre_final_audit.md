# Priority34B extension — LOCAL client-side update DP

Status: analysis complete; post-exit checkpoint audit pending.

264 new CPU/thread1 jobs;44 immutable baselines reused. Honest-but-curious server sees individual uploads already clipped/noised on each client. BN never transmitted. No central noise added.

## Accounting and release boundary

Unit: one client contribution transcript across50rounds, replace-one update-level adjacency; NOT record-level DP. q1, delta1e-5, joint C=.01, sensitivity2C=.02. Per-tensor radiusC/sqrt(L). Each arm/replicate is a separate hypothetical deployment; releasing all trials further composes. Checkpoint/localBN/validation/evaluation/audit material and private noise keys are NOT private releases. Ideal Gaussian simulation, not certified production finite-precision DP.

| ε whole-training | σ / sensitivity | Noise SD EACH upload | Rounds |
| --- | --- | --- | --- |
| 1 | 34.6521579132 | 0.693043158263 | 50 |
| 3 | 12.0044226396 | 0.240088452793 | 50 |
| 10 | 4.01563651964 | 0.0803127303927 | 50 |

## Descriptive paired n11 utility

Central column is historical context only: **central DP; requires secure aggregation to protect against the server**. Never use central recovery as a P34C comparator. No new NI/equivalence claim.

| Dataset | Mechanism | ε | Endpoint | Baseline | LOCAL DP | CENTRAL DP context | Local Δ [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paysim | global | 1 | f1 | 0.78322535 | 0.4718409 | 0.55188619 | -0.31138445 [-0.41144376,-0.21132515] |
| paysim | global | 1 | auc_roc | 0.99759361 | 0.94171476 | 0.95293394 | -0.055878854 [-0.094961554,-0.016796153] |
| paysim | global | 1 | pr_auc | 0.80333529 | 0.41735578 | 0.499903 | -0.38597951 [-0.4851833,-0.28677571] |
| paysim | global | 3 | f1 | 0.78322535 | 0.50039966 | 0.49414399 | -0.28282569 [-0.39525491,-0.17039647] |
| paysim | global | 3 | auc_roc | 0.99759361 | 0.92528182 | 0.94386736 | -0.072311786 [-0.17189007,0.0272665] |
| paysim | global | 3 | pr_auc | 0.80333529 | 0.45638876 | 0.44937225 | -0.34694653 [-0.45073869,-0.24315436] |
| paysim | global | 10 | f1 | 0.78322535 | 0.49916222 | 0.50356443 | -0.28406313 [-0.35570753,-0.21241873] |
| paysim | global | 10 | auc_roc | 0.99759361 | 0.96751729 | 0.96998151 | -0.030076323 [-0.040699195,-0.01945345] |
| paysim | global | 10 | pr_auc | 0.80333529 | 0.45331121 | 0.46087437 | -0.35002408 [-0.43338038,-0.26666778] |
| paysim | per_tensor | 1 | f1 | 0.78322535 | 0.54899644 | 0.5211386 | -0.23422891 [-0.26992623,-0.19853158] |
| paysim | per_tensor | 1 | auc_roc | 0.99759361 | 0.97376194 | 0.93433422 | -0.023831668 [-0.028460743,-0.019202594] |
| paysim | per_tensor | 1 | pr_auc | 0.80333529 | 0.49472867 | 0.47220817 | -0.30860662 [-0.3479199,-0.26929333] |
| paysim | per_tensor | 3 | f1 | 0.78322535 | 0.54058545 | 0.53638482 | -0.2426399 [-0.29688285,-0.18839696] |
| paysim | per_tensor | 3 | auc_roc | 0.99759361 | 0.97052216 | 0.96940798 | -0.027071454 [-0.032638581,-0.021504327] |
| paysim | per_tensor | 3 | pr_auc | 0.80333529 | 0.47968346 | 0.49565156 | -0.32365183 [-0.3848878,-0.26241586] |
| paysim | per_tensor | 10 | f1 | 0.78322535 | 0.50191953 | 0.42763828 | -0.28130582 [-0.34724814,-0.21536349] |
| paysim | per_tensor | 10 | auc_roc | 0.99759361 | 0.96970385 | 0.94817204 | -0.027889759 [-0.036387655,-0.019391862] |
| paysim | per_tensor | 10 | pr_auc | 0.80333529 | 0.47443417 | 0.38744231 | -0.32890112 [-0.40278166,-0.25502058] |
| ieee_cis | global | 1 | f1 | 0.34943956 | 0.15480631 | 0.16173672 | -0.19463326 [-0.20491591,-0.1843506] |
| ieee_cis | global | 1 | auc_roc | 0.83812534 | 0.6608408 | 0.69381332 | -0.17728454 [-0.21835962,-0.13620946] |
| ieee_cis | global | 1 | pr_auc | 0.32748308 | 0.073067658 | 0.078386142 | -0.25441542 [-0.26355836,-0.24527249] |
| ieee_cis | global | 3 | f1 | 0.34943956 | 0.15976807 | 0.1630319 | -0.1896715 [-0.19769991,-0.18164308] |
| ieee_cis | global | 3 | auc_roc | 0.83812534 | 0.70482079 | 0.73567546 | -0.13330454 [-0.15752707,-0.10908202] |
| ieee_cis | global | 3 | pr_auc | 0.32748308 | 0.076891913 | 0.081271128 | -0.25059117 [-0.25750145,-0.24368089] |
| ieee_cis | global | 10 | f1 | 0.34943956 | 0.16623234 | 0.16435197 | -0.18320722 [-0.19170461,-0.17470984] |
| ieee_cis | global | 10 | auc_roc | 0.83812534 | 0.74961544 | 0.74786944 | -0.088509904 [-0.09600428,-0.081015528] |
| ieee_cis | global | 10 | pr_auc | 0.32748308 | 0.083623255 | 0.084000151 | -0.24385983 [-0.24947604,-0.23824361] |
| ieee_cis | per_tensor | 1 | f1 | 0.34943956 | 0.15536922 | 0.16076122 | -0.19407034 [-0.20755388,-0.18058679] |
| ieee_cis | per_tensor | 1 | auc_roc | 0.83812534 | 0.67875688 | 0.70212752 | -0.15936846 [-0.19260041,-0.1261365] |
| ieee_cis | per_tensor | 1 | pr_auc | 0.32748308 | 0.073438536 | 0.076133062 | -0.25404455 [-0.26387129,-0.2442178] |
| ieee_cis | per_tensor | 3 | f1 | 0.34943956 | 0.16462393 | 0.15907424 | -0.18481563 [-0.19189455,-0.17773671] |
| ieee_cis | per_tensor | 3 | auc_roc | 0.83812534 | 0.73952656 | 0.71733084 | -0.098598782 [-0.11550461,-0.081692954] |
| ieee_cis | per_tensor | 3 | pr_auc | 0.32748308 | 0.082147136 | 0.078366312 | -0.24533595 [-0.25154897,-0.23912292] |
| ieee_cis | per_tensor | 10 | f1 | 0.34943956 | 0.16262877 | 0.16347188 | -0.18681079 [-0.19627828,-0.1773433] |
| ieee_cis | per_tensor | 10 | auc_roc | 0.83812534 | 0.73328702 | 0.74559508 | -0.10483832 [-0.12996883,-0.079707813] |
| ieee_cis | per_tensor | 10 | pr_auc | 0.32748308 | 0.081075635 | 0.081649636 | -0.24640745 [-0.25400661,-0.23880829] |
| baf | global | 1 | f1 | 0.2257117 | 0.048659797 | 0.056349928 | -0.1770519 [-0.19035501,-0.16374879] |
| baf | global | 1 | auc_roc | 0.8713404 | 0.60749926 | 0.64766805 | -0.26384114 [-0.30417491,-0.22350737] |
| baf | global | 1 | pr_auc | 0.15868126 | 0.022046023 | 0.025350498 | -0.13663524 [-0.14207694,-0.13119354] |
| baf | global | 3 | f1 | 0.2257117 | 0.058887714 | 0.067396044 | -0.16682398 [-0.17953935,-0.15410861] |
| baf | global | 3 | auc_roc | 0.8713404 | 0.64975231 | 0.67843448 | -0.22158808 [-0.24378081,-0.19939536] |
| baf | global | 3 | pr_auc | 0.15868126 | 0.026417651 | 0.030573012 | -0.13226361 [-0.13783148,-0.12669574] |
| baf | global | 10 | f1 | 0.2257117 | 0.064074174 | 0.070217552 | -0.16163752 [-0.16929688,-0.15397816] |
| baf | global | 10 | auc_roc | 0.8713404 | 0.67563369 | 0.68856768 | -0.19570671 [-0.20969436,-0.18171905] |
| baf | global | 10 | pr_auc | 0.15868126 | 0.027814312 | 0.029405441 | -0.13086695 [-0.1338236,-0.1279103] |
| baf | per_tensor | 1 | f1 | 0.2257117 | 0.042503266 | 0.05822049 | -0.18320843 [-0.19046162,-0.17595524] |
| baf | per_tensor | 1 | auc_roc | 0.8713404 | 0.60255457 | 0.64626099 | -0.26878583 [-0.29496785,-0.24260382] |
| baf | per_tensor | 1 | pr_auc | 0.15868126 | 0.019579013 | 0.026182109 | -0.13910225 [-0.14184714,-0.13635736] |
| baf | per_tensor | 3 | f1 | 0.2257117 | 0.053389903 | 0.069005339 | -0.17232179 [-0.18118515,-0.16345844] |
| baf | per_tensor | 3 | auc_roc | 0.8713404 | 0.64191387 | 0.67254286 | -0.22942653 [-0.25760028,-0.20125278] |
| baf | per_tensor | 3 | pr_auc | 0.15868126 | 0.024345527 | 0.030461159 | -0.13433574 [-0.13806997,-0.1306015] |
| baf | per_tensor | 10 | f1 | 0.2257117 | 0.061658623 | 0.062733411 | -0.16405307 [-0.17496067,-0.15314548] |
| baf | per_tensor | 10 | auc_roc | 0.8713404 | 0.666131 | 0.67424987 | -0.20520939 [-0.22623254,-0.18418625] |
| baf | per_tensor | 10 | pr_auc | 0.15868126 | 0.028309998 | 0.028155334 | -0.13037126 [-0.13615117,-0.12459136] |
| cifar10 | global | 1 | validation_accuracy | 0.36409091 | 0.10090909 | 0.10269697 | -0.26318182 [-0.27228993,-0.25407371] |
| cifar10 | global | 3 | validation_accuracy | 0.36409091 | 0.097787879 | 0.10324242 | -0.26630303 [-0.27428375,-0.25832231] |
| cifar10 | global | 10 | validation_accuracy | 0.36409091 | 0.10587879 | 0.10175758 | -0.25821212 [-0.26856625,-0.24785799] |
| cifar10 | per_tensor | 1 | validation_accuracy | 0.36409091 | 0.098606061 | 0.096363636 | -0.26548485 [-0.27251899,-0.25845071] |
| cifar10 | per_tensor | 3 | validation_accuracy | 0.36409091 | 0.1049697 | 0.093090909 | -0.25912121 [-0.26806089,-0.25018154] |
| cifar10 | per_tensor | 10 | validation_accuracy | 0.36409091 | 0.10118182 | 0.10157576 | -0.26290909 [-0.27319462,-0.25262356] |

## Same eleven-seed local-BN DNA context

| Dataset | DNA | Endpoint | Mean | Paired Δ [95% CI] |
| --- | --- | --- | --- | --- |
| paysim | dna_v1_conservative | f1 | 0.78160692 | -0.0016184296 [-0.007508671,0.0042718119] |
| paysim | dna_v1_conservative | auc_roc | 0.99763718 | +4.356689e-05 [-5.9681736e-05,0.00014681551] |
| paysim | dna_v1_conservative | pr_auc | 0.80584327 | +0.0025079811 [-0.001803508,0.0068194702] |
| paysim | dna_v2_0p95 | f1 | 0.78247155 | -0.00075380306 [-0.0076738774,0.0061662713] |
| paysim | dna_v2_0p95 | auc_roc | 0.9977114 | +0.00011778525 [-8.385328e-05,0.00031942379] |
| paysim | dna_v2_0p95 | pr_auc | 0.80938477 | +0.0060494815 [-0.00037848824,0.012477451] |
| ieee_cis | dna_v1_conservative | f1 | 0.34587489 | -0.0035646731 [-0.012566504,0.0054371577] |
| ieee_cis | dna_v1_conservative | auc_roc | 0.83610992 | -0.002015417 [-0.004761589,0.00073075496] |
| ieee_cis | dna_v1_conservative | pr_auc | 0.32195304 | -0.0055300432 [-0.014176348,0.0031162612] |
| ieee_cis | dna_v2_0p95 | f1 | 0.33653802 | -0.012901539 [-0.020610728,-0.0051923492] |
| ieee_cis | dna_v2_0p95 | auc_roc | 0.83632193 | -0.0018034125 [-0.0031561314,-0.00045069364] |
| ieee_cis | dna_v2_0p95 | pr_auc | 0.32021131 | -0.0072717708 [-0.013742666,-0.00080087548] |
| baf | dna_v1_conservative | f1 | 0.22578096 | +6.9265744e-05 [-0.0028706856,0.0030092171] |
| baf | dna_v1_conservative | auc_roc | 0.87146053 | +0.00012013199 [-0.00082801655,0.0010682805] |
| baf | dna_v1_conservative | pr_auc | 0.15844004 | -0.00024122472 [-0.00093284012,0.00045039068] |
| baf | dna_v2_0p95 | f1 | 0.22620504 | +0.00049334588 [-0.0023547112,0.003341403] |
| baf | dna_v2_0p95 | auc_roc | 0.87060258 | -0.00073782143 [-0.0020930713,0.00061742843] |
| baf | dna_v2_0p95 | pr_auc | 0.15826862 | -0.000412637 [-0.0013404823,0.00051520827] |

## Execution, interpretation and preservation

Pre-run amendment: protocols/amendments/2026-10-03_priority34b_local_dp_extension.md. Seeds321000–321010; unchanged data/partitions/model/50round training from P34A/B. CIFAR uses P34B three-client50round FedAvg, NOT P31 centralized100epochs. No C/sigma utility tuning, clamp, seed exclusion or scope reduction. Utility cost is not an optimal DP bound. Training nonfinite/negative-BN fails closed. Earlier central sources/results/reports untouched; scope label addendum reports/priority34b_central_dp_scope_addendum.md preserves prior hashes.

Synthetic preflight caught duplicate image receipt metadata before research-data runs. The preflight failure and pre-correction annex remain preserved. Wrapper removed only duplicate embedding fields, not numerical values, seeds, noise, training or objectives. All9 synthetic tests must pass before execution freeze.

Independent prediction metric reload, Gaussian RDP optimizer and78 paired CIs PASS. Full checkpoint bit-exact/post-exit audit remains mandatory. Commands/results/source hashes are in runs.jsonl, execution_freeze.json and final manifest. P34C epsilon10 uses this LOCAL release, not a noisy aggregate. No record-recovery claims from utility scores.
