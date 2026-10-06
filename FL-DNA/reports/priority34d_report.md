# Priority34D — checkpoint robustness and harder image recovery

Status: verified final audit PASS.

## Scope and paired units

Three independently initialized LeNet-Zhu models (342042/342043/342044), 39 fresh single-image targets each. The fixed trained P31/P33C checkpoint uses 39 fresh four-image recovery batches. Batch 4 is recovery only: utility training remains SGD lr0.1/momentum0, batch256, 100epochs, 16 paired order seeds51016–51031. Warm-start utility calibration is computed for this checkpoint; no sigma is transferred from another initialization. The recovery budget is4800 iterations, one restart, known labels, TV0.01. Three additional BAF checkpoints use seeds342000/342001/342002. BAF BN recovery is a batch-mean statistic instrument, not individual-record reconstruction.

## Utility calibration and DP scope

| Setting | DNA | Mechanism | Gate | Baseline validation | DNA delta | Threshold | Sigma | Next bracket | Clip(s) | One-release epsilon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| init342042 | dna_v1_conservative | per_tensor | MATCHED | 0.45779167 | -0.00275 | -0.00775 | 0.001 | 0.003 | 0.96849559, 0.42418731, 5.2521472, 0.63037242, 7.3087441, 0.99953659, 15.325729, 0.90289772 | 4013572.3 |
| init342042 | dna_v1_conservative | single | MATCHED | 0.45779167 | -0.00275 | -0.00775 | 0.001 | 0.003 | 17.848502 | 504798.53 |
| init342042 | dna_v2_0p95 | per_tensor | MATCHED | 0.45779167 | 0.00027083333 | -0.0047291667 | 0.001 | 0.003 | 0.96849559, 0.42418731, 5.2521472, 0.63037242, 7.3087441, 0.99953659, 15.325729, 0.90289772 | 4013572.3 |
| init342042 | dna_v2_0p95 | single | MATCHED | 0.45779167 | 0.00027083333 | -0.0047291667 | 0.0003 | 0.001 | 17.848502 | 5571550.6 |
| init342043 | dna_v1_conservative | per_tensor | MATCHED | 0.46533333 | -0.0014166667 | -0.0064166667 | 0.001 | 0.003 | 0.64559782, 0.13415791, 1.3809789, 0.1701976, 3.3463895, 0.30709667, 7.8658574, 0.46965906 | 4013572.3 |
| init342043 | dna_v1_conservative | single | MATCHED | 0.46533333 | -0.0014166667 | -0.0064166667 | 0.001 | 0.003 | 8.6567411 | 504798.53 |
| init342043 | dna_v2_0p95 | per_tensor | MATCHED | 0.46533333 | 0.0016666667 | -0.0033333333 | 0.001 | 0.003 | 0.64559782, 0.13415791, 1.3809789, 0.1701976, 3.3463895, 0.30709667, 7.8658574, 0.46965906 | 4013572.3 |
| init342043 | dna_v2_0p95 | single | MATCHED | 0.46533333 | 0.0016666667 | -0.0033333333 | 0.001 | 0.003 | 8.6567411 | 504798.53 |
| init342044 | dna_v1_conservative | per_tensor | MATCHED | 0.46035417 | -0.0020625 | -0.0070625 | 0.001 | 0.003 | 0.81840323, 0.22864529, 3.6933641, 0.37171576, 6.8110134, 0.80482135, 10.478618, 0.6559934 | 4013572.3 |
| init342044 | dna_v1_conservative | single | MATCHED | 0.46035417 | -0.0020625 | -0.0070625 | 0.001 | 0.003 | 13.040618 | 504798.53 |
| init342044 | dna_v2_0p95 | per_tensor | MATCHED | 0.46035417 | 0.00085416667 | -0.0041458333 | 0.001 | 0.003 | 0.81840323, 0.22864529, 3.6933641, 0.37171576, 6.8110134, 0.80482135, 10.478618, 0.6559934 | 4013572.3 |
| init342044 | dna_v2_0p95 | single | MATCHED | 0.46035417 | 0.00085416667 | -0.0041458333 | 0.001 | 0.003 | 13.040618 | 504798.53 |
| trained_batch4 | dna_v1_conservative | per_tensor | MATCHED | 0.48504167 | 0.0001875 | -0.0048125 | 0.01 | 0.03 | 0.54324129, 0.11071388, 0.89684687, 0.11492268, 1.0334976, 0.12768771, 1.0449039, 0.12190348 | 41357.228 |
| trained_batch4 | dna_v1_conservative | single | MATCHED | 0.48504167 | 0.0001875 | -0.0048125 | 0.003 | 0.01 | 1.7336099 | 57155.064 |
| trained_batch4 | dna_v2_0p95 | per_tensor | MATCHED | 0.48504167 | -0.00041666667 | -0.0054166667 | 0.01 | 0.03 | 0.54324129, 0.11071388, 0.89684687, 0.11492268, 1.0334976, 0.12768771, 1.0449039, 0.12190348 | 41357.228 |
| trained_batch4 | dna_v2_0p95 | single | MATCHED | 0.48504167 | -0.00041666667 | -0.0054166667 | 0.003 | 0.01 | 1.7336099 | 57155.064 |

Matching: baseline validation >=0.40; largest passing sigma with paired mean delta >=DNA delta−0.005 and the next higher grid point below threshold. Missing brackets are NOT_ASSESSABLE, never privacy evidence. Initial grid0.0001/0.0003/0.001/0.003/0.01/0.03/0.1/0.3; conditional1/3/10 only when preregistered top-grid trigger passes. Epsilon above is an analytic Gaussian RDP bound for one bounded-update release, delta1e-5, under the original add/remove convention, not record-level or whole-training DP. Per-tensor composition uses sqrt(number of tensors) sensitivity ratio. Noise is applied to the same individual client gradient as each transform; no noisy aggregate comparator.

## Image endpoint medians and intervals

| Setting | Arm | PSNR dB | SSIM | MSE |
| --- | --- | --- | --- | --- |
| init342042 | dna_v1_conservative | 24.086289 [22.493374, 25.585273] (ranks 13/27) | 0.86975622 [0.85476297, 0.89588594] (ranks 13/27) | 0.0039027531 [0.0027635843, 0.0056319996] (ranks 13/27) |
| init342042 | dna_v2_0p95 | 24.158828 [22.394904, 25.916683] (ranks 13/27) | 0.87146044 [0.85609299, 0.89792436] (ranks 13/27) | 0.0038381077 [0.002560541, 0.0057611559] (ranks 13/27) |
| init342042 | dp_per_tensor_for_dna_v1_conservative | 20.137785 [18.540629, 21.621453] (ranks 13/27) | 0.70852423 [0.65704232, 0.75157231] (ranks 13/27) | 0.0096877189 [0.0068842187, 0.013993845] (ranks 13/27) |
| init342042 | dp_per_tensor_for_dna_v2_0p95 | 20.269395 [18.665101, 21.707287] (ranks 13/27) | 0.70755213 [0.65361857, 0.73960286] (ranks 13/27) | 0.0093985423 [0.0067494959, 0.013598464] (ranks 13/27) |
| init342042 | dp_single_for_dna_v1_conservative | 23.854851 [22.234858, 25.182467] (ranks 13/27) | 0.85722661 [0.84728307, 0.88391346] (ranks 13/27) | 0.004116375 [0.0030321684, 0.0059774253] (ranks 13/27) |
| init342042 | dp_single_for_dna_v2_0p95 | 24.438403 [22.672301, 26.045645] (ranks 13/27) | 0.87227631 [0.85979563, 0.89770728] (ranks 13/27) | 0.0035988165 [0.0024856243, 0.0054046786] (ranks 13/27) |
| init342042 | unprotected | 24.239095 [22.756291, 25.821217] (ranks 13/27) | 0.87328887 [0.8572796, 0.89884919] (ranks 13/27) | 0.0037678231 [0.0026174493, 0.0053011603] (ranks 13/27) |
| init342043 | dna_v1_conservative | 17.656704 [16.353589, 19.780585] (ranks 13/27) | 0.65686709 [0.60517496, 0.68058014] (ranks 13/27) | 0.017152587 [0.010518201, 0.023154804] (ranks 13/27) |
| init342043 | dna_v2_0p95 | 17.997307 [15.997298, 19.417412] (ranks 13/27) | 0.60965198 [0.54259789, 0.67999464] (ranks 13/27) | 0.015858762 [0.011435595, 0.025134495] (ranks 13/27) |
| init342043 | dp_per_tensor_for_dna_v1_conservative | 16.433781 [14.619839, 16.928224] (ranks 13/27) | 0.45737243 [0.42140183, 0.47585943] (ranks 13/27) | 0.022731176 [0.020285122, 0.034515653] (ranks 13/27) |
| init342043 | dp_per_tensor_for_dna_v2_0p95 | 16.224158 [14.675912, 16.671628] (ranks 13/27) | 0.44439921 [0.41259563, 0.48629788] (ranks 13/27) | 0.023855262 [0.02151975, 0.03407288] (ranks 13/27) |
| init342043 | dp_single_for_dna_v1_conservative | 18.139863 [16.425354, 19.765382] (ranks 13/27) | 0.63947946 [0.55501199, 0.69462651] (ranks 13/27) | 0.015346654 [0.010555086, 0.022775324] (ranks 13/27) |
| init342043 | dp_single_for_dna_v2_0p95 | 18.106403 [16.092562, 18.897552] (ranks 13/27) | 0.64648432 [0.54736143, 0.68934536] (ranks 13/27) | 0.015465349 [0.012889758, 0.024589166] (ranks 13/27) |
| init342043 | unprotected | 18.454354 [16.067779, 19.628958] (ranks 13/27) | 0.621409 [0.53904045, 0.70431995] (ranks 13/27) | 0.014274622 [0.010891914, 0.024729883] (ranks 13/27) |
| init342044 | dna_v1_conservative | 19.43624 [17.406257, 20.073455] (ranks 13/27) | 0.65454549 [0.59272534, 0.78382587] (ranks 13/27) | 0.011386125 [0.0098322863, 0.018170811] (ranks 13/27) |
| init342044 | dna_v2_0p95 | 18.71651 [17.572745, 20.12169] (ranks 13/27) | 0.67415541 [0.58675432, 0.77863425] (ranks 13/27) | 0.013438445 [0.0097236875, 0.017487412] (ranks 13/27) |
| init342044 | dp_per_tensor_for_dna_v1_conservative | 17.633921 [16.384774, 18.587775] (ranks 13/27) | 0.58804041 [0.52762502, 0.6803599] (ranks 13/27) | 0.017242806 [0.013842754, 0.022989132] (ranks 13/27) |
| init342044 | dp_per_tensor_for_dna_v2_0p95 | 17.900175 [16.326367, 18.538477] (ranks 13/27) | 0.5701738 [0.49178424, 0.68307143] (ranks 13/27) | 0.016217446 [0.014000784, 0.023300394] (ranks 13/27) |
| init342044 | dp_single_for_dna_v1_conservative | 18.291615 [17.402805, 19.893034] (ranks 13/27) | 0.62655711 [0.56513715, 0.77276975] (ranks 13/27) | 0.01481967 [0.010249357, 0.01818526] (ranks 13/27) |
| init342044 | dp_single_for_dna_v2_0p95 | 18.238946 [16.913535, 20.088298] (ranks 13/27) | 0.65863782 [0.57166123, 0.76769942] (ranks 13/27) | 0.015000489 [0.00979874, 0.020353848] (ranks 13/27) |
| init342044 | unprotected | 19.217736 [17.597355, 20.637453] (ranks 13/27) | 0.67242718 [0.59384841, 0.79919845] (ranks 13/27) | 0.011973646 [0.0086348485, 0.017388595] (ranks 13/27) |
| pooled_initializations | dna_v1_conservative | 20.274412 [19.437277, 21.670066] (ranks 48/70) | 0.72797084 [0.67658401, 0.79942745] (ranks 48/70) | 0.0093876915 [0.0068075894, 0.011383408] (ranks 48/70) |
| pooled_initializations | dna_v2_0p95 | 20.341518 [19.083507, 21.271512] (ranks 48/70) | 0.73006445 [0.67415541, 0.79742432] (ranks 48/70) | 0.00924375 [0.0074618892, 0.012349498] (ranks 48/70) |
| pooled_initializations | dp_per_tensor_for_dna_v1_conservative | 17.633921 [16.92924, 18.484849] (ranks 48/70) | 0.58619386 [0.53214163, 0.63314158] (ranks 48/70) | 0.017242806 [0.01417474, 0.020280376] (ranks 48/70) |
| pooled_initializations | dp_per_tensor_for_dna_v2_0p95 | 17.709425 [17.016442, 18.530934] (ranks 48/70) | 0.5701738 [0.50834495, 0.60959107] (ranks 48/70) | 0.016945621 [0.014025122, 0.019877227] (ranks 48/70) |
| pooled_initializations | dp_single_for_dna_v1_conservative | 20.040226 [18.879481, 21.026579] (ranks 48/70) | 0.74223304 [0.6704216, 0.78004879] (ranks 48/70) | 0.0099078035 [0.0078948168, 0.012943505] (ranks 48/70) |
| pooled_initializations | dp_single_for_dna_v2_0p95 | 20.020784 [18.593519, 21.26885] (ranks 48/70) | 0.74262816 [0.67240959, 0.78173691] (ranks 48/70) | 0.0099522574 [0.0074664652, 0.013824458] (ranks 48/70) |
| pooled_initializations | unprotected | 20.5135 [19.419176, 21.26181] (ranks 48/70) | 0.75701755 [0.69739085, 0.80348855] (ranks 48/70) | 0.0088848481 [0.0074785776, 0.011430952] (ranks 48/70) |
| trained_batch4 | dna_v1_conservative | 15.036854 [14.279721, 15.737744] (ranks 13/27) | 0.37909418 [0.32080407, 0.43303329] (ranks 13/27) | 0.036340192 [0.03198806, 0.041001405] (ranks 13/27) |
| trained_batch4 | dna_v2_0p95 | 14.864825 [14.273018, 16.037369] (ranks 13/27) | 0.38461756 [0.30786848, 0.44052193] (ranks 13/27) | 0.036595527 [0.029949925, 0.040671741] (ranks 13/27) |
| trained_batch4 | dp_per_tensor_for_dna_v1_conservative | 10.939575 [10.720749, 11.176961] (ranks 13/27) | 0.12335179 [0.10979112, 0.13670628] (ranks 13/27) | 0.083358291 [0.079146306, 0.088423368] (ranks 13/27) |
| trained_batch4 | dp_per_tensor_for_dna_v2_0p95 | 10.922588 [10.576655, 11.087141] (ranks 13/27) | 0.12085633 [0.10480821, 0.12898912] (ranks 13/27) | 0.085721571 [0.078784659, 0.090063902] (ranks 13/27) |
| trained_batch4 | dp_single_for_dna_v1_conservative | 12.404507 [11.844262, 12.810678] (ranks 13/27) | 0.20089062 [0.17659859, 0.21569196] (ranks 13/27) | 0.059978173 [0.055234705, 0.068630843] (ranks 13/27) |
| trained_batch4 | dp_single_for_dna_v2_0p95 | 12.394029 [11.866126, 12.915099] (ranks 13/27) | 0.19397647 [0.17752306, 0.22382044] (ranks 13/27) | 0.06378222 [0.053305308, 0.070708886] (ranks 13/27) |
| trained_batch4 | unprotected | 15.133784 [14.208434, 16.153386] (ranks 13/27) | 0.39276198 [0.32421999, 0.43978291] (ranks 13/27) | 0.035754654 [0.028662688, 0.04172792] (ranks 13/27) |

Intervals are distribution-free two-sided >=95% median intervals. For n39 use ranks13/27. Image batch4 metrics use post-optimization Hungarian assignment and the mean across four images; the paired unit is the batch, not156 independent records. PSNR is the primary endpoint; SSIM and MSE are descriptive.

## BAF qualification, calibration and endpoint medians

| Checkpoint | Qualification status | Development / distortion context |
| --- | --- | --- |
| 342000 | QUALIFIED | n8: decoy p=0.00390625, prior p=0.00390625; n24: decoy p=5.9604645e-08, prior p=5.9604645e-08; dna_v1_conservative: C=3.7225798, sigma=0.0088487726, one-release epsilon=6927.9143; dna_v2_0p95: C=3.7225798, sigma=0.011477194, one-release epsilon=4213.8508 |
| 342001 | QUALIFIED | n8: decoy p=0.00390625, prior p=0.00390625; n24: decoy p=5.9604645e-08, prior p=5.9604645e-08; dna_v1_conservative: C=9.0186482, sigma=0.0036920166, one-release epsilon=37980.835; dna_v2_0p95: C=9.0186482, sigma=0.0093129318, one-release epsilon=6280.2255 |
| 342002 | QUALIFIED | n8: decoy p=0.00390625, prior p=0.00390625; n24: decoy p=5.9604645e-08, prior p=5.9604645e-08; dna_v1_conservative: C=43.44797, sigma=0.00074253342, one-release epsilon=913317.67; dna_v2_0p95: C=43.44797, sigma=0.0013268768, one-release epsilon=287610.16 |

| Checkpoint | Arm | Standardized batch-mean MSE |
| --- | --- | --- |
| 342000 | unprotected | 2.9127459e-10 [1.9008561e-10, 4.3579264e-10] (ranks 13/27) |
| 342000 | dna_v1_conservative | 14.097695 [11.724571, 22.678863] (ranks 13/27) |
| 342000 | dna_v2_0p95 | 0.038544073 [0.027772224, 0.047491679] (ranks 13/27) |
| 342000 | dp_dna_v1_conservative | 38.575244 [27.968431, 56.295886] (ranks 13/27) |
| 342000 | dp_dna_v2_0p95 | 54.57709 [41.777734, 68.516009] (ranks 13/27) |
| 342001 | unprotected | 7.918797e-10 [4.541047e-10, 9.8535625e-10] (ranks 13/27) |
| 342001 | dna_v1_conservative | 21.340621 [16.205594, 33.563825] (ranks 13/27) |
| 342001 | dna_v2_0p95 | 0.026464823 [0.021397159, 0.042758979] (ranks 13/27) |
| 342001 | dp_dna_v1_conservative | 49.468892 [32.43773, 61.079671] (ranks 13/27) |
| 342001 | dp_dna_v2_0p95 | 235.10603 [160.42753, 311.4981] (ranks 13/27) |
| 342002 | unprotected | 2.4199792e-10 [1.6701972e-10, 3.0046295e-10] (ranks 13/27) |
| 342002 | dna_v1_conservative | 21.057633 [14.570768, 36.253473] (ranks 13/27) |
| 342002 | dna_v2_0p95 | 0.053876507 [0.041492547, 0.065048408] (ranks 13/27) |
| 342002 | dp_dna_v1_conservative | 39.054495 [29.799169, 48.188184] (ranks 13/27) |
| 342002 | dp_dna_v2_0p95 | 134.81571 [90.356451, 192.41022] (ranks 13/27) |
| pooled_checkpoints | unprotected | 3.1540628e-10 [2.6667062e-10, 4.0668364e-10] (ranks 48/70) |
| pooled_checkpoints | dna_v1_conservative | 19.917639 [16.353109, 22.678863] (ranks 48/70) |
| pooled_checkpoints | dna_v2_0p95 | 0.039768548 [0.033665361, 0.045456402] (ranks 48/70) |
| pooled_checkpoints | dp_dna_v1_conservative | 39.195847 [35.356853, 49.468892] (ranks 48/70) |
| pooled_checkpoints | dp_dna_v2_0p95 | 116.28638 [90.356451, 152.36399] (ranks 48/70) |

Fresh qualification n8 then n24 must beat both training-prior and cyclic-decoy controls with exact one-sided p<0.05. Failed checkpoints are retained without replacement. Development n24 calibrates C=1.01×maximum raw norm and a median Gaussian-norm distortion match. All scheduled confirmatory pairs must be complete and finite; no seed exclusion or clamping.

## Fixed confirmatory statistics — all76 reservations

| Hypothesis | Status | n | DNA median | Comparator median | Paired DNA−comparator | Raw p | Holm76 | Holm211 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P34D::image::init342042::dna_v1_conservative::unprotected::dna_greater | ASSESSABLE | 39 | 24.086289 [22.493374, 25.585273] (ranks 13/27) | 24.239095 [22.756291, 25.821217] (ranks 13/27) | -0.17776773 [-0.27543781, -0.053159967] (ranks 13/27) | 0.99996487 | 1 | 1 |
| P34D::image::init342042::dna_v1_conservative::unprotected::dna_less | ASSESSABLE | 39 | 24.086289 [22.493374, 25.585273] (ranks 13/27) | 24.239095 [22.756291, 25.821217] (ranks 13/27) | -0.17776773 [-0.27543781, -0.053159967] (ranks 13/27) | 0.00014703844 | 0.0082341525 | 0.020879458 |
| P34D::image::init342042::dna_v1_conservative::single::dna_greater | ASSESSABLE | 39 | 24.086289 [22.493374, 25.585273] (ranks 13/27) | 23.854851 [22.234858, 25.182467] (ranks 13/27) | 0.37803145 [0.16925221, 0.51774011] (ranks 13/27) | 0.00053250982 | 0.02928804 | 0.074551374 |
| P34D::image::init342042::dna_v1_conservative::single::dna_less | ASSESSABLE | 39 | 24.086289 [22.493374, 25.585273] (ranks 13/27) | 23.854851 [22.234858, 25.182467] (ranks 13/27) | 0.37803145 [0.16925221, 0.51774011] (ranks 13/27) | 0.99985296 | 1 | 1 |
| P34D::image::init342042::dna_v1_conservative::per_tensor::dna_greater | ASSESSABLE | 39 | 24.086289 [22.493374, 25.585273] (ranks 13/27) | 20.137785 [18.540629, 21.621453] (ranks 13/27) | 4.1051529 [2.9938444, 5.0693501] (ranks 13/27) | 1.8189894e-12 | 1.3278623e-10 | 3.783498e-10 |
| P34D::image::init342042::dna_v1_conservative::per_tensor::dna_less | ASSESSABLE | 39 | 24.086289 [22.493374, 25.585273] (ranks 13/27) | 20.137785 [18.540629, 21.621453] (ranks 13/27) | 4.1051529 [2.9938444, 5.0693501] (ranks 13/27) | 1 | 1 | 1 |
| P34D::image::init342042::dna_v2_0p95::unprotected::dna_greater | ASSESSABLE | 39 | 24.158828 [22.394904, 25.916683] (ranks 13/27) | 24.239095 [22.756291, 25.821217] (ranks 13/27) | -0.01505768 [-0.079717512, 0.024021379] (ranks 13/27) | 0.90020457 | 1 | 1 |
| P34D::image::init342042::dna_v2_0p95::unprotected::dna_less | ASSESSABLE | 39 | 24.158828 [22.394904, 25.916683] (ranks 13/27) | 24.239095 [22.756291, 25.821217] (ranks 13/27) | -0.01505768 [-0.079717512, 0.024021379] (ranks 13/27) | 0.16839182 | 1 | 1 |
| P34D::image::init342042::dna_v2_0p95::single::dna_greater | ASSESSABLE | 39 | 24.158828 [22.394904, 25.916683] (ranks 13/27) | 24.438403 [22.672301, 26.045645] (ranks 13/27) | -0.032700699 [-0.086845405, 0.029412771] (ranks 13/27) | 0.94593549 | 1 | 1 |
| P34D::image::init342042::dna_v2_0p95::single::dna_less | ASSESSABLE | 39 | 24.158828 [22.394904, 25.916683] (ranks 13/27) | 24.438403 [22.672301, 26.045645] (ranks 13/27) | -0.032700699 [-0.086845405, 0.029412771] (ranks 13/27) | 0.099795433 | 1 | 1 |
| P34D::image::init342042::dna_v2_0p95::per_tensor::dna_greater | ASSESSABLE | 39 | 24.158828 [22.394904, 25.916683] (ranks 13/27) | 20.269395 [18.665101, 21.707287] (ranks 13/27) | 4.385316 [3.6255399, 5.1618477] (ranks 13/27) | 1.8189894e-12 | 1.3278623e-10 | 3.783498e-10 |
| P34D::image::init342042::dna_v2_0p95::per_tensor::dna_less | ASSESSABLE | 39 | 24.158828 [22.394904, 25.916683] (ranks 13/27) | 20.269395 [18.665101, 21.707287] (ranks 13/27) | 4.385316 [3.6255399, 5.1618477] (ranks 13/27) | 1 | 1 | 1 |
| P34D::image::init342043::dna_v1_conservative::unprotected::dna_greater | ASSESSABLE | 39 | 17.656704 [16.353589, 19.780585] (ranks 13/27) | 18.454354 [16.067779, 19.628958] (ranks 13/27) | 0.14034186 [-0.17016212, 0.71625705] (ranks 13/27) | 0.26119869 | 1 | 1 |
| P34D::image::init342043::dna_v1_conservative::unprotected::dna_less | ASSESSABLE | 39 | 17.656704 [16.353589, 19.780585] (ranks 13/27) | 18.454354 [16.067779, 19.628958] (ranks 13/27) | 0.14034186 [-0.17016212, 0.71625705] (ranks 13/27) | 0.83160818 | 1 | 1 |
| P34D::image::init342043::dna_v1_conservative::single::dna_greater | ASSESSABLE | 39 | 17.656704 [16.353589, 19.780585] (ranks 13/27) | 18.139863 [16.425354, 19.765382] (ranks 13/27) | 0.24551745 [-0.2394534, 0.60476413] (ranks 13/27) | 0.16839182 | 1 | 1 |
| P34D::image::init342043::dna_v1_conservative::single::dna_less | ASSESSABLE | 39 | 17.656704 [16.353589, 19.780585] (ranks 13/27) | 18.139863 [16.425354, 19.765382] (ranks 13/27) | 0.24551745 [-0.2394534, 0.60476413] (ranks 13/27) | 0.90020457 | 1 | 1 |
| P34D::image::init342043::dna_v1_conservative::per_tensor::dna_greater | ASSESSABLE | 39 | 17.656704 [16.353589, 19.780585] (ranks 13/27) | 16.433781 [14.619839, 16.928224] (ranks 13/27) | 2.3247447 [1.733439, 3.1382057] (ranks 13/27) | 7.2759576e-11 | 4.6566129e-09 | 1.200533e-08 |
| P34D::image::init342043::dna_v1_conservative::per_tensor::dna_less | ASSESSABLE | 39 | 17.656704 [16.353589, 19.780585] (ranks 13/27) | 16.433781 [14.619839, 16.928224] (ranks 13/27) | 2.3247447 [1.733439, 3.1382057] (ranks 13/27) | 1 | 1 | 1 |
| P34D::image::init342043::dna_v2_0p95::unprotected::dna_greater | ASSESSABLE | 39 | 17.997307 [15.997298, 19.417412] (ranks 13/27) | 18.454354 [16.067779, 19.628958] (ranks 13/27) | -0.017156068 [-0.49028468, 0.31533316] (ranks 13/27) | 0.73880131 | 1 | 1 |
| P34D::image::init342043::dna_v2_0p95::unprotected::dna_less | ASSESSABLE | 39 | 17.997307 [15.997298, 19.417412] (ranks 13/27) | 18.454354 [16.067779, 19.628958] (ranks 13/27) | -0.017156068 [-0.49028468, 0.31533316] (ranks 13/27) | 0.37462931 | 1 | 1 |
| P34D::image::init342043::dna_v2_0p95::single::dna_greater | ASSESSABLE | 39 | 17.997307 [15.997298, 19.417412] (ranks 13/27) | 18.106403 [16.092562, 18.897552] (ranks 13/27) | 0.045512007 [-0.47699849, 0.40729801] (ranks 13/27) | 0.5 | 1 | 1 |
| P34D::image::init342043::dna_v2_0p95::single::dna_less | ASSESSABLE | 39 | 17.997307 [15.997298, 19.417412] (ranks 13/27) | 18.106403 [16.092562, 18.897552] (ranks 13/27) | 0.045512007 [-0.47699849, 0.40729801] (ranks 13/27) | 0.62537069 | 1 | 1 |
| P34D::image::init342043::dna_v2_0p95::per_tensor::dna_greater | ASSESSABLE | 39 | 17.997307 [15.997298, 19.417412] (ranks 13/27) | 16.224158 [14.675912, 16.671628] (ranks 13/27) | 2.2810571 [1.4254467, 3.0229438] (ranks 13/27) | 1.6765807e-07 | 1.03948e-05 | 2.6489975e-05 |
| P34D::image::init342043::dna_v2_0p95::per_tensor::dna_less | ASSESSABLE | 39 | 17.997307 [15.997298, 19.417412] (ranks 13/27) | 16.224158 [14.675912, 16.671628] (ranks 13/27) | 2.2810571 [1.4254467, 3.0229438] (ranks 13/27) | 0.99999998 | 1 | 1 |
| P34D::image::init342044::dna_v1_conservative::unprotected::dna_greater | ASSESSABLE | 39 | 19.43624 [17.406257, 20.073455] (ranks 13/27) | 19.217736 [17.597355, 20.637453] (ranks 13/27) | -0.064832552 [-0.17282382, 0.38096247] (ranks 13/27) | 0.83160818 | 1 | 1 |
| P34D::image::init342044::dna_v1_conservative::unprotected::dna_less | ASSESSABLE | 39 | 19.43624 [17.406257, 20.073455] (ranks 13/27) | 19.217736 [17.597355, 20.637453] (ranks 13/27) | -0.064832552 [-0.17282382, 0.38096247] (ranks 13/27) | 0.26119869 | 1 | 1 |
| P34D::image::init342044::dna_v1_conservative::single::dna_greater | ASSESSABLE | 39 | 19.43624 [17.406257, 20.073455] (ranks 13/27) | 18.291615 [17.402805, 19.893034] (ranks 13/27) | 0.64901964 [0.25764229, 0.95720622] (ranks 13/27) | 7.1496306e-06 | 0.00041467858 | 0.0010581453 |
| P34D::image::init342044::dna_v1_conservative::single::dna_less | ASSESSABLE | 39 | 19.43624 [17.406257, 20.073455] (ranks 13/27) | 18.291615 [17.402805, 19.893034] (ranks 13/27) | 0.64901964 [0.25764229, 0.95720622] (ranks 13/27) | 0.99999879 | 1 | 1 |
| P34D::image::init342044::dna_v1_conservative::per_tensor::dna_greater | ASSESSABLE | 39 | 19.43624 [17.406257, 20.073455] (ranks 13/27) | 17.633921 [16.384774, 18.587775] (ranks 13/27) | 1.4758595 [1.082243, 1.6599048] (ranks 13/27) | 1.214954e-06 | 7.2897237e-05 | 0.00018345805 |
| P34D::image::init342044::dna_v1_conservative::per_tensor::dna_less | ASSESSABLE | 39 | 19.43624 [17.406257, 20.073455] (ranks 13/27) | 17.633921 [16.384774, 18.587775] (ranks 13/27) | 1.4758595 [1.082243, 1.6599048] (ranks 13/27) | 0.99999983 | 1 | 1 |
| P34D::image::init342044::dna_v2_0p95::unprotected::dna_greater | ASSESSABLE | 39 | 18.71651 [17.572745, 20.12169] (ranks 13/27) | 19.217736 [17.597355, 20.637453] (ranks 13/27) | -0.070470086 [-0.53577018, 0.03563642] (ranks 13/27) | 0.83160818 | 1 | 1 |
| P34D::image::init342044::dna_v2_0p95::unprotected::dna_less | ASSESSABLE | 39 | 18.71651 [17.572745, 20.12169] (ranks 13/27) | 19.217736 [17.597355, 20.637453] (ranks 13/27) | -0.070470086 [-0.53577018, 0.03563642] (ranks 13/27) | 0.26119869 | 1 | 1 |
| P34D::image::init342044::dna_v2_0p95::single::dna_greater | ASSESSABLE | 39 | 18.71651 [17.572745, 20.12169] (ranks 13/27) | 18.238946 [16.913535, 20.088298] (ranks 13/27) | 0.21396239 [-0.24393581, 0.57999664] (ranks 13/27) | 0.16839182 | 1 | 1 |
| P34D::image::init342044::dna_v2_0p95::single::dna_less | ASSESSABLE | 39 | 18.71651 [17.572745, 20.12169] (ranks 13/27) | 18.238946 [16.913535, 20.088298] (ranks 13/27) | 0.21396239 [-0.24393581, 0.57999664] (ranks 13/27) | 0.90020457 | 1 | 1 |
| P34D::image::init342044::dna_v2_0p95::per_tensor::dna_greater | ASSESSABLE | 39 | 18.71651 [17.572745, 20.12169] (ranks 13/27) | 17.900175 [16.326367, 18.538477] (ranks 13/27) | 1.2711451 [0.73489349, 1.9190928] (ranks 13/27) | 1.214954e-06 | 7.2897237e-05 | 0.00018345805 |
| P34D::image::init342044::dna_v2_0p95::per_tensor::dna_less | ASSESSABLE | 39 | 18.71651 [17.572745, 20.12169] (ranks 13/27) | 17.900175 [16.326367, 18.538477] (ranks 13/27) | 1.2711451 [0.73489349, 1.9190928] (ranks 13/27) | 0.99999983 | 1 | 1 |
| P34D::image::pooled_initializations::dna_v1_conservative::unprotected::dna_greater | ASSESSABLE | 117 | 20.274412 [19.437277, 21.670066] (ranks 48/70) | 20.5135 [19.419176, 21.26181] (ranks 48/70) | -0.064832552 [-0.16545625, -0.00068220751] (ranks 48/70) | 0.98696222 | 1 | 1 |
| P34D::image::pooled_initializations::dna_v1_conservative::unprotected::dna_less | ASSESSABLE | 117 | 20.274412 [19.437277, 21.670066] (ranks 48/70) | 20.5135 [19.419176, 21.26181] (ranks 48/70) | -0.064832552 [-0.16545625, -0.00068220751] (ranks 48/70) | 0.020750394 | 1 | 1 |
| P34D::image::pooled_initializations::dna_v1_conservative::single::dna_greater | ASSESSABLE | 117 | 20.274412 [19.437277, 21.670066] (ranks 48/70) | 20.040226 [18.879481, 21.026579] (ranks 48/70) | 0.39745678 [0.25764229, 0.53475954] (ranks 48/70) | 1.8407208e-07 | 1.1228397e-05 | 2.7978956e-05 |
| P34D::image::pooled_initializations::dna_v1_conservative::single::dna_less | ASSESSABLE | 117 | 20.274412 [19.437277, 21.670066] (ranks 48/70) | 20.040226 [18.879481, 21.026579] (ranks 48/70) | 0.39745678 [0.25764229, 0.53475954] (ranks 48/70) | 0.99999994 | 1 | 1 |
| P34D::image::pooled_initializations::dna_v1_conservative::per_tensor::dna_greater | ASSESSABLE | 117 | 20.274412 [19.437277, 21.670066] (ranks 48/70) | 17.633921 [16.92924, 18.484849] (ranks 48/70) | 2.3247447 [1.9992299, 2.9566733] (ranks 48/70) | 1.9878158e-26 | 1.4908618e-24 | 4.1744131e-24 |
| P34D::image::pooled_initializations::dna_v1_conservative::per_tensor::dna_less | ASSESSABLE | 117 | 20.274412 [19.437277, 21.670066] (ranks 48/70) | 17.633921 [16.92924, 18.484849] (ranks 48/70) | 2.3247447 [1.9992299, 2.9566733] (ranks 48/70) | 1 | 1 | 1 |
| P34D::image::pooled_initializations::dna_v2_0p95::unprotected::dna_greater | ASSESSABLE | 117 | 20.341518 [19.083507, 21.271512] (ranks 48/70) | 20.5135 [19.419176, 21.26181] (ranks 48/70) | -0.039517149 [-0.088016118, 0.0011004309] (ranks 48/70) | 0.93062342 | 1 | 1 |
| P34D::image::pooled_initializations::dna_v2_0p95::unprotected::dna_less | ASSESSABLE | 117 | 20.341518 [19.083507, 21.271512] (ranks 48/70) | 20.5135 [19.419176, 21.26181] (ranks 48/70) | -0.039517149 [-0.088016118, 0.0011004309] (ranks 48/70) | 0.09767453 | 1 | 1 |
| P34D::image::pooled_initializations::dna_v2_0p95::single::dna_greater | ASSESSABLE | 117 | 20.341518 [19.083507, 21.271512] (ranks 48/70) | 20.020784 [18.593519, 21.26885] (ranks 48/70) | -0.0006807913 [-0.064632427, 0.14330737] (ranks 48/70) | 0.57329578 | 1 | 1 |
| P34D::image::pooled_initializations::dna_v2_0p95::single::dna_less | ASSESSABLE | 117 | 20.341518 [19.083507, 21.271512] (ranks 48/70) | 20.020784 [18.593519, 21.26885] (ranks 48/70) | -0.0006807913 [-0.064632427, 0.14330737] (ranks 48/70) | 0.5 | 1 | 1 |
| P34D::image::pooled_initializations::dna_v2_0p95::per_tensor::dna_greater | ASSESSABLE | 117 | 20.341518 [19.083507, 21.271512] (ranks 48/70) | 17.709425 [17.016442, 18.530934] (ranks 48/70) | 2.2955621 [1.9190928, 2.7273835] (ranks 48/70) | 5.4129005e-23 | 4.0055464e-21 | 1.1312962e-20 |
| P34D::image::pooled_initializations::dna_v2_0p95::per_tensor::dna_less | ASSESSABLE | 117 | 20.341518 [19.083507, 21.271512] (ranks 48/70) | 17.709425 [17.016442, 18.530934] (ranks 48/70) | 2.2955621 [1.9190928, 2.7273835] (ranks 48/70) | 1 | 1 | 1 |
| P34D::image::trained_batch4::dna_v1_conservative::unprotected::dna_greater | ASSESSABLE | 39 | 15.036854 [14.279721, 15.737744] (ranks 13/27) | 15.133784 [14.208434, 16.153386] (ranks 13/27) | -0.085259635 [-0.40892776, 0.21583555] (ranks 13/27) | 0.73880131 | 1 | 1 |
| P34D::image::trained_batch4::dna_v1_conservative::unprotected::dna_less | ASSESSABLE | 39 | 15.036854 [14.279721, 15.737744] (ranks 13/27) | 15.133784 [14.208434, 16.153386] (ranks 13/27) | -0.085259635 [-0.40892776, 0.21583555] (ranks 13/27) | 0.37462931 | 1 | 1 |
| P34D::image::trained_batch4::dna_v1_conservative::single::dna_greater | ASSESSABLE | 39 | 15.036854 [14.279721, 15.737744] (ranks 13/27) | 12.404507 [11.844262, 12.810678] (ranks 13/27) | 2.6334169 [1.9676501, 3.1004301] (ranks 13/27) | 1.8189894e-12 | 1.3278623e-10 | 3.783498e-10 |
| P34D::image::trained_batch4::dna_v1_conservative::single::dna_less | ASSESSABLE | 39 | 15.036854 [14.279721, 15.737744] (ranks 13/27) | 12.404507 [11.844262, 12.810678] (ranks 13/27) | 2.6334169 [1.9676501, 3.1004301] (ranks 13/27) | 1 | 1 | 1 |
| P34D::image::trained_batch4::dna_v1_conservative::per_tensor::dna_greater | ASSESSABLE | 39 | 15.036854 [14.279721, 15.737744] (ranks 13/27) | 10.939575 [10.720749, 11.176961] (ranks 13/27) | 3.9862478 [3.2400562, 4.8465316] (ranks 13/27) | 1.8189894e-12 | 1.3278623e-10 | 3.783498e-10 |
| P34D::image::trained_batch4::dna_v1_conservative::per_tensor::dna_less | ASSESSABLE | 39 | 15.036854 [14.279721, 15.737744] (ranks 13/27) | 10.939575 [10.720749, 11.176961] (ranks 13/27) | 3.9862478 [3.2400562, 4.8465316] (ranks 13/27) | 1 | 1 | 1 |
| P34D::image::trained_batch4::dna_v2_0p95::unprotected::dna_greater | ASSESSABLE | 39 | 14.864825 [14.273018, 16.037369] (ranks 13/27) | 15.133784 [14.208434, 16.153386] (ranks 13/27) | -0.24870827 [-0.51452225, 0.096558114] (ranks 13/27) | 0.97337404 | 1 | 1 |
| P34D::image::trained_batch4::dna_v2_0p95::unprotected::dna_less | ASSESSABLE | 39 | 14.864825 [14.273018, 16.037369] (ranks 13/27) | 15.133784 [14.208434, 16.153386] (ranks 13/27) | -0.24870827 [-0.51452225, 0.096558114] (ranks 13/27) | 0.054064511 | 1 | 1 |
| P34D::image::trained_batch4::dna_v2_0p95::single::dna_greater | ASSESSABLE | 39 | 14.864825 [14.273018, 16.037369] (ranks 13/27) | 12.394029 [11.866126, 12.915099] (ranks 13/27) | 2.5156076 [1.9285662, 3.0066053] (ranks 13/27) | 1.8189894e-12 | 1.3278623e-10 | 3.783498e-10 |
| P34D::image::trained_batch4::dna_v2_0p95::single::dna_less | ASSESSABLE | 39 | 14.864825 [14.273018, 16.037369] (ranks 13/27) | 12.394029 [11.866126, 12.915099] (ranks 13/27) | 2.5156076 [1.9285662, 3.0066053] (ranks 13/27) | 1 | 1 | 1 |
| P34D::image::trained_batch4::dna_v2_0p95::per_tensor::dna_greater | ASSESSABLE | 39 | 14.864825 [14.273018, 16.037369] (ranks 13/27) | 10.922588 [10.576655, 11.087141] (ranks 13/27) | 4.1437042 [3.5424658, 4.8276824] (ranks 13/27) | 1.8189894e-12 | 1.3278623e-10 | 3.783498e-10 |
| P34D::image::trained_batch4::dna_v2_0p95::per_tensor::dna_less | ASSESSABLE | 39 | 14.864825 [14.273018, 16.037369] (ranks 13/27) | 10.922588 [10.576655, 11.087141] (ranks 13/27) | 4.1437042 [3.5424658, 4.8276824] (ranks 13/27) | 1 | 1 | 1 |
| P34D::BN::342000::dna_v1_conservative::distortion::dna_greater | ASSESSABLE | 39 | 14.097695 [11.724571, 22.678863] (ranks 13/27) | 38.575244 [27.968431, 56.295886] (ranks 13/27) | -22.440426 [-32.760348, -13.086325] (ranks 13/27) | 0.99999879 | 1 | 1 |
| P34D::BN::342000::dna_v1_conservative::distortion::dna_less | ASSESSABLE | 39 | 14.097695 [11.724571, 22.678863] (ranks 13/27) | 38.575244 [27.968431, 56.295886] (ranks 13/27) | -22.440426 [-32.760348, -13.086325] (ranks 13/27) | 7.1496306e-06 | 0.00041467858 | 0.0010581453 |
| P34D::BN::342000::dna_v2_0p95::distortion::dna_greater | ASSESSABLE | 39 | 0.038544073 [0.027772224, 0.047491679] (ranks 13/27) | 54.57709 [41.777734, 68.516009] (ranks 13/27) | -54.479889 [-68.468517, -41.752476] (ranks 13/27) | 1 | 1 | 1 |
| P34D::BN::342000::dna_v2_0p95::distortion::dna_less | ASSESSABLE | 39 | 0.038544073 [0.027772224, 0.047491679] (ranks 13/27) | 54.57709 [41.777734, 68.516009] (ranks 13/27) | -54.479889 [-68.468517, -41.752476] (ranks 13/27) | 1.8189894e-12 | 1.3278623e-10 | 3.783498e-10 |
| P34D::BN::342001::dna_v1_conservative::distortion::dna_greater | ASSESSABLE | 39 | 21.340621 [16.205594, 33.563825] (ranks 13/27) | 49.468892 [32.43773, 61.079671] (ranks 13/27) | -17.36284 [-43.091259, -6.2221497] (ranks 13/27) | 0.99985296 | 1 | 1 |
| P34D::BN::342001::dna_v1_conservative::distortion::dna_less | ASSESSABLE | 39 | 21.340621 [16.205594, 33.563825] (ranks 13/27) | 49.468892 [32.43773, 61.079671] (ranks 13/27) | -17.36284 [-43.091259, -6.2221497] (ranks 13/27) | 0.00053250982 | 0.02928804 | 0.074551374 |
| P34D::BN::342001::dna_v2_0p95::distortion::dna_greater | ASSESSABLE | 39 | 0.026464823 [0.021397159, 0.042758979] (ranks 13/27) | 235.10603 [160.42753, 311.4981] (ranks 13/27) | -235.06212 [-311.46228, -160.39124] (ranks 13/27) | 1 | 1 | 1 |
| P34D::BN::342001::dna_v2_0p95::distortion::dna_less | ASSESSABLE | 39 | 0.026464823 [0.021397159, 0.042758979] (ranks 13/27) | 235.10603 [160.42753, 311.4981] (ranks 13/27) | -235.06212 [-311.46228, -160.39124] (ranks 13/27) | 1.8189894e-12 | 1.3278623e-10 | 3.783498e-10 |
| P34D::BN::342002::dna_v1_conservative::distortion::dna_greater | ASSESSABLE | 39 | 21.057633 [14.570768, 36.253473] (ranks 13/27) | 39.054495 [29.799169, 48.188184] (ranks 13/27) | -15.228401 [-33.178689, -10.67379] (ranks 13/27) | 0.99946749 | 1 | 1 |
| P34D::BN::342002::dna_v1_conservative::distortion::dna_less | ASSESSABLE | 39 | 21.057633 [14.570768, 36.253473] (ranks 13/27) | 39.054495 [29.799169, 48.188184] (ranks 13/27) | -15.228401 [-33.178689, -10.67379] (ranks 13/27) | 0.001688924 | 0.08951297 | 0.22800473 |
| P34D::BN::342002::dna_v2_0p95::distortion::dna_greater | ASSESSABLE | 39 | 0.053876507 [0.041492547, 0.065048408] (ranks 13/27) | 134.81571 [90.356451, 192.41022] (ranks 13/27) | -134.76836 [-192.10367, -90.303561] (ranks 13/27) | 1 | 1 | 1 |
| P34D::BN::342002::dna_v2_0p95::distortion::dna_less | ASSESSABLE | 39 | 0.053876507 [0.041492547, 0.065048408] (ranks 13/27) | 134.81571 [90.356451, 192.41022] (ranks 13/27) | -134.76836 [-192.10367, -90.303561] (ranks 13/27) | 1.8189894e-12 | 1.3278623e-10 | 3.783498e-10 |
| P34D::BN::pooled_checkpoints::dna_v1_conservative::distortion::dna_greater | ASSESSABLE | 117 | 19.917639 [16.353109, 22.678863] (ranks 48/70) | 39.195847 [35.356853, 49.468892] (ranks 48/70) | -19.276941 [-23.336222, -13.813332] (ranks 48/70) | 1 | 1 | 1 |
| P34D::BN::pooled_checkpoints::dna_v1_conservative::distortion::dna_less | ASSESSABLE | 117 | 19.917639 [16.353109, 22.678863] (ranks 48/70) | 39.195847 [35.356853, 49.468892] (ranks 48/70) | -19.276941 [-23.336222, -13.813332] (ranks 48/70) | 1.6828814e-10 | 1.0602153e-08 | 2.709439e-08 |
| P34D::BN::pooled_checkpoints::dna_v2_0p95::distortion::dna_greater | ASSESSABLE | 117 | 0.039768548 [0.033665361, 0.045456402] (ranks 48/70) | 116.28638 [90.356451, 152.36399] (ranks 48/70) | -116.17131 [-152.3387, -90.303561] (ranks 48/70) | 1 | 1 | 1 |
| P34D::BN::pooled_checkpoints::dna_v2_0p95::distortion::dna_less | ASSESSABLE | 117 | 0.039768548 [0.033665361, 0.045456402] (ranks 48/70) | 116.28638 [90.356451, 152.36399] (ranks 48/70) | -116.17131 [-152.3387, -90.303561] (ranks 48/70) | 6.0185311e-36 | 4.5740836e-34 | 1.2699101e-33 |

Directions are numeric DNA greater/less, tested separately. Lower PSNR means less image recovery; higher BN MSE means less BN-statistic recovery. Exact signs omit exact ties; all ties give p1. Gated tests reserve p1. Holm76 is P34D only; Holm211 combines existing 135 raw tests with76 new tests, never shrinking the family. Pooled117 comparisons require all three complete eligible strata and use their exact binomial order-statistic ranks.

## Descriptive heterogeneity (outside Holm families)

| DNA | Comparator | Status | Kruskal-Wallis statistic | Exploratory p |
| --- | --- | --- | --- | --- |
| dna_v1_conservative | unprotected | DESCRIPTIVE_ONLY | 5.1864184 | 0.074779672 |
| dna_v1_conservative | single | DESCRIPTIVE_ONLY | 4.3766033 | 0.11210699 |
| dna_v1_conservative | per_tensor | DESCRIPTIVE_ONLY | 46.191734 | 9.3237893e-11 |
| dna_v2_0p95 | unprotected | DESCRIPTIVE_ONLY | 1.5710895 | 0.4558713 |
| dna_v2_0p95 | single | DESCRIPTIVE_ONLY | 1.5089092 | 0.47026704 |
| dna_v2_0p95 | per_tensor | DESCRIPTIVE_ONLY | 52.126099 | 4.7969083e-12 |

## Independent checks and disclosures

Independent metric reload/Hungarian pairing, public BN decoding, complete source-ID pairs, checkpoint validation inference, receipt/source/input hashes, calibration reload, SciPy exact binomial tests and separate vectorized NumPy Holm verification are required. Probe arrays and quantiles are hash-verified; this utility auditor does not regenerate probe gradients. All code runs CPU/thread1. Numerical failures and negative BN variance fail closed; no clamp, tune, replacement or outcome-based design change is allowed. This is conditional known-label client-gradient recovery with public model and known v2 key, not final-model inversion or cryptographic key secrecy. v1 candidate selection uses observable objective only; v2 uses sketch-space matching/proper least squares. Private truth/captures/noise seeds are audit-only bundles outside the DP release and never provided to the attacker. P34B central-DP results are excluded from individual-gradient comparisons. Earlier artifacts, Latex and external defenses are preserved.

## SHA-256 evidence

| File / evidence | SHA-256 |
| --- | --- |
| CHECKPOINTS_COMPLETE.json | 4143b085c1867339e0d1b498fbfb7103f692f67a4f26fe04300dfd5856e962fd |
| audits/final_analysis_freeze.json | 09ede95b40902e086218b912f46ce30641d55c21099197180e45ab543ac1a4ac |
| audits/final_statistics.json | 773156d31e3b9f0a176ec18327d82328267691f476b6ea59fa19f624895217bf |
| audits/independent_bn_results.json | 0cb0e65c007b2c9fe148429899e3ab1f427dcfffed4f29e76d594e275d459bfd |
| audits/independent_image_results.json | 7029fe7e9e469b37b6bcedf55d55d8796db451f0959eb61f21cd1f503f514d1d |
| audits/independent_utility_results.json | b1043921dbabc45c282f75a2228fa46d0e1f41c47e8a4f666577d3331c00bf45 |
| audits/json_key_preflight_repair.json | 2f71a90de5f88f09d5c514220b6590bc1c09d5f4632b081c68a47951e46dcf51 |
| bn/baf/BN_COMPLETE.json | f1fdd9f2d239e3864da19da48a69352c6708593de45bba39b987ec3b01e66721 |
| bn/baf/execution_freeze.json | 08181d79bacedce07e7120e4171879b5cf02ade3cca8d51af9aa9e53df1743b1 |
| checkpoint_execution_freeze.json | bb7283d73e391c0bbcca55b59f39ff4ad46ecb155ac7554b4302dcdd9771d364 |
| image_recovery/IMAGE_COMPLETE.json | c3e7967afec04c12c6373cabc32802ab31464674bbc05e43c284c2c284307be8 |
| image_recovery/execution_freeze.json | ab82ab050520d0cf52009e4fc362434f5865536c98be04b51e6b7e6a855cd350 |
| image_utility/UTILITY_COMPLETE.json | 866877846086060cc612e7d3159e41b1f2a1540e05790228e13610ffc3f034fb |
| image_utility/execution_freeze.json | b371a091383ff9335a4c16f162213f2759622789df41050f80cc43ab15f6575f |
| image_utility/preparation_freeze.json | fc552a2d4fee6312af73902a08e616eaedb9b74ef66e16d767707cfc2e5caa38 |


## Descriptive paired PSNR/SSIM/MSE effects

| Setting | DNA | Comparator | PSNR DNA−comparator | SSIM DNA−comparator | MSE DNA−comparator |
| --- | --- | --- | --- | --- | --- |
| init342042 | dna_v1_conservative | unprotected | -0.17776773 [-0.27543781, -0.053159967] (ranks 13/27) | -0.0039227009 [-0.0061340928, -0.0030404925] (ranks 13/27) | 9.970658e-05 [4.5006163e-05, 0.0002191253] (ranks 13/27) |
| init342042 | dna_v1_conservative | dp_single_for_dna_v1_conservative | 0.37803145 [0.16925221, 0.51774011] (ranks 13/27) | 0.011905849 [0.0077062845, 0.016655743] (ranks 13/27) | -0.00024200196 [-0.00036828069, -0.00013969536] (ranks 13/27) |
| init342042 | dna_v1_conservative | dp_per_tensor_for_dna_v1_conservative | 4.1051529 [2.9938444, 5.0693501] (ranks 13/27) | 0.1654253 [0.12872678, 0.2031877] (ranks 13/27) | -0.0058313922 [-0.0078495904, -0.0040008998] (ranks 13/27) |
| init342042 | dna_v2_0p95 | unprotected | -0.01505768 [-0.079717512, 0.024021379] (ranks 13/27) | -0.00092482567 [-0.0014715195, 5.7816505e-06] (ranks 13/27) | 1.0355143e-05 [-8.4941275e-06, 4.6915375e-05] (ranks 13/27) |
| init342042 | dna_v2_0p95 | dp_single_for_dna_v2_0p95 | -0.032700699 [-0.086845405, 0.029412771] (ranks 13/27) | 5.7220459e-06 [-0.0012325048, 0.0013830066] (ranks 13/27) | 2.3168512e-05 [-1.9402389e-05, 7.4916752e-05] (ranks 13/27) |
| init342042 | dna_v2_0p95 | dp_per_tensor_for_dna_v2_0p95 | 4.385316 [3.6255399, 5.1618477] (ranks 13/27) | 0.17699778 [0.13266015, 0.21238691] (ranks 13/27) | -0.0062794287 [-0.0080373739, -0.004009713] (ranks 13/27) |
| init342043 | dna_v1_conservative | unprotected | 0.14034186 [-0.17016212, 0.71625705] (ranks 13/27) | 0.017380416 [-0.013576448, 0.053628623] (ranks 13/27) | -0.00046369433 [-0.0030906713, 0.00084483624] (ranks 13/27) |
| init342043 | dna_v1_conservative | dp_single_for_dna_v1_conservative | 0.24551745 [-0.2394534, 0.60476413] (ranks 13/27) | 0.013964236 [-0.019860387, 0.035791457] (ranks 13/27) | -0.0004299616 [-0.0029049364, 0.00087371096] (ranks 13/27) |
| init342043 | dna_v1_conservative | dp_per_tensor_for_dna_v1_conservative | 2.3247447 [1.733439, 3.1382057] (ranks 13/27) | 0.20109171 [0.15028179, 0.22282378] (ranks 13/27) | -0.011358378 [-0.014007412, -0.0069559896] (ranks 13/27) |
| init342043 | dna_v2_0p95 | unprotected | -0.017156068 [-0.49028468, 0.31533316] (ranks 13/27) | 0.0010777712 [-0.039638698, 0.026710808] (ranks 13/27) | 9.4769523e-05 [-0.00088157225, 0.0013711415] (ranks 13/27) |
| init342043 | dna_v2_0p95 | dp_single_for_dna_v2_0p95 | 0.045512007 [-0.47699849, 0.40729801] (ranks 13/27) | 0.0034352541 [-0.022789776, 0.022731006] (ranks 13/27) | -0.00071344897 [-0.0014721593, 0.00095653627] (ranks 13/27) |
| init342043 | dna_v2_0p95 | dp_per_tensor_for_dna_v2_0p95 | 2.2810571 [1.4254467, 3.0229438] (ranks 13/27) | 0.1748696 [0.095084071, 0.21771494] (ranks 13/27) | -0.010995415 [-0.016094744, -0.0063899732] (ranks 13/27) |
| init342044 | dna_v1_conservative | unprotected | -0.064832552 [-0.17282382, 0.38096247] (ranks 13/27) | -0.0035486817 [-0.012857199, 0.0059714913] (ranks 13/27) | 8.9030713e-05 [-0.00062493398, 0.00067201257] (ranks 13/27) |
| init342044 | dna_v1_conservative | dp_single_for_dna_v1_conservative | 0.64901964 [0.25764229, 0.95720622] (ranks 13/27) | 0.036092103 [0.016386747, 0.049716771] (ranks 13/27) | -0.001174761 [-0.003251872, -0.00042487122] (ranks 13/27) |
| init342044 | dna_v1_conservative | dp_per_tensor_for_dna_v1_conservative | 1.4758595 [1.082243, 1.6599048] (ranks 13/27) | 0.091978252 [0.064580142, 0.11516237] (ranks 13/27) | -0.0042732814 [-0.0055369763, -0.0025363774] (ranks 13/27) |
| init342044 | dna_v2_0p95 | unprotected | -0.070470086 [-0.53577018, 0.03563642] (ranks 13/27) | -0.0046599507 [-0.014821887, 0.00047433376] (ranks 13/27) | 0.00015142187 [-0.00011072401, 0.00083883945] (ranks 13/27) |
| init342044 | dna_v2_0p95 | dp_single_for_dna_v2_0p95 | 0.21396239 [-0.24393581, 0.57999664] (ranks 13/27) | 0.0097724795 [-0.011273086, 0.031350374] (ranks 13/27) | -0.00042838044 [-0.0013660453, 0.00074351206] (ranks 13/27) |
| init342044 | dna_v2_0p95 | dp_per_tensor_for_dna_v2_0p95 | 1.2711451 [0.73489349, 1.9190928] (ranks 13/27) | 0.089523017 [0.060295284, 0.122904] (ranks 13/27) | -0.0032274332 [-0.0049846712, -0.0018762706] (ranks 13/27) |
| pooled_initializations | dna_v1_conservative | unprotected | -0.064832552 [-0.16545625, -0.00068220751] (ranks 48/70) | -0.003253758 [-0.0046173334, -1.4185905e-05] (ranks 48/70) | 6.3079526e-05 [1.876615e-06, 0.00015176809] (ranks 48/70) |
| pooled_initializations | dna_v1_conservative | dp_single_for_dna_v1_conservative | 0.39745678 [0.25764229, 0.53475954] (ranks 48/70) | 0.016386747 [0.011800826, 0.021161199] (ranks 48/70) | -0.00041030231 [-0.00062535517, -0.00026085257] (ranks 48/70) |
| pooled_initializations | dna_v1_conservative | dp_per_tensor_for_dna_v1_conservative | 2.3247447 [1.9992299, 2.9566733] (ranks 48/70) | 0.14081669 [0.1189065, 0.1654253] (ranks 48/70) | -0.0062103737 [-0.0074878447, -0.0053004203] (ranks 48/70) |
| pooled_initializations | dna_v2_0p95 | unprotected | -0.039517149 [-0.088016118, 0.0011004309] (ranks 48/70) | -0.0014428496 [-0.0020852089, -0.00031685829] (ranks 48/70) | 1.7118931e-05 [-1.0500662e-06, 0.00011451542] (ranks 48/70) |
| pooled_initializations | dna_v2_0p95 | dp_single_for_dna_v2_0p95 | -0.0006807913 [-0.064632427, 0.14330737] (ranks 48/70) | 0.0011469722 [-0.00060653687, 0.0035418272] (ranks 48/70) | 1.417473e-06 [-0.00010034349, 7.4916752e-05] (ranks 48/70) |
| pooled_initializations | dna_v2_0p95 | dp_per_tensor_for_dna_v2_0p95 | 2.2955621 [1.9190928, 2.7273835] (ranks 48/70) | 0.13321108 [0.11575341, 0.16826814] (ranks 48/70) | -0.0060285309 [-0.0072215134, -0.004375306] (ranks 48/70) |
| trained_batch4 | dna_v1_conservative | unprotected | -0.085259635 [-0.40892776, 0.21583555] (ranks 13/27) | -0.0069828257 [-0.030682556, 0.012985766] (ranks 13/27) | 0.0008775969 [-0.001503902, 0.0023519578] (ranks 13/27) |
| trained_batch4 | dna_v1_conservative | dp_single_for_dna_v1_conservative | 2.6334169 [1.9676501, 3.1004301] (ranks 13/27) | 0.19072975 [0.12237211, 0.21135575] (ranks 13/27) | -0.024393338 [-0.02977754, -0.02187895] (ranks 13/27) |
| trained_batch4 | dna_v1_conservative | dp_per_tensor_for_dna_v1_conservative | 3.9862478 [3.2400562, 4.8465316] (ranks 13/27) | 0.25008012 [0.20904529, 0.31858647] (ranks 13/27) | -0.047117637 [-0.049234566, -0.041561706] (ranks 13/27) |
| trained_batch4 | dna_v2_0p95 | unprotected | -0.24870827 [-0.51452225, 0.096558114] (ranks 13/27) | -0.010169849 [-0.036517985, 0.0053604599] (ranks 13/27) | 0.0015041574 [-0.00099832891, 0.0033540851] (ranks 13/27) |
| trained_batch4 | dna_v2_0p95 | dp_single_for_dna_v2_0p95 | 2.5156076 [1.9285662, 3.0066053] (ranks 13/27) | 0.17911329 [0.13068116, 0.21247243] (ranks 13/27) | -0.024781799 [-0.026652469, -0.021140721] (ranks 13/27) |
| trained_batch4 | dna_v2_0p95 | dp_per_tensor_for_dna_v2_0p95 | 4.1437042 [3.5424658, 4.8276824] (ranks 13/27) | 0.26888929 [0.20450762, 0.30935447] (ranks 13/27) | -0.04682384 [-0.053845977, -0.042593113] (ranks 13/27) |

## Infrastructure-only preflight amendment

The initial recovery preparation failed before target creation or attack launch
because calibration grid dictionary keys were numeric in memory and strings
after JSON serialization. The failed PREFLIGHT_FAILURE.json and existing logs
were retained unchanged. After explicit user approval, an additive preparation
adapter normalized JSON keys on both sides and used exact equality, with no
tolerance change. Six unit tests covered integer/string and float/string key
equivalence, genuine value differences, no tolerance, value preservation and
ambiguous-key rejection; five reservation audit tests also passed.

The rerun confirmed16/16 calibration cells matched. No calibration value, sigma,
target list, seed or original frozen scientific source was changed. The273
prespecified reservation IDs matched the independent historical audit. No
training or recovery was replayed. The original frozen recovery launcher and
worker were then used for1092 scheduled jobs. The amendment and adapter were
sealed into the new recovery execution freeze, together with the retained
failure and repair receipt. This infrastructure repair is not a scientific
result or a completion claim.

Recovery freeze SHA256:
ab82ab050520d0cf52009e4fc362434f5865536c98be04b51e6b7e6a855cd350.
Repair receipt SHA256:
2f71a90de5f88f09d5c514220b6590bc1c09d5f4632b081c68a47951e46dcf51.
Include this disclosure in the final P34D report; final independent audits are
still required before verified COMPLETE.


## Confirmatory decision summary

21 of76 P34D directional tests have Holm211 p<0.05. Non-significance is not equivalence. All76 tests remain reserved. Significant directions are listed below.

| Hypothesis | Holm211 p | Paired effect median |
| --- | --- | --- |
| P34D::image::init342042::dna_v1_conservative::unprotected::dna_less | 0.020879458 | -0.17776773 |
| P34D::image::init342042::dna_v1_conservative::per_tensor::dna_greater | 3.783498e-10 | 4.1051529 |
| P34D::image::init342042::dna_v2_0p95::per_tensor::dna_greater | 3.783498e-10 | 4.385316 |
| P34D::image::init342043::dna_v1_conservative::per_tensor::dna_greater | 1.200533e-08 | 2.3247447 |
| P34D::image::init342043::dna_v2_0p95::per_tensor::dna_greater | 2.6489975e-05 | 2.2810571 |
| P34D::image::init342044::dna_v1_conservative::single::dna_greater | 0.0010581453 | 0.64901964 |
| P34D::image::init342044::dna_v1_conservative::per_tensor::dna_greater | 0.00018345805 | 1.4758595 |
| P34D::image::init342044::dna_v2_0p95::per_tensor::dna_greater | 0.00018345805 | 1.2711451 |
| P34D::image::pooled_initializations::dna_v1_conservative::single::dna_greater | 2.7978956e-05 | 0.39745678 |
| P34D::image::pooled_initializations::dna_v1_conservative::per_tensor::dna_greater | 4.1744131e-24 | 2.3247447 |
| P34D::image::pooled_initializations::dna_v2_0p95::per_tensor::dna_greater | 1.1312962e-20 | 2.2955621 |
| P34D::image::trained_batch4::dna_v1_conservative::single::dna_greater | 3.783498e-10 | 2.6334169 |
| P34D::image::trained_batch4::dna_v1_conservative::per_tensor::dna_greater | 3.783498e-10 | 3.9862478 |
| P34D::image::trained_batch4::dna_v2_0p95::single::dna_greater | 3.783498e-10 | 2.5156076 |
| P34D::image::trained_batch4::dna_v2_0p95::per_tensor::dna_greater | 3.783498e-10 | 4.1437042 |
| P34D::BN::342000::dna_v1_conservative::distortion::dna_less | 0.0010581453 | -22.440426 |
| P34D::BN::342000::dna_v2_0p95::distortion::dna_less | 3.783498e-10 | -54.479889 |
| P34D::BN::342001::dna_v2_0p95::distortion::dna_less | 3.783498e-10 | -235.06212 |
| P34D::BN::342002::dna_v2_0p95::distortion::dna_less | 3.783498e-10 | -134.76836 |
| P34D::BN::pooled_checkpoints::dna_v1_conservative::distortion::dna_less | 2.709439e-08 | -19.276941 |
| P34D::BN::pooled_checkpoints::dna_v2_0p95::distortion::dna_less | 1.2699101e-33 | -116.17131 |

## Interpretation and final verification

Higher image PSNR means better attacker recovery (less protection), whereas higher BN batch-mean MSE means worse attacker recovery (more protection). Thus the significant DNA-greater image results versus matched DP, and DNA-less BN results versus distortion-matched DP, are evidence against claiming superior DNA protection under these instruments. Image DNA generally remains close to the unprotected arm; non-significance is not equivalence. In the trained batch4 setting, DNA has significantly higher PSNR than both matched DP mechanisms for both versions. BN v2 preserves substantially more batch-mean information than its DP comparator. These are scoped empirical findings, not record-level DP guarantees.

Initialization heterogeneity is descriptive and outside both Holm families. Across the three initializations, median paired PSNR effects (DNA minus comparator, dB) range as follows: v1 versus unprotected −0.177768 to 0.140342; versus single DP 0.245517 to 0.649020; versus per-tensor DP 1.475860 to 4.105153. Corresponding v2 ranges are −0.070470 to −0.015058, −0.032701 to 0.213962, and 1.271145 to 4.385316. The full per-initialization medians, intervals and descriptive heterogeneity statistics are retained above; pooled results do not replace them.

Final independent checks passed: utility 1216/1216 checkpoints, BN 285/285 scheduled jobs (three qualified checkpoints and 117 confirmatory paired units), and image recovery 1092/1092 jobs. All 16 utility calibration cells matched. Independent endpoint reloads, Hungarian assignments, payload/control/distortion receipts, source reservations, finite-value checks, Gaussian accounting, exact SciPy sign tests and separate NumPy Holm recomputation passed. All 76 P34D and 211 combined reservations were retained; no missing-pair analysis or replacement seeds were used. CSV exports contain 76 and 211 tests, 35 image metric summaries and 30 descriptive effect summaries, with exact typed-table/CSV roundtrip checks.

The final synthetic suites passed 92 unique tests (24 tabular and 68 image/pure/administrative, in separate processes to avoid the preserved import-name collision). All 33 P34D Python source/test files compiled and git diff --check passed. No scientific workload remains live. Original frozen sources, calibration values, tolerances, datasets and earlier reports were preserved. The infrastructure disclosure above reproduces the earlier repair receipt; its “still required” wording describes that historical stage, now superseded by these completed independent checks.

Final analysis freeze SHA256: 09ede95b40902e086218b912f46ce30641d55c21099197180e45ab543ac1a4ac. The full immutable file manifest is artifacts/priority34d/audits/full_immutable_manifest.json; the final verification seal is artifacts/priority34d/COMPLETE.json. Those separate seals bind this report's final hash and the preserved PROJECT snapshot. P34B's paused Goal was not changed.
