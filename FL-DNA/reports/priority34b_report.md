# Priority34B — utility at meaningful whole-training DP levels

Status: COMPLETE — full post-exit audit PASS.

Descriptive study, no new noninferiority/equivalence claim. 470 new jobs; 33 immutable P34A baselines reused.

## Formal mechanism and release boundary

q=1, fixed three slots, replace-one whole-client update-generating data/state; 50 adaptive releases, delta1e-5. Central trusted aggregator adds isotropic noise once after clipping/weighted averaging. Global C=.01; per-tensor C/sqrt(L), concatenated radius C. Sensitivity2 max(w) C; sigma multiplies sensitivity, NOT C. BN affine/buffers stay private/local. Global model transcript only is the accounted release. LocalBN/evaluation/audit files and disclosed deterministic noise seeds are not protected releases. Ideal Gaussian/RDP certificate; finite precision/PRNG simulation is not a certified private deployment.

| Target ε | Certified ε | σ (noise/sensitivity) | RDP order | Rounds | δ |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 34.65215791 | 24.515441 | 50 | 1e-5 |
| 3 | 3 | 12.00442264 | 9.1463698 | 50 | 1e-5 |
| 10 | 10 | 4.01563652 | 3.7250673 | 50 | 1e-5 |

## Utility versus epsilon — paired n11

| Dataset | Mechanism | ε | Endpoint | Baseline | DP mean | Mean Δ [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| paysim | global | 1 | f1 | 0.78322535 | 0.55188619 | -0.23133916 [-0.27193381, -0.1907445] |
| paysim | global | 1 | auc_roc | 0.99759361 | 0.95293394 | -0.044659668 [-0.065862743, -0.023456592] |
| paysim | global | 1 | pr_auc | 0.80333529 | 0.499903 | -0.30343229 [-0.35043849, -0.25642608] |
| paysim | global | 3 | f1 | 0.78322535 | 0.49414399 | -0.28908135 [-0.38932934, -0.18883337] |
| paysim | global | 3 | auc_roc | 0.99759361 | 0.94386736 | -0.053726254 [-0.094486276, -0.012966232] |
| paysim | global | 3 | pr_auc | 0.80333529 | 0.44937225 | -0.35396304 [-0.45355435, -0.25437172] |
| paysim | global | 10 | f1 | 0.78322535 | 0.50356443 | -0.27966092 [-0.3610146, -0.19830723] |
| paysim | global | 10 | auc_roc | 0.99759361 | 0.96998151 | -0.027612103 [-0.032586046, -0.02263816] |
| paysim | global | 10 | pr_auc | 0.80333529 | 0.46087437 | -0.34246092 [-0.4236795, -0.26124234] |
| paysim | per_tensor | 1 | f1 | 0.78322535 | 0.5211386 | -0.26208675 [-0.35235318, -0.17182032] |
| paysim | per_tensor | 1 | auc_roc | 0.99759361 | 0.93433422 | -0.063259393 [-0.13828883, 0.01177004] |
| paysim | per_tensor | 1 | pr_auc | 0.80333529 | 0.47220817 | -0.33112712 [-0.42960708, -0.23264715] |
| paysim | per_tensor | 3 | f1 | 0.78322535 | 0.53638482 | -0.24684053 [-0.32669031, -0.16699075] |
| paysim | per_tensor | 3 | auc_roc | 0.99759361 | 0.96940798 | -0.028185629 [-0.035002022, -0.021369236] |
| paysim | per_tensor | 3 | pr_auc | 0.80333529 | 0.49565156 | -0.30768373 [-0.39370966, -0.2216578] |
| paysim | per_tensor | 10 | f1 | 0.78322535 | 0.42763828 | -0.35558707 [-0.48568947, -0.22548467] |
| paysim | per_tensor | 10 | auc_roc | 0.99759361 | 0.94817204 | -0.049421572 [-0.08530461, -0.013538533] |
| paysim | per_tensor | 10 | pr_auc | 0.80333529 | 0.38744231 | -0.41589298 [-0.54775402, -0.28403194] |
| ieee_cis | global | 1 | f1 | 0.34943956 | 0.16173672 | -0.18770284 [-0.19517028, -0.18023539] |
| ieee_cis | global | 1 | auc_roc | 0.83812534 | 0.69381332 | -0.14431202 [-0.19024063, -0.098383401] |
| ieee_cis | global | 1 | pr_auc | 0.32748308 | 0.078386142 | -0.24909694 [-0.25656488, -0.241629] |
| ieee_cis | global | 3 | f1 | 0.34943956 | 0.1630319 | -0.18640767 [-0.19642318, -0.17639216] |
| ieee_cis | global | 3 | auc_roc | 0.83812534 | 0.73567546 | -0.10244988 [-0.12370009, -0.081199663] |
| ieee_cis | global | 3 | pr_auc | 0.32748308 | 0.081271128 | -0.24621195 [-0.25475123, -0.23767268] |
| ieee_cis | global | 10 | f1 | 0.34943956 | 0.16435197 | -0.18508759 [-0.1914944, -0.17868078] |
| ieee_cis | global | 10 | auc_roc | 0.83812534 | 0.74786944 | -0.090255897 [-0.098671864, -0.081839929] |
| ieee_cis | global | 10 | pr_auc | 0.32748308 | 0.084000151 | -0.24348293 [-0.25067102, -0.23629484] |
| ieee_cis | per_tensor | 1 | f1 | 0.34943956 | 0.16076122 | -0.18867834 [-0.196757, -0.18059969] |
| ieee_cis | per_tensor | 1 | auc_roc | 0.83812534 | 0.70212752 | -0.13599782 [-0.17036376, -0.10163188] |
| ieee_cis | per_tensor | 1 | pr_auc | 0.32748308 | 0.076133062 | -0.25135002 [-0.26059925, -0.24210079] |
| ieee_cis | per_tensor | 3 | f1 | 0.34943956 | 0.15907424 | -0.19036533 [-0.1999861, -0.18074456] |
| ieee_cis | per_tensor | 3 | auc_roc | 0.83812534 | 0.71733084 | -0.1207945 [-0.15136784, -0.090221152] |
| ieee_cis | per_tensor | 3 | pr_auc | 0.32748308 | 0.078366312 | -0.24911677 [-0.25696282, -0.24127072] |
| ieee_cis | per_tensor | 10 | f1 | 0.34943956 | 0.16347188 | -0.18596768 [-0.19444634, -0.17748902] |
| ieee_cis | per_tensor | 10 | auc_roc | 0.83812534 | 0.74559508 | -0.092530262 [-0.099120663, -0.085939861] |
| ieee_cis | per_tensor | 10 | pr_auc | 0.32748308 | 0.081649636 | -0.24583345 [-0.25474674, -0.23692015] |
| baf | global | 1 | f1 | 0.2257117 | 0.056349928 | -0.16936177 [-0.18116841, -0.15755512] |
| baf | global | 1 | auc_roc | 0.8713404 | 0.64766805 | -0.22367235 [-0.2470094, -0.20033529] |
| baf | global | 1 | pr_auc | 0.15868126 | 0.025350498 | -0.13333076 [-0.13911792, -0.12754361] |
| baf | global | 3 | f1 | 0.2257117 | 0.067396044 | -0.15831565 [-0.1724089, -0.14422241] |
| baf | global | 3 | auc_roc | 0.8713404 | 0.67843448 | -0.19290592 [-0.21448329, -0.17132855] |
| baf | global | 3 | pr_auc | 0.15868126 | 0.030573012 | -0.12810825 [-0.13406949, -0.12214701] |
| baf | global | 10 | f1 | 0.2257117 | 0.070217552 | -0.15549414 [-0.16449734, -0.14649094] |
| baf | global | 10 | auc_roc | 0.8713404 | 0.68856768 | -0.18277272 [-0.19559957, -0.16994587] |
| baf | global | 10 | pr_auc | 0.15868126 | 0.029405441 | -0.12927582 [-0.13312293, -0.12542872] |
| baf | per_tensor | 1 | f1 | 0.2257117 | 0.05822049 | -0.16749121 [-0.17966987, -0.15531254] |
| baf | per_tensor | 1 | auc_roc | 0.8713404 | 0.64626099 | -0.22507941 [-0.25392086, -0.19623796] |
| baf | per_tensor | 1 | pr_auc | 0.15868126 | 0.026182109 | -0.13249915 [-0.13643904, -0.12855927] |
| baf | per_tensor | 3 | f1 | 0.2257117 | 0.069005339 | -0.15670636 [-0.16880172, -0.144611] |
| baf | per_tensor | 3 | auc_roc | 0.8713404 | 0.67254286 | -0.19879754 [-0.22357218, -0.1740229] |
| baf | per_tensor | 3 | pr_auc | 0.15868126 | 0.030461159 | -0.1282201 [-0.13386375, -0.12257645] |
| baf | per_tensor | 10 | f1 | 0.2257117 | 0.062733411 | -0.16297829 [-0.17344075, -0.15251582] |
| baf | per_tensor | 10 | auc_roc | 0.8713404 | 0.67424987 | -0.19709053 [-0.21887269, -0.17530838] |
| baf | per_tensor | 10 | pr_auc | 0.15868126 | 0.028155334 | -0.13052593 [-0.13620015, -0.1248517] |
| cifar10 | global | 1 | validation_accuracy | 0.36409091 | 0.10269697 | -0.26139394 [-0.26668928, -0.2560986] |
| cifar10 | global | 3 | validation_accuracy | 0.36409091 | 0.10324242 | -0.26084848 [-0.26989306, -0.25180391] |
| cifar10 | global | 10 | validation_accuracy | 0.36409091 | 0.10175758 | -0.26233333 [-0.26964066, -0.255026] |
| cifar10 | per_tensor | 1 | validation_accuracy | 0.36409091 | 0.096363636 | -0.26772727 [-0.27398133, -0.26147322] |
| cifar10 | per_tensor | 3 | validation_accuracy | 0.36409091 | 0.093090909 | -0.271 [-0.27787306, -0.26412694] |
| cifar10 | per_tensor | 10 | validation_accuracy | 0.36409091 | 0.10157576 | -0.26251515 [-0.26693568, -0.25809462] |

## P34A local-BN DNA on the same eleven seeds

These n11 comparisons are not P34A's full21-seed NI claims, which remain unchanged in priority34a_report.md.

| Dataset | DNA | Endpoint | Mean | Mean Δ [95% CI] |
| --- | --- | --- | --- | --- |
| paysim | dna_v1_conservative | f1 | 0.78160692 | -0.0016184296 [-0.007508671, 0.0042718119] |
| paysim | dna_v1_conservative | auc_roc | 0.99763718 | +4.356689e-05 [-5.9681736e-05, 0.00014681551] |
| paysim | dna_v1_conservative | pr_auc | 0.80584327 | +0.0025079811 [-0.001803508, 0.0068194702] |
| paysim | dna_v2_0p95 | f1 | 0.78247155 | -0.00075380306 [-0.0076738774, 0.0061662713] |
| paysim | dna_v2_0p95 | auc_roc | 0.9977114 | +0.00011778525 [-8.385328e-05, 0.00031942379] |
| paysim | dna_v2_0p95 | pr_auc | 0.80938477 | +0.0060494815 [-0.00037848824, 0.012477451] |
| ieee_cis | dna_v1_conservative | f1 | 0.34587489 | -0.0035646731 [-0.012566504, 0.0054371577] |
| ieee_cis | dna_v1_conservative | auc_roc | 0.83610992 | -0.002015417 [-0.004761589, 0.00073075496] |
| ieee_cis | dna_v1_conservative | pr_auc | 0.32195304 | -0.0055300432 [-0.014176348, 0.0031162612] |
| ieee_cis | dna_v2_0p95 | f1 | 0.33653802 | -0.012901539 [-0.020610728, -0.0051923492] |
| ieee_cis | dna_v2_0p95 | auc_roc | 0.83632193 | -0.0018034125 [-0.0031561314, -0.00045069364] |
| ieee_cis | dna_v2_0p95 | pr_auc | 0.32021131 | -0.0072717708 [-0.013742666, -0.00080087548] |
| baf | dna_v1_conservative | f1 | 0.22578096 | +6.9265744e-05 [-0.0028706856, 0.0030092171] |
| baf | dna_v1_conservative | auc_roc | 0.87146053 | +0.00012013199 [-0.00082801655, 0.0010682805] |
| baf | dna_v1_conservative | pr_auc | 0.15844004 | -0.00024122472 [-0.00093284012, 0.00045039068] |
| baf | dna_v2_0p95 | f1 | 0.22620504 | +0.00049334588 [-0.0023547112, 0.003341403] |
| baf | dna_v2_0p95 | auc_roc | 0.87060258 | -0.00073782143 [-0.0020930713, 0.00061742843] |
| baf | dna_v2_0p95 | pr_auc | 0.15826862 | -0.000412637 [-0.0013404823, 0.00051520827] |

## Frozen recovery — conditional single-round instrument

Not final-model or trained-client inversion. Same noise/sensitivity ratio as ε10 whole50-round accounting; known label and known-zero other contributions. C=.01 is in gradient coordinates here, versus update coordinates for utility: no utility-matched claim.

Unprotected qualification: True. All reference gates (including DP instrument limitations): `{"dna_v1_conservative/cifar_mean": {"effective_n": 39, "losses": 0, "median_effect": 10.597429246302369, "order_interval": [9.738584157203231, 11.7707515638157], "raw_p": 1.8189894035458565e-12, "ties": 0, "wins": 39}, "dna_v1_conservative/gray": {"effective_n": 39, "losses": 0, "median_effect": 11.225317355180753, "order_interval": [9.952944055077266, 12.238615223573849], "raw_p": 1.8189894035458565e-12, "ties": 0, "wins": 39}, "dna_v2_0p95/cifar_mean": {"effective_n": 39, "losses": 0, "median_effect": 10.690720043421724, "order_interval": [9.803691199862776, 11.850889070483229], "raw_p": 1.8189894035458565e-12, "ties": 0, "wins": 39}, "dna_v2_0p95/gray": {"effective_n": 39, "losses": 0, "median_effect": 11.204899985831915, "order_interval": [9.955711662364052, 12.27328306506024], "raw_p": 1.8189894035458565e-12, "ties": 0, "wins": 39}, "dp_global_eps10/cifar_mean": {"effective_n": 39, "losses": 39, "median_effect": -5.963475666974995, "order_interval": [-7.08660125973811, -5.239293235684519], "raw_p": 1.0, "ties": 0, "wins": 0}, "dp_global_eps10/gray": {"effective_n": 39, "losses": 39, "median_effect": -5.768761277146795, "order_interval": [-6.63008051047804, -5.303217365664109], "raw_p": 1.0, "ties": 0, "wins": 0}, "dp_per_tensor_eps10/cifar_mean": {"effective_n": 39, "losses": 39, "median_effect": -6.081210524611677, "order_interval": [-7.035143057913155, -5.368310876586475], "raw_p": 1.0, "ties": 0, "wins": 0}, "dp_per_tensor_eps10/gray": {"effective_n": 39, "losses": 39, "median_effect": -5.894167785867634, "order_interval": [-6.705327577329112, -5.284391548808843], "raw_p": 1.0, "ties": 0, "wins": 0}, "unprotected/cifar_mean": {"effective_n": 39, "losses": 0, "median_effect": 10.85441697980264, "order_interval": [9.736020976271067, 11.994312420103093], "raw_p": 1.8189894035458565e-12, "ties": 0, "wins": 39}, "unprotected/gray": {"effective_n": 39, "losses": 0, "median_effect": 11.321631504320457, "order_interval": [9.888041438772344, 12.437855401197048], "raw_p": 1.8189894035458565e-12, "ties": 0, "wins": 39}}`.

| Arm | Metric | Median [ranks13/27] |
| --- | --- | --- |
| unprotected | psnr_db | 23.978547 [22.968393698015035, 26.327320032879953] |
| unprotected | ssim | 0.86907959 [0.8405165076255798, 0.8828274607658386] |
| unprotected | mse | 0.0040007862 [0.002329528331756592, 0.0050484798848629] |
| dna_v1_conservative | psnr_db | 23.814139 [22.744832841727643, 26.24625882863047] |
| dna_v1_conservative | ssim | 0.861269 [0.8394004702568054, 0.8784525990486145] |
| dna_v1_conservative | mse | 0.0041551446 [0.0023734173737466335, 0.005315164569765329] |
| dna_v2_0p95 | psnr_db | 24.066783 [23.0711105188564, 26.209577059216258] |
| dna_v2_0p95 | ssim | 0.86532974 [0.8422365188598633, 0.8821175694465637] |
| dna_v2_0p95 | mse | 0.0039203214 [0.0023935488425195217, 0.004930477123707533] |
| dp_global_eps10 | psnr_db | 7.2853725 [7.061061784927575, 7.4706424505520985] |
| dp_global_eps10 | ssim | 0.0032960076 [0.0001807257067412138, 0.007009059190750122] |
| dp_global_eps10 | mse | 0.18683694 [0.17903409898281097, 0.19674052298069] |
| dp_per_tensor_eps10 | psnr_db | 7.244022 [7.0754155492588975, 7.458344697441555] |
| dp_per_tensor_eps10 | ssim | 0.0096150599 [0.0033124834299087524, 0.014147507958114147] |
| dp_per_tensor_eps10 | mse | 0.18862437 [0.17954178154468536, 0.19609135389328003] |

Positive effect = DNA metric minus DP metric; lower PSNR/SSIM or higher MSE indicates poorer recovery. Both signed directions reported, fixed Holm24 family.

| DNA | DP | Metric | Direction | W/L/T | Median Δ [ranks13/27] | Raw p | Holm p | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dna_v1_conservative | dp_global_eps10 | psnr_db | 1 | 39/0/0 | 16.749352 [16.105901087897543, 18.617365143888705] | 1.818989404e-12 | 4.365574569e-11 | ASSESSABLE |
| dna_v1_conservative | dp_global_eps10 | psnr_db | -1 | 0/39/0 | 16.749352 [16.105901087897543, 18.617365143888705] | 1 | 1 | ASSESSABLE |
| dna_v1_conservative | dp_global_eps10 | ssim | 1 | 39/0/0 | 0.86183505 [0.836812686175108, 0.8693758547306061] | 1.818989404e-12 | 4.365574569e-11 | ASSESSABLE |
| dna_v1_conservative | dp_global_eps10 | ssim | -1 | 0/39/0 | 0.86183505 [0.836812686175108, 0.8693758547306061] | 1 | 1 | ASSESSABLE |
| dna_v1_conservative | dp_global_eps10 | mse | 1 | 0/39/0 | -0.1826148 [-0.19018240342848003, -0.17349684052169323] | 1 | 1 | ASSESSABLE |
| dna_v1_conservative | dp_global_eps10 | mse | -1 | 39/0/0 | -0.1826148 [-0.19018240342848003, -0.17349684052169323] | 1.818989404e-12 | 4.365574569e-11 | ASSESSABLE |
| dna_v1_conservative | dp_per_tensor_eps10 | psnr_db | 1 | 39/0/0 | 16.793815 [16.018708651649476, 18.6632355328926] | 1.818989404e-12 | 4.365574569e-11 | ASSESSABLE |
| dna_v1_conservative | dp_per_tensor_eps10 | psnr_db | -1 | 0/39/0 | 16.793815 [16.018708651649476, 18.6632355328926] | 1 | 1 | ASSESSABLE |
| dna_v1_conservative | dp_per_tensor_eps10 | ssim | 1 | 39/0/0 | 0.85337231 [0.824445073842071, 0.8694383511319757] | 1.818989404e-12 | 4.365574569e-11 | ASSESSABLE |
| dna_v1_conservative | dp_per_tensor_eps10 | ssim | -1 | 0/39/0 | 0.85337231 [0.824445073842071, 0.8694383511319757] | 1 | 1 | ASSESSABLE |
| dna_v1_conservative | dp_per_tensor_eps10 | mse | 1 | 0/39/0 | -0.18393981 [-0.18935993034392595, -0.1750829826341942] | 1 | 1 | ASSESSABLE |
| dna_v1_conservative | dp_per_tensor_eps10 | mse | -1 | 39/0/0 | -0.18393981 [-0.18935993034392595, -0.1750829826341942] | 1.818989404e-12 | 4.365574569e-11 | ASSESSABLE |
| dna_v2_0p95 | dp_global_eps10 | psnr_db | 1 | 39/0/0 | 17.019105 [16.156717260941235, 18.545504962188186] | 1.818989404e-12 | 4.365574569e-11 | ASSESSABLE |
| dna_v2_0p95 | dp_global_eps10 | psnr_db | -1 | 0/39/0 | 17.019105 [16.156717260941235, 18.545504962188186] | 1 | 1 | ASSESSABLE |
| dna_v2_0p95 | dp_global_eps10 | ssim | 1 | 39/0/0 | 0.86321657 [0.8386607957072556, 0.873315304517746] | 1.818989404e-12 | 4.365574569e-11 | ASSESSABLE |
| dna_v2_0p95 | dp_global_eps10 | ssim | -1 | 0/39/0 | 0.86321657 [0.8386607957072556, 0.873315304517746] | 1 | 1 | ASSESSABLE |
| dna_v2_0p95 | dp_global_eps10 | mse | 1 | 0/39/0 | -0.18145426 [-0.19029048550873995, -0.1735564055852592] | 1 | 1 | ASSESSABLE |
| dna_v2_0p95 | dp_global_eps10 | mse | -1 | 39/0/0 | -0.18145426 [-0.19029048550873995, -0.1735564055852592] | 1.818989404e-12 | 4.365574569e-11 | ASSESSABLE |
| dna_v2_0p95 | dp_per_tensor_eps10 | psnr_db | 1 | 39/0/0 | 16.876142 [16.15544711759692, 18.626553763478384] | 1.818989404e-12 | 4.365574569e-11 | ASSESSABLE |
| dna_v2_0p95 | dp_per_tensor_eps10 | psnr_db | -1 | 0/39/0 | 16.876142 [16.15544711759692, 18.626553763478384] | 1 | 1 | ASSESSABLE |
| dna_v2_0p95 | dp_per_tensor_eps10 | ssim | 1 | 39/0/0 | 0.85910336 [0.8285973081365228, 0.8742763912305236] | 1.818989404e-12 | 4.365574569e-11 | ASSESSABLE |
| dna_v2_0p95 | dp_per_tensor_eps10 | ssim | -1 | 0/39/0 | 0.85910336 [0.8285973081365228, 0.8742763912305236] | 1 | 1 | ASSESSABLE |
| dna_v2_0p95 | dp_per_tensor_eps10 | mse | 1 | 0/39/0 | -0.18317975 [-0.18966307258233428, -0.17509699368383735] | 1 | 1 | ASSESSABLE |
| dna_v2_0p95 | dp_per_tensor_eps10 | mse | -1 | 39/0/0 | -0.18317975 [-0.18966307258233428, -0.17509699368383735] | 1.818989404e-12 | 4.365574569e-11 | ASSESSABLE |

## Commands, preservation and limitations

Pre-run amendment: protocols/amendments/2026-10-03_priority34b_meaningful_dp.md. CPU/thread1; no negative-BN/nonfinite output permitted; no clamp, excluded/replaced seed or scientific tuning. Per-client scores and raw arrays remain in the new namespace. Metrics use the frozen image [0,1] metric clipping, not a repair of training or BN.

CIFAR uses P31 LeNet42/splits/lr.1 but50-round three-client FedAvg, NOT P31's100-epoch centralized SGD. Clipping C preselected, not optimized; three users offer no sampling amplification. Utility collapse or weak recovery cannot establish DNA privacy superiority at matched utility. Different epsilon arms are separate hypothetical deployments; releasing all arms/seeds composes further.

Privacy noise uses independent OS-entropy keys per DP job, privately retained mode0600; public training/noise namespace IDs do not determine it. Private research keys/targets/localBN/evaluation scores are outside the accounted release boundary; publishing them is not DP. Torch secret-seeded PRNG/float32 simulates ideal Gaussian DP, not a certified production sampler. Secure aggregation and cryptographically secret Gaussian noise are deployment assumptions, not implemented security claims.

Preflight initially exposed a generic models namespace collision between project FraudMLP and the read-only TabLeak import chain. A new P34B-only import isolation helper resolved it before freeze/training; no earlier source, data, budget, target or outcome changed. See preflight_import_failure.json and the preflight annex.

[Gaussian RDP: Mironov2017](https://arxiv.org/abs/1702.07476), [user-level DP-FedAvg: McMahan et al., ICLR2018](https://arxiv.org/abs/1710.06963).

```text
python -B -m unittest discover -s tests -p test_priority34b.py -v
python -B experiments/supervise_priority34b.py --freeze
python -B -u experiments/supervise_priority34b.py --supervise
python -B experiments/verify_priority34b.py
git diff --check
```

## Final verified completion

470 new jobs,33 reused baselines; 1,265 new client/split or validation arrays bit-exact from checkpoints;41,250 no-BN upload assertions. Independent metrics, RDP calibration, CIs and fixed24test Holm PASS. Sources/input/result hashes, tests, py_compile/git diff --check PASS; no live worker. Pre-exit report archived byte-identically; scientific-manifest report digest resolves to archive/priority34b_report_pre_exit_audit.md. No scientific outcome replaced.

## Administrative close-out and noise scale clarification

No training or recovery was repeated during completion. All 470 jobs succeeded;
zero numerical/BN failures or execution interruptions occurred. The pre-run
import failure remains preserved and disclosed above. The post-exit report
version covered by `sha256_manifest_final.json` and `final_verified_receipt.json`
is preserved byte-identically at
`artifacts/priority34b/report_post_exit_pre_admin.md`; the earlier scientific
manifest's report version is the pre-exit archive. The administrative seal
records these relocations and the present report without changing either prior
manifest or any result. Goal auto-polling was paused at the user's request;
completion was checked by the scheduled monitor instead.

For both mechanisms, these actual per-coordinate server noise scales were
checked across every seed and round. Sensitivity S=2 max(w) C is constant per
dataset under the fixed public partition weights. Noise is added once per round
after averaging; these SDs are not independent per-client perturbations.

| Dataset | S | Noise SD, ε=1 | Noise SD, ε=3 | Noise SD, ε=10 |
| --- | --- | --- | --- | --- |
| PaySim | 0.008123877 | 0.281509866 | 0.097522452 | 0.032622537 |
| IEEE-CIS | 0.009589169 | 0.332285406 | 0.115112440 | 0.038506618 |
| BAF | 0.007085868 | 0.245540612 | 0.085061753 | 0.028454270 |
| CIFAR-10 | 0.006666667 | 0.231014386 | 0.080029484 | 0.026770910 |

At ε=10, conditional recovery divides the noisy aggregate by the target weight
1/3, so its equivalent gradient-coordinate noise SD is 0.0803127304.
The noise keys themselves remain private; the earlier phrase about disclosed
seeds describes an excluded release, not disclosure of these keys.

DP lowered utility substantially at the frozen C=.01 and K=3; this is not a
bound on the best attainable DP utility. Observed utility need not be monotone
in ε with eleven independent noise realizations and fixed clipping. DNA's
key-known recovery remains close to unprotected recovery. DP recovery is below
both prior references on all 39 targets: its lower reconstruction scores are
descriptive results of this frozen instrument, not proof against all attacks
or a matched-utility comparison. No new noninferiority claim is made.
