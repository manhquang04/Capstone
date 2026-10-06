# Citation audit: `Latex/references.bib`

**Scope.** Audited all 52 entries in the read-only source bibliography on 2026-09-30. A DOI was resolved directly through Crossref; an `eprint` through the arXiv Atom API; entries without either were title-resolved through OpenAlex. `OK` means the bibliographic representation is supported by the resolved record (minor capitalization/diacritic/style differences ignored). `MISMATCH` means a material field differs. `UNVERIFIED` means no exact authoritative record was returned, not that the work is nonexistent.

**Field convention.** Each cell is `title; authors; venue; year; pages; DOI/arXiv`. `—` means absent from the record, not inferred. Full author lists are retained when the source record supplied them; `et al.` is used only where the live metadata provider itself returned a truncated author list.

## Results

| Key | Status | Original | Resolved live record (source) | Corrective edit recommendation |
|---|---|---|---|---|
| chawla2002smote | OK | *SMOTE: Synthetic Minority Over-sampling Technique*; Chawla, Bowyer, Hall, Kegelmeyer; Journal of Artificial Intelligence Research 16; 2002; 321--357; DOI 10.1613/jair.953 | Same title/authors/venue/year; 321--357; DOI 10.1613/jair.953 ([Crossref](https://api.crossref.org/works/10.1613%2Fjair.953)) | None. |
| davis2006prroc | OK | *The Relationship between Precision-Recall and ROC Curves*; Davis, Goadrich; ICML 2006; 2006; 233--240; DOI 10.1145/1143844.1143874 | *The relationship between Precision-Recall and ROC curves*; Davis, Goadrich; Proceedings of the 23rd International Conference on Machine Learning; 2006; 233--240; same DOI ([Crossref](https://api.crossref.org/works/10.1145%2F1143844.1143874)) | None; capitalization is stylistic. |
| konecny2016communication | OK | *Federated Learning: Strategies for Improving Communication Efficiency*; Konečný, McMahan, Yu, Richtárik, Suresh, Bacon; arXiv; 2016; —; arXiv:1610.05492 | Same title/authors; arXiv cs.LG; 2016; —; arXiv:1610.05492 ([arXiv](https://export.arxiv.org/api/query?id_list=1610.05492)) | None. |
| kairouz2021advances | MISMATCH | *Advances and Open Problems in Federated Learning*; Kairouz, McMahan, Avent, Bellet, Bennis, Bhagoji, Bonawitz, Charles, Cormode, Cummings, D'Oliveira, Eichner, El Rouayheb, Evans, Gardner, Garrett, Gascón, Ghazi, Gibbons, Gruteser, Harchaoui, He, He, Huo, Hutchinson, Hsu, Jaggi, Javidi, Joshi, Khodak, Konečný, Korolova, Koushanfar, Koyejo, Lepoint, Liu, Mittal, Mohri, Nock, Özgür, Pagh, Raykova, Qi, Ramage, Raskar, Song, Song, Stich, Sun, Suresh, Tramèr, Vepakomma, Wang, Xiong, Xu, Yang, Yu, Yu, Zhao; Foundations and Trends in Machine Learning 14(1--2); 2021; 1--210; DOI 10.1561/2200000083 | Same title/venue/year/pages/DOI, but Crossref deposits only Kairouz and McMahan as authors ([Crossref](https://api.crossref.org/works/10.1561%2F2200000083)). | Do **not** truncate the existing author list solely to the incomplete Crossref deposit; verify against the publisher before any author-list edit. |
| yang2019ffd | OK | *FFD: A Federated Learning Based Method for Credit Card Fraud Detection*; Yang, Zhang, Ye, Li, Xu; Big Data--BigData 2019 / LNCS 11514; 2019; 18--32; DOI 10.1007/978-3-030-23551-2_2 | Same title/authors; LNCS / *Big Data – BigData 2019*; 2019; 18--32; same DOI ([Crossref](https://api.crossref.org/works/10.1007%2F978-3-030-23551-2_2)) | None. |
| aurna2023flfraud | OK | *Federated Learning-Based Credit Card Fraud Detection: Performance Analysis with Sampling Methods and Deep Learning Algorithms*; Aurna, Hossain, Taenaka, Kadobayashi; IEEE CSR; 2023; 180--186; DOI 10.1109/CSR57506.2023.10224978 | Same title/authors; 2023 IEEE International Conference on Cyber Security and Resilience; 2023; 180--186; same DOI ([Crossref](https://api.crossref.org/works/10.1109%2FCSR57506.2023.10224978)) | None. |
| dwork2014dp | MISMATCH | *The Algorithmic Foundations of Differential Privacy*; Dwork, Roth; Foundations and Trends in Theoretical Computer Science 9(3--4); 2014; **211--407**; DOI 10.1561/0400000042 | Same title/authors/venue/year/DOI; **211--487** ([Crossref](https://api.crossref.org/works/10.1561%2F0400000042)) | Change `pages = {211--407}` to `pages = {211--487}`. |
| abadi2016dpsgd | OK | *Deep Learning with Differential Privacy*; Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar, Zhang; CCS; 2016; 308--318; DOI 10.1145/2976749.2978318 | Same title/authors; Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security; 2016; 308--318; same DOI ([Crossref](https://api.crossref.org/works/10.1145%2F2976749.2978318)) | None. |
| mironov2017renyi | OK | *Rényi Differential Privacy*; Mironov; IEEE CSF; 2017; 263--275; DOI 10.1109/CSF.2017.11 | Same title/author; 2017 IEEE 30th Computer Security Foundations Symposium; 2017; 263--275; same DOI ([Crossref](https://api.crossref.org/works/10.1109%2FCSF.2017.11)) | None. |
| wang2019subsampled | OK | *Subsampled Rényi Differential Privacy and Analytical Moments Accountant*; Wang, Balle, Kasiviswanathan; AISTATS / PMLR 89; 2019; 1226--1235; no DOI/arXiv in source | Same title/authors; OpenAlex resolves the journal version (*Journal of Privacy and Confidentiality* 10(2), 2020, no pages) and DOI 10.29012/jpc.723 ([OpenAlex W2887654530](https://openalex.org/W2887654530)). The cited AISTATS pages are a valid conference representation. | Optional: add `eprint = {1808.00087}` only after independently checking the desired version; otherwise none. |
| geyer2017clientdp | OK | *Differentially Private Federated Learning: A Client Level Perspective*; Geyer, Klein, Nabi; arXiv; 2017; —; arXiv:1712.07557 | Same title/authors; arXiv cs.CR; 2017; —; arXiv:1712.07557 ([arXiv](https://export.arxiv.org/api/query?id_list=1712.07557)) | None. |
| bonawitz2017secureagg | OK | *Practical Secure Aggregation for Privacy-Preserving Machine Learning*; Bonawitz, Ivanov, Kreuter, Marcedone, McMahan, Patel, Ramage, Segal, Seth; CCS; 2017; 1175--1191; DOI 10.1145/3133956.3133982 | Same title/authors/venue/year/pages/DOI ([Crossref](https://api.crossref.org/works/10.1145%2F3133956.3133982)) | None. |
| bonawitz2019system | OK | *Towards Federated Learning at Scale: System Design*; Bonawitz, Eichner, Grieskamp, Huba, Ingerman, Ivanov, Kiddon, Konečný, Mazzocchi, McMahan, Van Overveldt, Petrou, Ramage, Roselander; MLSys 1; 2019; 374--388; — | Same title/authors; OpenAlex W2913570153 (arXiv:1902.01046 version), 2019; record has no venue/pages; DOI 10.48550/arXiv.1902.01046 ([OpenAlex](https://openalex.org/W2913570153)). Source's MLSys pages are not contradicted. | Optional: add `eprint = {1902.01046}`; retain published MLSys venue/pages. |
| pittaluga2019privacy | OK | *Learning Privacy Preserving Encodings Through Adversarial Training*; Pittaluga, Koppal, Chakrabarti; WACV; 2019; 791--799; DOI 10.1109/WACV.2019.00089 | Same title/authors; 2019 IEEE WACV; 2019; 791--799; same DOI ([Crossref](https://api.crossref.org/works/10.1109%2FWACV.2019.00089)) | None. |
| jamshidi2024obfuscator | UNVERIFIED | *Customizable Utility-Privacy Trade-Off: A Flexible Autoencoder-Based Obfuscator*; Jamshidi, Mojahedian, Aref; The ISC International Journal of Information Security 16(2); 2024; 137--147; — | No exact title/author record returned by OpenAlex; exact-title query returned zero results ([OpenAlex query](https://api.openalex.org/works?filter=title.search:Customizable%20Utility%20Privacy%20Trade%20Off%20A%20Flexible%20Autoencoder%20Based%20Obfuscator&per-page=1)). | Retain unchanged. Obtain a publisher or Crossref record before adding/revising DOI, pages, or venue. |
| du2025sok | UNVERIFIED | *SoK: On Gradient Leakage in Federated Learning*; Du, Hu, Wang, Sun, Gong, Ren, Chen; 34th USENIX Security Symposium; 2025; 3045--3064; — | No exact-title OpenAlex record; returned result was the unrelated *Gradient-Leakage Resilient Federated Learning* ([query](https://api.openalex.org/works?filter=title.search:On%20Gradient%20Leakage%20in%20Federated%20Learning&per-page=1)). | Retain unchanged; recheck when USENIX/OpenAlex metadata is indexed. |
| lopez2016paysim | OK | *PaySim: A Financial Mobile Money Simulator for Fraud Detection*; Lopez-Rojas, Elmir, Axelsson; EMSS; 2016; 249--255; — | *Paysim: a financial mobile money simulator for fraud detection*; same authors; OpenAlex W2785726556 / Annual Simulation Symposium; 2016; 249--255; no DOI ([OpenAlex](https://openalex.org/W2785726556)). | None. |
| dalpozzolo2018fraud | OK | *Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy*; Dal Pozzolo, Boracchi, Caelen, Alippi, Bontempi; IEEE TNNLS 29(8); 2018; 3784--3797; DOI 10.1109/TNNLS.2017.2736643 | Same title/authors/venue/year/pages/DOI ([Crossref](https://api.crossref.org/works/10.1109%2FTNNLS.2017.2736643)) | None. |
| zhao2018noniid | OK | *Federated Learning with Non-IID Data*; Zhao, Li, Lai, Suda, Civin, Chandra; arXiv; 2018; —; arXiv:1806.00582 | Same title/authors; arXiv cs.LG; 2018; —; arXiv:1806.00582 ([arXiv](https://export.arxiv.org/api/query?id_list=1806.00582)) | None. |
| li2020fedprox | OK | *Federated Optimization in Heterogeneous Networks*; Li, Sahu, Zaheer, Sanjabi, Talwalkar, Smith; MLSys 2; 2020; 429--450; — | Same title/authors; OpenAlex W2914328083 resolves arXiv:1812.06127, 2018, without proceedings fields ([OpenAlex](https://openalex.org/W2914328083)). This is the preprint date, not a contradiction of the cited 2020 MLSys version. | Optional: add `eprint = {1812.06127}`; retain the published MLSys fields. |
| mcmahan2017fedavg | OK | *Communication-Efficient Learning of Deep Networks from Decentralized Data*; McMahan, Moore, Ramage, Hampson, Agüera y Arcas; AISTATS / PMLR 54; 2017; 1273--1282; — | Same title/authors; OpenAlex W2541884796 resolves arXiv:1602.05629, 2016, with pages 1273--1282 ([OpenAlex](https://openalex.org/W2541884796)). The different year is the preprint date. | Optional: add `eprint = {1602.05629}`; retain the published 2017 AISTATS fields. |
| zhu2019dlg | OK | *Deep Leakage from Gradients*; Zhu, Liu, Han; NeurIPS 32; 2019; —; arXiv:1906.08935 | Same title/authors; arXiv cs.LG; 2019; —; arXiv:1906.08935 ([arXiv](https://export.arxiv.org/api/query?id_list=1906.08935)) | None. |
| zhao2020idlg | OK | *iDLG: Improved Deep Leakage from Gradients*; Zhao, Mopuri, Bilen; arXiv; 2020; —; arXiv:2001.02610 | Same title/authors; arXiv cs.LG; 2020; —; arXiv:2001.02610 ([arXiv](https://export.arxiv.org/api/query?id_list=2001.02610)) | None. |
| geiping2020inverting | OK | *Inverting Gradients---How Easy Is It to Break Privacy in Federated Learning?*; Geiping, Bauermeister, Dröge, Moeller; NeurIPS 33; 2020; —; arXiv:2003.14053 | Same title/authors; arXiv cs.CV; 2020; —; arXiv:2003.14053 ([arXiv](https://export.arxiv.org/api/query?id_list=2003.14053)) | None. |
| yin2021gradinversion | MISMATCH | *See Through Gradients: Image Batch Recovery via GradInversion*; Yin, Mallya, Vahdat, Alvarez, Kautz, Molchanov; CVPR; 2021; **16337--16346**; DOI 10.1109/CVPR46437.2021.01607 | Same title/authors/venue/year/DOI; **16332--16341** ([Crossref](https://api.crossref.org/works/10.1109%2FCVPR46437.2021.01607)) | Change pages to `16332--16341`. |
| sun2021soteria | MISMATCH | *Soteria: Provable Defense Against Privacy Leakage in Federated Learning From Representation Perspective*; Sun, Li, Wang, Yang, Li, Chen; CVPR; 2021; **9311--9319**; DOI 10.1109/CVPR46437.2021.00919 | Same title/authors/venue/year/DOI; **9307--9315** ([Crossref](https://api.crossref.org/works/10.1109%2FCVPR46437.2021.00919)) | Change pages to `9307--9315`. |
| huang2021evaluating | OK | *Evaluating Gradient Inversion Attacks and Defenses in Federated Learning*; Huang, Gupta, Song, Li, Arora; NeurIPS 34; 2021; —; arXiv:2112.00059 | Same title/authors; arXiv cs.CR, journal reference NeurIPS 2021; 2021; —; arXiv:2112.00059 ([arXiv](https://export.arxiv.org/api/query?id_list=2112.00059)) | None. |
| vero2023tableak | OK | *TabLeak: Tabular Data Leakage in Federated Learning*; Vero, Balunović, Dimitrov, Vechev; ICML / PMLR 202; 2023; 35051--35083; — | Same title/authors; OpenAlex W4302307937 resolves arXiv:2210.01785, 2022, without published proceedings fields ([OpenAlex](https://openalex.org/W4302307937)). | Optional: add `eprint = {2210.01785}`; retain the published ICML 2023 fields. |
| wu2022federated | OK | *Federated Learning for Tabular Data: Exploring Potential Risk to Privacy*; Wu, Zhao, Chen, van Moorsel; ISSRE; 2022; 193--204; DOI 10.1109/ISSRE55969.2022.00028 | Same title/authors; IEEE ISSRE; 2022; 193--204; same DOI ([Crossref](https://api.crossref.org/works/10.1109%2FISSRE55969.2022.00028)) | None. |
| jin2021cafe | OK | *CAFE: Catastrophic Data Leakage in Vertical Federated Learning*; Jin, Chen, Hsu, Yu, Chen; NeurIPS 34; 2021; —; arXiv:2110.15122 | Same title/authors; arXiv cs.LG; 2021; —; arXiv:2110.15122 ([arXiv](https://export.arxiv.org/api/query?id_list=2110.15122)) | None. |
| ailon2006fast | OK | *Approximate Nearest Neighbors and the Fast Johnson--Lindenstrauss Transform*; Ailon, Chazelle; STOC; 2006; 557--563; DOI 10.1145/1132516.1132597 | Same title/authors/venue/year/pages/DOI ([Crossref](https://api.crossref.org/works/10.1145%2F1132516.1132597)) | None. |
| tropp2011srht | OK | *Improved Analysis of the Subsampled Randomized Hadamard Transform*; Tropp; Advances in Adaptive Data Analysis 3(1--2); 2011; 115--126; DOI 10.1142/S1793536911000787 | Same title/author/venue/year/pages/DOI ([Crossref](https://api.crossref.org/works/10.1142%2FS1793536911000787)) | None. |
| blackwelder1982noninferiority | OK | *Proving the Null Hypothesis in Clinical Trials*; Blackwelder; Controlled Clinical Trials 3(4); 1982; 345--353; DOI 10.1016/0197-2456(82)90024-1 | Same title/author/venue/year/pages/DOI ([Crossref](https://api.crossref.org/works/10.1016%2F0197-2456%2882%2990024-1)) | None. |
| lecun1998lenet | OK | *Gradient-Based Learning Applied to Document Recognition*; LeCun, Bottou, Bengio, Haffner; Proceedings of the IEEE 86(11); 1998; 2278--2324; DOI 10.1109/5.726791 | Same title/authors/venue/year/pages/DOI ([Crossref](https://api.crossref.org/works/10.1109%2F5.726791)) | None. |
| cifar10 | MISMATCH | *Learning Multiple Layers of Features from Tiny Images*; Krizhevsky; University of Toronto technical report; 2009; —; — | OpenAlex W3118608800 has same title/author but reports 2024 and DOI 10.57702/zp44cu3g, while its primary location is the original Toronto report URL ([OpenAlex](https://openalex.org/W3118608800)). | Retain the source's 2009 technical-report year; **do not add this DOI** without publisher confirmation. Flag the OpenAlex date as a record conflict. |
| huang2020instahide | OK | *InstaHide: Instance-hiding Schemes for Private Distributed Learning*; Huang, Song, Li, Arora; ICML / PMLR 119; 2020; 4507--4518; — | Same title/authors; OpenAlex W3091830290 resolves arXiv:2010.02772, 2020, without proceedings pages ([OpenAlex](https://openalex.org/W3091830290)). | Optional: add `eprint = {2010.02772}`; retain published ICML pages. |
| carlini2021instance | OK | *Is Private Learning Possible with Instance Encoding?*; Carlini, Deng, Garg, Jha, Mahloujifar, Mahmoody, Thakurta, Tramèr; IEEE S&P; 2021; 410--427; DOI 10.1109/SP40001.2021.00099 | Same title/authors/venue/year/pages/DOI ([Crossref](https://api.crossref.org/works/10.1109%2FSP40001.2021.00099)) | None. |
| blocki2012jl | OK | *The Johnson-Lindenstrauss Transform Itself Preserves Differential Privacy*; Blocki, Blum, Datta, Sheffet; FOCS; 2012; 410--419; DOI 10.1109/FOCS.2012.67 | Same title/authors/venue/year/pages/DOI ([Crossref](https://api.crossref.org/works/10.1109%2FFOCS.2012.67)) | None. |
| scheliga2022precode | OK | *PRECODE - A Generic Model Extension to Prevent Deep Gradient Leakage*; Scheliga, Mäder, Seeland; WACV; 2022; 3605--3614; DOI 10.1109/WACV51458.2022.00366 | Same title/authors/venue/year/pages/DOI ([Crossref](https://api.crossref.org/works/10.1109%2FWACV51458.2022.00366)) | None. |
| gehani2003dnacrypto | OK | *DNA-based Cryptography*; Gehani, LaBean, Reif; *Aspects of Molecular Computing* / LNCS 2950; 2003; 167--188; DOI 10.1007/978-3-540-24635-0_12 | Same title/authors/book/year/pages/DOI ([Crossref](https://api.crossref.org/works/10.1007%2F978-3-540-24635-0_12)) | None. |
| tramer2020adaptive | OK | *On Adaptive Attacks to Adversarial Example Defenses*; Tramèr, Carlini, Brendel, Madry; NeurIPS 33; 2020; —; arXiv:2002.08347 | Same title/authors; arXiv cs.LG, comment “NeurIPS 2020”; 2020; —; arXiv:2002.08347 ([arXiv](https://export.arxiv.org/api/query?id_list=2002.08347)) | None. |
| athalye2018obfuscated | OK | *Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples*; Athalye, Carlini, Wagner; ICML / PMLR 80; 2018; 274--283; — | Same title/authors; OpenAlex W2787708942 resolves arXiv:1802.00420, 2018, without published proceedings fields ([OpenAlex](https://openalex.org/W2787708942)). | Optional: add `eprint = {1802.00420}`; retain ICML pages. |
| carlini2019evaluating | OK | *On Evaluating Adversarial Robustness*; Carlini, Athalye, Papernot, Brendel, Rauber, Tsipras, Goodfellow, Madry, Kurakin; arXiv; 2019; —; arXiv:1902.06705 | Same title/authors; arXiv cs.LG; 2019; —; arXiv:1902.06705 ([arXiv](https://export.arxiv.org/api/query?id_list=1902.06705)) | None. |
| yue2023obfuscation | OK | *Gradient Obfuscation Gives a False Sense of Security in Federated Learning*; Yue, Jin, Wong, Baron, Dai; USENIX Security; 2023; —; arXiv:2206.04055 | Same title/authors; arXiv cs.CR, comment “Accepted by USENIX Security 2023”; submitted 2022; —; arXiv:2206.04055 ([arXiv](https://export.arxiv.org/api/query?id_list=2206.04055)) | None. |
| balunovic2022bayesian | OK | *Bayesian Framework for Gradient Leakage*; Balunović, Dimitrov, Staab, Vechev; ICLR; 2022; —; arXiv:2111.04706 | Same title/authors; arXiv cs.LG; first submitted 2021; —; arXiv:2111.04706 ([arXiv](https://export.arxiv.org/api/query?id_list=2111.04706)) | None. |
| suresh2017dme | OK | *Distributed Mean Estimation with Limited Communication*; Suresh, Yu, Kumar, McMahan; ICML / PMLR 70; 2017; 3329--3337; — | Same title/authors; OpenAlex W2547352193 resolves arXiv:1611.00429, 2016, and pages 3329--3337 ([OpenAlex](https://openalex.org/W2547352193)). | Optional: add `eprint = {1611.00429}`; retain published ICML year. |
| kairouz2021ddg | OK | *The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation*; Kairouz, Liu, Steinke; ICML / PMLR 139; 2021; 5201--5212; — | Same title/authors; OpenAlex W3132435311 resolves arXiv:2102.06387, 2021, and pages 5201--5212 ([OpenAlex](https://openalex.org/W3132435311)). | Optional: add `eprint = {2102.06387}`. |
| agarwal2021skellam | OK | *The Skellam Mechanism for Differentially Private Federated Learning*; Agarwal, Kairouz, Liu; NeurIPS 34; 2021; —; arXiv:2110.04995 | Same title/authors; arXiv cs.LG, comment “Paper published in NeurIPS 2021”; 2021; —; arXiv:2110.04995 ([arXiv](https://export.arxiv.org/api/query?id_list=2110.04995)) | None. |
| hatamizadeh2023gia | OK | *Do Gradient Inversion Attacks Make Federated Learning Unsafe?*; Hatamizadeh, Yin, Molchanov, Myronenko, Li, Dogra, Feng, Flores, Kautz, Xu, Roth; IEEE TMI 42(7); 2023; 2044--2056; DOI 10.1109/TMI.2023.3239391 | Same title/authors/venue/year/pages/DOI ([Crossref](https://api.crossref.org/works/10.1109%2FTMI.2023.3239391)) | None. |
| li2021fedbn | OK | *FedBN: Federated Learning on Non-IID Features via Local Batch Normalization*; Li, Jiang, Zhang, Kamp, Dou; ICLR; 2021; —; arXiv:2102.07623 | Same title/authors; arXiv cs.LG, comment “Accepted at ICLR 2021”; 2021; —; arXiv:2102.07623 ([arXiv](https://export.arxiv.org/api/query?id_list=2102.07623)) | None. |
| mcmahan2018dplm | OK | *Learning Differentially Private Recurrent Language Models*; McMahan, Ramage, Talwar, Zhang; ICLR; 2018; —; arXiv:1710.06963 | Same title/authors; arXiv cs.LG, comment “Camera-ready ICLR 2018 version”; 2017 preprint / 2018 venue; —; arXiv:1710.06963 ([arXiv](https://export.arxiv.org/api/query?id_list=1710.06963)) | None. |
| wu2023learning | OK | *Learning To Invert: Simple Adaptive Attacks for Gradient Inversion in Federated Learning*; Wu, Chen, Guo, Weinberger; UAI / PMLR 216; 2023; 2293--2303; — | Same title/authors; OpenAlex W4307074185 resolves arXiv:2210.10880, 2022, without UAI pages ([OpenAlex](https://openalex.org/W4307074185)). | Optional: add `eprint = {2210.10880}`; retain published UAI year/pages. |

## Counts

| Status | Entries |
|---|---:|
| OK | 45 |
| MISMATCH | 5 |
| UNVERIFIED | 2 |
| **Total audited** | **52** |

Resolution routes: Crossref direct DOI records **23**; arXiv direct ID records **15**; OpenAlex exact title/author records **12**; OpenAlex no exact record **2**. The title-resolution results that surfaced a preprint record rather than its proceedings manifestation were treated as supporting the work identity, not as a reason to overwrite an already-valid published citation.

## Exact query log

All requests below were made live during this audit. Each DOI request was `GET`; each arXiv request was an Atom API `GET`; each OpenAlex request was a `GET` of the displayed URL. The URLs in the result table are also the record locators.

### Crossref (23 exact-DOI requests)

`https://api.crossref.org/works/10.1613%2Fjair.953`  
`https://api.crossref.org/works/10.1145%2F1143844.1143874`  
`https://api.crossref.org/works/10.1561%2F2200000083`  
`https://api.crossref.org/works/10.1007%2F978-3-030-23551-2_2`  
`https://api.crossref.org/works/10.1109%2FCSR57506.2023.10224978`  
`https://api.crossref.org/works/10.1561%2F0400000042`  
`https://api.crossref.org/works/10.1145%2F2976749.2978318`  
`https://api.crossref.org/works/10.1109%2FCSF.2017.11`  
`https://api.crossref.org/works/10.1145%2F3133956.3133982`  
`https://api.crossref.org/works/10.1109%2FWACV.2019.00089`  
`https://api.crossref.org/works/10.1109%2FTNNLS.2017.2736643`  
`https://api.crossref.org/works/10.1109%2FCVPR46437.2021.01607`  
`https://api.crossref.org/works/10.1109%2FCVPR46437.2021.00919`  
`https://api.crossref.org/works/10.1109%2FISSRE55969.2022.00028`  
`https://api.crossref.org/works/10.1145%2F1132516.1132597`  
`https://api.crossref.org/works/10.1142%2FS1793536911000787`  
`https://api.crossref.org/works/10.1016%2F0197-2456%2882%2990024-1`  
`https://api.crossref.org/works/10.1109%2F5.726791`  
`https://api.crossref.org/works/10.1109%2FSP40001.2021.00099`  
`https://api.crossref.org/works/10.1109%2FFOCS.2012.67`  
`https://api.crossref.org/works/10.1109%2FWACV51458.2022.00366`  
`https://api.crossref.org/works/10.1007%2F978-3-540-24635-0_12`  
`https://api.crossref.org/works/10.1109%2FTMI.2023.3239391`

### arXiv (15 exact-ID requests)

`https://export.arxiv.org/api/query?id_list=1610.05492`  
`https://export.arxiv.org/api/query?id_list=1712.07557`  
`https://export.arxiv.org/api/query?id_list=1806.00582`  
`https://export.arxiv.org/api/query?id_list=1906.08935`  
`https://export.arxiv.org/api/query?id_list=2001.02610`  
`https://export.arxiv.org/api/query?id_list=2003.14053`  
`https://export.arxiv.org/api/query?id_list=2112.00059`  
`https://export.arxiv.org/api/query?id_list=2110.15122`  
`https://export.arxiv.org/api/query?id_list=2002.08347`  
`https://export.arxiv.org/api/query?id_list=1902.06705`  
`https://export.arxiv.org/api/query?id_list=2206.04055`  
`https://export.arxiv.org/api/query?id_list=2111.04706`  
`https://export.arxiv.org/api/query?id_list=2110.04995`  
`https://export.arxiv.org/api/query?id_list=2102.07623`  
`https://export.arxiv.org/api/query?id_list=1710.06963`

### OpenAlex (exact-title searches and failed/ambiguous attempts)

Exact matches:  
`https://api.openalex.org/works?search=Subsampled%20Renyi%20Differential%20Privacy%20and%20Analytical%20Moments%20Accountant&per-page=1&select=id,doi,title,publication_year,authorships,primary_location,biblio`  
`https://api.openalex.org/works?search=Towards%20Federated%20Learning%20at%20Scale%20System%20Design&per-page=1&select=id,doi,title,publication_year,authorships,primary_location,biblio`  
`https://api.openalex.org/works?search=PaySim%20A%20Financial%20Mobile%20Money%20Simulator%20for%20Fraud%20Detection&per-page=1&select=id,doi,title,publication_year,authorships,primary_location,biblio`  
`https://api.openalex.org/works?search=Federated%20Optimization%20in%20Heterogeneous%20Networks&per-page=1&select=id,doi,title,publication_year,authorships,primary_location,biblio`  
`https://api.openalex.org/works?search=Communication-Efficient%20Learning%20of%20Deep%20Networks%20from%20Decentralized%20Data&per-page=1&select=id,doi,title,publication_year,authorships,primary_location,biblio`  
`https://api.openalex.org/works?search=TabLeak%20Tabular%20Data%20Leakage%20in%20Federated%20Learning&per-page=1&select=id,doi,title,publication_year,authorships,primary_location,biblio`  
`https://api.openalex.org/works?search=Learning%20Multiple%20Layers%20of%20Features%20from%20Tiny%20Images&per-page=1&select=id,doi,title,publication_year,authorships,primary_location,biblio`  
`https://api.openalex.org/works?search=InstaHide%20Instance-hiding%20Schemes%20for%20Private%20Distributed%20Learning&per-page=1&select=id,doi,title,publication_year,authorships,primary_location,biblio`  
`https://api.openalex.org/works?filter=title.search:Obfuscated%20Gradients%20Give%20a%20False%20Sense%20of%20Security&per-page=1&select=id,doi,title,publication_year,authorships,primary_location,biblio`  
`https://api.openalex.org/works?filter=title.search:Distributed%20Mean%20Estimation%20with%20Limited%20Communication&per-page=1&select=id,doi,title,publication_year,authorships,primary_location,biblio`  
`https://api.openalex.org/works?filter=title.search:The%20Distributed%20Discrete%20Gaussian%20Mechanism%20for%20Federated%20Learning%20with%20Secure%20Aggregation&per-page=1&select=id,doi,title,publication_year,authorships,primary_location,biblio`  
`https://api.openalex.org/works?filter=title.search:Learning%20To%20Invert%20Simple%20Adaptive%20Attacks%20for%20Gradient%20Inversion%20in%20Federated%20Learning&per-page=1&select=id,doi,title,publication_year,authorships,primary_location,biblio`

No exact result / unrelated top result (retained to document the negative lookup):  
`https://api.openalex.org/works?search=Customizable%20Utility-Privacy%20Trade-Off%20A%20Flexible%20Autoencoder-Based%20Obfuscator&per-page=1&select=id,doi,title,publication_year,authorships,primary_location,biblio` (unrelated top result)  
`https://api.openalex.org/works?search=On%20Gradient%20Leakage%20in%20Federated%20Learning&per-page=1&select=id,doi,title,publication_year,authorships,primary_location,biblio` (unrelated top result)  
`https://api.openalex.org/works?filter=title.search:Customizable%20Utility%20Privacy%20Trade%20Off%20A%20Flexible%20Autoencoder%20Based%20Obfuscator&per-page=1&select=id,doi,title,publication_year,authorships,primary_location,biblio` (zero results)  
`https://api.openalex.org/works?filter=title.search:On%20Gradient%20Leakage%20in%20Federated%20Learning&per-page=1&select=id,doi,title,publication_year,authorships,primary_location,biblio` (unrelated top result)

The source bibliography was not edited.
