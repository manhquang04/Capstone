# Priority34E — client scaling and v2 seed-space feasibility

Status: independently verified execution and descriptive analysis complete.

## More clients: utility

Fixed total P32/P34A data: PaySim13, IEEE-CIS476, BAF58. K10/K20, 11 paired seeds343000–343010 each, baseline/v1/v2,198 jobs.50 rounds, Adam0.001, focal0.95/2, batch1024, one local epoch. All BN state stays local. Fraud split evenly; natural-category nonfraud55% to primary client,45% across others. Validation-only client thresholds; unweighted mean of K client test metrics per replicate. Clients are not independent replicates.

Pointwise descriptive Student-t95% CIs (df10) for mean paired DNA−baseline; not simultaneous confidence, NI, equivalence or superiority tests. No incomplete pairs/excluded seeds.

| Dataset | K | DNA | Endpoint | Mean delta | 95% CI | SD | Median | IQR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paysim | 10 | dna_v1_conservative | f1 | -0.00105072835 | [-0.005893655332108359, 0.003792198636542137] | 0.00720878399 | -0.000224464111 | [-0.006604758678848355, 0.0033680694997628713] |
| paysim | 10 | dna_v1_conservative | auc_roc | 0.00016009553 | [-0.00030430886849726227, 0.0006244999276485471] | 0.000691274306 | 2.53505316e-05 | [-0.0001781387136692847, 0.0003864655948809248] |
| paysim | 10 | dna_v1_conservative | pr_auc | -0.00302529944 | [-0.007231010630235371, 0.0011804117410176085] | 0.00626027681 | -0.00360910426 | [-0.006080520733418804, -0.0013343592486055322] |
| paysim | 10 | dna_v2_0p95 | f1 | -0.00239137455 | [-0.00719134020394442, 0.0024085911018222853] | 0.00714483528 | -0.0020620967 | [-0.008014950472017557, 0.0023687343643558623] |
| paysim | 10 | dna_v2_0p95 | auc_roc | 0.000193748173 | [-0.0001620160901028386, 0.0005495124367819253] | 0.00052956151 | 9.60619044e-05 | [-0.00021908106418511153, 0.0004516504585488357] |
| paysim | 10 | dna_v2_0p95 | pr_auc | -0.000722679018 | [-0.005265918362350614, 0.003820560326769796] | 0.00676269355 | -0.00236233546 | [-0.003948665286452269, 0.003099867487793062] |
| paysim | 20 | dna_v1_conservative | f1 | -0.00942637766 | [-0.022591457000596726, 0.0037387016756239254] | 0.0195964576 | -0.00532072644 | [-0.011015309076347346, 0.002676835126707411] |
| paysim | 20 | dna_v1_conservative | auc_roc | 0.000114125023 | [-0.0005847096962512143, 0.0008129597413463244] | 0.00104022806 | 0.000305075718 | [-0.0004183313140617151, 0.0008064502044847766] |
| paysim | 20 | dna_v1_conservative | pr_auc | -0.00238230984 | [-0.007659523581115289, 0.0028949039014661295] | 0.00785522765 | -0.00311559091 | [-0.006554654462258669, 0.00285293598469355] |
| paysim | 20 | dna_v2_0p95 | f1 | -1.50569531e-05 | [-0.01005338019198852, 0.010023266285739445] | 0.0149422248 | 0.00379531453 | [-0.007375422205519788, 0.009921720303100623] |
| paysim | 20 | dna_v2_0p95 | auc_roc | -1.06097048e-05 | [-0.0008439505604828323, 0.0008227311507987058] | 0.00124044286 | -0.000154744194 | [-0.0006531225668710561, 0.0009547966101074445] |
| paysim | 20 | dna_v2_0p95 | pr_auc | 0.00316235197 | [-0.004339805661313185, 0.01066450960831248] | 0.0111670967 | 0.00518295244 | [0.0024276365102687825, 0.008876890929347236] |
| ieee_cis | 10 | dna_v1_conservative | f1 | -0.00356074641 | [-0.012963090147514586, 0.005841597328675037] | 0.0139955579 | -0.00134199772 | [-0.011925716631667471, 0.005062537620084684] |
| ieee_cis | 10 | dna_v1_conservative | auc_roc | -0.000963271301 | [-0.0031926162728015133, 0.0012660736710775442] | 0.00331842012 | -0.00111910338 | [-0.0032338087806598192, 0.0002773307301391914] |
| ieee_cis | 10 | dna_v1_conservative | pr_auc | -0.00210879753 | [-0.01262802912972355, 0.00841043407607706] | 0.0156580656 | -0.00424114231 | [-0.014212576938324228, 0.007098882899563952] |
| ieee_cis | 10 | dna_v2_0p95 | f1 | -0.00550443398 | [-0.011126883885086421, 0.00011801592762297666] | 0.0083691179 | -0.00467191587 | [-0.01044358252327679, 0.0009760124614715537] |
| ieee_cis | 10 | dna_v2_0p95 | auc_roc | -0.00118948059 | [-0.003177362418033418, 0.0007984012301673797] | 0.00295899788 | -0.000915035334 | [-0.0024644855246421238, -0.00022241424481206362] |
| ieee_cis | 10 | dna_v2_0p95 | pr_auc | -0.00901969503 | [-0.022404610949481256, 0.00436522089362106] | 0.0199236883 | -0.00528065591 | [-0.014972492929580847, -0.002792408760206422] |
| ieee_cis | 20 | dna_v1_conservative | f1 | -0.00582673195 | [-0.017663421532359114, 0.006009957623818947] | 0.0176191255 | -0.00448277619 | [-0.006448036488268055, 0.001220184729662982] |
| ieee_cis | 20 | dna_v1_conservative | auc_roc | -0.000482820527 | [-0.0041087366883930596, 0.003143095635327697] | 0.00539724148 | -0.000865270846 | [-0.0037925466478508363, 0.003139766064911509] |
| ieee_cis | 20 | dna_v1_conservative | pr_auc | -0.00807381785 | [-0.03364502805744309, 0.017497392359893844] | 0.0380632067 | -0.00371546224 | [-0.02474034249087094, 0.007348870259575951] |
| ieee_cis | 20 | dna_v2_0p95 | f1 | -8.41995662e-05 | [-0.011913768139017894, 0.011745369006631637] | 0.0176085258 | -0.00436579113 | [-0.010130550430596588, 0.006018851226215677] |
| ieee_cis | 20 | dna_v2_0p95 | auc_roc | 0.000771970817 | [-0.0035554389238477823, 0.005099380557452056] | 0.0064414273 | -0.000618170476 | [-0.003012494353848705, 0.0016330435818152078] |
| ieee_cis | 20 | dna_v2_0p95 | pr_auc | 0.00759907023 | [-0.011501088397706926, 0.026699228862600833] | 0.0284309299 | -0.00295271624 | [-0.006759839117625066, 0.00814752126381331] |
| baf | 10 | dna_v1_conservative | f1 | 0.00123060237 | [0.0004655461048294889, 0.0019956586375614377] | 0.00113880002 | 0.00120284277 | [0.0005567027259577029, 0.0016997589601854546] |
| baf | 10 | dna_v1_conservative | auc_roc | 7.34355421e-05 | [-0.0006654173139116044, 0.0008122883982055328] | 0.00109979578 | 0.000249791703 | [-0.0004921393057718548, 0.0007031035953125042] |
| baf | 10 | dna_v1_conservative | pr_auc | 0.000218834316 | [-0.00047769735207245844, 0.0009153659833096646] | 0.00103679993 | -0.00027109671 | [-0.0005419063036399413, 0.0007993650567476585] |
| baf | 10 | dna_v2_0p95 | f1 | -0.000934612626 | [-0.0023656848120165762, 0.0004964595609291126] | 0.00213017671 | -0.000799140419 | [-0.0015857323493779496, -0.00011355280201760798] |
| baf | 10 | dna_v2_0p95 | auc_roc | 0.000171332419 | [-0.0007386815723314306, 0.0010813464099911475] | 0.00135457221 | -0.000126023167 | [-0.00037613889997945016, 0.00081019256889725] |
| baf | 10 | dna_v2_0p95 | pr_auc | -0.000646942395 | [-0.0017640701815151562, 0.00047018539171314686] | 0.00166286482 | -0.0009848768 | [-0.0016492542254391979, 0.0001438064734136657] |
| baf | 20 | dna_v1_conservative | f1 | -0.000463762423 | [-0.0021289083616741755, 0.0012013835159391149] | 0.0024785997 | 0.000163337021 | [-0.0021195230732252435, 0.0014680791593304349] |
| baf | 20 | dna_v1_conservative | auc_roc | -9.73122471e-05 | [-0.0007841336384464893, 0.0005895091441974216] | 0.001022346 | -0.000291490866 | [-0.0006025751589516237, 0.00039720060527242973] |
| baf | 20 | dna_v1_conservative | pr_auc | 0.000873801109 | [-0.00030944538152002606, 0.0020570476002358387] | 0.0017612837 | 0.000408527395 | [5.193529523397911e-06, 0.0013260833786647719] |
| baf | 20 | dna_v2_0p95 | f1 | -0.000110541991 | [-0.0019549079563663707, 0.0017338239749091666] | 0.00274537194 | 0.00116688674 | [-0.00216703876990329, 0.0017193668002628698] |
| baf | 20 | dna_v2_0p95 | auc_roc | -4.61736084e-05 | [-0.0010347637542842084, 0.0009424165374826139] | 0.00147153423 | 0.000177693522 | [-0.0005791749354248088, 0.0009343158188058198] |
| baf | 20 | dna_v2_0p95 | pr_auc | 0.000462451642 | [-0.0009598361410132721, 0.001884739425842929] | 0.00211710097 | 0.00105951752 | [-0.0004398278884756718, 0.0015994773796815798] |

## Seed-space recognition

Original API metadata includes the derived seed directly; literal observation requires no search. The experiment removes seed but retains sampled_indices. Recognition regenerates the ordered index list for each candidate base seed and scores exact equality. One first-layer128×58 BAF update sketch, ratio0.95/eta0.01; observer receives no raw update/true seed. This is protocol-metadata-assisted recognition, NOT identification from a metadata-free sketch. This simulation lifts locally and has no implemented wire channel whose metadata exposure is proven here.

Exhaustive20-bit candidates: 1048576; true rank: 1; tied at true score: 1; false positives: 0; unique-rank1 validation: True. Matching candidates: [343101].
Active search time 561.75s, throughput 1866.63 candidates/s. Full32 extrapolated worst case 2.30093e+06s (639.146h); uniform-position expected 1.15046e+06s. Chunk-rate sensitivity seconds: [2282899.3413120005, 2398292.039172096]. No full32 search executed. Setup/downtime/audit excluded from throughput.
Quantization-grid rounding residual: 1.29833426e-05; decode diagnostics: [{'candidate': 343101, 'decoded_l2': 0.078512324133045}].
Grid consistency of q is seed independent, and full padded orthogonal lift L2 is seed invariant; neither alone identifies the seed.20-bit uniqueness does not prove full32 uniqueness, cryptographic security or recovery of original records. Benchmark uses only this machine/protocol/observation; it is not an optimized GPU or distributed attack.

## Reproducibility and independent audit

{"cpu_count": 10, "device": "cpu", "interop_threads": 1, "numpy": "1.26.4", "platform": "macOS-15.7.7-arm64-arm-64bit", "processor": "arm", "python": "3.9.6 (default, Jan  9 2026, 11:03:41) \n[Clang 17.0.0 (clang-1700.6.4.2)]", "torch": "2.2.2", "torch_threads": 1}
CPU only, torch intra/inter-op1 per process; sequential seed timing before four utility workers. Independently reloaded all198 results, 5940 checkpoint prediction arrays and 148500 BN-firewall uploads. All20-bit candidate scores independently recomputed. Exact completed-job/chunk hash skip on explicit resume; interrupted attempts and failures retained. No numerical/BN clamp, seed replacement or scientific retuning.
All P34E unit tests, Python compilation and git diff --check PASS. Earlier artifacts, datasets, Latex/ and external_defenses/ untouched. Old P34B Goal remains paused. Amendment: protocols/amendments/2026-10-06_priority34e_clients_seed_search.md. Full hashes: artifacts/priority34e/final_manifest.json; verification seal: artifacts/priority34e/COMPLETE.json.
