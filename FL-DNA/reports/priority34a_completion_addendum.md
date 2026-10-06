# Priority34A — verified completion addendum

Status: COMPLETE after supervisor exit and final independent audit.

The pre-exit scientific report is preserved byte-identically in archive/priority34a_report_pre_final_audit.md; priority34a_report.md receives only completion wording and administrative audit details, with all scientific tables unchanged.

189/189 jobs and567client evaluations; zero failures, no extra/replaced seed. All28350upload assertions confirm no BN tensor in any simulated transmission. All1134client/split probability arrays reproduce bit-exactly from final global-non-BN plus client-local-BN checkpoints. Original and raw-BN P32 evidence remain unchanged. Source/data/result/checkpoint/probability/round manifest verified; independent18endpoint NI recomputation and compile/diff checks PASS, no live supervisor/worker.

Research audit files contain local BN checkpoints/minima for verification; they are not simulated transmitted payloads. Keeping BN local removes that explicit transmission channel but does not establish a general privacy guarantee.

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
