# Priority35B — protocol and tests readiness

Date: 2026-10-10. Status: **PREPARATION COMPLETE; MEASUREMENTS NOT AUTHORIZED**.
This is not a privacy–utility result or completion of the full experiment.

## Frozen prospective protocol

Amendment: `protocols/amendments/2026-10-10_priority35b_frontier_protocol.md`.
SHA-256: `37634510777594f4ac7b65e8face6c305778e9e1c805a0fa3fae5b5032357987`.
The amendment was frozen before tests; its scientific rules and figure specification remain unchanged.

Utility uses Config-C local-BN training, K=3, 50 rounds, Adam learning rate
0.001 and 11 paired seeds (321000–321010) on BAF, IEEE-CIS and PaySim.
The read-only identity audit validated 297 existing utility jobs for potential
reuse: 99 baseline/v1/v2 jobs and 198 local-DP jobs at epsilon 1/3/10.
No new F1 aggregation or measurement was performed.

The leakage axis uses the different Priority34C single-gradient setting.
Eight previously qualified dataset/estimator cells are eligible. IEEE-CIS
TabLeak b=2 did not qualify previously and remains NOT_ASSESSABLE; no gate
is reopened. Future evaluation uses 39 fresh, paired row sets per eligible
cell, subject to a source-ID firewall against Priorities24–35, including
reserved Priority35A targets. No new real target was allocated in this step.

DP uses the Priority34B client-side mechanism, not its central comparator:
joint clip C=0.01, replace-one sensitivity 0.02, full participation,
50-round RDP composition and delta=1e-5. Joint and per-tensor clipping are
separate curves. The epsilon grid is 1, 3, 10, 30, 100, 1e3, 1e4, 1e6.
Noise SD reproduction for epsilon 1/3/10 is exact:

| Whole-training epsilon | Per-upload noise SD |
| --- | --- |
| 1 | 0.6930431582631877 |
| 3 | 0.24008845279270588 |
| 10 | 0.08031273039270019 |

Matching requires the largest-noise passing grid point and a failing next
noisier neighbor at the fixed mean-F1 margin 0.005. Missing brackets are
NOT_ASSESSABLE. The fixed Holm family reserves 144 hypotheses, including
ineligible or unmatched slots at p=1; no outcome-dependent family shrinkage.
The experimental unit is the paired row set, not an optimization ensemble
member. Median and paired-effect intervals use ranks 13/27 of 39; exact
sign tests exclude ties from the effective count and retain both directions.

## Locked figure specification

Three dataset panels share scales at IEEE single-column width 8.8 cm.
The horizontal coordinate is utility loss, minus paired Delta-F1 relative
to baseline, with 95% t intervals over 11 seeds; gains are not clamped.
The vertical coordinate is median feature accuracy with rank-13/27 bars.
Each panel's estimator is selected only from its unprotected-upload medians
before transform points are inspected, with a fixed tie rule. Mean/mode and
empirical-marginal references, separately labelled joint/per-tensor DP
curves and epsilon markers, and v1/v2 points are specified in the amendment.
The caption must explicitly disclose that the axes come from different
configurations. PDF, PNG and exact data CSV are reserved under
`artifacts/priority35b/figure/`; none was generated here.

## Verification and commands

From FL-DNA:

```sh
.venv-phase1/bin/python -B experiments/priority35b_preflight.py
```

The preflight ran the unit-test discovery, Python compilation and
`git diff --check`, with command output preserved under
`artifacts/priority35b/preflight/1791624198038289000/`.
All **15 tests PASS**, including exact historical noise SD reproduction,
clipping bounds, grid matching and rejection cases, independent sign-test
and Holm checks, interval coverage, panel selection, source firewall, and
non-finite/negative-BN guards. CPU/thread-1 settings were checked.
Historical source/input and reusable output identities were audited read-only.

Evidence: `protocol_freeze.json`, `readiness_freeze.json`,
`historical_reuse_audit.json`, `historical_firewall_audit.json`,
`checklist.json`, `progress.json` and `progress.log`, all under
`artifacts/priority35b/`. A separate preparation seal includes this report
without rewriting the existing readiness freeze.

## Limits and next step

There are **zero new measurements and zero new real target allocations**.
No empirical frontier, matched-utility superiority or privacy conclusion is
available yet. These cross-configuration axes must not be interpreted as
record recovery from Config-C Adam uploads or as a record-level DP guarantee.
Historical utility results were already observed; the prospective matching
rules and hash-conditional reuse do not make them new blinded data.

Latex/, external_defenses/, datasets and earlier artifacts/reports were not
edited. No commit or push was made. Execution requires separate user
authorization and a subsequent source/input/target freeze. Stop and wait.


## Priority 35B B3 — verified frontier results

This section is appended to the unchanged historical readiness report. B2b adds post-hoc clip norms and arms; it does not replace B2 results, matching rules or the Holm family.

### Design, experimental units and statistics

Utility uses Config-C local-BN Adam training: 50 rounds, K=3, 11 paired seeds. The client-average endpoint is computed within each seed; clients and rounds are not additional independent replicates. Utility deltas are arm minus baseline with descriptive two-sided 95% Student-t intervals (df=10). Leakage uses the different Priority 34C single-gradient configuration with 39 fresh row sets paired across all arms. The batch, not each feature or ensemble member, is the comparison unit. Leakage medians and paired differences use order-statistic ranks 13/27. Feature accuracy is higher for more recovery; standardized MSE and bounded min(MSE, MSE_prior) are lower for more recovery.

Exact one-sided sign tests are reported in both directions on transform-minus-comparator differences; ties are excluded from the effective sign-test n. The fixed Holm family has 144 entries, including NOT_ASSESSABLE reservations with p=1. No family shrinkage or outcome-dependent exclusions are used. Failure to reject is not evidence of equivalence or privacy.

### Exact execution and provenance

```text
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python /Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/experiments/priority35b_scipy_verifier_adapter.py --stage prepare
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python /Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/experiments/priority35b_label_coordinates.py --stage render --ready artifacts/priority35b/b3/verified_preparation_attempt_1791785722977843000/READY_FOR_RENDER.json
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage_statistics
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage_integrity
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage_metrics_audit
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage_payload_audit
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage_postexit
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage_objectives_audit
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage_completion_audit
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage_scientific_verify
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage_tables
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage_report_guard
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage_report_text
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage_figure
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage_data_bridge
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage_prepare_publication
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage_manifest_audit
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage_verified_preparation
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage_render_publication
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage_publication_contract
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python -B -m unittest tests.test_priority35b_leakage_final_validation
git diff --check
/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/.venv-phase1/bin/python /Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/experiments/priority35b_report_optional_columns.py --stage publish --ready artifacts/priority35b/b3/verified_preparation_attempt_1791785722977843000/READY_FOR_RENDER.json --render artifacts/priority35b/b3/publication_attempt_1791785736353163000/presentation_attempt_1791787208409293000/RENDERED.json --qa artifacts/priority35b/b3/publication_attempt_1791785736353163000/presentation_attempt_1791787208409293000/VISUAL_QA.json --validation artifacts/priority35b/b3/publication_attempt_1791785736353163000/validation_attempt_1791786291296702000/VALIDATION.json --interpretation artifacts/priority35b/b3/publication_attempt_1791785736353163000/sensitivity_attempt_1791787109768044000/interpretation.json
```

| path | sha256 |
| --- | --- |
| protocols/amendments/2026-10-10_priority35b_b2b_baseline_clip_values.md | e185e292df12f29dec2f0c43a51d896e5d8c5373880a2ed2d9e409411b05be16 |
| protocols/amendments/2026-10-10_priority35b_b2b_baseline_telemetry.md | 4180b481aef73a5e8151d48882abc688992d2e20dbc79e6b3a41e8a0ee7cc824 |
| protocols/amendments/2026-10-10_priority35b_b2b_csv_infrastructure.md | e614befdb1a258c270d89a8e6e8e9507979d8bde532d8f680605ec38370fdcdb |
| protocols/amendments/2026-10-10_priority35b_b2b_execution.md | 97c088555282c1ec7e2407befbd8c3e8ea6abe58d402dc22a766951d4a0c4376 |
| protocols/amendments/2026-10-10_priority35b_frontier_protocol.md | 37634510777594f4ac7b65e8face6c305778e9e1c805a0fa3fae5b5032357987 |
| protocols/amendments/2026-10-11_priority35b_b3_completion_audit.md | 183e6cdc3697c587d0f0af1aa61499c3cc2df79fffe26c7b34d971b7c05b6f43 |
| protocols/amendments/2026-10-11_priority35b_b3_data_bridge.md | d18481ba79d9649e40430c59160038fb938ce0e8d5545bccece922790ccb8aeb |
| protocols/amendments/2026-10-11_priority35b_b3_execution.md | 0cbd59a245795998ab811d6dad69a634038a7dc50d7d6081a7674e642f17ef29 |
| protocols/amendments/2026-10-11_priority35b_b3_figure_rendering.md | ce792d3889b97e961f71bf6f119d1719f6989a2cda67ce4b0a223d7274fb2201 |
| protocols/amendments/2026-10-11_priority35b_b3_final_validation.md | 3c8a060286d043b8385bfcd8d36cfa14fc48d93bc0a9558be6813cccf8b0d755 |
| protocols/amendments/2026-10-11_priority35b_b3_integrity_audit.md | 1a97dfbe85fe054339c89522b52dccd49c6c9184dc53ed1ce9f4548eb4d1745e |
| protocols/amendments/2026-10-11_priority35b_b3_manifest_audit.md | 760f70578d5b43639bf9a13ae85f78bc3726571c4808ad94447346832e2405c7 |
| protocols/amendments/2026-10-11_priority35b_b3_metrics_audit.md | 9c04f7ab3b81b0b25ca5099cb7d28d8e5e494e853d02c5512e85d484c73eef39 |
| protocols/amendments/2026-10-11_priority35b_b3_objective_audit.md | 6f89aefabaddbf2ddbc5b42297cedd4b4753b692d70a733187716bab28b14850 |
| protocols/amendments/2026-10-11_priority35b_b3_payload_audit.md | a60052787a86eac145c93365a488ed0f02c3f2cb05dbf4ec0fcec1d26bef7cdc |
| protocols/amendments/2026-10-11_priority35b_b3_postexit_integration.md | 30a40d60b8494e65b3d0bed7e3d21290e33658b6cf4cf5292fe1941a6f62ac4a |
| protocols/amendments/2026-10-11_priority35b_b3_publication_contract.md | 391eeb8bc3d9227022331d0393897c304e64765db7a9e9708df89ee4452bf9c7 |
| protocols/amendments/2026-10-11_priority35b_b3_publication_preparation.md | c5e68ed0affa88f7ae0ba752d5b8bd4afb5b3bf8caf53c2c95593edbd56f6000 |
| protocols/amendments/2026-10-11_priority35b_b3_render_integration.md | e21357499a22215b34c280b161df53d00d952fa54b5946e5cea5001b2829e6bc |
| protocols/amendments/2026-10-11_priority35b_b3_report_guard.md | 92bfeeeb36b07622ede9dab96640f9ad8d414759c689506b76f66d00a48ab832 |
| protocols/amendments/2026-10-11_priority35b_b3_report_rendering.md | d57bc223518730a80fb9bf9c07eca9ca995f7b3b7349f8b5b9b53da2fbaed2aa |
| protocols/amendments/2026-10-11_priority35b_b3_scientific_coordinator.md | 7d0b163cac186ffee70edc4e9566fb968d478e8b4b9325161aca4b1c8a2ce1e5 |
| protocols/amendments/2026-10-11_priority35b_b3_statistics_audit.md | 33dc21c69d9f3c0ea781106fa9c7e0eff9eac5647ce73821e092f04b8355c397 |
| protocols/amendments/2026-10-11_priority35b_b3_table_serialization.md | d7df590d84747f34d7f4b170f9a03cb8d0a8025efc1655a39c5584c592a5386a |
| protocols/amendments/2026-10-11_priority35b_b3_verified_preparation.md | 652620952367efca56d268edb9a05a63def7dcf87157f4cb9747aaf38ca93781 |
| protocols/amendments/2026-10-12_priority35b_b3_final_seal.md | 75772012e20b8405b3c7eb2cce5893682515f1e9ee81b6ac571c19045025b6f8 |
| protocols/amendments/2026-10-12_priority35b_b3_float64_reduction_infrastructure.md | f53726ca4018deefc4011f94b874ec95db231261eed5986f38bd9ab5bc79e5ca |
| protocols/amendments/2026-10-12_priority35b_b3_report_publication.md | 8323f903aa4950a4c66e0efd9c29c3e2d6442584557da8929aac0571245c7bab |
| protocols/amendments/2026-10-12_priority35b_b3_scipy_crosscheck_infrastructure.md | 7bbf94ed715fa1aed9e5fe01ca954109da756d0fba89fb47bd14f5f49e0421d9 |
| protocols/amendments/2026-10-12_priority35b_label_coordinates.md | 2d6ef3332ccdcd4bbbacf3aa356f6a24e32a037df255bc815aca15bbd63ea2f9 |
| protocols/amendments/2026-10-12_priority35b_presentation_sensitivity.md | 42f0088e9b93baae54073a4353bbe0009d667a41fc9eb33030d327ffe24289c7 |
| protocols/amendments/2026-10-12_priority35b_report_optional_columns.md | 7a0ff7c629a96988ef47fa78ef0f83304a0e05192a84b86253063bf6d2fc5c8c |

Software: {"matplotlib": "3.7.5", "numpy": "1.26.4", "scipy": "1.13.1", "torch": "2.2.2"}

### Complete utility results

| baseline_mean | ci_high | ci_low | dataset | df | endpoint | mean | mean_delta | method | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.22571169625881407 | 0 | 0 | baf | 10 | f1 | 0.22571169625881407 | 0 | baseline | 11 |
| 0.87134039862506862 | 0 | 0 | baf | 10 | auc_roc | 0.87134039862506862 | 0 | baseline | 11 |
| 0.15868126190169332 | 0 | 0 | baf | 10 | pr_auc | 0.15868126190169332 | 0 | baseline | 11 |
| 0.22571169625881407 | 0.0030092171073262614 | -0.0028706856192922212 | baf | 10 | f1 | 0.22578096200283107 | 6.9265744017019877e-05 | dna_v1_conservative | 11 |
| 0.87134039862506862 | 0.0010682805271964436 | -0.00082801654696190545 | baf | 10 | auc_roc | 0.87146053061518591 | 0.00012013199011726908 | dna_v1_conservative | 11 |
| 0.15868126190169332 | 0.0004503906829602701 | -0.00093284012161728579 | baf | 10 | pr_auc | 0.15844003718236482 | -0.00024122471932850784 | dna_v1_conservative | 11 |
| 0.22571169625881407 | 0.0033414029778549175 | -0.0023547112124832283 | baf | 10 | f1 | 0.22620504214149992 | 0.00049334588268584473 | dna_v2_0p95 | 11 |
| 0.87134039862506862 | 0.00061742842646211871 | -0.0020930712914545239 | baf | 10 | auc_roc | 0.87060257719257261 | -0.00073782143249620264 | dna_v2_0p95 | 11 |
| 0.15868126190169332 | 0.00051520826873067071 | -0.0013404822658895062 | baf | 10 | pr_auc | 0.15826862490311389 | -0.00041263699857941769 | dna_v2_0p95 | 11 |
| 0.22571169625881407 | -0.16374878638613483 | -0.19035501298366625 | baf | 10 | f1 | 0.048659796573913533 | -0.17705189968490054 | local_dp_global_eps1 | 11 |
| 0.87134039862506862 | -0.22350736589905035 | -0.30417490541997944 | baf | 10 | auc_roc | 0.60749926296555379 | -0.26384113565951489 | local_dp_global_eps1 | 11 |
| 0.15868126190169332 | -0.13119353767670069 | -0.14207694009013352 | baf | 10 | pr_auc | 0.022046023018276198 | -0.13663523888341711 | local_dp_global_eps1 | 11 |
| 0.22571169625881407 | -0.15410861378669358 | -0.17953935052746964 | baf | 10 | f1 | 0.058887714101732477 | -0.16682398215708161 | local_dp_global_eps3 | 11 |
| 0.87134039862506862 | -0.19939535947266115 | -0.24378080796206222 | baf | 10 | auc_roc | 0.64975231490770702 | -0.22158808371736169 | local_dp_global_eps3 | 11 |
| 0.15868126190169332 | -0.12669573978609458 | -0.13783148258913833 | baf | 10 | pr_auc | 0.026417650714076858 | -0.13226361118761645 | local_dp_global_eps3 | 11 |
| 0.22571169625881407 | -0.15397816086316382 | -0.16929688350169461 | baf | 10 | f1 | 0.064074174076384882 | -0.16163752218242922 | local_dp_global_eps10 | 11 |
| 0.87134039862506862 | -0.18171904698423608 | -0.2096943632501066 | baf | 10 | auc_roc | 0.67563369350789737 | -0.19570670511717134 | local_dp_global_eps10 | 11 |
| 0.15868126190169332 | -0.12791029877829094 | -0.13382360083268982 | baf | 10 | pr_auc | 0.027814312096202953 | -0.13086694980549038 | local_dp_global_eps10 | 11 |
| 0.22571169625881407 | -0.15176012801749453 | -0.16736371453659407 | baf | 10 | f1 | 0.066149774981769799 | -0.1595619212770443 | local_dp_global_eps30 | 11 |
| 0.87134039862506862 | -0.16928265823628272 | -0.1924692611771566 | baf | 10 | auc_roc | 0.69046443891834908 | -0.18087595970671966 | local_dp_global_eps30 | 11 |
| 0.15868126190169332 | -0.1260504524739979 | -0.13258115848931076 | baf | 10 | pr_auc | 0.029365456420039002 | -0.12931580548165433 | local_dp_global_eps30 | 11 |
| 0.22571169625881407 | -0.12997333613548387 | -0.15194230769184805 | baf | 10 | f1 | 0.084753874345148139 | -0.14095782191366596 | local_dp_global_eps100 | 11 |
| 0.87134039862506862 | -0.13478993219849775 | -0.16161654653261445 | baf | 10 | auc_roc | 0.72313715925951261 | -0.1482032393655561 | local_dp_global_eps100 | 11 |
| 0.15868126190169332 | -0.11478355849450629 | -0.12844849090393495 | baf | 10 | pr_auc | 0.037065237202472706 | -0.12161602469922062 | local_dp_global_eps100 | 11 |
| 0.22571169625881407 | -0.11798249341390861 | -0.14176034417181277 | baf | 10 | f1 | 0.095840277465953388 | -0.12987141879286068 | local_dp_global_eps1000 | 11 |
| 0.87134039862506862 | -0.11650705851860546 | -0.13063613025865431 | baf | 10 | auc_roc | 0.74776880423643877 | -0.12357159438862989 | local_dp_global_eps1000 | 11 |
| 0.15868126190169332 | -0.10661937234930061 | -0.12184994895684459 | baf | 10 | pr_auc | 0.04444660124862073 | -0.1142346606530726 | local_dp_global_eps1000 | 11 |
| 0.22571169625881407 | -0.11417206194912113 | -0.14383023426488503 | baf | 10 | f1 | 0.096710548151811002 | -0.12900114810700308 | local_dp_global_eps10000 | 11 |
| 0.87134039862506862 | -0.11028988384091394 | -0.12748497644369083 | baf | 10 | auc_roc | 0.75245296848276633 | -0.11888743014230239 | local_dp_global_eps10000 | 11 |
| 0.15868126190169332 | -0.10349599430323927 | -0.12293488343423736 | baf | 10 | pr_auc | 0.045465823032954995 | -0.11321543886873832 | local_dp_global_eps10000 | 11 |
| 0.22571169625881407 | -0.11151287832966809 | -0.14442252620172857 | baf | 10 | f1 | 0.097743993993115744 | -0.12796770226569834 | local_dp_global_eps1000000 | 11 |
| 0.87134039862506862 | -0.11102969505853499 | -0.13375505037536406 | baf | 10 | auc_roc | 0.74894802590811915 | -0.12239237271694953 | local_dp_global_eps1000000 | 11 |
| 0.15868126190169332 | -0.10479483884728363 | -0.12262591714176224 | baf | 10 | pr_auc | 0.044970883907170404 | -0.11371037799452294 | local_dp_global_eps1000000 | 11 |
| 0.22571169625881407 | -0.17595524194377674 | -0.1904616193786117 | baf | 10 | f1 | 0.042503265597619869 | -0.18320843066119422 | local_dp_per_tensor_eps1 | 11 |
| 0.87134039862506862 | -0.24260381606857645 | -0.29496784785322078 | baf | 10 | auc_roc | 0.60255456666417018 | -0.26878583196089861 | local_dp_per_tensor_eps1 | 11 |
| 0.15868126190169332 | -0.13635735525666479 | -0.14184714312568711 | baf | 10 | pr_auc | 0.019579012710517367 | -0.13910224919117595 | local_dp_per_tensor_eps1 | 11 |
| 0.22571169625881407 | -0.16345844140642951 | -0.18118514537951091 | baf | 10 | f1 | 0.053389902865843865 | -0.17232179339297021 | local_dp_per_tensor_eps3 | 11 |
| 0.87134039862506862 | -0.20125278441457917 | -0.25760027810048597 | baf | 10 | auc_roc | 0.6419138673675362 | -0.22942653125753257 | local_dp_per_tensor_eps3 | 11 |
| 0.15868126190169332 | -0.13060150213934474 | -0.13806996849160746 | baf | 10 | pr_auc | 0.024345526586217234 | -0.1343357353154761 | local_dp_per_tensor_eps3 | 11 |
| 0.22571169625881407 | -0.15314547821207738 | -0.17496066829096638 | baf | 10 | f1 | 0.061658623007292211 | -0.16405307325152188 | local_dp_per_tensor_eps10 | 11 |
| 0.87134039862506862 | -0.18418624734577493 | -0.22623254183722236 | baf | 10 | auc_roc | 0.66613100403357006 | -0.20520939459149865 | local_dp_per_tensor_eps10 | 11 |
| 0.15868126190169332 | -0.12459135700625046 | -0.13615117156731704 | baf | 10 | pr_auc | 0.028309997614909585 | -0.13037126428678375 | local_dp_per_tensor_eps10 | 11 |
| 0.22571169625881407 | -0.14331038117639402 | -0.16318329939285783 | baf | 10 | f1 | 0.072464855974188155 | -0.15324684028462593 | local_dp_per_tensor_eps30 | 11 |
| 0.87134039862506862 | -0.15513960351878908 | -0.18622576369172206 | baf | 10 | auc_roc | 0.70065771501981311 | -0.17068268360525557 | local_dp_per_tensor_eps30 | 11 |
| 0.15868126190169332 | -0.12289399438970151 | -0.1314366366857066 | baf | 10 | pr_auc | 0.031515946363989286 | -0.12716531553770405 | local_dp_per_tensor_eps30 | 11 |
| 0.22571169625881407 | -0.14115750934368437 | -0.1686564318294278 | baf | 10 | f1 | 0.070804725672257984 | -0.15490697058655609 | local_dp_per_tensor_eps100 | 11 |
| 0.87134039862506862 | -0.15339308119701234 | -0.20414029846862364 | baf | 10 | auc_roc | 0.69257370879225066 | -0.17876668983281799 | local_dp_per_tensor_eps100 | 11 |
| 0.15868126190169332 | -0.12017854047081418 | -0.13303438532401446 | baf | 10 | pr_auc | 0.032074799004279009 | -0.12660646289741431 | local_dp_per_tensor_eps100 | 11 |
| 0.22571169625881407 | -0.12405033194806504 | -0.14709243350072304 | baf | 10 | f1 | 0.09014031353442005 | -0.13557138272439404 | local_dp_per_tensor_eps1000 | 11 |
| 0.87134039862506862 | -0.13506312895166689 | -0.15573661224223051 | baf | 10 | auc_roc | 0.72594052802811992 | -0.1453998705969487 | local_dp_per_tensor_eps1000 | 11 |
| 0.15868126190169332 | -0.11049325233383391 | -0.12388865005043796 | baf | 10 | pr_auc | 0.041490310709557394 | -0.11719095119213593 | local_dp_per_tensor_eps1000 | 11 |
| 0.22571169625881407 | -0.12712559724313907 | -0.15327341087801324 | baf | 10 | f1 | 0.085512192198237927 | -0.14019950406057616 | local_dp_per_tensor_eps10000 | 11 |
| 0.87134039862506862 | -0.12752355759635106 | -0.14724025919847009 | baf | 10 | auc_roc | 0.73395849022765813 | -0.13738190839741057 | local_dp_per_tensor_eps10000 | 11 |
| 0.15868126190169332 | -0.11215216646442344 | -0.12695786436176829 | baf | 10 | pr_auc | 0.039126246488597444 | -0.11955501541309586 | local_dp_per_tensor_eps10000 | 11 |
| 0.22571169625881407 | -0.12696191845450044 | -0.15073457326636153 | baf | 10 | f1 | 0.086863450398383085 | -0.13884824586043099 | local_dp_per_tensor_eps1000000 | 11 |
| 0.87134039862506862 | -0.1312721523852122 | -0.14621200747766191 | baf | 10 | auc_roc | 0.73259831869363168 | -0.13874207993143706 | local_dp_per_tensor_eps1000000 | 11 |
| 0.15868126190169332 | -0.11192013249152528 | -0.12557902590561998 | baf | 10 | pr_auc | 0.039931682703120683 | -0.11874957919857264 | local_dp_per_tensor_eps1000000 | 11 |
| 0.22571169625881407 | -0.1129754900550804 | -0.142679281744515 | baf | 10 | f1 | 0.097884310359016402 | -0.1278273858997977 | b2b_global_c001_clip_only | 11 |
| 0.87134039862506862 | -0.11340227722338998 | -0.12672349846490841 | baf | 10 | auc_roc | 0.75127751078091942 | -0.12006288784414919 | b2b_global_c001_clip_only | 11 |
| 0.15868126190169332 | -0.10409266250802554 | -0.12234843295298997 | baf | 10 | pr_auc | 0.045460714171185572 | -0.11322054773050776 | b2b_global_c001_clip_only | 11 |
| 0.22571169625881407 | -0.014111063347429049 | -0.02092202382207704 | baf | 10 | f1 | 0.20819515267406105 | -0.017516543584753044 | b2b_global_c01_clip_only | 11 |
| 0.87134039862506862 | -0.016865032584247076 | -0.019254982907727224 | baf | 10 | auc_roc | 0.85328039087908147 | -0.01806000774598715 | b2b_global_c01_clip_only | 11 |
| 0.15868126190169332 | -0.021868258075461014 | -0.025348316142132225 | baf | 10 | pr_auc | 0.13507297479289671 | -0.02360828710879662 | b2b_global_c01_clip_only | 11 |
| 0.22571169625881407 | -0.17668966651408288 | -0.18892886467025236 | baf | 10 | f1 | 0.042902430666646459 | -0.18280926559216762 | b2b_global_c01_eps10 | 11 |
| 0.87134039862506862 | -0.24929893293318189 | -0.29215011936280627 | baf | 10 | auc_roc | 0.6006158724770746 | -0.27072452614799408 | b2b_global_c01_eps10 | 11 |
| 0.15868126190169332 | -0.13578348868374443 | -0.14141057870578086 | baf | 10 | pr_auc | 0.02008422820693068 | -0.13859703369476264 | b2b_global_c01_eps10 | 11 |
| 0.22571169625881407 | -0.06835051627595462 | -0.11269532667218576 | baf | 10 | f1 | 0.13518877478474389 | -0.090522921474070192 | b2b_global_c01_eps1000 | 11 |
| 0.87134039862506862 | -0.064918209940622362 | -0.094843027599834651 | baf | 10 | auc_roc | 0.79145977985484006 | -0.079880618770228506 | b2b_global_c01_eps1000 | 11 |
| 0.15868126190169332 | -0.070846292079737638 | -0.10267026100786789 | baf | 10 | pr_auc | 0.071922985357890559 | -0.086758276543802762 | b2b_global_c01_eps1000 | 11 |
| 0.22571169625881407 | -0.013866952572652648 | -0.019814796977912448 | baf | 10 | f1 | 0.20887082148353153 | -0.016840874775282549 | b2b_global_c01_eps1000000 | 11 |
| 0.87134039862506862 | -0.015723891661169099 | -0.019561547672742322 | baf | 10 | auc_roc | 0.85369767895811288 | -0.017642719666955711 | b2b_global_c01_eps1000000 | 11 |
| 0.15868126190169332 | -0.020869740204933106 | -0.028426642772498224 | baf | 10 | pr_auc | 0.13403307041297766 | -0.024648191488715665 | b2b_global_c01_eps1000000 | 11 |
| 0.22571169625881407 | 0.0029197927646958007 | -0.00237488703606568 | baf | 10 | f1 | 0.22598414912312917 | 0.00027245286431506024 | b2b_global_cmed_half_clip_only | 11 |
| 0.87134039862506862 | 0.0007036508641126097 | -0.0015486011977153165 | baf | 10 | auc_roc | 0.87091792345826735 | -0.00042247516680135345 | b2b_global_cmed_half_clip_only | 11 |
| 0.15868126190169332 | 0.00027404303903098132 | -0.001609717072397209 | baf | 10 | pr_auc | 0.15801342488501022 | -0.00066783701668311388 | b2b_global_cmed_half_clip_only | 11 |
| 0.22571169625881407 | -0.18075006471506869 | -0.19974431887678265 | baf | 10 | f1 | 0.035464504462888423 | -0.19024719179592567 | b2b_global_cmed_half_eps10 | 11 |
| 0.87134039862506862 | -0.28323963731624557 | -0.35378738682782862 | baf | 10 | auc_roc | 0.55282688655303158 | -0.3185135120720371 | b2b_global_cmed_half_eps10 | 11 |
| 0.15868126190169332 | -0.13822523663500924 | -0.14526270835979627 | baf | 10 | pr_auc | 0.016937289404290588 | -0.14174397249740275 | b2b_global_cmed_half_eps10 | 11 |
| 0.22571169625881407 | -0.074217404825431349 | -0.12010760508878565 | baf | 10 | f1 | 0.12854919130170558 | -0.0971625049571085 | b2b_global_cmed_half_eps1000 | 11 |
| 0.87134039862506862 | -0.085024447517600382 | -0.1088610065848164 | baf | 10 | auc_roc | 0.77439767157386019 | -0.096942727051208391 | b2b_global_cmed_half_eps1000 | 11 |
| 0.15868126190169332 | -0.078319008500511725 | -0.11010257057162424 | baf | 10 | pr_auc | 0.06447047236562535 | -0.094210789536067985 | b2b_global_cmed_half_eps1000 | 11 |
| 0.22571169625881407 | 0.00018103991314616548 | -0.0025781495673731949 | baf | 10 | f1 | 0.22451314143170056 | -0.0011985548271135147 | b2b_global_cmed_half_eps1000000 | 11 |
| 0.87134039862506862 | -0.001579181240018457 | -0.0051301559864920928 | baf | 10 | auc_roc | 0.86798573001181334 | -0.003354668613255275 | b2b_global_cmed_half_eps1000000 | 11 |
| 0.15868126190169332 | -0.0014989882871658928 | -0.0041756902498657367 | baf | 10 | pr_auc | 0.15584392263317751 | -0.002837339268515815 | b2b_global_cmed_half_eps1000000 | 11 |
| 0.22571169625881407 | 0.0017684948778282014 | -0.0023658569647731428 | baf | 10 | f1 | 0.2254130152153416 | -0.00029868104347247071 | b2b_global_cmed_clip_only | 11 |
| 0.87134039862506862 | 0.00094025842509969115 | -0.0013807855199626962 | baf | 10 | auc_roc | 0.8711201350776373 | -0.0002202635474315025 | b2b_global_cmed_clip_only | 11 |
| 0.15868126190169332 | 0.0005330777652316882 | -0.0013721811955214392 | baf | 10 | pr_auc | 0.15826171018654844 | -0.00041955171514487544 | b2b_global_cmed_clip_only | 11 |
| 0.22571169625881407 | -0.18464293168202095 | -0.19460640660721501 | baf | 10 | f1 | 0.036087027114196112 | -0.18962466914461798 | b2b_global_cmed_eps10 | 11 |
| 0.87134039862506862 | -0.27336900918164081 | -0.33446614618458864 | baf | 10 | auc_roc | 0.56742282094195395 | -0.30391757768311473 | b2b_global_cmed_eps10 | 11 |
| 0.15868126190169332 | -0.13874149127234689 | -0.14350330406366854 | baf | 10 | pr_auc | 0.017558864233685589 | -0.14112239766800772 | b2b_global_cmed_eps10 | 11 |
| 0.22571169625881407 | -0.10398145656096873 | -0.16015119548125639 | baf | 10 | f1 | 0.093645370237701503 | -0.13206632602111257 | b2b_global_cmed_eps1000 | 11 |
| 0.87134039862506862 | -0.1226909163801954 | -0.19557659496276603 | baf | 10 | auc_roc | 0.71220664295358793 | -0.15913375567148072 | b2b_global_cmed_eps1000 | 11 |
| 0.15868126190169332 | -0.1032636271201076 | -0.1303338520675747 | baf | 10 | pr_auc | 0.041882522307852173 | -0.11679873959384114 | b2b_global_cmed_eps1000 | 11 |
| 0.22571169625881407 | 0.0019795192523630869 | -0.012519186328017833 | baf | 10 | f1 | 0.22044186272098673 | -0.0052698335378273727 | b2b_global_cmed_eps1000000 | 11 |
| 0.87134039862506862 | -0.0016361914422220419 | -0.0063315166707767177 | baf | 10 | auc_roc | 0.86735654456856925 | -0.00398385405649938 | b2b_global_cmed_eps1000000 | 11 |
| 0.15868126190169332 | 0.00016941669789527665 | -0.01878699385085706 | baf | 10 | pr_auc | 0.14937247332521242 | -0.0093087885764808927 | b2b_global_cmed_eps1000000 | 11 |
| 0.34943956286275646 | 0 | 0 | ieee_cis | 10 | f1 | 0.34943956286275646 | 0 | baseline | 11 |
| 0.83812533978168957 | 0 | 0 | ieee_cis | 10 | auc_roc | 0.83812533978168957 | 0 | baseline | 11 |
| 0.32748308162421763 | 0 | 0 | ieee_cis | 10 | pr_auc | 0.32748308162421763 | 0 | baseline | 11 |
| 0.34943956286275646 | 0.0054371576737062225 | -0.012566503799712041 | ieee_cis | 10 | f1 | 0.34587488979975362 | -0.0035646730630029089 | dna_v1_conservative | 11 |
| 0.83812533978168957 | 0.00073075496022543315 | -0.0047615889713255926 | ieee_cis | 10 | auc_roc | 0.8361099227761396 | -0.0020154170055500797 | dna_v1_conservative | 11 |
| 0.32748308162421763 | 0.0031162612256062329 | -0.014176347713234946 | ieee_cis | 10 | pr_auc | 0.32195303838040323 | -0.0055300432438143568 | dna_v1_conservative | 11 |
| 0.34943956286275646 | -0.0051923491806149715 | -0.020610727971500334 | ieee_cis | 10 | f1 | 0.33653802428669882 | -0.012901538576057652 | dna_v2_0p95 | 11 |
| 0.83812533978168957 | -0.0004506936357498248 | -0.0031561313635782382 | ieee_cis | 10 | auc_roc | 0.83632192728202559 | -0.0018034124996640316 | dna_v2_0p95 | 11 |
| 0.32748308162421763 | -0.00080087547610759469 | -0.013742666197952932 | ieee_cis | 10 | pr_auc | 0.32021131078718734 | -0.007271770837030264 | dna_v2_0p95 | 11 |
| 0.34943956286275646 | -0.18435060353109503 | -0.20491590649706784 | ieee_cis | 10 | f1 | 0.15480630784867505 | -0.19463325501408144 | local_dp_global_eps1 | 11 |
| 0.83812533978168957 | -0.13620945742100687 | -0.21835962001515274 | ieee_cis | 10 | auc_roc | 0.66084080106360987 | -0.17728453871807981 | local_dp_global_eps1 | 11 |
| 0.32748308162421763 | -0.2452724903393351 | -0.26355835648774451 | ieee_cis | 10 | pr_auc | 0.073067658210677816 | -0.25441542341353979 | local_dp_global_eps1 | 11 |
| 0.34943956286275646 | -0.18164308247344976 | -0.19769991154174685 | ieee_cis | 10 | f1 | 0.15976806585515821 | -0.18967149700759831 | local_dp_global_eps3 | 11 |
| 0.83812533978168957 | -0.10908202169185913 | -0.15752706802120628 | ieee_cis | 10 | auc_roc | 0.70482079492515692 | -0.1333045448565327 | local_dp_global_eps3 | 11 |
| 0.32748308162421763 | -0.24368088803746885 | -0.25750144948893544 | ieee_cis | 10 | pr_auc | 0.076891912861015474 | -0.25059116876320214 | local_dp_global_eps3 | 11 |
| 0.34943956286275646 | -0.17470983763405912 | -0.19170460768707492 | ieee_cis | 10 | f1 | 0.16623234020218947 | -0.18320722266056702 | local_dp_global_eps10 | 11 |
| 0.83812533978168957 | -0.081015527615238683 | -0.096004279968399456 | ieee_cis | 10 | auc_roc | 0.74961543598987046 | -0.08850990379181907 | local_dp_global_eps10 | 11 |
| 0.32748308162421763 | -0.23824361188198273 | -0.2494760404149392 | ieee_cis | 10 | pr_auc | 0.08362325547575665 | -0.24385982614846097 | local_dp_global_eps10 | 11 |
| 0.34943956286275646 | -0.17795409961648639 | -0.19446048197183707 | ieee_cis | 10 | f1 | 0.16323227206859475 | -0.18620729079416173 | local_dp_global_eps30 | 11 |
| 0.83812533978168957 | -0.082078089812708352 | -0.094954277686527408 | ieee_cis | 10 | auc_roc | 0.74960915603207179 | -0.08851618374961788 | local_dp_global_eps30 | 11 |
| 0.32748308162421763 | -0.23936707084450237 | -0.25418205783892434 | ieee_cis | 10 | pr_auc | 0.080708517282504236 | -0.24677456434171335 | local_dp_global_eps30 | 11 |
| 0.34943956286275646 | -0.17845203811073113 | -0.19601929205275775 | ieee_cis | 10 | f1 | 0.16220389778101205 | -0.18723566508174444 | local_dp_global_eps100 | 11 |
| 0.83812533978168957 | -0.080131382295399664 | -0.09570843237195463 | ieee_cis | 10 | auc_roc | 0.75020543244801263 | -0.087919907333677147 | local_dp_global_eps100 | 11 |
| 0.32748308162421763 | -0.23927261932315017 | -0.2563911846078934 | ieee_cis | 10 | pr_auc | 0.07965117965869585 | -0.24783190196552177 | local_dp_global_eps100 | 11 |
| 0.34943956286275646 | -0.17583765768737925 | -0.19288790058235344 | ieee_cis | 10 | f1 | 0.16507678372789014 | -0.18436277913486634 | local_dp_global_eps1000 | 11 |
| 0.83812533978168957 | -0.075561371055644685 | -0.087355610897266997 | ieee_cis | 10 | auc_roc | 0.75666684880523372 | -0.081458490976455841 | local_dp_global_eps1000 | 11 |
| 0.32748308162421763 | -0.23894296598592268 | -0.25368123922586305 | ieee_cis | 10 | pr_auc | 0.081170979018324735 | -0.24631210260589287 | local_dp_global_eps1000 | 11 |
| 0.34943956286275646 | -0.17843645122579338 | -0.19380010752405663 | ieee_cis | 10 | f1 | 0.16332128348783145 | -0.18611827937492501 | local_dp_global_eps10000 | 11 |
| 0.83812533978168957 | -0.0765727001605601 | -0.099037534382053885 | ieee_cis | 10 | auc_roc | 0.75032022251038266 | -0.087805117271306993 | local_dp_global_eps10000 | 11 |
| 0.32748308162421763 | -0.24164009367655845 | -0.25472340647667541 | ieee_cis | 10 | pr_auc | 0.07930133154760069 | -0.24818175007661691 | local_dp_global_eps10000 | 11 |
| 0.34943956286275646 | -0.17930531403177993 | -0.19264935062883673 | ieee_cis | 10 | f1 | 0.16346223053244816 | -0.18597733233030833 | local_dp_global_eps1000000 | 11 |
| 0.83812533978168957 | -0.074420241008784957 | -0.10598846626750021 | ieee_cis | 10 | auc_roc | 0.74792098614354707 | -0.090204353638142581 | local_dp_global_eps1000000 | 11 |
| 0.32748308162421763 | -0.24257275120020858 | -0.25452092597242293 | ieee_cis | 10 | pr_auc | 0.078936243037901796 | -0.24854683858631577 | local_dp_global_eps1000000 | 11 |
| 0.34943956286275646 | -0.18058679226533716 | -0.20755388360060947 | ieee_cis | 10 | f1 | 0.15536922492978317 | -0.19407033793297332 | local_dp_per_tensor_eps1 | 11 |
| 0.83812533978168957 | -0.12613650245084887 | -0.19260040934248954 | ieee_cis | 10 | auc_roc | 0.67875688388502042 | -0.1593684558966692 | local_dp_per_tensor_eps1 | 11 |
| 0.32748308162421763 | -0.24421780296403928 | -0.26387128796136627 | ieee_cis | 10 | pr_auc | 0.073438536161514828 | -0.25404454546270278 | local_dp_per_tensor_eps1 | 11 |
| 0.34943956286275646 | -0.17773670854770887 | -0.19189454972160255 | ieee_cis | 10 | f1 | 0.16462393372810077 | -0.18481562913465571 | local_dp_per_tensor_eps3 | 11 |
| 0.83812533978168957 | -0.081692954090429482 | -0.11550461071523996 | ieee_cis | 10 | auc_roc | 0.73952655737885498 | -0.09859878240283472 | local_dp_per_tensor_eps3 | 11 |
| 0.32748308162421763 | -0.23912292034327537 | -0.25154897127155157 | ieee_cis | 10 | pr_auc | 0.082147135816804148 | -0.24533594580741347 | local_dp_per_tensor_eps3 | 11 |
| 0.34943956286275646 | -0.17734329786982225 | -0.19627827971229869 | ieee_cis | 10 | f1 | 0.16262877407169601 | -0.18681078879106047 | local_dp_per_tensor_eps10 | 11 |
| 0.83812533978168957 | -0.079707812683393664 | -0.12996883112762025 | ieee_cis | 10 | auc_roc | 0.73328701787618267 | -0.10483832190550696 | local_dp_per_tensor_eps10 | 11 |
| 0.32748308162421763 | -0.23880828600183132 | -0.25400660787128154 | ieee_cis | 10 | pr_auc | 0.081075634687661158 | -0.24640744693655645 | local_dp_per_tensor_eps10 | 11 |
| 0.34943956286275646 | -0.17795889466868337 | -0.19394980914964499 | ieee_cis | 10 | f1 | 0.16348521095359234 | -0.18595435190916418 | local_dp_per_tensor_eps30 | 11 |
| 0.83812533978168957 | -0.081284924328104108 | -0.098747082702539013 | ieee_cis | 10 | auc_roc | 0.74810933626636811 | -0.09001600351532156 | local_dp_per_tensor_eps30 | 11 |
| 0.32748308162421763 | -0.24074857442095074 | -0.25279994421296298 | ieee_cis | 10 | pr_auc | 0.080708822307260744 | -0.24677425931695687 | local_dp_per_tensor_eps30 | 11 |
| 0.34943956286275646 | -0.17754145471473162 | -0.1947437185652455 | ieee_cis | 10 | f1 | 0.16329697622276793 | -0.18614258663998856 | local_dp_per_tensor_eps100 | 11 |
| 0.83812533978168957 | -0.085505616591260825 | -0.097921788885575289 | ieee_cis | 10 | auc_roc | 0.74641163704327163 | -0.091713702738418057 | local_dp_per_tensor_eps100 | 11 |
| 0.32748308162421763 | -0.23948943208640014 | -0.25590666369072357 | ieee_cis | 10 | pr_auc | 0.079785033735655694 | -0.24769804788856187 | local_dp_per_tensor_eps100 | 11 |
| 0.34943956286275646 | -0.17981422803590824 | -0.19500666802691385 | ieee_cis | 10 | f1 | 0.16202911483134544 | -0.18741044803141105 | local_dp_per_tensor_eps1000 | 11 |
| 0.83812533978168957 | -0.08023304909460445 | -0.093061810326303937 | ieee_cis | 10 | auc_roc | 0.75147791007123554 | -0.086647429710454194 | local_dp_per_tensor_eps1000 | 11 |
| 0.32748308162421763 | -0.2418515070918667 | -0.25418435366601866 | ieee_cis | 10 | pr_auc | 0.079465151245274965 | -0.24801793037894268 | local_dp_per_tensor_eps1000 | 11 |
| 0.34943956286275646 | -0.17902515582254463 | -0.19343557772114073 | ieee_cis | 10 | f1 | 0.16320919609091381 | -0.18623036677184268 | local_dp_per_tensor_eps10000 | 11 |
| 0.83812533978168957 | -0.07933111945134623 | -0.097036713013427159 | ieee_cis | 10 | auc_roc | 0.74994142354930304 | -0.088183916232386694 | local_dp_per_tensor_eps10000 | 11 |
| 0.32748308162421763 | -0.24216079345206276 | -0.25481900885727565 | ieee_cis | 10 | pr_auc | 0.07899318046954841 | -0.24848990115466921 | local_dp_per_tensor_eps10000 | 11 |
| 0.34943956286275646 | -0.17997585014858614 | -0.1933470648011566 | ieee_cis | 10 | f1 | 0.16277810538788515 | -0.18666145747487137 | local_dp_per_tensor_eps1000000 | 11 |
| 0.83812533978168957 | -0.077611637903460054 | -0.10468001271292206 | ieee_cis | 10 | auc_roc | 0.74697951447349853 | -0.091145825308191056 | local_dp_per_tensor_eps1000000 | 11 |
| 0.32748308162421763 | -0.2416798982281349 | -0.25516389888547586 | ieee_cis | 10 | pr_auc | 0.079061183067412236 | -0.24842189855680538 | local_dp_per_tensor_eps1000000 | 11 |
| 0.34943956286275646 | -0.17827651149553286 | -0.19301384614374012 | ieee_cis | 10 | f1 | 0.16379438404311999 | -0.18564517881963649 | b2b_global_c001_clip_only | 11 |
| 0.83812533978168957 | -0.074405045676992812 | -0.09702720007651873 | ieee_cis | 10 | auc_roc | 0.7524092169049339 | -0.085716122876755771 | b2b_global_c001_clip_only | 11 |
| 0.32748308162421763 | -0.24190761132493147 | -0.25480688028005205 | ieee_cis | 10 | pr_auc | 0.079125835821725846 | -0.24835724580249174 | b2b_global_c001_clip_only | 11 |
| 0.34943956286275646 | -0.15222273793010632 | -0.17402024392086207 | ieee_cis | 10 | f1 | 0.18631807193727229 | -0.1631214909254842 | b2b_global_c01_clip_only | 11 |
| 0.83812533978168957 | -0.051283802918883521 | -0.06059769084512303 | ieee_cis | 10 | auc_roc | 0.78218459289968645 | -0.055940746882003275 | b2b_global_c01_clip_only | 11 |
| 0.32748308162421763 | -0.22141850998513929 | -0.23768667908966618 | ieee_cis | 10 | pr_auc | 0.097930487086814871 | -0.22955259453740273 | b2b_global_c01_clip_only | 11 |
| 0.34943956286275646 | -0.17951625240724858 | -0.1975585604514152 | ieee_cis | 10 | f1 | 0.16090215643342456 | -0.18853740642933189 | b2b_global_c01_eps10 | 11 |
| 0.83812533978168957 | -0.1120130344215607 | -0.18339556800741946 | ieee_cis | 10 | auc_roc | 0.69042103856719961 | -0.14770430121449007 | b2b_global_c01_eps10 | 11 |
| 0.32748308162421763 | -0.24345094939600143 | -0.25686985649675842 | ieee_cis | 10 | pr_auc | 0.077322678677837681 | -0.25016040294637992 | b2b_global_c01_eps10 | 11 |
| 0.34943956286275646 | -0.17364416238537289 | -0.19328141816475386 | ieee_cis | 10 | f1 | 0.16597677258769308 | -0.18346279027506338 | b2b_global_c01_eps1000 | 11 |
| 0.83812533978168957 | -0.071362924815907267 | -0.080373152125183936 | ieee_cis | 10 | auc_roc | 0.76225730131114411 | -0.075868038470545601 | b2b_global_c01_eps1000 | 11 |
| 0.32748308162421763 | -0.23543301982334375 | -0.25115422575157892 | ieee_cis | 10 | pr_auc | 0.084189458836756256 | -0.24329362278746133 | b2b_global_c01_eps1000 | 11 |
| 0.34943956286275646 | -0.1536950730715369 | -0.17674811213511032 | ieee_cis | 10 | f1 | 0.1842179702594329 | -0.16522159260332361 | b2b_global_c01_eps1000000 | 11 |
| 0.83812533978168957 | -0.052285566585699897 | -0.062685134976051396 | ieee_cis | 10 | auc_roc | 0.78063998900081399 | -0.057485350780875646 | b2b_global_c01_eps1000000 | 11 |
| 0.32748308162421763 | -0.22177629844324814 | -0.23824597625653876 | ieee_cis | 10 | pr_auc | 0.097471944274324152 | -0.23001113734989345 | b2b_global_c01_eps1000000 | 11 |
| 0.34943956286275646 | -0.011459837927022681 | -0.025288801267989394 | ieee_cis | 10 | f1 | 0.33106524326525044 | -0.018374319597506038 | b2b_global_cmed_half_clip_only | 11 |
| 0.83812533978168957 | -0.0036016943253263617 | -0.0083853523531959885 | ieee_cis | 10 | auc_roc | 0.8321318164424284 | -0.0059935233392611753 | b2b_global_cmed_half_clip_only | 11 |
| 0.32748308162421763 | -0.019476345639976047 | -0.031670996036952864 | ieee_cis | 10 | pr_auc | 0.30190941078575312 | -0.025573670838464457 | b2b_global_cmed_half_clip_only | 11 |
| 0.34943956286275646 | -0.18150091938024754 | -0.2042548375626137 | ieee_cis | 10 | f1 | 0.15656168439132587 | -0.19287787847143062 | b2b_global_cmed_half_eps10 | 11 |
| 0.83812533978168957 | -0.14837251757133438 | -0.23495786376594838 | ieee_cis | 10 | auc_roc | 0.64646014911304828 | -0.19166519066864138 | b2b_global_cmed_half_eps10 | 11 |
| 0.32748308162421763 | -0.24670604292641962 | -0.2670170093222205 | ieee_cis | 10 | pr_auc | 0.070621555499897529 | -0.25686152612432006 | b2b_global_cmed_half_eps10 | 11 |
| 0.34943956286275646 | -0.17626122711050923 | -0.19569103986135766 | ieee_cis | 10 | f1 | 0.16346342937682307 | -0.18597613348593345 | b2b_global_cmed_half_eps1000 | 11 |
| 0.83812533978168957 | -0.077753841357278489 | -0.12368119257729031 | ieee_cis | 10 | auc_roc | 0.73740782281440531 | -0.1007175169672844 | b2b_global_cmed_half_eps1000 | 11 |
| 0.32748308162421763 | -0.23794825281852064 | -0.25504183462411567 | ieee_cis | 10 | pr_auc | 0.080988037902899448 | -0.24649504372131814 | b2b_global_cmed_half_eps1000 | 11 |
| 0.34943956286275646 | -0.080268792363791322 | -0.13077671866994431 | ieee_cis | 10 | f1 | 0.24391680734588866 | -0.10552275551686781 | b2b_global_cmed_half_eps1000000 | 11 |
| 0.83812533978168957 | -0.020408006432508098 | -0.030907159814227675 | ieee_cis | 10 | auc_roc | 0.81246775665832172 | -0.025657583123367887 | b2b_global_cmed_half_eps1000000 | 11 |
| 0.32748308162421763 | -0.086290006807371283 | -0.142597306870514 | ieee_cis | 10 | pr_auc | 0.21303942478527496 | -0.11444365683894264 | b2b_global_cmed_half_eps1000000 | 11 |
| 0.34943956286275646 | 0.0046188897070912754 | -0.015608740408675499 | ieee_cis | 10 | f1 | 0.34394463751196436 | -0.0054949253507921118 | b2b_global_cmed_clip_only | 11 |
| 0.83812533978168957 | 0.0014379865800497073 | -0.0029608362625905419 | ieee_cis | 10 | auc_roc | 0.83736391494041917 | -0.00076142484127041739 | b2b_global_cmed_clip_only | 11 |
| 0.32748308162421763 | 0.0032908392181346183 | -0.012461653431409422 | ieee_cis | 10 | pr_auc | 0.32289767451758017 | -0.0045854071066374017 | b2b_global_cmed_clip_only | 11 |
| 0.34943956286275646 | -0.18651671191804059 | -0.22813853662310807 | ieee_cis | 10 | f1 | 0.14211193859218213 | -0.20732762427057433 | b2b_global_cmed_eps10 | 11 |
| 0.83812533978168957 | -0.17405649544459534 | -0.29069460227133853 | ieee_cis | 10 | auc_roc | 0.60574979092372272 | -0.23237554885796693 | b2b_global_cmed_eps10 | 11 |
| 0.32748308162421763 | -0.25345373608097882 | -0.27522089912376396 | ieee_cis | 10 | pr_auc | 0.063145764021846212 | -0.26433731760237139 | b2b_global_cmed_eps10 | 11 |
| 0.34943956286275646 | -0.17953989917946395 | -0.20204588092321812 | ieee_cis | 10 | f1 | 0.15864667281141545 | -0.19079289005134104 | b2b_global_cmed_eps1000 | 11 |
| 0.83812533978168957 | -0.11956463726139563 | -0.1968867220275535 | ieee_cis | 10 | auc_roc | 0.67989966013721503 | -0.15822567964447457 | b2b_global_cmed_eps1000 | 11 |
| 0.32748308162421763 | -0.2430727240908459 | -0.26119154443103099 | ieee_cis | 10 | pr_auc | 0.07535094736327913 | -0.25213213426093845 | b2b_global_cmed_eps1000 | 11 |
| 0.34943956286275646 | -0.060483207436467734 | -0.11306482309287119 | ieee_cis | 10 | f1 | 0.26266554759808702 | -0.086774015264669463 | b2b_global_cmed_eps1000000 | 11 |
| 0.83812533978168957 | -0.016258215998426841 | -0.032477065295988511 | ieee_cis | 10 | auc_roc | 0.81375769913448193 | -0.024367640647207676 | b2b_global_cmed_eps1000000 | 11 |
| 0.32748308162421763 | -0.075683006136761793 | -0.12862066787760831 | ieee_cis | 10 | pr_auc | 0.22533124461703255 | -0.10215183700718505 | b2b_global_cmed_eps1000000 | 11 |
| 0.7832253489110883 | 0 | 0 | paysim | 10 | f1 | 0.7832253489110883 | 0 | baseline | 11 |
| 0.99759361058029361 | 0 | 0 | paysim | 10 | auc_roc | 0.99759361058029361 | 0 | baseline | 11 |
| 0.80333529003778459 | 0 | 0 | paysim | 10 | pr_auc | 0.80333529003778459 | 0 | baseline | 11 |
| 0.7832253489110883 | 0.0042718119045868181 | -0.0075086710441081778 | paysim | 10 | f1 | 0.78160691934132764 | -0.00161842956976068 | dna_v1_conservative | 11 |
| 0.99759361058029361 | 0.00014681551472444 | -5.9681735593082756e-05 | paysim | 10 | auc_roc | 0.99763717746985925 | 4.3566889565678629e-05 | dna_v1_conservative | 11 |
| 0.80333529003778459 | 0.0068194701817754673 | -0.0018035079792296338 | paysim | 10 | pr_auc | 0.8058432711390574 | 0.002507981101272917 | dna_v1_conservative | 11 |
| 0.7832253489110883 | 0.0061662713208122536 | -0.0076738774375483343 | paysim | 10 | f1 | 0.78247154585272038 | -0.00075380305836804005 | dna_v2_0p95 | 11 |
| 0.99759361058029361 | 0.00031942378587027456 | -8.3853280232729462e-05 | paysim | 10 | auc_roc | 0.99771139583311241 | 0.00011778525281877253 | dna_v2_0p95 | 11 |
| 0.80333529003778459 | 0.012477451226186933 | -0.00037848823612515736 | paysim | 10 | pr_auc | 0.80938477153281552 | 0.0060494814950308876 | dna_v2_0p95 | 11 |
| 0.7832253489110883 | -0.21132514765096777 | -0.4114437579128416 | paysim | 10 | f1 | 0.47184089612918362 | -0.31138445278190469 | local_dp_global_eps1 | 11 |
| 0.99759361058029361 | -0.016796153267382687 | -0.094961554226964645 | paysim | 10 | auc_roc | 0.94171475683311989 | -0.055878853747173669 | local_dp_global_eps1 | 11 |
| 0.80333529003778459 | -0.28677571414277675 | -0.48518329919774666 | paysim | 10 | pr_auc | 0.41735578336752288 | -0.38597950667026171 | local_dp_global_eps1 | 11 |
| 0.7832253489110883 | -0.17039647423362272 | -0.39525490742857605 | paysim | 10 | f1 | 0.50039965807998898 | -0.28282569083109937 | local_dp_global_eps3 | 11 |
| 0.99759361058029361 | 0.027266499799654051 | -0.17189007148925717 | paysim | 10 | auc_roc | 0.92528182473549203 | -0.072311785844801565 | local_dp_global_eps3 | 11 |
| 0.80333529003778459 | -0.24315436207383634 | -0.45073869015660051 | paysim | 10 | pr_auc | 0.45638876392256617 | -0.34694652611521842 | local_dp_global_eps3 | 11 |
| 0.7832253489110883 | -0.21241873340431927 | -0.35570753113409725 | paysim | 10 | f1 | 0.49916221664188004 | -0.28406313226920826 | local_dp_global_eps10 | 11 |
| 0.99759361058029361 | -0.019453450120678537 | -0.040699194947437664 | paysim | 10 | auc_roc | 0.96751728804623549 | -0.030076322534058099 | local_dp_global_eps10 | 11 |
| 0.80333529003778459 | -0.26666778391954987 | -0.43338037832465898 | paysim | 10 | pr_auc | 0.45331120891568016 | -0.35002408112210442 | local_dp_global_eps10 | 11 |
| 0.7832253489110883 | -0.16534479382396516 | -0.27807648250055222 | paysim | 10 | f1 | 0.56151471074882964 | -0.22171063816225869 | local_dp_global_eps30 | 11 |
| 0.99759361058029361 | -0.018540788213369234 | -0.035632708470267838 | paysim | 10 | auc_roc | 0.97050686223847504 | -0.027086748341818536 | local_dp_global_eps30 | 11 |
| 0.80333529003778459 | -0.21631959812029772 | -0.35940742104069118 | paysim | 10 | pr_auc | 0.51547178045729014 | -0.28786350958049445 | local_dp_global_eps30 | 11 |
| 0.7832253489110883 | -0.17337758804336365 | -0.20358026054277281 | paysim | 10 | f1 | 0.59474642461802008 | -0.18847892429306823 | local_dp_global_eps100 | 11 |
| 0.99759361058029361 | -0.024912907267221805 | -0.029764953929653566 | paysim | 10 | auc_roc | 0.97025467998185599 | -0.027338930598437686 | local_dp_global_eps100 | 11 |
| 0.80333529003778459 | -0.23219194981350605 | -0.26750572447747883 | paysim | 10 | pr_auc | 0.55348645289229215 | -0.24984883714549244 | local_dp_global_eps100 | 11 |
| 0.7832253489110883 | -0.17240355437154228 | -0.19093180197110662 | paysim | 10 | f1 | 0.60155767073976385 | -0.18166767817132445 | local_dp_global_eps1000 | 11 |
| 0.99759361058029361 | -0.019287237669839474 | -0.023773996076214873 | paysim | 10 | auc_roc | 0.97606299370726646 | -0.021530616873027174 | local_dp_global_eps1000 | 11 |
| 0.80333529003778459 | -0.22065773359836444 | -0.24224158459571982 | paysim | 10 | pr_auc | 0.57188563094074241 | -0.23144965909704213 | local_dp_global_eps1000 | 11 |
| 0.7832253489110883 | -0.17440943860442951 | -0.18311245037738336 | paysim | 10 | f1 | 0.6044644044201819 | -0.17876094449090643 | local_dp_global_eps10000 | 11 |
| 0.99759361058029361 | -0.018684586762628061 | -0.021637715594210467 | paysim | 10 | auc_roc | 0.9774324594018744 | -0.020161151178419264 | local_dp_global_eps10000 | 11 |
| 0.80333529003778459 | -0.22226598318700333 | -0.24081624224534023 | paysim | 10 | pr_auc | 0.57179417732161275 | -0.23154111271617178 | local_dp_global_eps10000 | 11 |
| 0.7832253489110883 | -0.17458455695994538 | -0.18391348932367027 | paysim | 10 | f1 | 0.60397632576928051 | -0.17924902314180782 | local_dp_global_eps1000000 | 11 |
| 0.99759361058029361 | -0.018555139228781242 | -0.022120496541503352 | paysim | 10 | auc_roc | 0.97725579269515139 | -0.020337817885142297 | local_dp_global_eps1000000 | 11 |
| 0.80333529003778459 | -0.22169734046868686 | -0.23977790714606054 | paysim | 10 | pr_auc | 0.57259766623041086 | -0.2307376238073737 | local_dp_global_eps1000000 | 11 |
| 0.7832253489110883 | -0.19853158346528757 | -0.26992622841124136 | paysim | 10 | f1 | 0.54899644297282391 | -0.23422890593826445 | local_dp_per_tensor_eps1 | 11 |
| 0.99759361058029361 | -0.019202593839712352 | -0.028460742933553017 | paysim | 10 | auc_roc | 0.97376194219366097 | -0.023831668386632684 | local_dp_per_tensor_eps1 | 11 |
| 0.80333529003778459 | -0.26929333485297591 | -0.34791989622843467 | paysim | 10 | pr_auc | 0.49472867449707925 | -0.30860661554070529 | local_dp_per_tensor_eps1 | 11 |
| 0.7832253489110883 | -0.18839695515253563 | -0.29688284969434087 | paysim | 10 | f1 | 0.54058544648765006 | -0.24263990242343825 | local_dp_per_tensor_eps3 | 11 |
| 0.99759361058029361 | -0.021504326835647691 | -0.032638580705119956 | paysim | 10 | auc_roc | 0.97052215680990983 | -0.027071453770383822 | local_dp_per_tensor_eps3 | 11 |
| 0.80333529003778459 | -0.26241585819339303 | -0.38488780434901199 | paysim | 10 | pr_auc | 0.47968345876658208 | -0.32365183127120251 | local_dp_per_tensor_eps3 | 11 |
| 0.7832253489110883 | -0.21536349254693027 | -0.34724814355575667 | paysim | 10 | f1 | 0.50191953085974494 | -0.28130581805134347 | local_dp_per_tensor_eps10 | 11 |
| 0.99759361058029361 | -0.019391862088791838 | -0.036387655151591335 | paysim | 10 | auc_roc | 0.969703851960102 | -0.027889758620191586 | local_dp_per_tensor_eps10 | 11 |
| 0.80333529003778459 | -0.2550205752895891 | -0.40278166210625166 | paysim | 10 | pr_auc | 0.47443417133986415 | -0.32890111869792038 | local_dp_per_tensor_eps10 | 11 |
| 0.7832253489110883 | -0.20993750160715108 | -0.3262476932287891 | paysim | 10 | f1 | 0.51513275149311832 | -0.26809259741797009 | local_dp_per_tensor_eps30 | 11 |
| 0.99759361058029361 | -0.023639137690795675 | -0.044016867708153939 | paysim | 10 | auc_roc | 0.96376560788081878 | -0.033828002699474807 | local_dp_per_tensor_eps30 | 11 |
| 0.80333529003778459 | -0.26232207667657947 | -0.40284486237800732 | paysim | 10 | pr_auc | 0.47075182051049114 | -0.3325834695272934 | local_dp_per_tensor_eps30 | 11 |
| 0.7832253489110883 | -0.18711558811153106 | -0.29285292175453476 | paysim | 10 | f1 | 0.54324109397805542 | -0.23998425493303291 | local_dp_per_tensor_eps100 | 11 |
| 0.99759361058029361 | -0.027810595461556273 | -0.041145905994031742 | paysim | 10 | auc_roc | 0.96311535985249963 | -0.034478250727794008 | local_dp_per_tensor_eps100 | 11 |
| 0.80333529003778459 | -0.24786777997631373 | -0.37501700718727993 | paysim | 10 | pr_auc | 0.49189289645598772 | -0.31144239358179682 | local_dp_per_tensor_eps100 | 11 |
| 0.7832253489110883 | -0.17590987015008622 | -0.21017988568021692 | paysim | 10 | f1 | 0.59018047099593673 | -0.19304487791515157 | local_dp_per_tensor_eps1000 | 11 |
| 0.99759361058029361 | -0.019504793403574777 | -0.025126224349048903 | paysim | 10 | auc_roc | 0.97527810170398188 | -0.02231550887631184 | local_dp_per_tensor_eps1000 | 11 |
| 0.80333529003778459 | -0.22932831586066177 | -0.25744033165358271 | paysim | 10 | pr_auc | 0.55995096628066232 | -0.24338432375712224 | local_dp_per_tensor_eps1000 | 11 |
| 0.7832253489110883 | -0.17555373984983902 | -0.195235206711242 | paysim | 10 | f1 | 0.59783087563054782 | -0.18539447328054051 | local_dp_per_tensor_eps10000 | 11 |
| 0.99759361058029361 | -0.019972597712638221 | -0.023990908727230177 | paysim | 10 | auc_roc | 0.97561185736035938 | -0.021981753219934199 | local_dp_per_tensor_eps10000 | 11 |
| 0.80333529003778459 | -0.22677683237407048 | -0.24715696441785145 | paysim | 10 | pr_auc | 0.56636839164182351 | -0.23696689839596097 | local_dp_per_tensor_eps10000 | 11 |
| 0.7832253489110883 | -0.18003366764813461 | -0.19373613659894662 | paysim | 10 | f1 | 0.59634044678754772 | -0.18688490212354061 | local_dp_per_tensor_eps1000000 | 11 |
| 0.99759361058029361 | -0.020341527487292635 | -0.023684874275938454 | paysim | 10 | auc_roc | 0.97558040969867799 | -0.022013200881615545 | local_dp_per_tensor_eps1000000 | 11 |
| 0.80333529003778459 | -0.22920610328288529 | -0.24864088466625148 | paysim | 10 | pr_auc | 0.56441179606321612 | -0.23892349397456839 | local_dp_per_tensor_eps1000000 | 11 |
| 0.7832253489110883 | -0.17242106705128321 | -0.18390595669122087 | paysim | 10 | f1 | 0.60506183703983629 | -0.17816351187125204 | b2b_global_c001_clip_only | 11 |
| 0.99759361058029361 | -0.018196349755512976 | -0.022182730096114395 | paysim | 10 | auc_roc | 0.97740407065447987 | -0.020189539925813686 | b2b_global_c001_clip_only | 11 |
| 0.80333529003778459 | -0.2205199209339033 | -0.2404203361606492 | paysim | 10 | pr_auc | 0.57286516149050826 | -0.23047012854727625 | b2b_global_c001_clip_only | 11 |
| 0.7832253489110883 | -0.12278981206253795 | -0.15331440800694909 | paysim | 10 | f1 | 0.64517323887634481 | -0.13805211003474352 | b2b_global_c01_clip_only | 11 |
| 0.99759361058029361 | -0.0043042621227193925 | -0.0055033768116092242 | paysim | 10 | auc_roc | 0.99268979111312927 | -0.0049038194671643083 | b2b_global_c01_clip_only | 11 |
| 0.80333529003778459 | -0.10868161263212277 | -0.13906992088413878 | paysim | 10 | pr_auc | 0.67945952327965375 | -0.12387576675813078 | b2b_global_c01_clip_only | 11 |
| 0.7832253489110883 | -0.19408685840118656 | -0.27229861333928024 | paysim | 10 | f1 | 0.55003261304085493 | -0.2331927358702334 | b2b_global_c01_eps10 | 11 |
| 0.99759361058029361 | -0.03276475228980847 | -0.092384881952834813 | paysim | 10 | auc_roc | 0.93501879345897199 | -0.062574817121321638 | b2b_global_c01_eps10 | 11 |
| 0.80333529003778459 | -0.26174327541357811 | -0.36946556303407108 | paysim | 10 | pr_auc | 0.48773087081395999 | -0.31560441922382459 | b2b_global_c01_eps10 | 11 |
| 0.7832253489110883 | -0.15912705915187486 | -0.1789313549072396 | paysim | 10 | f1 | 0.61419614188153104 | -0.16902920702955723 | b2b_global_c01_eps1000 | 11 |
| 0.99759361058029361 | -0.0097655052399764838 | -0.012645420357920774 | paysim | 10 | auc_roc | 0.98638814778134498 | -0.011205462798948629 | b2b_global_c01_eps1000 | 11 |
| 0.80333529003778459 | -0.19286330399078161 | -0.22014380429570701 | paysim | 10 | pr_auc | 0.59683173589454031 | -0.20650355414324431 | b2b_global_c01_eps1000 | 11 |
| 0.7832253489110883 | -0.11491274944140405 | -0.14872403557380814 | paysim | 10 | f1 | 0.65140695640348223 | -0.1318183925076061 | b2b_global_c01_eps1000000 | 11 |
| 0.99759361058029361 | -0.0041485548738428761 | -0.0053506970469700864 | paysim | 10 | auc_roc | 0.9928439846198871 | -0.0047496259604064813 | b2b_global_c01_eps1000000 | 11 |
| 0.80333529003778459 | -0.10756067605244155 | -0.13895716828921181 | paysim | 10 | pr_auc | 0.68007636786695791 | -0.12325892217082668 | b2b_global_c01_eps1000000 | 11 |
| 0.7832253489110883 | -0.02402061205817238 | -0.037262017428173859 | paysim | 10 | f1 | 0.75258403416791519 | -0.03064131474317312 | b2b_global_cmed_half_clip_only | 11 |
| 0.99759361058029361 | -0.00051058365388983291 | -0.00089942882575127754 | paysim | 10 | auc_roc | 0.99688860434047299 | -0.00070500623982055523 | b2b_global_cmed_half_clip_only | 11 |
| 0.80333529003778459 | -0.020724751270355137 | -0.034176179726652323 | paysim | 10 | pr_auc | 0.77588482453928087 | -0.027450465498503728 | b2b_global_cmed_half_clip_only | 11 |
| 0.7832253489110883 | -0.22758017931489946 | -0.30889000958740598 | paysim | 10 | f1 | 0.51499025445993563 | -0.26823509445115273 | b2b_global_cmed_half_eps10 | 11 |
| 0.99759361058029361 | -0.02227186392317055 | -0.044061298056864269 | paysim | 10 | auc_roc | 0.96442702959027615 | -0.033166580990017409 | b2b_global_cmed_half_eps10 | 11 |
| 0.80333529003778459 | -0.35036842581168171 | -0.48333317614105187 | paysim | 10 | pr_auc | 0.3864844890614178 | -0.41685080097636679 | b2b_global_cmed_half_eps10 | 11 |
| 0.7832253489110883 | -0.16915200035089456 | -0.19374666117905004 | paysim | 10 | f1 | 0.60177601814611603 | -0.1814493307649723 | b2b_global_cmed_half_eps1000 | 11 |
| 0.99759361058029361 | -0.012907967123723846 | -0.017749349195817637 | paysim | 10 | auc_roc | 0.98226495242052292 | -0.015328658159770742 | b2b_global_cmed_half_eps1000 | 11 |
| 0.80333529003778459 | -0.21262951154200208 | -0.24200521549776297 | paysim | 10 | pr_auc | 0.57601792651790207 | -0.22731736351988252 | b2b_global_cmed_half_eps1000 | 11 |
| 0.7832253489110883 | -0.026126795068120115 | -0.038989615429242466 | paysim | 10 | f1 | 0.75066714366240705 | -0.032558205248681291 | b2b_global_cmed_half_eps1000000 | 11 |
| 0.99759361058029361 | -0.00047834889348898079 | -0.00098969240151054478 | paysim | 10 | auc_roc | 0.99685958993279378 | -0.00073402064749976273 | b2b_global_cmed_half_eps1000000 | 11 |
| 0.80333529003778459 | -0.022465375675306255 | -0.040204517485223551 | paysim | 10 | pr_auc | 0.77200034345751967 | -0.031334946580264901 | b2b_global_cmed_half_eps1000000 | 11 |
| 0.7832253489110883 | 0.004261257499296899 | -0.011950997789943355 | paysim | 10 | f1 | 0.77938047876576499 | -0.0038448701453232285 | b2b_global_cmed_clip_only | 11 |
| 0.99759361058029361 | 0.00019853686911132425 | -0.00015301185102291388 | paysim | 10 | auc_roc | 0.99761637308933782 | 2.2762509044205181e-05 | b2b_global_cmed_clip_only | 11 |
| 0.80333529003778459 | 0.0056711742069704193 | -0.0065148287419696978 | paysim | 10 | pr_auc | 0.80291346277028486 | -0.00042182726749963946 | b2b_global_cmed_clip_only | 11 |
| 0.7832253489110883 | -0.26327346861254131 | -0.42215651489501083 | paysim | 10 | f1 | 0.44051035715731229 | -0.34271499175377607 | b2b_global_cmed_eps10 | 11 |
| 0.99759361058029361 | -0.061145654835033439 | -0.17773055749437533 | paysim | 10 | auc_roc | 0.87815550441558932 | -0.11943810616470439 | b2b_global_cmed_eps10 | 11 |
| 0.80333529003778459 | -0.43905641939403023 | -0.56212144754003379 | paysim | 10 | pr_auc | 0.3027463565707525 | -0.50058893346703204 | b2b_global_cmed_eps10 | 11 |
| 0.7832253489110883 | -0.15895038184786239 | -0.18359025476692784 | paysim | 10 | f1 | 0.61195503060369327 | -0.17127031830739511 | b2b_global_cmed_eps1000 | 11 |
| 0.99759361058029361 | -0.014857388127530244 | -0.018119163839682394 | paysim | 10 | auc_roc | 0.98110533459668725 | -0.016488275983606318 | b2b_global_cmed_eps1000 | 11 |
| 0.80333529003778459 | -0.1990700525408475 | -0.23840081496764234 | paysim | 10 | pr_auc | 0.58459985628353961 | -0.21873543375424492 | b2b_global_cmed_eps1000 | 11 |
| 0.7832253489110883 | -0.0037579824417394055 | -0.020280388536004779 | paysim | 10 | f1 | 0.77120616342221626 | -0.012019185488872092 | b2b_global_cmed_eps1000000 | 11 |
| 0.99759361058029361 | 7.3401083982511264e-05 | -0.0002626374469527325 | paysim | 10 | auc_roc | 0.99749899239880857 | -9.4618181485110625e-05 | b2b_global_cmed_eps1000000 | 11 |
| 0.80333529003778459 | -0.0010019328530612836 | -0.015723922303831954 | paysim | 10 | pr_auc | 0.79497236245933789 | -0.0083629275784466179 | b2b_global_cmed_eps1000000 | 11 |

### Clipping/noise decomposition (post hoc B2b)

Decomposition contrasts are descriptive; they do not establish causality or introduce new tests. Clip-only arms have NO DP guarantee.

| ci_high | ci_low | dataset | df | endpoint | from_arm | mean_delta | n | to_arm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -0.1129754900550804 | -0.142679281744515 | baf | 10 | f1 | baseline | -0.1278273858997977 | 11 | b2b_global_c001_clip_only |
| -0.11340227722338998 | -0.12672349846490841 | baf | 10 | auc_roc | baseline | -0.12006288784414919 | 11 | b2b_global_c001_clip_only |
| -0.10409266250802554 | -0.12234843295298997 | baf | 10 | pr_auc | baseline | -0.11322054773050776 | 11 | b2b_global_c001_clip_only |
| -0.031829930489732899 | -0.066619097080472839 | baf | 10 | f1 | b2b_global_c001_clip_only | -0.049224513785102869 | 11 | local_dp_global_eps1 |
| -0.10347698026010332 | -0.18407951537062811 | baf | 10 | auc_roc | b2b_global_c001_clip_only | -0.14377824781536572 | 11 | local_dp_global_eps1 |
| -0.014410747709380195 | -0.032418634596438532 | baf | 10 | pr_auc | b2b_global_c001_clip_only | -0.023414691152909364 | 11 | local_dp_global_eps1 |
| -0.026164066741404163 | -0.05182912577316369 | baf | 10 | f1 | b2b_global_c001_clip_only | -0.038996596257283925 | 11 | local_dp_global_eps3 |
| -0.081059572897387286 | -0.12199081884903772 | baf | 10 | auc_roc | b2b_global_c001_clip_only | -0.1015251958732125 | 11 | local_dp_global_eps3 |
| -0.010717163212036073 | -0.027368963702181344 | baf | 10 | pr_auc | b2b_global_c001_clip_only | -0.019043063457108707 | 11 | local_dp_global_eps3 |
| -0.021327840860444907 | -0.046292431704818143 | baf | 10 | f1 | b2b_global_c001_clip_only | -0.033810136282631527 | 11 | local_dp_global_eps10 |
| -0.058204250675620794 | -0.093083383870423475 | baf | 10 | auc_roc | b2b_global_c001_clip_only | -0.075643817273022135 | 11 | local_dp_global_eps10 |
| -0.010038810591621511 | -0.025253993558343714 | baf | 10 | pr_auc | b2b_global_c001_clip_only | -0.017646402074982612 | 11 | local_dp_global_eps10 |
| -0.018477995420044538 | -0.044991075334448669 | baf | 10 | f1 | b2b_global_c001_clip_only | -0.031734535377246603 | 11 | local_dp_global_eps30 |
| -0.050672200420267488 | -0.070953943304873465 | baf | 10 | auc_roc | b2b_global_c001_clip_only | -0.060813071862570477 | 11 | local_dp_global_eps30 |
| -0.0085044658056675371 | -0.023686049696625598 | baf | 10 | pr_auc | b2b_global_c001_clip_only | -0.016095257751146567 | 11 | local_dp_global_eps30 |
| 0.0016088702911589599 | -0.027869742318895489 | baf | 10 | f1 | b2b_global_c001_clip_only | -0.013130436013868265 | 11 | local_dp_global_eps100 |
| -0.016196617438189152 | -0.040084085604624675 | baf | 10 | auc_roc | b2b_global_c001_clip_only | -0.028140351521406915 | 11 | local_dp_global_eps100 |
| 0.0011188104519396364 | -0.017909764389365351 | baf | 10 | pr_auc | b2b_global_c001_clip_only | -0.0083954769687128571 | 11 | local_dp_global_eps100 |
| 0.013578360793982116 | -0.017666426580108124 | baf | 10 | f1 | b2b_global_c001_clip_only | -0.0020440328930630036 | 11 | local_dp_global_eps1000 |
| 0.0019616163741047574 | -0.0089790294630661491 | baf | 10 | auc_roc | b2b_global_c001_clip_only | -0.0035087065444806957 | 11 | local_dp_global_eps1000 |
| 0.0084489848070048532 | -0.010477210652134526 | baf | 10 | pr_auc | b2b_global_c001_clip_only | -0.0010141129225648361 | 11 | local_dp_global_eps1000 |
| 0.0039050078263117415 | -0.0062525322407225425 | baf | 10 | f1 | b2b_global_c001_clip_only | -0.0011737622072054005 | 11 | local_dp_global_eps10000 |
| 0.0051961878651775361 | -0.0028452724614839485 | baf | 10 | auc_roc | b2b_global_c001_clip_only | 0.0011754577018467938 | 11 | local_dp_global_eps10000 |
| 0.0027666509674981734 | -0.0027564332439593015 | baf | 10 | pr_auc | b2b_global_c001_clip_only | 5.108861769435995e-06 | 11 | local_dp_global_eps10000 |
| 0.0076276467964559039 | -0.0079082795282572199 | baf | 10 | f1 | b2b_global_c001_clip_only | -0.00014031636590065781 | 11 | local_dp_global_eps1000000 |
| 0.0055896151509201384 | -0.010248584896520798 | baf | 10 | auc_roc | b2b_global_c001_clip_only | -0.00232948487280033 | 11 | local_dp_global_eps1000000 |
| 0.0036093716080973838 | -0.0045890321361277183 | baf | 10 | pr_auc | b2b_global_c001_clip_only | -0.00048983026401516717 | 11 | local_dp_global_eps1000000 |
| -0.17827651149553286 | -0.19301384614374012 | ieee_cis | 10 | f1 | baseline | -0.18564517881963649 | 11 | b2b_global_c001_clip_only |
| -0.074405045676992812 | -0.09702720007651873 | ieee_cis | 10 | auc_roc | baseline | -0.085716122876755771 | 11 | b2b_global_c001_clip_only |
| -0.24190761132493147 | -0.25480688028005205 | ieee_cis | 10 | pr_auc | baseline | -0.24835724580249174 | 11 | b2b_global_c001_clip_only |
| -0.0010947904953612364 | -0.016881361893528686 | ieee_cis | 10 | f1 | b2b_global_c001_clip_only | -0.0089880761944449612 | 11 | local_dp_global_eps1 |
| -0.049044789402191258 | -0.13409204228045685 | ieee_cis | 10 | auc_roc | b2b_global_c001_clip_only | -0.091568415841324052 | 11 | local_dp_global_eps1 |
| 0.00044358768932412692 | -0.01255994291142019 | ieee_cis | 10 | pr_auc | b2b_global_c001_clip_only | -0.0060581776110480321 | 11 | local_dp_global_eps1 |
| 0.00028294426278113738 | -0.0083355806387047544 | ieee_cis | 10 | f1 | b2b_global_c001_clip_only | -0.0040263181879618085 | 11 | local_dp_global_eps3 |
| -0.018112554250902951 | -0.077064289708650938 | ieee_cis | 10 | auc_roc | b2b_global_c001_clip_only | -0.047588421979776946 | 11 | local_dp_global_eps3 |
| 0.001873251478349088 | -0.0063410973997698314 | ieee_cis | 10 | pr_auc | b2b_global_c001_clip_only | -0.0022339229607103717 | 11 | local_dp_global_eps3 |
| 0.0055305256225961271 | -0.00065461330445723033 | ieee_cis | 10 | f1 | b2b_global_c001_clip_only | 0.0024379561590694484 | 11 | local_dp_global_eps10 |
| 0.012617730002176598 | -0.018205291832303211 | ieee_cis | 10 | auc_roc | b2b_global_c001_clip_only | -0.002793780915063306 | 11 | local_dp_global_eps10 |
| 0.0068028312966116108 | 0.0021920080114499994 | ieee_cis | 10 | pr_auc | b2b_global_c001_clip_only | 0.0044974196540308051 | 11 | local_dp_global_eps10 |
| 0.0017769713894955218 | -0.0029011953385460619 | ieee_cis | 10 | f1 | b2b_global_c001_clip_only | -0.00056211197452527006 | 11 | local_dp_global_eps30 |
| 0.0068183463162712777 | -0.012418468061995517 | ieee_cis | 10 | auc_roc | b2b_global_c001_clip_only | -0.002800060872862119 | 11 | local_dp_global_eps30 |
| 0.0049185910897542578 | -0.0017532281681974719 | ieee_cis | 10 | pr_auc | b2b_global_c001_clip_only | 0.0015826814607783927 | 11 | local_dp_global_eps30 |
| 0.00044986044709236558 | -0.003630832971308293 | ieee_cis | 10 | f1 | b2b_global_c001_clip_only | -0.0015904862621079637 | 11 | local_dp_global_eps100 |
| 0.010241057503999175 | -0.014648626417841927 | ieee_cis | 10 | auc_roc | b2b_global_c001_clip_only | -0.002203784456921376 | 11 | local_dp_global_eps100 |
| 0.0032154039245221207 | -0.0021647162505821216 | ieee_cis | 10 | pr_auc | b2b_global_c001_clip_only | 0.00052534383696999966 | 11 | local_dp_global_eps100 |
| 0.0032390281419010861 | -0.00067422877236081292 | ieee_cis | 10 | f1 | b2b_global_c001_clip_only | 0.0012823996847701367 | 11 | local_dp_global_eps1000 |
| 0.018191399015887094 | -0.0096761352152872349 | ieee_cis | 10 | auc_roc | b2b_global_c001_clip_only | 0.0042576319002999297 | 11 | local_dp_global_eps1000 |
| 0.0040823832655558518 | 7.903127641920981e-06 | ieee_cis | 10 | pr_auc | b2b_global_c001_clip_only | 0.0020451431965988866 | 11 | local_dp_global_eps1000 |
| 0.00077385761207789203 | -0.0017200587226549749 | ieee_cis | 10 | f1 | b2b_global_c001_clip_only | -0.00047310055528854145 | 11 | local_dp_global_eps10000 |
| 0.0048405053686209169 | -0.0090184941577233693 | ieee_cis | 10 | auc_roc | b2b_global_c001_clip_only | -0.0020889943945512257 | 11 | local_dp_global_eps10000 |
| 0.0014087215063423378 | -0.0010577300545926496 | ieee_cis | 10 | pr_auc | b2b_global_c001_clip_only | 0.00017549572587484408 | 11 | local_dp_global_eps10000 |
| 0.00085380392834283119 | -0.0015181109496865564 | ieee_cis | 10 | f1 | b2b_global_c001_clip_only | -0.00033215351067186256 | 11 | local_dp_global_eps1000000 |
| 0.0028527465285663346 | -0.011829208051339963 | ieee_cis | 10 | auc_roc | b2b_global_c001_clip_only | -0.004488230761386814 | 11 | local_dp_global_eps1000000 |
| 0.00087112533450843131 | -0.0012503109021565128 | ieee_cis | 10 | pr_auc | b2b_global_c001_clip_only | -0.00018959278382404081 | 11 | local_dp_global_eps1000000 |
| -0.17242106705128321 | -0.18390595669122087 | paysim | 10 | f1 | baseline | -0.17816351187125204 | 11 | b2b_global_c001_clip_only |
| -0.018196349755512976 | -0.022182730096114395 | paysim | 10 | auc_roc | baseline | -0.020189539925813686 | 11 | b2b_global_c001_clip_only |
| -0.2205199209339033 | -0.2404203361606492 | paysim | 10 | pr_auc | baseline | -0.23047012854727625 | 11 | b2b_global_c001_clip_only |
| -0.032304669306735104 | -0.2341372125145702 | paysim | 10 | f1 | b2b_global_c001_clip_only | -0.13322094091065265 | 11 | local_dp_global_eps1 |
| 0.0026851468682359547 | -0.074063774510955915 | paysim | 10 | auc_roc | b2b_global_c001_clip_only | -0.03568931382135998 | 11 | local_dp_global_eps1 |
| -0.053501722609410093 | -0.2575170336365607 | paysim | 10 | pr_auc | b2b_global_c001_clip_only | -0.1555093781229854 | 11 | local_dp_global_eps1 |
| 0.0096211046628729424 | -0.21894546258256758 | paysim | 10 | f1 | b2b_global_c001_clip_only | -0.10466217895984732 | 11 | local_dp_global_eps3 |
| 0.047697777644954836 | -0.1519422694829306 | paysim | 10 | auc_roc | b2b_global_c001_clip_only | -0.052122245918987883 | 11 | local_dp_global_eps3 |
| -0.010210325128796729 | -0.22274247000708758 | paysim | 10 | pr_auc | b2b_global_c001_clip_only | -0.11647639756794215 | 11 | local_dp_global_eps3 |
| -0.03605274650636224 | -0.17574649428955019 | paysim | 10 | f1 | b2b_global_c001_clip_only | -0.10589962039795622 | 11 | local_dp_global_eps10 |
| 0.0009744665690784287 | -0.02074803178556725 | paysim | 10 | auc_roc | b2b_global_c001_clip_only | -0.0098867826082444116 | 11 | local_dp_global_eps10 |
| -0.036858328361149176 | -0.20224957678850713 | paysim | 10 | pr_auc | b2b_global_c001_clip_only | -0.11955395257482815 | 11 | local_dp_global_eps10 |
| 0.011421820846338852 | -0.098516073428352169 | paysim | 10 | f1 | b2b_global_c001_clip_only | -0.043547126291006655 | 11 | local_dp_global_eps30 |
| 0.0017820276968349242 | -0.015576444528844623 | paysim | 10 | auc_roc | b2b_global_c001_clip_only | -0.0068972084160048497 | 11 | local_dp_global_eps30 |
| 0.014260421366386278 | -0.12904718343282259 | paysim | 10 | pr_auc | b2b_global_c001_clip_only | -0.057393381033218158 | 11 | local_dp_global_eps30 |
| 0.0049413941225640839 | -0.025572218966196478 | paysim | 10 | f1 | b2b_global_c001_clip_only | -0.010315412421816196 | 11 | local_dp_global_eps100 |
| -0.0042487857332105554 | -0.010049995612037444 | paysim | 10 | auc_roc | b2b_global_c001_clip_only | -0.0071493906726240002 | 11 | local_dp_global_eps100 |
| -0.00063695473964868812 | -0.038120462456783646 | paysim | 10 | pr_auc | b2b_global_c001_clip_only | -0.019378708598216169 | 11 | local_dp_global_eps100 |
| 0.0051470176912837918 | -0.012155350291428616 | paysim | 10 | f1 | b2b_global_c001_clip_only | -0.0035041663000724118 | 11 | local_dp_global_eps1000 |
| 0.00032954464541225638 | -0.0030116985398392286 | paysim | 10 | auc_roc | b2b_global_c001_clip_only | -0.0013410769472134861 | 11 | local_dp_global_eps1000 |
| 0.0049144325479849888 | -0.0068734936475167499 | paysim | 10 | pr_auc | b2b_global_c001_clip_only | -0.00097953054976588055 | 11 | local_dp_global_eps1000 |
| 0.0020819709652596281 | -0.0032768362045684524 | paysim | 10 | f1 | b2b_global_c001_clip_only | -0.00059743261965441237 | 11 | local_dp_global_eps10000 |
| 0.0011377652960626889 | -0.0010809878012738396 | paysim | 10 | auc_roc | b2b_global_c001_clip_only | 2.8388747394424531e-05 | 11 | local_dp_global_eps10000 |
| 0.00026907185510447297 | -0.0024110401928954845 | paysim | 10 | pr_auc | b2b_global_c001_clip_only | -0.0010709841688955057 | 11 | local_dp_global_eps10000 |
| 0.0017984598911947754 | -0.0039694824323063824 | paysim | 10 | f1 | b2b_global_c001_clip_only | -0.0010855112705558036 | 11 | local_dp_global_eps1000000 |
| 0.00074445486092675553 | -0.0010410107795839779 | paysim | 10 | auc_roc | b2b_global_c001_clip_only | -0.0001482779593286112 | 11 | local_dp_global_eps1000000 |
| 0.00089603336326446426 | -0.0014310238834592736 | paysim | 10 | pr_auc | b2b_global_c001_clip_only | -0.00026749526009740463 | 11 | local_dp_global_eps1000000 |

### Frozen matched-utility decisions

Largest-noise passing point requires the next noisier point to fail the frozen mean-F1 threshold. MATCHED_CLIP_ONLY_NO_DP means a clip-only comparator, not a DP match. NOT_ASSESSABLE receives no leakage test.

| C | clipping | dataset | epsilon | matched_arm | next_noisier_arms | next_noisier_sd | noise_sd | points | status | threshold | transform | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.74207586050033569 | global | baf | 1000000 | b2b_global_cmed_half_eps1000000 | ["local_dp_global_eps100"] | 0.013953038273947622 | 0.0074459804771753839 | [{"C": 1.4841517210006714, "epsilon": 10, "mean_f1": 0.03608702711419611, "method": "b2b_global_cmed_eps10", "noise_sd": 11.919627703058891, "passes": false}, {"C": 0.7420758605003357, "epsilon": 10, "mean_f1": 0.03546450446288842, "method": "b2b_global_cmed_half_eps10", "noise_sd": 5.959813851529446, "passes": false}, {"C": 0.1, "epsilon": 10, "mean_f1": 0.04290243066664646, "method": "b2b_global_c01_eps10", "noise_sd": 0.8031273039270019, "passes": false}, {"C": 0.01, "epsilon": 1, "mean_f1": 0.04865979657391353, "method": "local_dp_global_eps1", "noise_sd": 0.6930431582631877, "passes": false}, {"C": 1.4841517210006714, "epsilon": 1000, "mean_f1": 0.0936453702377015, "method": "b2b_global_cmed_eps1000", "noise_sd": 0.5223822420500062, "passes": false}, {"C": 0.7420758605003357, "epsilon": 1000, "mean_f1": 0.12854919130170558, "method": "b2b_global_cmed_half_eps1000", "noise_sd": 0.2611911210250031, "passes": false}, {"C": 0.01, "epsilon": 3, "mean_f1": 0.05888771410173248, "method": "local_dp_global_eps3", "noise_sd": 0.24008845279270588, "passes": false}, {"C": 0.01, "epsilon": 10, "mean_f1": 0.06407417407638488, "method": "local_dp_global_eps10", "noise_sd": 0.08031273039270019, "passes": false}, {"C": 0.1, "epsilon": 1000, "mean_f1": 0.1351887747847439, "method": "b2b_global_c01_eps1000", "noise_sd": 0.035197361203596914, "passes": false}, {"C": 0.01, "epsilon": 30, "mean_f1": 0.0661497749817698, "method": "local_dp_global_eps30", "noise_sd": 0.03278707569370402, "passes": false}, {"C": 1.4841517210006714, "epsilon": 1000000, "mean_f1": 0.22044186272098673, "method": "b2b_global_cmed_eps1000000", "noise_sd": 0.014891960954350768, "passes": false}, {"C": 0.01, "epsilon": 100, "mean_f1": 0.08475387434514814, "method": "local_dp_global_eps100", "noise_sd": 0.013953038273947622, "passes": false}, {"C": 0.7420758605003357, "epsilon": 1000000, "mean_f1": 0.22451314143170056, "method": "b2b_global_cmed_half_eps1000000", "noise_sd": 0.007445980477175384, "passes": true}, {"C": 0.01, "epsilon": 1000, "mean_f1": 0.09584027746595339, "method": "local_dp_global_eps1000", "noise_sd": 0.0035197361203596913, "passes": false}, {"C": 0.01, "epsilon": 10000, "mean_f1": 0.096710548151811, "method": "local_dp_global_eps10000", "noise_sd": 0.0010345061828063154, "passes": false}, {"C": 0.1, "epsilon": 1000000, "mean_f1": 0.20887082148353153, "method": "b2b_global_c01_eps1000000", "noise_sd": 0.0010033988266583717, "passes": false}, {"C": 0.01, "epsilon": 1000000, "mean_f1": 0.09774399399311574, "method": "local_dp_global_eps1000000", "noise_sd": 0.00010033988266583718, "passes": false}, {"C": 0.01, "epsilon": null, "mean_f1": 0.0978843103590164, "method": "b2b_global_c001_clip_only", "noise_sd": 0.0, "passes": false}, {"C": 0.1, "epsilon": null, "mean_f1": 0.20819515267406105, "method": "b2b_global_c01_clip_only", "noise_sd": 0.0, "passes": false}, {"C": 0.7420758605003357, "epsilon": null, "mean_f1": 0.22598414912312917, "method": "b2b_global_cmed_half_clip_only", "noise_sd": 0.0, "passes": true}, {"C": 1.4841517210006714, "epsilon": null, "mean_f1": 0.2254130152153416, "method": "b2b_global_cmed_clip_only", "noise_sd": 0.0, "passes": true}] | MATCHED | 0.22078096200283107 | dna_v1_conservative | NA |
| NA | per_tensor | baf | NA | NA | NA | NA | NA | [{"C": 0.01, "epsilon": 1, "mean_f1": 0.04250326559761987, "method": "local_dp_per_tensor_eps1", "noise_sd": 0.6930431582631877, "passes": false}, {"C": 0.01, "epsilon": 3, "mean_f1": 0.053389902865843865, "method": "local_dp_per_tensor_eps3", "noise_sd": 0.24008845279270588, "passes": false}, {"C": 0.01, "epsilon": 10, "mean_f1": 0.06165862300729221, "method": "local_dp_per_tensor_eps10", "noise_sd": 0.08031273039270019, "passes": false}, {"C": 0.01, "epsilon": 30, "mean_f1": 0.07246485597418816, "method": "local_dp_per_tensor_eps30", "noise_sd": 0.03278707569370402, "passes": false}, {"C": 0.01, "epsilon": 100, "mean_f1": 0.07080472567225798, "method": "local_dp_per_tensor_eps100", "noise_sd": 0.013953038273947622, "passes": false}, {"C": 0.01, "epsilon": 1000, "mean_f1": 0.09014031353442005, "method": "local_dp_per_tensor_eps1000", "noise_sd": 0.0035197361203596913, "passes": false}, {"C": 0.01, "epsilon": 10000, "mean_f1": 0.08551219219823793, "method": "local_dp_per_tensor_eps10000", "noise_sd": 0.0010345061828063154, "passes": false}, {"C": 0.01, "epsilon": 1000000, "mean_f1": 0.08686345039838309, "method": "local_dp_per_tensor_eps1000000", "noise_sd": 0.00010033988266583718, "passes": false}] | NOT_ASSESSABLE | 0.22078096200283107 | dna_v1_conservative | no passing point |
| 0.74207586050033569 | global | baf | 1000000 | b2b_global_cmed_half_eps1000000 | ["local_dp_global_eps100"] | 0.013953038273947622 | 0.0074459804771753839 | [{"C": 1.4841517210006714, "epsilon": 10, "mean_f1": 0.03608702711419611, "method": "b2b_global_cmed_eps10", "noise_sd": 11.919627703058891, "passes": false}, {"C": 0.7420758605003357, "epsilon": 10, "mean_f1": 0.03546450446288842, "method": "b2b_global_cmed_half_eps10", "noise_sd": 5.959813851529446, "passes": false}, {"C": 0.1, "epsilon": 10, "mean_f1": 0.04290243066664646, "method": "b2b_global_c01_eps10", "noise_sd": 0.8031273039270019, "passes": false}, {"C": 0.01, "epsilon": 1, "mean_f1": 0.04865979657391353, "method": "local_dp_global_eps1", "noise_sd": 0.6930431582631877, "passes": false}, {"C": 1.4841517210006714, "epsilon": 1000, "mean_f1": 0.0936453702377015, "method": "b2b_global_cmed_eps1000", "noise_sd": 0.5223822420500062, "passes": false}, {"C": 0.7420758605003357, "epsilon": 1000, "mean_f1": 0.12854919130170558, "method": "b2b_global_cmed_half_eps1000", "noise_sd": 0.2611911210250031, "passes": false}, {"C": 0.01, "epsilon": 3, "mean_f1": 0.05888771410173248, "method": "local_dp_global_eps3", "noise_sd": 0.24008845279270588, "passes": false}, {"C": 0.01, "epsilon": 10, "mean_f1": 0.06407417407638488, "method": "local_dp_global_eps10", "noise_sd": 0.08031273039270019, "passes": false}, {"C": 0.1, "epsilon": 1000, "mean_f1": 0.1351887747847439, "method": "b2b_global_c01_eps1000", "noise_sd": 0.035197361203596914, "passes": false}, {"C": 0.01, "epsilon": 30, "mean_f1": 0.0661497749817698, "method": "local_dp_global_eps30", "noise_sd": 0.03278707569370402, "passes": false}, {"C": 1.4841517210006714, "epsilon": 1000000, "mean_f1": 0.22044186272098673, "method": "b2b_global_cmed_eps1000000", "noise_sd": 0.014891960954350768, "passes": false}, {"C": 0.01, "epsilon": 100, "mean_f1": 0.08475387434514814, "method": "local_dp_global_eps100", "noise_sd": 0.013953038273947622, "passes": false}, {"C": 0.7420758605003357, "epsilon": 1000000, "mean_f1": 0.22451314143170056, "method": "b2b_global_cmed_half_eps1000000", "noise_sd": 0.007445980477175384, "passes": true}, {"C": 0.01, "epsilon": 1000, "mean_f1": 0.09584027746595339, "method": "local_dp_global_eps1000", "noise_sd": 0.0035197361203596913, "passes": false}, {"C": 0.01, "epsilon": 10000, "mean_f1": 0.096710548151811, "method": "local_dp_global_eps10000", "noise_sd": 0.0010345061828063154, "passes": false}, {"C": 0.1, "epsilon": 1000000, "mean_f1": 0.20887082148353153, "method": "b2b_global_c01_eps1000000", "noise_sd": 0.0010033988266583717, "passes": false}, {"C": 0.01, "epsilon": 1000000, "mean_f1": 0.09774399399311574, "method": "local_dp_global_eps1000000", "noise_sd": 0.00010033988266583718, "passes": false}, {"C": 0.01, "epsilon": null, "mean_f1": 0.0978843103590164, "method": "b2b_global_c001_clip_only", "noise_sd": 0.0, "passes": false}, {"C": 0.1, "epsilon": null, "mean_f1": 0.20819515267406105, "method": "b2b_global_c01_clip_only", "noise_sd": 0.0, "passes": false}, {"C": 0.7420758605003357, "epsilon": null, "mean_f1": 0.22598414912312917, "method": "b2b_global_cmed_half_clip_only", "noise_sd": 0.0, "passes": true}, {"C": 1.4841517210006714, "epsilon": null, "mean_f1": 0.2254130152153416, "method": "b2b_global_cmed_clip_only", "noise_sd": 0.0, "passes": true}] | MATCHED | 0.22120504214149991 | dna_v2_0p95 | NA |
| NA | per_tensor | baf | NA | NA | NA | NA | NA | [{"C": 0.01, "epsilon": 1, "mean_f1": 0.04250326559761987, "method": "local_dp_per_tensor_eps1", "noise_sd": 0.6930431582631877, "passes": false}, {"C": 0.01, "epsilon": 3, "mean_f1": 0.053389902865843865, "method": "local_dp_per_tensor_eps3", "noise_sd": 0.24008845279270588, "passes": false}, {"C": 0.01, "epsilon": 10, "mean_f1": 0.06165862300729221, "method": "local_dp_per_tensor_eps10", "noise_sd": 0.08031273039270019, "passes": false}, {"C": 0.01, "epsilon": 30, "mean_f1": 0.07246485597418816, "method": "local_dp_per_tensor_eps30", "noise_sd": 0.03278707569370402, "passes": false}, {"C": 0.01, "epsilon": 100, "mean_f1": 0.07080472567225798, "method": "local_dp_per_tensor_eps100", "noise_sd": 0.013953038273947622, "passes": false}, {"C": 0.01, "epsilon": 1000, "mean_f1": 0.09014031353442005, "method": "local_dp_per_tensor_eps1000", "noise_sd": 0.0035197361203596913, "passes": false}, {"C": 0.01, "epsilon": 10000, "mean_f1": 0.08551219219823793, "method": "local_dp_per_tensor_eps10000", "noise_sd": 0.0010345061828063154, "passes": false}, {"C": 0.01, "epsilon": 1000000, "mean_f1": 0.08686345039838309, "method": "local_dp_per_tensor_eps1000000", "noise_sd": 0.00010033988266583718, "passes": false}] | NOT_ASSESSABLE | 0.22120504214149991 | dna_v2_0p95 | no passing point |
| 3.4925928115844727 | global | ieee_cis | NA | b2b_global_cmed_clip_only | ["local_dp_global_eps1000000"] | 0.00010033988266583718 | 0 | [{"C": 3.4925928115844727, "epsilon": 10, "mean_f1": 0.14211193859218213, "method": "b2b_global_cmed_eps10", "noise_sd": 28.049966484826648, "passes": false}, {"C": 1.7462964057922363, "epsilon": 10, "mean_f1": 0.15656168439132587, "method": "b2b_global_cmed_half_eps10", "noise_sd": 14.024983242413324, "passes": false}, {"C": 3.4925928115844727, "epsilon": 1000, "mean_f1": 0.15864667281141545, "method": "b2b_global_cmed_eps1000", "noise_sd": 1.2293005072642478, "passes": false}, {"C": 0.1, "epsilon": 10, "mean_f1": 0.16090215643342456, "method": "b2b_global_c01_eps10", "noise_sd": 0.8031273039270019, "passes": false}, {"C": 0.01, "epsilon": 1, "mean_f1": 0.15480630784867505, "method": "local_dp_global_eps1", "noise_sd": 0.6930431582631877, "passes": false}, {"C": 1.7462964057922363, "epsilon": 1000, "mean_f1": 0.16346342937682307, "method": "b2b_global_cmed_half_eps1000", "noise_sd": 0.6146502536321239, "passes": false}, {"C": 0.01, "epsilon": 3, "mean_f1": 0.1597680658551582, "method": "local_dp_global_eps3", "noise_sd": 0.24008845279270588, "passes": false}, {"C": 0.01, "epsilon": 10, "mean_f1": 0.16623234020218947, "method": "local_dp_global_eps10", "noise_sd": 0.08031273039270019, "passes": false}, {"C": 0.1, "epsilon": 1000, "mean_f1": 0.16597677258769308, "method": "b2b_global_c01_eps1000", "noise_sd": 0.035197361203596914, "passes": false}, {"C": 3.4925928115844727, "epsilon": 1000000, "mean_f1": 0.262665547598087, "method": "b2b_global_cmed_eps1000000", "noise_sd": 0.035044635291393234, "passes": false}, {"C": 0.01, "epsilon": 30, "mean_f1": 0.16323227206859475, "method": "local_dp_global_eps30", "noise_sd": 0.03278707569370402, "passes": false}, {"C": 1.7462964057922363, "epsilon": 1000000, "mean_f1": 0.24391680734588866, "method": "b2b_global_cmed_half_eps1000000", "noise_sd": 0.017522317645696617, "passes": false}, {"C": 0.01, "epsilon": 100, "mean_f1": 0.16220389778101205, "method": "local_dp_global_eps100", "noise_sd": 0.013953038273947622, "passes": false}, {"C": 0.01, "epsilon": 1000, "mean_f1": 0.16507678372789014, "method": "local_dp_global_eps1000", "noise_sd": 0.0035197361203596913, "passes": false}, {"C": 0.01, "epsilon": 10000, "mean_f1": 0.16332128348783145, "method": "local_dp_global_eps10000", "noise_sd": 0.0010345061828063154, "passes": false}, {"C": 0.1, "epsilon": 1000000, "mean_f1": 0.1842179702594329, "method": "b2b_global_c01_eps1000000", "noise_sd": 0.0010033988266583717, "passes": false}, {"C": 0.01, "epsilon": 1000000, "mean_f1": 0.16346223053244816, "method": "local_dp_global_eps1000000", "noise_sd": 0.00010033988266583718, "passes": false}, {"C": 0.01, "epsilon": null, "mean_f1": 0.16379438404312, "method": "b2b_global_c001_clip_only", "noise_sd": 0.0, "passes": false}, {"C": 0.1, "epsilon": null, "mean_f1": 0.1863180719372723, "method": "b2b_global_c01_clip_only", "noise_sd": 0.0, "passes": false}, {"C": 1.7462964057922363, "epsilon": null, "mean_f1": 0.33106524326525044, "method": "b2b_global_cmed_half_clip_only", "noise_sd": 0.0, "passes": false}, {"C": 3.4925928115844727, "epsilon": null, "mean_f1": 0.34394463751196436, "method": "b2b_global_cmed_clip_only", "noise_sd": 0.0, "passes": true}] | MATCHED_CLIP_ONLY_NO_DP | 0.34087488979975361 | dna_v1_conservative | NA |
| NA | per_tensor | ieee_cis | NA | NA | NA | NA | NA | [{"C": 0.01, "epsilon": 1, "mean_f1": 0.15536922492978317, "method": "local_dp_per_tensor_eps1", "noise_sd": 0.6930431582631877, "passes": false}, {"C": 0.01, "epsilon": 3, "mean_f1": 0.16462393372810077, "method": "local_dp_per_tensor_eps3", "noise_sd": 0.24008845279270588, "passes": false}, {"C": 0.01, "epsilon": 10, "mean_f1": 0.16262877407169601, "method": "local_dp_per_tensor_eps10", "noise_sd": 0.08031273039270019, "passes": false}, {"C": 0.01, "epsilon": 30, "mean_f1": 0.16348521095359234, "method": "local_dp_per_tensor_eps30", "noise_sd": 0.03278707569370402, "passes": false}, {"C": 0.01, "epsilon": 100, "mean_f1": 0.16329697622276793, "method": "local_dp_per_tensor_eps100", "noise_sd": 0.013953038273947622, "passes": false}, {"C": 0.01, "epsilon": 1000, "mean_f1": 0.16202911483134544, "method": "local_dp_per_tensor_eps1000", "noise_sd": 0.0035197361203596913, "passes": false}, {"C": 0.01, "epsilon": 10000, "mean_f1": 0.1632091960909138, "method": "local_dp_per_tensor_eps10000", "noise_sd": 0.0010345061828063154, "passes": false}, {"C": 0.01, "epsilon": 1000000, "mean_f1": 0.16277810538788515, "method": "local_dp_per_tensor_eps1000000", "noise_sd": 0.00010033988266583718, "passes": false}] | NOT_ASSESSABLE | 0.34087488979975361 | dna_v1_conservative | no passing point |
| 3.4925928115844727 | global | ieee_cis | NA | b2b_global_cmed_clip_only | ["local_dp_global_eps1000000"] | 0.00010033988266583718 | 0 | [{"C": 3.4925928115844727, "epsilon": 10, "mean_f1": 0.14211193859218213, "method": "b2b_global_cmed_eps10", "noise_sd": 28.049966484826648, "passes": false}, {"C": 1.7462964057922363, "epsilon": 10, "mean_f1": 0.15656168439132587, "method": "b2b_global_cmed_half_eps10", "noise_sd": 14.024983242413324, "passes": false}, {"C": 3.4925928115844727, "epsilon": 1000, "mean_f1": 0.15864667281141545, "method": "b2b_global_cmed_eps1000", "noise_sd": 1.2293005072642478, "passes": false}, {"C": 0.1, "epsilon": 10, "mean_f1": 0.16090215643342456, "method": "b2b_global_c01_eps10", "noise_sd": 0.8031273039270019, "passes": false}, {"C": 0.01, "epsilon": 1, "mean_f1": 0.15480630784867505, "method": "local_dp_global_eps1", "noise_sd": 0.6930431582631877, "passes": false}, {"C": 1.7462964057922363, "epsilon": 1000, "mean_f1": 0.16346342937682307, "method": "b2b_global_cmed_half_eps1000", "noise_sd": 0.6146502536321239, "passes": false}, {"C": 0.01, "epsilon": 3, "mean_f1": 0.1597680658551582, "method": "local_dp_global_eps3", "noise_sd": 0.24008845279270588, "passes": false}, {"C": 0.01, "epsilon": 10, "mean_f1": 0.16623234020218947, "method": "local_dp_global_eps10", "noise_sd": 0.08031273039270019, "passes": false}, {"C": 0.1, "epsilon": 1000, "mean_f1": 0.16597677258769308, "method": "b2b_global_c01_eps1000", "noise_sd": 0.035197361203596914, "passes": false}, {"C": 3.4925928115844727, "epsilon": 1000000, "mean_f1": 0.262665547598087, "method": "b2b_global_cmed_eps1000000", "noise_sd": 0.035044635291393234, "passes": false}, {"C": 0.01, "epsilon": 30, "mean_f1": 0.16323227206859475, "method": "local_dp_global_eps30", "noise_sd": 0.03278707569370402, "passes": false}, {"C": 1.7462964057922363, "epsilon": 1000000, "mean_f1": 0.24391680734588866, "method": "b2b_global_cmed_half_eps1000000", "noise_sd": 0.017522317645696617, "passes": false}, {"C": 0.01, "epsilon": 100, "mean_f1": 0.16220389778101205, "method": "local_dp_global_eps100", "noise_sd": 0.013953038273947622, "passes": false}, {"C": 0.01, "epsilon": 1000, "mean_f1": 0.16507678372789014, "method": "local_dp_global_eps1000", "noise_sd": 0.0035197361203596913, "passes": false}, {"C": 0.01, "epsilon": 10000, "mean_f1": 0.16332128348783145, "method": "local_dp_global_eps10000", "noise_sd": 0.0010345061828063154, "passes": false}, {"C": 0.1, "epsilon": 1000000, "mean_f1": 0.1842179702594329, "method": "b2b_global_c01_eps1000000", "noise_sd": 0.0010033988266583717, "passes": false}, {"C": 0.01, "epsilon": 1000000, "mean_f1": 0.16346223053244816, "method": "local_dp_global_eps1000000", "noise_sd": 0.00010033988266583718, "passes": false}, {"C": 0.01, "epsilon": null, "mean_f1": 0.16379438404312, "method": "b2b_global_c001_clip_only", "noise_sd": 0.0, "passes": false}, {"C": 0.1, "epsilon": null, "mean_f1": 0.1863180719372723, "method": "b2b_global_c01_clip_only", "noise_sd": 0.0, "passes": false}, {"C": 1.7462964057922363, "epsilon": null, "mean_f1": 0.33106524326525044, "method": "b2b_global_cmed_half_clip_only", "noise_sd": 0.0, "passes": false}, {"C": 3.4925928115844727, "epsilon": null, "mean_f1": 0.34394463751196436, "method": "b2b_global_cmed_clip_only", "noise_sd": 0.0, "passes": true}] | MATCHED_CLIP_ONLY_NO_DP | 0.33153802428669882 | dna_v2_0p95 | NA |
| NA | per_tensor | ieee_cis | NA | NA | NA | NA | NA | [{"C": 0.01, "epsilon": 1, "mean_f1": 0.15536922492978317, "method": "local_dp_per_tensor_eps1", "noise_sd": 0.6930431582631877, "passes": false}, {"C": 0.01, "epsilon": 3, "mean_f1": 0.16462393372810077, "method": "local_dp_per_tensor_eps3", "noise_sd": 0.24008845279270588, "passes": false}, {"C": 0.01, "epsilon": 10, "mean_f1": 0.16262877407169601, "method": "local_dp_per_tensor_eps10", "noise_sd": 0.08031273039270019, "passes": false}, {"C": 0.01, "epsilon": 30, "mean_f1": 0.16348521095359234, "method": "local_dp_per_tensor_eps30", "noise_sd": 0.03278707569370402, "passes": false}, {"C": 0.01, "epsilon": 100, "mean_f1": 0.16329697622276793, "method": "local_dp_per_tensor_eps100", "noise_sd": 0.013953038273947622, "passes": false}, {"C": 0.01, "epsilon": 1000, "mean_f1": 0.16202911483134544, "method": "local_dp_per_tensor_eps1000", "noise_sd": 0.0035197361203596913, "passes": false}, {"C": 0.01, "epsilon": 10000, "mean_f1": 0.1632091960909138, "method": "local_dp_per_tensor_eps10000", "noise_sd": 0.0010345061828063154, "passes": false}, {"C": 0.01, "epsilon": 1000000, "mean_f1": 0.16277810538788515, "method": "local_dp_per_tensor_eps1000000", "noise_sd": 0.00010033988266583718, "passes": false}] | NOT_ASSESSABLE | 0.33153802428669882 | dna_v2_0p95 | no passing point |
| 1.0199140906333923 | global | paysim | NA | b2b_global_cmed_clip_only | ["local_dp_global_eps1000000"] | 0.00010033988266583718 | 0 | [{"C": 1.0199140906333923, "epsilon": 10, "mean_f1": 0.4405103571573123, "method": "b2b_global_cmed_eps10", "noise_sd": 8.191208538475562, "passes": false}, {"C": 0.5099570453166962, "epsilon": 10, "mean_f1": 0.5149902544599356, "method": "b2b_global_cmed_half_eps10", "noise_sd": 4.095604269237781, "passes": false}, {"C": 0.1, "epsilon": 10, "mean_f1": 0.5500326130408549, "method": "b2b_global_c01_eps10", "noise_sd": 0.8031273039270019, "passes": false}, {"C": 0.01, "epsilon": 1, "mean_f1": 0.4718408961291836, "method": "local_dp_global_eps1", "noise_sd": 0.6930431582631877, "passes": false}, {"C": 1.0199140906333923, "epsilon": 1000, "mean_f1": 0.6119550306036933, "method": "b2b_global_cmed_eps1000", "noise_sd": 0.3589828464466159, "passes": false}, {"C": 0.01, "epsilon": 3, "mean_f1": 0.500399658079989, "method": "local_dp_global_eps3", "noise_sd": 0.24008845279270588, "passes": false}, {"C": 0.5099570453166962, "epsilon": 1000, "mean_f1": 0.601776018146116, "method": "b2b_global_cmed_half_eps1000", "noise_sd": 0.17949142322330794, "passes": false}, {"C": 0.01, "epsilon": 10, "mean_f1": 0.49916221664188004, "method": "local_dp_global_eps10", "noise_sd": 0.08031273039270019, "passes": false}, {"C": 0.1, "epsilon": 1000, "mean_f1": 0.614196141881531, "method": "b2b_global_c01_eps1000", "noise_sd": 0.035197361203596914, "passes": false}, {"C": 0.01, "epsilon": 30, "mean_f1": 0.5615147107488296, "method": "local_dp_global_eps30", "noise_sd": 0.03278707569370402, "passes": false}, {"C": 0.01, "epsilon": 100, "mean_f1": 0.5947464246180201, "method": "local_dp_global_eps100", "noise_sd": 0.013953038273947622, "passes": false}, {"C": 1.0199140906333923, "epsilon": 1000000, "mean_f1": 0.7712061634222163, "method": "b2b_global_cmed_eps1000000", "noise_sd": 0.010233806018338861, "passes": false}, {"C": 0.5099570453166962, "epsilon": 1000000, "mean_f1": 0.750667143662407, "method": "b2b_global_cmed_half_eps1000000", "noise_sd": 0.005116903009169431, "passes": false}, {"C": 0.01, "epsilon": 1000, "mean_f1": 0.6015576707397639, "method": "local_dp_global_eps1000", "noise_sd": 0.0035197361203596913, "passes": false}, {"C": 0.01, "epsilon": 10000, "mean_f1": 0.6044644044201819, "method": "local_dp_global_eps10000", "noise_sd": 0.0010345061828063154, "passes": false}, {"C": 0.1, "epsilon": 1000000, "mean_f1": 0.6514069564034822, "method": "b2b_global_c01_eps1000000", "noise_sd": 0.0010033988266583717, "passes": false}, {"C": 0.01, "epsilon": 1000000, "mean_f1": 0.6039763257692805, "method": "local_dp_global_eps1000000", "noise_sd": 0.00010033988266583718, "passes": false}, {"C": 0.01, "epsilon": null, "mean_f1": 0.6050618370398363, "method": "b2b_global_c001_clip_only", "noise_sd": 0.0, "passes": false}, {"C": 0.1, "epsilon": null, "mean_f1": 0.6451732388763448, "method": "b2b_global_c01_clip_only", "noise_sd": 0.0, "passes": false}, {"C": 0.5099570453166962, "epsilon": null, "mean_f1": 0.7525840341679152, "method": "b2b_global_cmed_half_clip_only", "noise_sd": 0.0, "passes": false}, {"C": 1.0199140906333923, "epsilon": null, "mean_f1": 0.779380478765765, "method": "b2b_global_cmed_clip_only", "noise_sd": 0.0, "passes": true}] | MATCHED_CLIP_ONLY_NO_DP | 0.77660691934132764 | dna_v1_conservative | NA |
| NA | per_tensor | paysim | NA | NA | NA | NA | NA | [{"C": 0.01, "epsilon": 1, "mean_f1": 0.5489964429728239, "method": "local_dp_per_tensor_eps1", "noise_sd": 0.6930431582631877, "passes": false}, {"C": 0.01, "epsilon": 3, "mean_f1": 0.5405854464876501, "method": "local_dp_per_tensor_eps3", "noise_sd": 0.24008845279270588, "passes": false}, {"C": 0.01, "epsilon": 10, "mean_f1": 0.501919530859745, "method": "local_dp_per_tensor_eps10", "noise_sd": 0.08031273039270019, "passes": false}, {"C": 0.01, "epsilon": 30, "mean_f1": 0.5151327514931183, "method": "local_dp_per_tensor_eps30", "noise_sd": 0.03278707569370402, "passes": false}, {"C": 0.01, "epsilon": 100, "mean_f1": 0.5432410939780554, "method": "local_dp_per_tensor_eps100", "noise_sd": 0.013953038273947622, "passes": false}, {"C": 0.01, "epsilon": 1000, "mean_f1": 0.5901804709959367, "method": "local_dp_per_tensor_eps1000", "noise_sd": 0.0035197361203596913, "passes": false}, {"C": 0.01, "epsilon": 10000, "mean_f1": 0.5978308756305478, "method": "local_dp_per_tensor_eps10000", "noise_sd": 0.0010345061828063154, "passes": false}, {"C": 0.01, "epsilon": 1000000, "mean_f1": 0.5963404467875477, "method": "local_dp_per_tensor_eps1000000", "noise_sd": 0.00010033988266583718, "passes": false}] | NOT_ASSESSABLE | 0.77660691934132764 | dna_v1_conservative | no passing point |
| 1.0199140906333923 | global | paysim | NA | b2b_global_cmed_clip_only | ["local_dp_global_eps1000000"] | 0.00010033988266583718 | 0 | [{"C": 1.0199140906333923, "epsilon": 10, "mean_f1": 0.4405103571573123, "method": "b2b_global_cmed_eps10", "noise_sd": 8.191208538475562, "passes": false}, {"C": 0.5099570453166962, "epsilon": 10, "mean_f1": 0.5149902544599356, "method": "b2b_global_cmed_half_eps10", "noise_sd": 4.095604269237781, "passes": false}, {"C": 0.1, "epsilon": 10, "mean_f1": 0.5500326130408549, "method": "b2b_global_c01_eps10", "noise_sd": 0.8031273039270019, "passes": false}, {"C": 0.01, "epsilon": 1, "mean_f1": 0.4718408961291836, "method": "local_dp_global_eps1", "noise_sd": 0.6930431582631877, "passes": false}, {"C": 1.0199140906333923, "epsilon": 1000, "mean_f1": 0.6119550306036933, "method": "b2b_global_cmed_eps1000", "noise_sd": 0.3589828464466159, "passes": false}, {"C": 0.01, "epsilon": 3, "mean_f1": 0.500399658079989, "method": "local_dp_global_eps3", "noise_sd": 0.24008845279270588, "passes": false}, {"C": 0.5099570453166962, "epsilon": 1000, "mean_f1": 0.601776018146116, "method": "b2b_global_cmed_half_eps1000", "noise_sd": 0.17949142322330794, "passes": false}, {"C": 0.01, "epsilon": 10, "mean_f1": 0.49916221664188004, "method": "local_dp_global_eps10", "noise_sd": 0.08031273039270019, "passes": false}, {"C": 0.1, "epsilon": 1000, "mean_f1": 0.614196141881531, "method": "b2b_global_c01_eps1000", "noise_sd": 0.035197361203596914, "passes": false}, {"C": 0.01, "epsilon": 30, "mean_f1": 0.5615147107488296, "method": "local_dp_global_eps30", "noise_sd": 0.03278707569370402, "passes": false}, {"C": 0.01, "epsilon": 100, "mean_f1": 0.5947464246180201, "method": "local_dp_global_eps100", "noise_sd": 0.013953038273947622, "passes": false}, {"C": 1.0199140906333923, "epsilon": 1000000, "mean_f1": 0.7712061634222163, "method": "b2b_global_cmed_eps1000000", "noise_sd": 0.010233806018338861, "passes": false}, {"C": 0.5099570453166962, "epsilon": 1000000, "mean_f1": 0.750667143662407, "method": "b2b_global_cmed_half_eps1000000", "noise_sd": 0.005116903009169431, "passes": false}, {"C": 0.01, "epsilon": 1000, "mean_f1": 0.6015576707397639, "method": "local_dp_global_eps1000", "noise_sd": 0.0035197361203596913, "passes": false}, {"C": 0.01, "epsilon": 10000, "mean_f1": 0.6044644044201819, "method": "local_dp_global_eps10000", "noise_sd": 0.0010345061828063154, "passes": false}, {"C": 0.1, "epsilon": 1000000, "mean_f1": 0.6514069564034822, "method": "b2b_global_c01_eps1000000", "noise_sd": 0.0010033988266583717, "passes": false}, {"C": 0.01, "epsilon": 1000000, "mean_f1": 0.6039763257692805, "method": "local_dp_global_eps1000000", "noise_sd": 0.00010033988266583718, "passes": false}, {"C": 0.01, "epsilon": null, "mean_f1": 0.6050618370398363, "method": "b2b_global_c001_clip_only", "noise_sd": 0.0, "passes": false}, {"C": 0.1, "epsilon": null, "mean_f1": 0.6451732388763448, "method": "b2b_global_c01_clip_only", "noise_sd": 0.0, "passes": false}, {"C": 0.5099570453166962, "epsilon": null, "mean_f1": 0.7525840341679152, "method": "b2b_global_cmed_half_clip_only", "noise_sd": 0.0, "passes": false}, {"C": 1.0199140906333923, "epsilon": null, "mean_f1": 0.779380478765765, "method": "b2b_global_cmed_clip_only", "noise_sd": 0.0, "passes": true}] | MATCHED_CLIP_ONLY_NO_DP | 0.77747154585272038 | dna_v2_0p95 | NA |
| NA | per_tensor | paysim | NA | NA | NA | NA | NA | [{"C": 0.01, "epsilon": 1, "mean_f1": 0.5489964429728239, "method": "local_dp_per_tensor_eps1", "noise_sd": 0.6930431582631877, "passes": false}, {"C": 0.01, "epsilon": 3, "mean_f1": 0.5405854464876501, "method": "local_dp_per_tensor_eps3", "noise_sd": 0.24008845279270588, "passes": false}, {"C": 0.01, "epsilon": 10, "mean_f1": 0.501919530859745, "method": "local_dp_per_tensor_eps10", "noise_sd": 0.08031273039270019, "passes": false}, {"C": 0.01, "epsilon": 30, "mean_f1": 0.5151327514931183, "method": "local_dp_per_tensor_eps30", "noise_sd": 0.03278707569370402, "passes": false}, {"C": 0.01, "epsilon": 100, "mean_f1": 0.5432410939780554, "method": "local_dp_per_tensor_eps100", "noise_sd": 0.013953038273947622, "passes": false}, {"C": 0.01, "epsilon": 1000, "mean_f1": 0.5901804709959367, "method": "local_dp_per_tensor_eps1000", "noise_sd": 0.0035197361203596913, "passes": false}, {"C": 0.01, "epsilon": 10000, "mean_f1": 0.5978308756305478, "method": "local_dp_per_tensor_eps10000", "noise_sd": 0.0010345061828063154, "passes": false}, {"C": 0.01, "epsilon": 1000000, "mean_f1": 0.5963404467875477, "method": "local_dp_per_tensor_eps1000000", "noise_sd": 0.00010033988266583718, "passes": false}] | NOT_ASSESSABLE | 0.77747154585272038 | dna_v2_0p95 | no passing point |

### Complete leakage frontier (all qualified estimators)

| dataset | cell | arm | metric | n | median | interval | ranks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baf | ratio_batch1 | unprotected | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | ratio_batch1 | unprotected | standardized_mse | 39 | 3.0650769278983306e-16 | [1.8400032455533306e-16, 4.2901506102433314e-16] | [13, 27] |
| baf | ratio_batch1 | unprotected | bounded_mse | 39 | 3.0650769278983306e-16 | [1.8400032455533306e-16, 4.2901506102433314e-16] | [13, 27] |
| baf | ratio_batch1 | dna_v1_conservative | feature_accuracy | 39 | 0.97297297297297303 | [0.972972972972973, 1.0] | [13, 27] |
| baf | ratio_batch1 | dna_v1_conservative | standardized_mse | 39 | 0.17090206687019305 | [0.03599763855399735, 0.3694430166784568] | [13, 27] |
| baf | ratio_batch1 | dna_v1_conservative | bounded_mse | 39 | 0.17090206687019305 | [0.03599763855399735, 0.36492400776196326] | [13, 27] |
| baf | ratio_batch1 | dna_v2_0p95 | feature_accuracy | 39 | 0.94594594594594594 | [0.9189189189189189, 0.972972972972973] | [13, 27] |
| baf | ratio_batch1 | dna_v2_0p95 | standardized_mse | 39 | 0.092119926259410762 | [0.04232488486704544, 0.86730680601281] | [13, 27] |
| baf | ratio_batch1 | dna_v2_0p95 | bounded_mse | 39 | 0.092119926259410762 | [0.04232488486704544, 0.4966083227310801] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps1 | feature_accuracy | 39 | 0.40540540540540537 | [0.3513513513513513, 0.43243243243243246] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps1 | standardized_mse | 39 | 19.794976249442421 | [15.420603357359271, 32.82893428931452] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps1 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps3 | feature_accuracy | 39 | 0.3783783783783784 | [0.3783783783783784, 0.4054054054054054] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps3 | standardized_mse | 39 | 24.565617051008655 | [19.999415988860658, 38.12981942182969] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps3 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps10 | feature_accuracy | 39 | 0.40540540540540537 | [0.3513513513513513, 0.43243243243243246] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps10 | standardized_mse | 39 | 20.944870994607996 | [17.37123220807155, 26.811853897161534] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps10 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps30 | feature_accuracy | 39 | 0.40540540540540537 | [0.3783783783783784, 0.43243243243243246] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps30 | standardized_mse | 39 | 23.129431294701263 | [17.924590785722593, 31.237125114054184] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps30 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps100 | feature_accuracy | 39 | 0.3783783783783784 | [0.3513513513513513, 0.4054054054054054] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps100 | standardized_mse | 39 | 25.237865917240089 | [18.6508139842085, 33.38344014187232] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps100 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps1000 | feature_accuracy | 39 | 0.40540540540540537 | [0.3783783783783784, 0.43243243243243246] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps1000 | standardized_mse | 39 | 27.631555982452941 | [21.28323799389745, 38.26165499916411] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps1000 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps10000 | feature_accuracy | 39 | 0.40540540540540537 | [0.3783783783783784, 0.43243243243243246] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps10000 | standardized_mse | 39 | 27.799564591265188 | [19.185119053763774, 36.29012310094616] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps10000 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps1000000 | feature_accuracy | 39 | 0.43243243243243246 | [0.4054054054054054, 0.45945945945945943] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps1000000 | standardized_mse | 39 | 19.449540188919858 | [15.420710367251635, 32.60070911230641] | [13, 27] |
| baf | ratio_batch1 | local_dp_global_eps1000000 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps1 | feature_accuracy | 39 | 0.40540540540540537 | [0.3783783783783784, 0.43243243243243246] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps1 | standardized_mse | 39 | 30.809452991696443 | [22.085314145049967, 40.243975285136216] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps1 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps3 | feature_accuracy | 39 | 0.3783783783783784 | [0.3513513513513513, 0.4054054054054054] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps3 | standardized_mse | 39 | 27.635934364120345 | [19.537389148398773, 35.23308398025342] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps3 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps10 | feature_accuracy | 39 | 0.40540540540540537 | [0.3783783783783784, 0.43243243243243246] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps10 | standardized_mse | 39 | 31.080446517156822 | [18.37179965875235, 43.30921480874114] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps10 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps30 | feature_accuracy | 39 | 0.3783783783783784 | [0.3513513513513513, 0.43243243243243246] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps30 | standardized_mse | 39 | 30.870431673855865 | [21.258689283302225, 35.578570900865834] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps30 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps100 | feature_accuracy | 39 | 0.40540540540540537 | [0.3783783783783784, 0.43243243243243246] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps100 | standardized_mse | 39 | 24.296187991586873 | [15.205016283164124, 34.849512283481964] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps100 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps1000 | feature_accuracy | 39 | 0.43243243243243246 | [0.3783783783783784, 0.45945945945945943] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps1000 | standardized_mse | 39 | 18.019193072277961 | [13.569441480085908, 26.6995000389458] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps1000 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps10000 | feature_accuracy | 39 | 0.40540540540540537 | [0.3783783783783784, 0.43243243243243246] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps10000 | standardized_mse | 39 | 20.643539606570581 | [16.47258599034395, 31.151559787179735] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps10000 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps1000000 | feature_accuracy | 39 | 0.45945945945945943 | [0.43243243243243246, 0.5135135135135135] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps1000000 | standardized_mse | 39 | 11.538101208739597 | [8.362591998547947, 16.27390087820638] | [13, 27] |
| baf | ratio_batch1 | local_dp_per_tensor_eps1000000 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | b2b_global_c001_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | ratio_batch1 | b2b_global_c001_clip_only | standardized_mse | 39 | 7.4294019212523946e-16 | [4.529422813826339e-16, 1.0415519021968332e-15] | [13, 27] |
| baf | ratio_batch1 | b2b_global_c001_clip_only | bounded_mse | 39 | 7.4294019212523946e-16 | [4.529422813826339e-16, 1.0415519021968332e-15] | [13, 27] |
| baf | ratio_batch1 | b2b_global_c01_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | ratio_batch1 | b2b_global_c01_clip_only | standardized_mse | 39 | 6.7498688630766526e-16 | [4.2901506102433314e-16, 1.0568653232261459e-15] | [13, 27] |
| baf | ratio_batch1 | b2b_global_c01_clip_only | bounded_mse | 39 | 6.7498688630766526e-16 | [4.2901506102433314e-16, 1.0568653232261459e-15] | [13, 27] |
| baf | ratio_batch1 | b2b_global_c01_eps10 | feature_accuracy | 39 | 0.3783783783783784 | [0.3513513513513513, 0.43243243243243246] | [13, 27] |
| baf | ratio_batch1 | b2b_global_c01_eps10 | standardized_mse | 39 | 22.037948068949021 | [17.590603993820384, 25.92347961020756] | [13, 27] |
| baf | ratio_batch1 | b2b_global_c01_eps10 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | b2b_global_c01_eps1000 | feature_accuracy | 39 | 0.3783783783783784 | [0.3513513513513513, 0.45945945945945943] | [13, 27] |
| baf | ratio_batch1 | b2b_global_c01_eps1000 | standardized_mse | 39 | 24.050467191116674 | [18.666803553706238, 33.14890198736187] | [13, 27] |
| baf | ratio_batch1 | b2b_global_c01_eps1000 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | b2b_global_c01_eps1000000 | feature_accuracy | 39 | 0.40540540540540537 | [0.3513513513513513, 0.43243243243243246] | [13, 27] |
| baf | ratio_batch1 | b2b_global_c01_eps1000000 | standardized_mse | 39 | 22.94182846450602 | [18.18485114467215, 30.267029284076724] | [13, 27] |
| baf | ratio_batch1 | b2b_global_c01_eps1000000 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_half_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_half_clip_only | standardized_mse | 39 | 7.3911183686791135e-16 | [4.529422813826339e-16, 1.0568653232261459e-15] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_half_clip_only | bounded_mse | 39 | 7.3911183686791135e-16 | [4.529422813826339e-16, 1.0568653232261459e-15] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_half_eps10 | feature_accuracy | 39 | 0.3783783783783784 | [0.3783783783783784, 0.4054054054054054] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_half_eps10 | standardized_mse | 39 | 26.93348224107519 | [20.525924921777655, 39.4148872893923] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_half_eps10 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_half_eps1000 | feature_accuracy | 39 | 0.40540540540540537 | [0.3783783783783784, 0.45945945945945943] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_half_eps1000 | standardized_mse | 39 | 25.041941939742756 | [18.87736940200382, 33.17834983692759] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_half_eps1000 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_half_eps1000000 | feature_accuracy | 39 | 0.40540540540540537 | [0.3783783783783784, 0.43243243243243246] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_half_eps1000000 | standardized_mse | 39 | 26.409047806610996 | [18.141834615615927, 39.537559759312444] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_half_eps1000000 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_clip_only | standardized_mse | 39 | 7.3911183686791135e-16 | [4.529422813826339e-16, 1.0568653232261459e-15] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_clip_only | bounded_mse | 39 | 7.3911183686791135e-16 | [4.529422813826339e-16, 1.0568653232261459e-15] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_eps10 | feature_accuracy | 39 | 0.40540540540540537 | [0.3783783783783784, 0.43243243243243246] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_eps10 | standardized_mse | 39 | 29.531387926030778 | [19.6515408012757, 36.246112056715425] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_eps10 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_eps1000 | feature_accuracy | 39 | 0.3783783783783784 | [0.3513513513513513, 0.43243243243243246] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_eps1000 | standardized_mse | 39 | 29.808570386357243 | [18.84218301810284, 39.74665239824012] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_eps1000 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_eps1000000 | feature_accuracy | 39 | 0.40540540540540537 | [0.3783783783783784, 0.43243243243243246] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_eps1000000 | standardized_mse | 39 | 29.066375345287412 | [20.870897863620705, 34.14441112036081] | [13, 27] |
| baf | ratio_batch1 | b2b_global_cmed_eps1000000 | bounded_mse | 39 | 0.91808612101652987 | [0.6756126375406015, 1.065390201734111] | [13, 27] |
| baf | tableak_batch1 | unprotected | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | tableak_batch1 | unprotected | standardized_mse | 39 | 5.6516632734701665e-06 | [2.989564843773286e-06, 1.4968996641225748e-05] | [13, 27] |
| baf | tableak_batch1 | unprotected | bounded_mse | 39 | 5.6516632734701665e-06 | [2.989564843773286e-06, 1.4968996641225748e-05] | [13, 27] |
| baf | tableak_batch1 | dna_v1_conservative | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | tableak_batch1 | dna_v1_conservative | standardized_mse | 39 | 0.00074419152277022235 | [0.0003792714806134176, 0.002190371046939964] | [13, 27] |
| baf | tableak_batch1 | dna_v1_conservative | bounded_mse | 39 | 0.00074419152277022235 | [0.0003792714806134176, 0.002190371046939964] | [13, 27] |
| baf | tableak_batch1 | dna_v2_0p95 | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | tableak_batch1 | dna_v2_0p95 | standardized_mse | 39 | 5.5027501366003215e-06 | [3.967698408354691e-06, 1.199806287940557e-05] | [13, 27] |
| baf | tableak_batch1 | dna_v2_0p95 | bounded_mse | 39 | 5.5027501366003215e-06 | [3.967698408354691e-06, 1.199806287940557e-05] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps1 | feature_accuracy | 39 | 0.35135135135135132 | [0.29729729729729726, 0.3783783783783784] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps1 | standardized_mse | 39 | 2.3411963957442068 | [1.9100875217452928, 2.5988060549978735] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps1 | bounded_mse | 39 | 0.95009301791737344 | [0.7841911052668353, 1.1977414639521684] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps3 | feature_accuracy | 39 | 0.32432432432432434 | [0.29729729729729726, 0.3783783783783784] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps3 | standardized_mse | 39 | 2.0778035745660559 | [1.773827171453945, 2.5660728480927038] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps3 | bounded_mse | 39 | 0.95009301791737344 | [0.7841911052668353, 1.1977414639521684] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps10 | feature_accuracy | 39 | 0.32432432432432434 | [0.29729729729729726, 0.3783783783783784] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps10 | standardized_mse | 39 | 2.0198597128540881 | [1.8333368357227149, 2.211116248761154] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps10 | bounded_mse | 39 | 0.95009301791737344 | [0.7841911052668353, 1.1977414639521684] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps30 | feature_accuracy | 39 | 0.35135135135135132 | [0.29729729729729726, 0.3513513513513513] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps30 | standardized_mse | 39 | 2.0439350378463419 | [1.7354908054746212, 2.5553115652774796] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps30 | bounded_mse | 39 | 0.95009301791737344 | [0.7841911052668353, 1.0836050577503948] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps100 | feature_accuracy | 39 | 0.32432432432432434 | [0.29729729729729726, 0.3783783783783784] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps100 | standardized_mse | 39 | 2.2500343954815447 | [1.9244463922621742, 2.615852670178401] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps100 | bounded_mse | 39 | 0.95009301791737344 | [0.7841911052668353, 1.1977414639521684] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps1000 | feature_accuracy | 39 | 0.35135135135135132 | [0.32432432432432434, 0.3783783783783784] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps1000 | standardized_mse | 39 | 1.4872215754573932 | [1.213499631064053, 2.00625527002344] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps1000 | bounded_mse | 39 | 0.91152658890637639 | [0.7841911052668353, 1.062666478756486] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps10000 | feature_accuracy | 39 | 0.48648648648648651 | [0.43243243243243246, 0.5135135135135135] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps10000 | standardized_mse | 39 | 0.61190181104380814 | [0.4823671713094517, 0.7174703599533278] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps10000 | bounded_mse | 39 | 0.53472623926726437 | [0.45257250805414984, 0.7174703599533278] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps1000000 | feature_accuracy | 39 | 1 | [0.972972972972973, 1.0] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps1000000 | standardized_mse | 39 | 0.0073422656045026048 | [0.006082961938499691, 0.011180507072728807] | [13, 27] |
| baf | tableak_batch1 | local_dp_global_eps1000000 | bounded_mse | 39 | 0.0073422656045026048 | [0.006082961938499691, 0.011180507072728807] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps1 | feature_accuracy | 39 | 0.35135135135135132 | [0.29729729729729726, 0.3783783783783784] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps1 | standardized_mse | 39 | 2.1366688688898501 | [1.8459684023021623, 2.529664433059772] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps1 | bounded_mse | 39 | 0.95009301791737344 | [0.7841911052668353, 1.1977414639521684] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps3 | feature_accuracy | 39 | 0.32432432432432434 | [0.29729729729729726, 0.3783783783783784] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps3 | standardized_mse | 39 | 2.18217115778757 | [1.97103483606575, 2.486396944822575] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps3 | bounded_mse | 39 | 0.95009301791737344 | [0.7841911052668353, 1.1977414639521684] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps10 | feature_accuracy | 39 | 0.32432432432432434 | [0.29729729729729726, 0.3513513513513513] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps10 | standardized_mse | 39 | 2.0814618861732939 | [1.8067199707277262, 2.8955275795191455] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps10 | bounded_mse | 39 | 0.95009301791737344 | [0.7841911052668353, 1.1977414639521684] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps30 | feature_accuracy | 39 | 0.32432432432432434 | [0.29729729729729726, 0.3513513513513513] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps30 | standardized_mse | 39 | 2.2415111381640229 | [1.900740451962324, 2.5835377760887592] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps30 | bounded_mse | 39 | 0.95009301791737344 | [0.7841911052668353, 1.1977414639521684] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps100 | feature_accuracy | 39 | 0.35135135135135132 | [0.32432432432432434, 0.4054054054054054] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps100 | standardized_mse | 39 | 1.9269463481980644 | [1.6624541472922107, 2.3123153688822082] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps100 | bounded_mse | 39 | 0.95009301791737344 | [0.7841911052668353, 1.1977414639521684] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps1000 | feature_accuracy | 39 | 0.35135135135135132 | [0.32432432432432434, 0.4054054054054054] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps1000 | standardized_mse | 39 | 1.5016805776059234 | [1.2678540524131814, 2.1157718639362515] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps1000 | bounded_mse | 39 | 0.91152658890637639 | [0.7596747370440937, 1.1977414639521684] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps10000 | feature_accuracy | 39 | 0.45945945945945943 | [0.3783783783783784, 0.4864864864864865] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps10000 | standardized_mse | 39 | 0.73510390516266444 | [0.5516581054937814, 0.9417821062453166] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps10000 | bounded_mse | 39 | 0.70400576793832026 | [0.5078611967405414, 0.9402862702362205] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps1000000 | feature_accuracy | 39 | 0.64864864864864868 | [0.6216216216216216, 0.7027027027027026] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps1000000 | standardized_mse | 39 | 0.19397224732251506 | [0.13107261584526622, 0.27227261239607214] | [13, 27] |
| baf | tableak_batch1 | local_dp_per_tensor_eps1000000 | bounded_mse | 39 | 0.19397224732251506 | [0.13107261584526622, 0.27227261239607214] | [13, 27] |
| baf | tableak_batch1 | b2b_global_c001_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | tableak_batch1 | b2b_global_c001_clip_only | standardized_mse | 39 | 5.1639939089409794e-06 | [2.6623022850498598e-06, 1.5340549711303958e-05] | [13, 27] |
| baf | tableak_batch1 | b2b_global_c001_clip_only | bounded_mse | 39 | 5.1639939089409794e-06 | [2.6623022850498598e-06, 1.5340549711303958e-05] | [13, 27] |
| baf | tableak_batch1 | b2b_global_c01_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | tableak_batch1 | b2b_global_c01_clip_only | standardized_mse | 39 | 6.1638191219690095e-06 | [2.816350542820299e-06, 1.629947351862827e-05] | [13, 27] |
| baf | tableak_batch1 | b2b_global_c01_clip_only | bounded_mse | 39 | 6.1638191219690095e-06 | [2.816350542820299e-06, 1.629947351862827e-05] | [13, 27] |
| baf | tableak_batch1 | b2b_global_c01_eps10 | feature_accuracy | 39 | 0.32432432432432434 | [0.29729729729729726, 0.3513513513513513] | [13, 27] |
| baf | tableak_batch1 | b2b_global_c01_eps10 | standardized_mse | 39 | 2.1065007689823902 | [1.7469479955488103, 2.435631981308022] | [13, 27] |
| baf | tableak_batch1 | b2b_global_c01_eps10 | bounded_mse | 39 | 0.95009301791737344 | [0.7841911052668353, 1.1977414639521684] | [13, 27] |
| baf | tableak_batch1 | b2b_global_c01_eps1000 | feature_accuracy | 39 | 0.3783783783783784 | [0.3513513513513513, 0.4054054054054054] | [13, 27] |
| baf | tableak_batch1 | b2b_global_c01_eps1000 | standardized_mse | 39 | 1.4925067634224527 | [1.2423593249059028, 1.8006679308587035] | [13, 27] |
| baf | tableak_batch1 | b2b_global_c01_eps1000 | bounded_mse | 39 | 0.95009301791737344 | [0.7841911052668353, 1.1977414639521684] | [13, 27] |
| baf | tableak_batch1 | b2b_global_c01_eps1000000 | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | tableak_batch1 | b2b_global_c01_eps1000000 | standardized_mse | 39 | 0.0070538769669590742 | [0.005693947568174419, 0.009543839890501333] | [13, 27] |
| baf | tableak_batch1 | b2b_global_c01_eps1000000 | bounded_mse | 39 | 0.0070538769669590742 | [0.005693947568174419, 0.009543839890501333] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_half_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_half_clip_only | standardized_mse | 39 | 5.4620900117146367e-06 | [2.568328274847684e-06, 1.1896455653688855e-05] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_half_clip_only | bounded_mse | 39 | 5.4620900117146367e-06 | [2.568328274847684e-06, 1.1896455653688855e-05] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_half_eps10 | feature_accuracy | 39 | 0.32432432432432434 | [0.29729729729729726, 0.3783783783783784] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_half_eps10 | standardized_mse | 39 | 2.031022649412316 | [1.8682338412197481, 2.370015439075348] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_half_eps10 | bounded_mse | 39 | 0.95009301791737344 | [0.7841911052668353, 1.1977414639521684] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_half_eps1000 | feature_accuracy | 39 | 0.3783783783783784 | [0.32432432432432434, 0.3783783783783784] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_half_eps1000 | standardized_mse | 39 | 1.3524640693989027 | [1.2111842377391708, 1.8147040024437848] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_half_eps1000 | bounded_mse | 39 | 0.91152658890637639 | [0.7596747370440937, 1.0836050577503948] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_half_eps1000000 | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_half_eps1000000 | standardized_mse | 39 | 0.0072979536731934231 | [0.005170771252805714, 0.01050061905037469] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_half_eps1000000 | bounded_mse | 39 | 0.0072979536731934231 | [0.005170771252805714, 0.01050061905037469] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_clip_only | standardized_mse | 39 | 5.4620900117146367e-06 | [2.568328274847684e-06, 1.1896455653688855e-05] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_clip_only | bounded_mse | 39 | 5.4620900117146367e-06 | [2.568328274847684e-06, 1.1896455653688855e-05] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_eps10 | feature_accuracy | 39 | 0.32432432432432434 | [0.29729729729729726, 0.3513513513513513] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_eps10 | standardized_mse | 39 | 1.9867917614209585 | [1.6730962405323355, 2.4377919596209114] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_eps10 | bounded_mse | 39 | 0.95009301791737344 | [0.7841911052668353, 1.1977414639521684] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_eps1000 | feature_accuracy | 39 | 0.35135135135135132 | [0.32432432432432434, 0.4054054054054054] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_eps1000 | standardized_mse | 39 | 1.6342332646447415 | [1.304682587616954, 2.1374068362673118] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_eps1000 | bounded_mse | 39 | 0.95009301791737344 | [0.7841911052668353, 1.1977414639521684] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_eps1000000 | feature_accuracy | 39 | 1 | [0.972972972972973, 1.0] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_eps1000000 | standardized_mse | 39 | 0.0076820412451929845 | [0.00563443663552471, 0.008888818868745027] | [13, 27] |
| baf | tableak_batch1 | b2b_global_cmed_eps1000000 | bounded_mse | 39 | 0.0076820412451929845 | [0.00563443663552471, 0.008888818868745027] | [13, 27] |
| baf | tableak_batch2 | unprotected | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | tableak_batch2 | unprotected | standardized_mse | 39 | 3.1314239009458762e-05 | [1.2020187133898772e-05, 0.00010351675310820033] | [13, 27] |
| baf | tableak_batch2 | unprotected | bounded_mse | 39 | 3.1314239009458762e-05 | [1.2020187133898772e-05, 0.00010351675310820033] | [13, 27] |
| baf | tableak_batch2 | dna_v1_conservative | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | tableak_batch2 | dna_v1_conservative | standardized_mse | 39 | 0.003078016234964921 | [0.0013582894876563675, 0.004234089919604867] | [13, 27] |
| baf | tableak_batch2 | dna_v1_conservative | bounded_mse | 39 | 0.003078016234964921 | [0.0013582894876563675, 0.004234089919604867] | [13, 27] |
| baf | tableak_batch2 | dna_v2_0p95 | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | tableak_batch2 | dna_v2_0p95 | standardized_mse | 39 | 3.1716804071425123e-05 | [1.3913447509254673e-05, 0.00010045643961937704] | [13, 27] |
| baf | tableak_batch2 | dna_v2_0p95 | bounded_mse | 39 | 3.1716804071425123e-05 | [1.3913447509254673e-05, 0.00010045643961937704] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps1 | feature_accuracy | 39 | 0.36486486486486491 | [0.33783783783783783, 0.3783783783783784] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps1 | standardized_mse | 39 | 2.5070443385403776 | [2.0952730551788945, 2.687474463592473] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps1 | bounded_mse | 39 | 0.89324359048010082 | [0.7823462005777662, 1.0605669809201397] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps3 | feature_accuracy | 39 | 0.33783783783783783 | [0.32432432432432434, 0.3783783783783783] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps3 | standardized_mse | 39 | 2.1173344997807377 | [1.950056072618603, 2.581622922000117] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps3 | bounded_mse | 39 | 0.89324359048010082 | [0.7823462005777662, 1.0605669809201397] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps10 | feature_accuracy | 39 | 0.36486486486486491 | [0.33783783783783783, 0.3918918918918919] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps10 | standardized_mse | 39 | 2.1879979409071662 | [1.870191438192598, 2.492739695164062] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps10 | bounded_mse | 39 | 0.89324359048010082 | [0.7823462005777662, 1.0605669809201397] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps30 | feature_accuracy | 39 | 0.36486486486486491 | [0.32432432432432434, 0.3783783783783784] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps30 | standardized_mse | 39 | 2.0810848961805291 | [1.7956875780791024, 2.4991793065020698] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps30 | bounded_mse | 39 | 0.89324359048010082 | [0.7823462005777662, 1.0605669809201397] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps100 | feature_accuracy | 39 | 0.3783783783783784 | [0.3513513513513513, 0.3918918918918919] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps100 | standardized_mse | 39 | 2.0472134806383879 | [1.8292615716757594, 2.495709861169733] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps100 | bounded_mse | 39 | 0.89324359048010082 | [0.7823462005777662, 1.0605669809201397] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps1000 | feature_accuracy | 39 | 0.39189189189189189 | [0.3648648648648649, 0.41891891891891886] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps1000 | standardized_mse | 39 | 1.7489188636519648 | [1.58261145051709, 2.0477180718795194] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps1000 | bounded_mse | 39 | 0.89324359048010082 | [0.7823462005777662, 1.0605669809201397] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps10000 | feature_accuracy | 39 | 0.44594594594594594 | [0.43243243243243246, 0.472972972972973] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps10000 | standardized_mse | 39 | 0.99770363800992568 | [0.8196071041460505, 1.270620873391675] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps10000 | bounded_mse | 39 | 0.80863224061347494 | [0.7252525851011581, 0.9412838432110665] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps1000000 | feature_accuracy | 39 | 0.91891891891891886 | [0.8513513513513513, 0.9459459459459459] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps1000000 | standardized_mse | 39 | 0.027616175067566542 | [0.020047252679968725, 0.03595086753436907] | [13, 27] |
| baf | tableak_batch2 | local_dp_global_eps1000000 | bounded_mse | 39 | 0.027616175067566542 | [0.020047252679968725, 0.03595086753436907] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps1 | feature_accuracy | 39 | 0.35135135135135132 | [0.33783783783783783, 0.3783783783783784] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps1 | standardized_mse | 39 | 2.2037348199032474 | [1.8461536228876942, 2.422565117684611] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps1 | bounded_mse | 39 | 0.89324359048010082 | [0.7823462005777662, 1.0605669809201397] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps3 | feature_accuracy | 39 | 0.35135135135135132 | [0.33783783783783783, 0.3918918918918919] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps3 | standardized_mse | 39 | 2.2426213602961349 | [1.9580443085391461, 2.423423270894194] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps3 | bounded_mse | 39 | 0.89324359048010082 | [0.7823462005777662, 1.0605669809201397] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps10 | feature_accuracy | 39 | 0.36486486486486491 | [0.33783783783783783, 0.3918918918918919] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps10 | standardized_mse | 39 | 2.2161030723061894 | [1.9695468403722531, 2.4056937025116323] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps10 | bounded_mse | 39 | 0.89324359048010082 | [0.7823462005777662, 1.0605669809201397] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps30 | feature_accuracy | 39 | 0.3783783783783784 | [0.3648648648648649, 0.4054054054054055] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps30 | standardized_mse | 39 | 1.7568050985523602 | [1.7075931030898448, 2.176932894050879] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps30 | bounded_mse | 39 | 0.89324359048010082 | [0.7823462005777662, 1.0605669809201397] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps100 | feature_accuracy | 39 | 0.36486486486486491 | [0.3513513513513513, 0.3918918918918919] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps100 | standardized_mse | 39 | 2.0951926975299204 | [1.807900756871331, 2.422925013575884] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps100 | bounded_mse | 39 | 0.89324359048010082 | [0.7823462005777662, 1.0605669809201397] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps1000 | feature_accuracy | 39 | 0.39189189189189189 | [0.3783783783783784, 0.41891891891891886] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps1000 | standardized_mse | 39 | 1.6332683744731322 | [1.3672106809712596, 1.8937934087083643] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps1000 | bounded_mse | 39 | 0.89324359048010082 | [0.7823462005777662, 1.0605669809201397] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps10000 | feature_accuracy | 39 | 0.44594594594594594 | [0.41891891891891886, 0.45945945945945943] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps10000 | standardized_mse | 39 | 0.95921854329530409 | [0.7811735485951239, 1.0414955478118493] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps10000 | bounded_mse | 39 | 0.84182706000579888 | [0.7252525851011581, 0.9592185432953041] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps1000000 | feature_accuracy | 39 | 0.71621621621621623 | [0.6756756756756757, 0.7837837837837838] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps1000000 | standardized_mse | 39 | 0.15245340812139913 | [0.0959300337741997, 0.2017684514150636] | [13, 27] |
| baf | tableak_batch2 | local_dp_per_tensor_eps1000000 | bounded_mse | 39 | 0.15245340812139913 | [0.0959300337741997, 0.2017684514150636] | [13, 27] |
| baf | tableak_batch2 | b2b_global_c001_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | tableak_batch2 | b2b_global_c001_clip_only | standardized_mse | 39 | 3.2365254989293548e-05 | [1.3126832288152554e-05, 0.000108225049895979] | [13, 27] |
| baf | tableak_batch2 | b2b_global_c001_clip_only | bounded_mse | 39 | 3.2365254989293548e-05 | [1.3126832288152554e-05, 0.000108225049895979] | [13, 27] |
| baf | tableak_batch2 | b2b_global_c01_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | tableak_batch2 | b2b_global_c01_clip_only | standardized_mse | 39 | 2.9880629820637848e-05 | [1.52075087449466e-05, 0.00011184947419922632] | [13, 27] |
| baf | tableak_batch2 | b2b_global_c01_clip_only | bounded_mse | 39 | 2.9880629820637848e-05 | [1.52075087449466e-05, 0.00011184947419922632] | [13, 27] |
| baf | tableak_batch2 | b2b_global_c01_eps10 | feature_accuracy | 39 | 0.36486486486486491 | [0.3513513513513513, 0.3918918918918919] | [13, 27] |
| baf | tableak_batch2 | b2b_global_c01_eps10 | standardized_mse | 39 | 2.1671100181357099 | [1.9753811519495974, 2.496199291605944] | [13, 27] |
| baf | tableak_batch2 | b2b_global_c01_eps10 | bounded_mse | 39 | 0.89324359048010082 | [0.7823462005777662, 1.0605669809201397] | [13, 27] |
| baf | tableak_batch2 | b2b_global_c01_eps1000 | feature_accuracy | 39 | 0.39189189189189189 | [0.3648648648648649, 0.4054054054054054] | [13, 27] |
| baf | tableak_batch2 | b2b_global_c01_eps1000 | standardized_mse | 39 | 1.7555659650323825 | [1.4185646657079707, 1.9512169616112387] | [13, 27] |
| baf | tableak_batch2 | b2b_global_c01_eps1000 | bounded_mse | 39 | 0.89324359048010082 | [0.7823462005777662, 1.0605669809201397] | [13, 27] |
| baf | tableak_batch2 | b2b_global_c01_eps1000000 | feature_accuracy | 39 | 0.91891891891891886 | [0.9054054054054054, 0.9459459459459459] | [13, 27] |
| baf | tableak_batch2 | b2b_global_c01_eps1000000 | standardized_mse | 39 | 0.024217289022650086 | [0.022844152735652313, 0.02948048826574959] | [13, 27] |
| baf | tableak_batch2 | b2b_global_c01_eps1000000 | bounded_mse | 39 | 0.024217289022650086 | [0.022844152735652313, 0.02948048826574959] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_half_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_half_clip_only | standardized_mse | 39 | 2.9193021860973958e-05 | [1.3793823785851542e-05, 0.00011180037426941221] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_half_clip_only | bounded_mse | 39 | 2.9193021860973958e-05 | [1.3793823785851542e-05, 0.00011180037426941221] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_half_eps10 | feature_accuracy | 39 | 0.35135135135135132 | [0.32432432432432434, 0.3783783783783783] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_half_eps10 | standardized_mse | 39 | 2.5821743713327123 | [2.315253953381866, 2.813480681485818] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_half_eps10 | bounded_mse | 39 | 0.89324359048010082 | [0.7823462005777662, 1.0605669809201397] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_half_eps1000 | feature_accuracy | 39 | 0.3783783783783784 | [0.3648648648648649, 0.43243243243243246] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_half_eps1000 | standardized_mse | 39 | 1.6868639873427083 | [1.4594681311054916, 2.0532903132466647] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_half_eps1000 | bounded_mse | 39 | 0.89324359048010082 | [0.7823462005777662, 1.0605669809201397] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_half_eps1000000 | feature_accuracy | 39 | 0.93243243243243246 | [0.8783783783783784, 0.9459459459459459] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_half_eps1000000 | standardized_mse | 39 | 0.024702000994710569 | [0.019011910984538995, 0.031535766999945126] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_half_eps1000000 | bounded_mse | 39 | 0.024702000994710569 | [0.019011910984538995, 0.031535766999945126] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_clip_only | standardized_mse | 39 | 3.157492852674418e-05 | [1.3793823785851542e-05, 0.00011180037426941221] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_clip_only | bounded_mse | 39 | 3.157492852674418e-05 | [1.3793823785851542e-05, 0.00011180037426941221] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_eps10 | feature_accuracy | 39 | 0.36486486486486491 | [0.3513513513513513, 0.3918918918918919] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_eps10 | standardized_mse | 39 | 2.0251995733707755 | [1.859387897687942, 2.242875217684397] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_eps10 | bounded_mse | 39 | 0.89324359048010082 | [0.7823462005777662, 1.0605669809201397] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_eps1000 | feature_accuracy | 39 | 0.37837837837837829 | [0.3513513513513513, 0.41891891891891886] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_eps1000 | standardized_mse | 39 | 1.835300196639476 | [1.5402672510708186, 2.0179432007734976] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_eps1000 | bounded_mse | 39 | 0.89324359048010082 | [0.7823462005777662, 1.0605669809201397] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_eps1000000 | feature_accuracy | 39 | 0.93243243243243246 | [0.9054054054054054, 0.9459459459459459] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_eps1000000 | standardized_mse | 39 | 0.023656120873701527 | [0.020443688253383826, 0.028739289569052667] | [13, 27] |
| baf | tableak_batch2 | b2b_global_cmed_eps1000000 | bounded_mse | 39 | 0.023656120873701527 | [0.020443688253383826, 0.028739289569052667] | [13, 27] |
| ieee_cis | ratio_batch1 | unprotected | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| ieee_cis | ratio_batch1 | unprotected | standardized_mse | 39 | 3.2595308338522188e-16 | [1.8691281015945058e-16, 5.059561546571682e-16] | [13, 27] |
| ieee_cis | ratio_batch1 | unprotected | bounded_mse | 39 | 3.2595308338522188e-16 | [1.8691281015945058e-16, 5.059561546571682e-16] | [13, 27] |
| ieee_cis | ratio_batch1 | dna_v1_conservative | feature_accuracy | 39 | 0.96666666666666667 | [0.8833333333333333, 0.9785714285714285] | [13, 27] |
| ieee_cis | ratio_batch1 | dna_v1_conservative | standardized_mse | 39 | 7.850644543156422 | [2.40360010628441, 51.66319001243269] | [13, 27] |
| ieee_cis | ratio_batch1 | dna_v1_conservative | bounded_mse | 39 | 0.3552153664997279 | [0.2683233801176002, 0.520987017678775] | [13, 27] |
| ieee_cis | ratio_batch1 | dna_v2_0p95 | feature_accuracy | 39 | 0.65476190476190477 | [0.5595238095238095, 0.7785714285714286] | [13, 27] |
| ieee_cis | ratio_batch1 | dna_v2_0p95 | standardized_mse | 39 | 123.94260353395855 | [32.166094375603016, 289.7837271363815] | [13, 27] |
| ieee_cis | ratio_batch1 | dna_v2_0p95 | bounded_mse | 39 | 0.3457460717883481 | [0.2625385277659348, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps1 | feature_accuracy | 39 | 0.60714285714285721 | [0.5928571428571429, 0.6214285714285714] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps1 | standardized_mse | 39 | 84.851862561720367 | [58.84110859527418, 106.12533810508282] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps1 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps3 | feature_accuracy | 39 | 0.60476190476190483 | [0.5833333333333333, 0.6190476190476191] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps3 | standardized_mse | 39 | 103.07767021524501 | [55.53979286517151, 120.76517636530423] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps3 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps10 | feature_accuracy | 39 | 0.60952380952380958 | [0.5976190476190476, 0.6238095238095238] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps10 | standardized_mse | 39 | 61.114965613404735 | [43.48163738302335, 93.4043236174297] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps10 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps30 | feature_accuracy | 39 | 0.59999999999999998 | [0.5785714285714285, 0.6142857142857143] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps30 | standardized_mse | 39 | 69.930689698886297 | [49.169880838776464, 111.46036914416976] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps30 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps100 | feature_accuracy | 39 | 0.61190476190476195 | [0.5904761904761905, 0.6190476190476191] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps100 | standardized_mse | 39 | 63.588877268547677 | [48.5661234224498, 97.906925391949] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps100 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps1000 | feature_accuracy | 39 | 0.61428571428571432 | [0.5833333333333333, 0.6261904761904762] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps1000 | standardized_mse | 39 | 76.855892087943545 | [59.52447752341808, 122.23515142310454] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps1000 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps10000 | feature_accuracy | 39 | 0.60476190476190483 | [0.5976190476190476, 0.6309523809523809] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps10000 | standardized_mse | 39 | 77.816479391636562 | [40.68297528260683, 118.13676965792384] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps10000 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps1000000 | feature_accuracy | 39 | 0.60238095238095246 | [0.5833333333333333, 0.6190476190476191] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps1000000 | standardized_mse | 39 | 93.269697686958835 | [61.03156713038648, 117.91731792229577] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_global_eps1000000 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps1 | feature_accuracy | 39 | 0.60476190476190483 | [0.5857142857142856, 0.6142857142857143] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps1 | standardized_mse | 39 | 60.938573928086385 | [45.92111625459868, 91.15529508659019] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps1 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps3 | feature_accuracy | 39 | 0.60714285714285721 | [0.5833333333333333, 0.6214285714285714] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps3 | standardized_mse | 39 | 45.758681948833704 | [31.012627843902447, 112.64052896684852] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps3 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps10 | feature_accuracy | 39 | 0.60476190476190483 | [0.5880952380952381, 0.611904761904762] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps10 | standardized_mse | 39 | 67.565385619131845 | [42.841951678981516, 95.73720705215507] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps10 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps30 | feature_accuracy | 39 | 0.60238095238095246 | [0.5809523809523809, 0.6238095238095238] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps30 | standardized_mse | 39 | 64.809132167318253 | [52.58801745343276, 100.88596013619258] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps30 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps100 | feature_accuracy | 39 | 0.60714285714285721 | [0.5880952380952381, 0.6214285714285714] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps100 | standardized_mse | 39 | 69.874524547654076 | [38.04274846918746, 103.45280436597899] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps100 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps1000 | feature_accuracy | 39 | 0.60952380952380958 | [0.5833333333333333, 0.6261904761904762] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps1000 | standardized_mse | 39 | 94.026962145123719 | [61.26323764267753, 118.90099573943176] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps1000 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps10000 | feature_accuracy | 39 | 0.59999999999999998 | [0.5833333333333333, 0.6214285714285714] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps10000 | standardized_mse | 39 | 67.943499752105879 | [47.59119932527858, 86.6180274472756] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps10000 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps1000000 | feature_accuracy | 39 | 0.65238095238095239 | [0.6357142857142857, 0.6761904761904762] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps1000000 | standardized_mse | 39 | 26.635449752145156 | [12.027692854620415, 44.355286847612334] | [13, 27] |
| ieee_cis | ratio_batch1 | local_dp_per_tensor_eps1000000 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_c001_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_c001_clip_only | standardized_mse | 39 | 5.7674453448358321e-16 | [4.313193126655611e-16, 8.810354406813972e-16] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_c001_clip_only | bounded_mse | 39 | 5.7674453448358321e-16 | [4.313193126655611e-16, 8.810354406813972e-16] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_c01_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_c01_clip_only | standardized_mse | 39 | 6.1782395259537006e-16 | [3.9190173048874356e-16, 8.713049539569055e-16] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_c01_clip_only | bounded_mse | 39 | 6.1782395259537006e-16 | [3.9190173048874356e-16, 8.713049539569055e-16] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_c01_eps10 | feature_accuracy | 39 | 0.60952380952380958 | [0.5761904761904761, 0.6214285714285714] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_c01_eps10 | standardized_mse | 39 | 63.91051842366403 | [41.6107390378112, 85.49426619267346] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_c01_eps10 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_c01_eps1000 | feature_accuracy | 39 | 0.60476190476190483 | [0.5952380952380952, 0.6190476190476191] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_c01_eps1000 | standardized_mse | 39 | 82.638713939546307 | [63.39698939473907, 119.86876685281305] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_c01_eps1000 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_c01_eps1000000 | feature_accuracy | 39 | 0.60952380952380958 | [0.5833333333333333, 0.6285714285714286] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_c01_eps1000000 | standardized_mse | 39 | 66.944316670756521 | [50.568957577787806, 83.81363547633323] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_c01_eps1000000 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_half_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_half_clip_only | standardized_mse | 39 | 5.6566562825045401e-16 | [3.660412309393078e-16, 7.689635576283746e-16] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_half_clip_only | bounded_mse | 39 | 5.6566562825045401e-16 | [3.660412309393078e-16, 7.689635576283746e-16] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_half_eps10 | feature_accuracy | 39 | 0.60714285714285721 | [0.5833333333333333, 0.6238095238095238] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_half_eps10 | standardized_mse | 39 | 78.090523645348398 | [57.26041341090271, 128.59666339106604] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_half_eps10 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_half_eps1000 | feature_accuracy | 39 | 0.59523809523809523 | [0.5785714285714285, 0.6214285714285714] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_half_eps1000 | standardized_mse | 39 | 93.87483115317869 | [61.655871410453486, 153.91874668207353] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_half_eps1000 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_half_eps1000000 | feature_accuracy | 39 | 0.59999999999999998 | [0.5833333333333333, 0.6142857142857143] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_half_eps1000000 | standardized_mse | 39 | 85.622009633591119 | [53.99290129834884, 118.19761211714135] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_half_eps1000000 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_clip_only | standardized_mse | 39 | 5.6566562825045401e-16 | [3.660412309393078e-16, 7.672142566441963e-16] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_clip_only | bounded_mse | 39 | 5.6566562825045401e-16 | [3.660412309393078e-16, 7.672142566441963e-16] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_eps10 | feature_accuracy | 39 | 0.60476190476190483 | [0.5857142857142856, 0.6190476190476191] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_eps10 | standardized_mse | 39 | 60.333197193578386 | [28.723657521815223, 86.98342506735182] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_eps10 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_eps1000 | feature_accuracy | 39 | 0.60714285714285721 | [0.6023809523809525, 0.6238095238095238] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_eps1000 | standardized_mse | 39 | 81.264420271966898 | [48.74082821729737, 140.83335257265455] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_eps1000 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_eps1000000 | feature_accuracy | 39 | 0.61190476190476195 | [0.6023809523809525, 0.6190476190476191] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_eps1000000 | standardized_mse | 39 | 68.206430222920886 | [47.79370368243157, 96.928070824193] | [13, 27] |
| ieee_cis | ratio_batch1 | b2b_global_cmed_eps1000000 | bounded_mse | 39 | 0.3552153664997279 | [0.27954952346791706, 0.5676439569995847] | [13, 27] |
| ieee_cis | tableak_batch1 | unprotected | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| ieee_cis | tableak_batch1 | unprotected | standardized_mse | 39 | 5.2188056762689603e-05 | [2.8327908400846805e-05, 6.196224510245767e-05] | [13, 27] |
| ieee_cis | tableak_batch1 | unprotected | bounded_mse | 39 | 5.2188056762689603e-05 | [2.8327908400846805e-05, 6.196224510245767e-05] | [13, 27] |
| ieee_cis | tableak_batch1 | dna_v1_conservative | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| ieee_cis | tableak_batch1 | dna_v1_conservative | standardized_mse | 39 | 0.00080972567292951546 | [0.000730349131213591, 0.001371099328235959] | [13, 27] |
| ieee_cis | tableak_batch1 | dna_v1_conservative | bounded_mse | 39 | 0.00080972567292951546 | [0.000730349131213591, 0.001371099328235959] | [13, 27] |
| ieee_cis | tableak_batch1 | dna_v2_0p95 | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| ieee_cis | tableak_batch1 | dna_v2_0p95 | standardized_mse | 39 | 4.926073196530291e-05 | [2.421027008980861e-05, 5.655552771291891e-05] | [13, 27] |
| ieee_cis | tableak_batch1 | dna_v2_0p95 | bounded_mse | 39 | 4.926073196530291e-05 | [2.421027008980861e-05, 5.655552771291891e-05] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps1 | feature_accuracy | 39 | 0.33333333333333337 | [0.31481481481481477, 0.35185185185185186] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps1 | standardized_mse | 39 | 13.416080011496497 | [9.550382045109245, 24.950312677109412] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps1 | bounded_mse | 39 | 0.58314912274509589 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps3 | feature_accuracy | 39 | 0.35185185185185186 | [0.31481481481481477, 0.38888888888888884] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps3 | standardized_mse | 39 | 18.546034777249677 | [12.541378860999428, 33.09155579580169] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps3 | bounded_mse | 39 | 0.58314912274509589 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps10 | feature_accuracy | 39 | 0.37037037037037035 | [0.31481481481481477, 0.40740740740740744] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps10 | standardized_mse | 39 | 16.488115571082332 | [10.299353135903795, 25.607037515654493] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps10 | bounded_mse | 39 | 0.58314912274509589 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps30 | feature_accuracy | 39 | 0.35185185185185186 | [0.31481481481481477, 0.38888888888888884] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps30 | standardized_mse | 39 | 16.225112627892443 | [9.93987375912527, 23.242385154516647] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps30 | bounded_mse | 39 | 0.58314912274509589 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps100 | feature_accuracy | 39 | 0.33333333333333337 | [0.2777777777777778, 0.37037037037037035] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps100 | standardized_mse | 39 | 17.427014739617302 | [9.090144081096874, 25.623458190372858] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps100 | bounded_mse | 39 | 0.58314912274509589 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps1000 | feature_accuracy | 39 | 0.38888888888888884 | [0.31481481481481477, 0.42592592592592593] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps1000 | standardized_mse | 39 | 7.0047358334451992 | [5.285309565369336, 9.687908434543715] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps1000 | bounded_mse | 39 | 0.58314912274509589 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps10000 | feature_accuracy | 39 | 0.5 | [0.4444444444444444, 0.5555555555555556] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps10000 | standardized_mse | 39 | 0.84233776483318101 | [0.40172350840504606, 1.5018905372834743] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps10000 | bounded_mse | 39 | 0.45293544421732751 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps1000000 | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps1000000 | standardized_mse | 39 | 0.0056520812186301042 | [0.0044461285407570075, 0.0069135060502596] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_global_eps1000000 | bounded_mse | 39 | 0.0056520812186301042 | [0.0044461285407570075, 0.0069135060502596] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps1 | feature_accuracy | 39 | 0.35185185185185186 | [0.31481481481481477, 0.37037037037037035] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps1 | standardized_mse | 39 | 13.123512014651929 | [8.183382387009482, 25.61226431443692] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps1 | bounded_mse | 39 | 0.58314912274509589 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps3 | feature_accuracy | 39 | 0.35185185185185186 | [0.2777777777777778, 0.38888888888888884] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps3 | standardized_mse | 39 | 17.538170813239283 | [9.625057031944994, 28.912669965333368] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps3 | bounded_mse | 39 | 0.58314912274509589 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps10 | feature_accuracy | 39 | 0.33333333333333337 | [0.2962962962962963, 0.37037037037037035] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps10 | standardized_mse | 39 | 13.766523082321603 | [9.565741622473148, 23.398310405062027] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps10 | bounded_mse | 39 | 0.58314912274509589 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps30 | feature_accuracy | 39 | 0.35185185185185186 | [0.31481481481481477, 0.37037037037037035] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps30 | standardized_mse | 39 | 15.704207692053105 | [9.436844034822816, 22.964718110780456] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps30 | bounded_mse | 39 | 0.58314912274509589 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps100 | feature_accuracy | 39 | 0.37037037037037035 | [0.31481481481481477, 0.40740740740740744] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps100 | standardized_mse | 39 | 10.867383536740606 | [8.548593902856373, 26.360544647164357] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps100 | bounded_mse | 39 | 0.58314912274509589 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps1000 | feature_accuracy | 39 | 0.38888888888888884 | [0.35185185185185186, 0.4444444444444444] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps1000 | standardized_mse | 39 | 5.5639913040243067 | [4.18496101765597, 9.361354997804021] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps1000 | bounded_mse | 39 | 0.58314912274509589 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps10000 | feature_accuracy | 39 | 0.53703703703703698 | [0.4814814814814815, 0.5925925925925926] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps10000 | standardized_mse | 39 | 0.70039671195024034 | [0.5397095208223237, 1.4957273079145117] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps10000 | bounded_mse | 39 | 0.43412621054663736 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps1000000 | feature_accuracy | 39 | 0.98148148148148151 | [0.962962962962963, 1.0] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps1000000 | standardized_mse | 39 | 0.018155214179349306 | [0.009041483009522492, 0.04290345654492659] | [13, 27] |
| ieee_cis | tableak_batch1 | local_dp_per_tensor_eps1000000 | bounded_mse | 39 | 0.018155214179349306 | [0.009041483009522492, 0.04290345654492659] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_c001_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_c001_clip_only | standardized_mse | 39 | 5.4118683233527987e-05 | [2.18361252106814e-05, 6.193816253780003e-05] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_c001_clip_only | bounded_mse | 39 | 5.4118683233527987e-05 | [2.18361252106814e-05, 6.193816253780003e-05] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_c01_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_c01_clip_only | standardized_mse | 39 | 5.4337951020298209e-05 | [2.3176000729694543e-05, 6.261548893946382e-05] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_c01_clip_only | bounded_mse | 39 | 5.4337951020298209e-05 | [2.3176000729694543e-05, 6.261548893946382e-05] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_c01_eps10 | feature_accuracy | 39 | 0.35185185185185186 | [0.2962962962962963, 0.37037037037037035] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_c01_eps10 | standardized_mse | 39 | 16.12489280795214 | [10.387747071732502, 33.8597562506382] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_c01_eps10 | bounded_mse | 39 | 0.58314912274509589 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_c01_eps1000 | feature_accuracy | 39 | 0.40740740740740744 | [0.33333333333333337, 0.4444444444444444] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_c01_eps1000 | standardized_mse | 39 | 7.5511222417985522 | [4.593154806469422, 10.725248036844368] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_c01_eps1000 | bounded_mse | 39 | 0.58314912274509589 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_c01_eps1000000 | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_c01_eps1000000 | standardized_mse | 39 | 0.0050493532683381064 | [0.003645092886761655, 0.007355109206516697] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_c01_eps1000000 | bounded_mse | 39 | 0.0050493532683381064 | [0.003645092886761655, 0.007355109206516697] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_half_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_half_clip_only | standardized_mse | 39 | 5.3677310361805152e-05 | [2.3959875099509226e-05, 6.186047586328982e-05] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_half_clip_only | bounded_mse | 39 | 5.3677310361805152e-05 | [2.3959875099509226e-05, 6.186047586328982e-05] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_half_eps10 | feature_accuracy | 39 | 0.35185185185185186 | [0.2962962962962963, 0.38888888888888884] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_half_eps10 | standardized_mse | 39 | 16.479943354491887 | [8.784707795755942, 20.7840079485377] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_half_eps10 | bounded_mse | 39 | 0.58314912274509589 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_half_eps1000 | feature_accuracy | 39 | 0.37037037037037035 | [0.31481481481481477, 0.38888888888888884] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_half_eps1000 | standardized_mse | 39 | 7.6215374933376321 | [5.227372005939197, 8.93173782886989] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_half_eps1000 | bounded_mse | 39 | 0.58314912274509589 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_half_eps1000000 | feature_accuracy | 39 | 1 | [0.9814814814814815, 1.0] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_half_eps1000000 | standardized_mse | 39 | 0.010448189444188532 | [0.007304992185239463, 0.014761051887764103] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_half_eps1000000 | bounded_mse | 39 | 0.010448189444188532 | [0.007304992185239463, 0.014761051887764103] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_clip_only | standardized_mse | 39 | 5.3677310361805152e-05 | [2.3959875099509226e-05, 6.186047586328982e-05] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_clip_only | bounded_mse | 39 | 5.3677310361805152e-05 | [2.3959875099509226e-05, 6.186047586328982e-05] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_eps10 | feature_accuracy | 39 | 0.35185185185185186 | [0.2777777777777778, 0.40740740740740744] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_eps10 | standardized_mse | 39 | 19.287710922632936 | [13.73741697201547, 31.64672210689715] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_eps10 | bounded_mse | 39 | 0.58314912274509589 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_eps1000 | feature_accuracy | 39 | 0.37037037037037035 | [0.33333333333333337, 0.40740740740740744] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_eps1000 | standardized_mse | 39 | 9.5971322313591294 | [6.705654445956932, 14.827078324503846] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_eps1000 | bounded_mse | 39 | 0.58314912274509589 | [0.04910925127010084, 0.5968965703203362] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_eps1000000 | feature_accuracy | 39 | 0.88888888888888884 | [0.8703703703703703, 0.9074074074074074] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_eps1000000 | standardized_mse | 39 | 0.032881681105592621 | [0.028963608724366516, 0.03846363193069558] | [13, 27] |
| ieee_cis | tableak_batch1 | b2b_global_cmed_eps1000000 | bounded_mse | 39 | 0.032881681105592621 | [0.028963608724366516, 0.03846363193069558] | [13, 27] |
| paysim | ratio_batch1 | unprotected | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | ratio_batch1 | unprotected | standardized_mse | 39 | 2.4085968262601053e-17 | [0.0, 6.832141690000964e-17] | [13, 27] |
| paysim | ratio_batch1 | unprotected | bounded_mse | 39 | 2.4085968262601053e-17 | [0.0, 6.832141690000964e-17] | [13, 27] |
| paysim | ratio_batch1 | dna_v1_conservative | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | ratio_batch1 | dna_v1_conservative | standardized_mse | 39 | 0.00086593691814288104 | [0.00023156655678353415, 0.004320375559346979] | [13, 27] |
| paysim | ratio_batch1 | dna_v1_conservative | bounded_mse | 39 | 0.00086593691814288104 | [0.00023156655678353415, 0.004320375559346979] | [13, 27] |
| paysim | ratio_batch1 | dna_v2_0p95 | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | ratio_batch1 | dna_v2_0p95 | standardized_mse | 39 | 0.0089326711665205141 | [0.003587299852116323, 0.01767797079774048] | [13, 27] |
| paysim | ratio_batch1 | dna_v2_0p95 | bounded_mse | 39 | 0.0089326711665205141 | [0.003587299852116323, 0.01767797079774048] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps1 | feature_accuracy | 39 | 0.77777777777777779 | [0.5555555555555556, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps1 | standardized_mse | 39 | 2.458054629502993 | [1.764402137854908, 3.434181429160591] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps1 | bounded_mse | 39 | 0.73963207692575295 | [0.49017495573479997, 1.0522810887292686] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps3 | feature_accuracy | 39 | 0.77777777777777779 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps3 | standardized_mse | 39 | 2.0512906663021506 | [1.0925352560521053, 3.4568930188510794] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps3 | bounded_mse | 39 | 0.73769794951440992 | [0.5599700049657133, 1.198087250074445] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps10 | feature_accuracy | 39 | 0.77777777777777779 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps10 | standardized_mse | 39 | 1.9717105468047802 | [1.4705512207943618, 3.055447851398985] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps10 | bounded_mse | 39 | 0.78007075918040425 | [0.49017495573479997, 1.198087250074445] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps30 | feature_accuracy | 39 | 0.77777777777777779 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps30 | standardized_mse | 39 | 2.0924210381806554 | [1.3701810892136885, 3.4281054443449794] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps30 | bounded_mse | 39 | 0.75162716597590395 | [0.49017495573479997, 1.0522810887292686] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps100 | feature_accuracy | 39 | 0.77777777777777779 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps100 | standardized_mse | 39 | 1.9995483530844045 | [1.4798813077484116, 3.0528681373451882] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps100 | bounded_mse | 39 | 0.73963207692575295 | [0.6207922523534786, 0.9308255885408591] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps1000 | feature_accuracy | 39 | 0.77777777777777779 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps1000 | standardized_mse | 39 | 2.0216989784192654 | [1.4666806850992662, 3.145428501529445] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps1000 | bounded_mse | 39 | 0.75162716597590395 | [0.5501180582969932, 1.0522810887292686] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps10000 | feature_accuracy | 39 | 0.77777777777777779 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps10000 | standardized_mse | 39 | 1.5001731669526772 | [0.936742988274264, 2.8905306973507856] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps10000 | bounded_mse | 39 | 0.75788760726406701 | [0.4216527671531517, 1.198087250074445] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps1000000 | feature_accuracy | 39 | 0.77777777777777779 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps1000000 | standardized_mse | 39 | 1.4179578867047593 | [0.9212835384888027, 2.205758537120665] | [13, 27] |
| paysim | ratio_batch1 | local_dp_global_eps1000000 | bounded_mse | 39 | 0.75078135255459622 | [0.5350810664359021, 0.9308255885408591] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps1 | feature_accuracy | 39 | 0.77777777777777779 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps1 | standardized_mse | 39 | 2.0898160331984692 | [1.3712038598681342, 2.8155301421467254] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps1 | bounded_mse | 39 | 0.78007075918040425 | [0.7178513773827089, 1.198087250074445] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps3 | feature_accuracy | 39 | 0.77777777777777779 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps3 | standardized_mse | 39 | 2.4790250843169068 | [1.5696695120413475, 3.391594939307574] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps3 | bounded_mse | 39 | 0.75788760726406701 | [0.7178513773827089, 1.198087250074445] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps10 | feature_accuracy | 39 | 0.66666666666666674 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps10 | standardized_mse | 39 | 2.0413374938652806 | [1.6725408968045579, 3.2786651312445563] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps10 | bounded_mse | 39 | 0.75788760726406701 | [0.6035520294616499, 1.198087250074445] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps30 | feature_accuracy | 39 | 0.77777777777777779 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps30 | standardized_mse | 39 | 2.8471369890527281 | [1.4285384296604566, 3.4798243144277725] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps30 | bounded_mse | 39 | 0.78007075918040425 | [0.6746629475007077, 1.198087250074445] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps100 | feature_accuracy | 39 | 0.77777777777777779 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps100 | standardized_mse | 39 | 1.6666657451266771 | [0.9293022716731685, 3.4364175941937343] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps100 | bounded_mse | 39 | 0.75162716597590395 | [0.5971769278316201, 1.0522810887292686] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps1000 | feature_accuracy | 39 | 0.77777777777777779 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps1000 | standardized_mse | 39 | 2.1740889420151004 | [1.2374797618028857, 3.071622062930208] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps1000 | bounded_mse | 39 | 0.75162716597590395 | [0.581454814277831, 1.198087250074445] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps10000 | feature_accuracy | 39 | 0.77777777777777779 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps10000 | standardized_mse | 39 | 1.683240887677421 | [1.0008045204247291, 3.1413686977719264] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps10000 | bounded_mse | 39 | 0.73504823645145556 | [0.44750223410183715, 1.198087250074445] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps1000000 | feature_accuracy | 39 | 0.88888888888888884 | [0.7777777777777778, 0.8888888888888888] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps1000000 | standardized_mse | 39 | 1.3338215102245867 | [0.5659355359373399, 2.06710352897071] | [13, 27] |
| paysim | ratio_batch1 | local_dp_per_tensor_eps1000000 | bounded_mse | 39 | 0.56593553593733992 | [0.39433477123343963, 0.8103067520826427] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_c001_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_c001_clip_only | standardized_mse | 39 | 6.8321416900009637e-17 | [6.832141690000964e-17, 1.5479071016408432e-16] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_c001_clip_only | bounded_mse | 39 | 6.8321416900009637e-17 | [6.832141690000964e-17, 1.5479071016408432e-16] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_c01_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_c01_clip_only | standardized_mse | 39 | 6.8321416900009637e-17 | [6.832141690000964e-17, 2.7328566760003855e-16] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_c01_clip_only | bounded_mse | 39 | 6.8321416900009637e-17 | [6.832141690000964e-17, 2.7328566760003855e-16] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_c01_eps10 | feature_accuracy | 39 | 0.66666666666666674 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_c01_eps10 | standardized_mse | 39 | 1.588941279651789 | [1.236087312918173, 2.422878627436879] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_c01_eps10 | bounded_mse | 39 | 0.73963207692575295 | [0.5963335856734735, 0.9308255885408591] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_c01_eps1000 | feature_accuracy | 39 | 0.66666666666666674 | [0.5555555555555556, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_c01_eps1000 | standardized_mse | 39 | 2.0321051752746189 | [1.3897773354435334, 2.33953078329639] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_c01_eps1000 | bounded_mse | 39 | 0.78007075918040425 | [0.5231103801669199, 1.198087250074445] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_c01_eps1000000 | feature_accuracy | 39 | 0.77777777777777779 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_c01_eps1000000 | standardized_mse | 39 | 3.6261827760896241 | [1.7812618751044296, 4.655353763645712] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_c01_eps1000000 | bounded_mse | 39 | 0.78007075918040425 | [0.49017495573479997, 1.0694372836669241] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_half_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_half_clip_only | standardized_mse | 39 | 6.8321416900009637e-17 | [6.832141690000964e-17, 2.7328566760003855e-16] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_half_clip_only | bounded_mse | 39 | 6.8321416900009637e-17 | [6.832141690000964e-17, 2.7328566760003855e-16] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_half_eps10 | feature_accuracy | 39 | 0.77777777777777779 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_half_eps10 | standardized_mse | 39 | 1.9864433755049107 | [1.3627724967000203, 3.3729234986555596] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_half_eps10 | bounded_mse | 39 | 0.75788760726406701 | [0.7178513773827089, 0.9308255885408591] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_half_eps1000 | feature_accuracy | 39 | 0.77777777777777779 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_half_eps1000 | standardized_mse | 39 | 1.5566914296782524 | [1.264782901671811, 2.128234814531694] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_half_eps1000 | bounded_mse | 39 | 0.75162716597590395 | [0.6315916150726832, 1.198087250074445] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_half_eps1000000 | feature_accuracy | 39 | 0.77777777777777779 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_half_eps1000000 | standardized_mse | 39 | 1.7484498863920912 | [1.215343903523873, 3.4180443794493414] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_half_eps1000000 | bounded_mse | 39 | 0.73769794951440992 | [0.5323160757755491, 0.9308255885408591] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_clip_only | standardized_mse | 39 | 6.8321416900009637e-17 | [6.832141690000964e-17, 2.7328566760003855e-16] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_clip_only | bounded_mse | 39 | 6.8321416900009637e-17 | [6.832141690000964e-17, 2.7328566760003855e-16] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_eps10 | feature_accuracy | 39 | 0.77777777777777779 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_eps10 | standardized_mse | 39 | 1.9396546022480554 | [1.1668550294027404, 2.6156981383292592] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_eps10 | bounded_mse | 39 | 0.73963207692575295 | [0.4873140611324358, 1.0522810887292686] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_eps1000 | feature_accuracy | 39 | 0.77777777777777779 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_eps1000 | standardized_mse | 39 | 2.7752171658808171 | [1.3825243868731307, 4.165795573765318] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_eps1000 | bounded_mse | 39 | 0.75162716597590395 | [0.49017495573479997, 1.0343902409940822] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_eps1000000 | feature_accuracy | 39 | 0.66666666666666674 | [0.6666666666666667, 0.7777777777777778] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_eps1000000 | standardized_mse | 39 | 1.8814658664097395 | [1.5547602802364284, 2.7555394630435703] | [13, 27] |
| paysim | ratio_batch1 | b2b_global_cmed_eps1000000 | bounded_mse | 39 | 0.75162716597590395 | [0.49017495573479997, 1.1836055623861546] | [13, 27] |
| paysim | tableak_batch1 | unprotected | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch1 | unprotected | standardized_mse | 39 | 3.0671600860971301e-05 | [1.2213518942291082e-05, 4.720446504260426e-05] | [13, 27] |
| paysim | tableak_batch1 | unprotected | bounded_mse | 39 | 3.0671600860971301e-05 | [1.2213518942291082e-05, 4.720446504260426e-05] | [13, 27] |
| paysim | tableak_batch1 | dna_v1_conservative | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch1 | dna_v1_conservative | standardized_mse | 39 | 0.0003657317981744 | [0.00019017187013169713, 0.0011143556713750608] | [13, 27] |
| paysim | tableak_batch1 | dna_v1_conservative | bounded_mse | 39 | 0.0003657317981744 | [0.00019017187013169713, 0.0011143556713750608] | [13, 27] |
| paysim | tableak_batch1 | dna_v2_0p95 | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch1 | dna_v2_0p95 | standardized_mse | 39 | 2.2349376027822695e-05 | [1.2419311021514791e-05, 3.4872409862684286e-05] | [13, 27] |
| paysim | tableak_batch1 | dna_v2_0p95 | bounded_mse | 39 | 2.2349376027822695e-05 | [1.2419311021514791e-05, 3.4872409862684286e-05] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps1 | feature_accuracy | 39 | 0.33333333333333337 | [0.2222222222222222, 0.33333333333333337] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps1 | standardized_mse | 39 | 6.8483101345273019 | [3.4562648394387177, 28.081940808799782] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps1 | bounded_mse | 39 | 0.72567004895866483 | [0.4122947504558796, 0.8718351994290529] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps3 | feature_accuracy | 39 | 0.22222222222222221 | [0.2222222222222222, 0.33333333333333337] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps3 | standardized_mse | 39 | 5.1165964582717187 | [3.339730737124388, 41.224771030510176] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps3 | bounded_mse | 39 | 0.7308004658161672 | [0.4122947504558796, 0.8718351994290529] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps10 | feature_accuracy | 39 | 0.22222222222222221 | [0.2222222222222222, 0.33333333333333337] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps10 | standardized_mse | 39 | 7.3294175292818755 | [4.533093835727757, 15.412620748347784] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps10 | bounded_mse | 39 | 0.72715056744668627 | [0.4122947504558796, 0.863440666168725] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps30 | feature_accuracy | 39 | 0.22222222222222221 | [0.2222222222222222, 0.33333333333333337] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps30 | standardized_mse | 39 | 7.3400681216018588 | [3.8708434369339946, 18.58342555856531] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps30 | bounded_mse | 39 | 0.72715056744668627 | [0.4122947504558796, 0.863440666168725] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps100 | feature_accuracy | 39 | 0.22222222222222221 | [0.2222222222222222, 0.33333333333333337] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps100 | standardized_mse | 39 | 3.1435890476328301 | [2.007390989358566, 7.207103702232404] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps100 | bounded_mse | 39 | 0.72715056744668627 | [0.4122947504558796, 0.863440666168725] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps1000 | feature_accuracy | 39 | 0.33333333333333337 | [0.2222222222222222, 0.4444444444444444] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps1000 | standardized_mse | 39 | 1.1643645828987548 | [0.7371061354933827, 2.5240285199956336] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps1000 | bounded_mse | 39 | 0.54109132680843652 | [0.32732452964383507, 0.7256700489586648] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps10000 | feature_accuracy | 39 | 0.66666666666666674 | [0.5555555555555556, 0.7777777777777778] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps10000 | standardized_mse | 39 | 0.11913672346271291 | [0.07525834241313455, 0.15426187383059822] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps10000 | bounded_mse | 39 | 0.10613512242629801 | [0.07458314710546882, 0.15426187383059822] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps1000000 | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps1000000 | standardized_mse | 39 | 0.00077414982477394317 | [0.000527742973919478, 0.0009885078421147779] | [13, 27] |
| paysim | tableak_batch1 | local_dp_global_eps1000000 | bounded_mse | 39 | 0.00077414982477394317 | [0.000527742973919478, 0.0009885078421147779] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps1 | feature_accuracy | 39 | 0.33333333333333337 | [0.11111111111111116, 0.33333333333333337] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps1 | standardized_mse | 39 | 5.7856882760873853 | [3.8343809106876896, 32.10366209995882] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps1 | bounded_mse | 39 | 0.72715056744668627 | [0.4122947504558796, 0.863440666168725] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps3 | feature_accuracy | 39 | 0.22222222222222221 | [0.2222222222222222, 0.33333333333333337] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps3 | standardized_mse | 39 | 6.2322954340752537 | [3.278687312874116, 18.721854811796145] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps3 | bounded_mse | 39 | 0.7308004658161672 | [0.4122947504558796, 0.863440666168725] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps10 | feature_accuracy | 39 | 0.22222222222222221 | [0.2222222222222222, 0.33333333333333337] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps10 | standardized_mse | 39 | 4.8621083130268055 | [2.9121899317896185, 26.001299996813025] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps10 | bounded_mse | 39 | 0.72567004895866483 | [0.4122947504558796, 0.863440666168725] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps30 | feature_accuracy | 39 | 0.33333333333333337 | [0.2222222222222222, 0.33333333333333337] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps30 | standardized_mse | 39 | 3.4900923613057295 | [2.210516319031688, 9.679297996154828] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps30 | bounded_mse | 39 | 0.7308004658161672 | [0.4122947504558796, 0.8718351994290529] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps100 | feature_accuracy | 39 | 0.33333333333333337 | [0.2222222222222222, 0.33333333333333337] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps100 | standardized_mse | 39 | 4.0970068893535458 | [1.9252576806548876, 20.01582890278211] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps100 | bounded_mse | 39 | 0.7308004658161672 | [0.4122947504558796, 0.8718351994290529] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps1000 | feature_accuracy | 39 | 0.33333333333333337 | [0.2222222222222222, 0.4444444444444444] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps1000 | standardized_mse | 39 | 1.6506571014130129 | [0.9569000687609063, 3.915874460175518] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps1000 | bounded_mse | 39 | 0.69101651938329012 | [0.2515470764854838, 0.7664165646023473] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps10000 | feature_accuracy | 39 | 0.66666666666666674 | [0.5555555555555556, 0.7777777777777778] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps10000 | standardized_mse | 39 | 0.15699775338063732 | [0.12235447336185394, 0.22891662372985325] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps10000 | bounded_mse | 39 | 0.15175338034423019 | [0.10219677223349413, 0.22847468558676967] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps1000000 | feature_accuracy | 39 | 1 | [0.8888888888888888, 1.0] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps1000000 | standardized_mse | 39 | 0.01888027321531446 | [0.014374306523376219, 0.04609440925030342] | [13, 27] |
| paysim | tableak_batch1 | local_dp_per_tensor_eps1000000 | bounded_mse | 39 | 0.01888027321531446 | [0.014374306523376219, 0.04609440925030342] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_c001_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_c001_clip_only | standardized_mse | 39 | 3.0671600860971301e-05 | [1.2213530140102991e-05, 4.720446504260426e-05] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_c001_clip_only | bounded_mse | 39 | 3.0671600860971301e-05 | [1.2213530140102991e-05, 4.720446504260426e-05] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_c01_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_c01_clip_only | standardized_mse | 39 | 3.0671600860971301e-05 | [1.224574055556606e-05, 4.720446504260426e-05] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_c01_clip_only | bounded_mse | 39 | 3.0671600860971301e-05 | [1.224574055556606e-05, 4.720446504260426e-05] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_c01_eps10 | feature_accuracy | 39 | 0.22222222222222221 | [0.2222222222222222, 0.33333333333333337] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_c01_eps10 | standardized_mse | 39 | 8.0829973708520679 | [3.4264725574044417, 37.74055415766265] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_c01_eps10 | bounded_mse | 39 | 0.7308004658161672 | [0.4122947504558796, 0.8718351994290529] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_c01_eps1000 | feature_accuracy | 39 | 0.33333333333333337 | [0.2222222222222222, 0.4444444444444444] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_c01_eps1000 | standardized_mse | 39 | 1.1158189248883146 | [0.7351559148706971, 2.2018056194595967] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_c01_eps1000 | bounded_mse | 39 | 0.52702993166897494 | [0.22847468558676967, 0.7333094168196078] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_c01_eps1000000 | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_c01_eps1000000 | standardized_mse | 39 | 0.0010944348256639548 | [0.0006586730469923444, 0.0013176966097780108] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_c01_eps1000000 | bounded_mse | 39 | 0.0010944348256639548 | [0.0006586730469923444, 0.0013176966097780108] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_half_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_half_clip_only | standardized_mse | 39 | 3.0671600860971301e-05 | [1.1164262746064694e-05, 4.720446504260426e-05] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_half_clip_only | bounded_mse | 39 | 3.0671600860971301e-05 | [1.1164262746064694e-05, 4.720446504260426e-05] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_half_eps10 | feature_accuracy | 39 | 0.22222222222222221 | [0.11111111111111116, 0.33333333333333337] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_half_eps10 | standardized_mse | 39 | 6.3518327965715145 | [3.0500381629331743, 47.10978407119138] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_half_eps10 | bounded_mse | 39 | 0.7308004658161672 | [0.4122947504558796, 0.8718351994290529] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_half_eps1000 | feature_accuracy | 39 | 0.33333333333333337 | [0.33333333333333337, 0.4444444444444444] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_half_eps1000 | standardized_mse | 39 | 1.4423421313886871 | [0.9326180684353697, 2.327981146936686] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_half_eps1000 | bounded_mse | 39 | 0.67992957387357444 | [0.3449023322430419, 0.7483168989788553] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_half_eps1000000 | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_half_eps1000000 | standardized_mse | 39 | 0.00075348110696739682 | [0.0006374886786971146, 0.0010538025527365852] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_half_eps1000000 | bounded_mse | 39 | 0.00075348110696739682 | [0.0006374886786971146, 0.0010538025527365852] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_clip_only | standardized_mse | 39 | 3.0671600860971301e-05 | [1.1164262746064694e-05, 4.720446504260426e-05] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_clip_only | bounded_mse | 39 | 3.0671600860971301e-05 | [1.1164262746064694e-05, 4.720446504260426e-05] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_eps10 | feature_accuracy | 39 | 0.22222222222222221 | [0.2222222222222222, 0.33333333333333337] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_eps10 | standardized_mse | 39 | 6.3227219808481161 | [2.5826897420496406, 15.609081912988389] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_eps10 | bounded_mse | 39 | 0.7308004658161672 | [0.3449023322430419, 0.8718351994290529] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_eps1000 | feature_accuracy | 39 | 0.33333333333333337 | [0.33333333333333337, 0.4444444444444444] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_eps1000 | standardized_mse | 39 | 1.0170019891589257 | [0.8040004600972863, 1.5269453732821843] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_eps1000 | bounded_mse | 39 | 0.71980779067974177 | [0.3247093608919634, 0.7483168989788553] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_eps1000000 | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_eps1000000 | standardized_mse | 39 | 0.00085697952023428752 | [0.000562593769505199, 0.0011578075633676068] | [13, 27] |
| paysim | tableak_batch1 | b2b_global_cmed_eps1000000 | bounded_mse | 39 | 0.00085697952023428752 | [0.000562593769505199, 0.0011578075633676068] | [13, 27] |
| paysim | tableak_batch2 | unprotected | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch2 | unprotected | standardized_mse | 39 | 0.00035413725705352812 | [0.00018117846756955338, 0.0013080786351845526] | [13, 27] |
| paysim | tableak_batch2 | unprotected | bounded_mse | 39 | 0.00035413725705352812 | [0.00018117846756955338, 0.0013080786351845526] | [13, 27] |
| paysim | tableak_batch2 | dna_v1_conservative | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch2 | dna_v1_conservative | standardized_mse | 39 | 0.0016943229953218135 | [0.0010805718800394133, 0.004186625003430727] | [13, 27] |
| paysim | tableak_batch2 | dna_v1_conservative | bounded_mse | 39 | 0.0016943229953218135 | [0.0010805718800394133, 0.004186625003430727] | [13, 27] |
| paysim | tableak_batch2 | dna_v2_0p95 | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch2 | dna_v2_0p95 | standardized_mse | 39 | 0.0003596946097653132 | [0.00020247747055678804, 0.0011657898267750009] | [13, 27] |
| paysim | tableak_batch2 | dna_v2_0p95 | bounded_mse | 39 | 0.0003596946097653132 | [0.00020247747055678804, 0.0011657898267750009] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps1 | feature_accuracy | 39 | 0.27777777777777779 | [0.2777777777777778, 0.33333333333333337] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps1 | standardized_mse | 39 | 15.722639655484658 | [6.43224633960252, 72.8815732371115] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps1 | bounded_mse | 39 | 0.72147232665357031 | [0.46493933301671536, 0.9219957575486094] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps3 | feature_accuracy | 39 | 0.22222222222222221 | [0.16666666666666674, 0.2777777777777778] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps3 | standardized_mse | 39 | 17.070938863979251 | [6.8767117531499355, 100.06052470757415] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps3 | bounded_mse | 39 | 0.72147232665357031 | [0.46493933301671536, 0.981720508338917] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps10 | feature_accuracy | 39 | 0.33333333333333326 | [0.2222222222222222, 0.38888888888888884] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps10 | standardized_mse | 39 | 37.78352693439588 | [8.706012639558159, 304.430120264931] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps10 | bounded_mse | 39 | 0.72147232665357031 | [0.46493933301671536, 0.981720508338917] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps30 | feature_accuracy | 39 | 0.27777777777777779 | [0.22222222222222232, 0.38888888888888884] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps30 | standardized_mse | 39 | 71.911635357292596 | [4.979550457923575, 344.7759111726248] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps30 | bounded_mse | 39 | 0.72147232665357031 | [0.46493933301671536, 0.981720508338917] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps100 | feature_accuracy | 39 | 0.33333333333333326 | [0.22222222222222232, 0.33333333333333337] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps100 | standardized_mse | 39 | 8.6942856936641526 | [3.3960572668724778, 18.33286637343075] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps100 | bounded_mse | 39 | 0.72147232665357031 | [0.46493933301671536, 0.9726433833316371] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps1000 | feature_accuracy | 39 | 0.33333333333333337 | [0.33333333333333326, 0.4444444444444444] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps1000 | standardized_mse | 39 | 2.3471027829904041 | [1.4716116117966962, 245.32903201149793] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps1000 | bounded_mse | 39 | 0.72147232665357031 | [0.46493933301671536, 0.9418307324392435] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps10000 | feature_accuracy | 39 | 0.5 | [0.4444444444444444, 0.5555555555555556] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps10000 | standardized_mse | 39 | 0.63425451695408985 | [0.4144096119370352, 1.454085271977685] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps10000 | bounded_mse | 39 | 0.51322775821622824 | [0.3981855074952023, 0.7620811949027246] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps1000000 | feature_accuracy | 39 | 1 | [0.9444444444444444, 1.0] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps1000000 | standardized_mse | 39 | 0.0051297959785931609 | [0.0033396501852860985, 0.014524876112854041] | [13, 27] |
| paysim | tableak_batch2 | local_dp_global_eps1000000 | bounded_mse | 39 | 0.0051297959785931609 | [0.0033396501852860985, 0.014524876112854041] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps1 | feature_accuracy | 39 | 0.27777777777777779 | [0.22222222222222232, 0.38888888888888884] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps1 | standardized_mse | 39 | 23.257127434277681 | [4.562736488908321, 174.4404527804506] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps1 | bounded_mse | 39 | 0.72147232665357031 | [0.46493933301671536, 0.981720508338917] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps3 | feature_accuracy | 39 | 0.27777777777777779 | [0.2222222222222222, 0.33333333333333326] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps3 | standardized_mse | 39 | 21.433360239891986 | [8.819862372341595, 180.5321959377505] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps3 | bounded_mse | 39 | 0.72147232665357031 | [0.46493933301671536, 0.981720508338917] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps10 | feature_accuracy | 39 | 0.27777777777777779 | [0.2222222222222222, 0.33333333333333337] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps10 | standardized_mse | 39 | 23.909849757087077 | [6.885109829916752, 236.08493337647286] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps10 | bounded_mse | 39 | 0.72147232665357031 | [0.46493933301671536, 0.981720508338917] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps30 | feature_accuracy | 39 | 0.27777777777777779 | [0.2222222222222222, 0.33333333333333337] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps30 | standardized_mse | 39 | 40.012143589685245 | [5.500025147367576, 340.82890036753355] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps30 | bounded_mse | 39 | 0.72147232665357031 | [0.46493933301671536, 0.981720508338917] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps100 | feature_accuracy | 39 | 0.27777777777777779 | [0.2222222222222222, 0.2777777777777778] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps100 | standardized_mse | 39 | 12.478280344994031 | [6.960555748770682, 82.49885155695128] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps100 | bounded_mse | 39 | 0.72147232665357031 | [0.46493933301671536, 0.981720508338917] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps1000 | feature_accuracy | 39 | 0.33333333333333337 | [0.2777777777777778, 0.38888888888888884] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps1000 | standardized_mse | 39 | 214.94676905941904 | [3.4421240932421076, 380.06279267879415] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps1000 | bounded_mse | 39 | 0.72147232665357031 | [0.46493933301671536, 0.9219957575486094] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps10000 | feature_accuracy | 39 | 0.5 | [0.4444444444444444, 0.5555555555555556] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps10000 | standardized_mse | 39 | 0.95443341350922861 | [0.5565370128101204, 1.645929171983565] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps10000 | bounded_mse | 39 | 0.51640263886108628 | [0.4039062009996645, 0.7842291709259067] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps1000000 | feature_accuracy | 39 | 0.88888888888888884 | [0.7777777777777778, 1.0] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps1000000 | standardized_mse | 39 | 0.055469712829344243 | [0.017904901023967267, 0.1316291446685976] | [13, 27] |
| paysim | tableak_batch2 | local_dp_per_tensor_eps1000000 | bounded_mse | 39 | 0.055469712829344243 | [0.017904901023967267, 0.1316291446685976] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_c001_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_c001_clip_only | standardized_mse | 39 | 0.00040216358288585951 | [0.0001816507436296761, 0.0012551451005906786] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_c001_clip_only | bounded_mse | 39 | 0.00040216358288585951 | [0.0001816507436296761, 0.0012551451005906786] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_c01_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_c01_clip_only | standardized_mse | 39 | 0.00035413725705352812 | [0.00016088793444179377, 0.001298040699358385] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_c01_clip_only | bounded_mse | 39 | 0.00035413725705352812 | [0.00016088793444179377, 0.001298040699358385] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_c01_eps10 | feature_accuracy | 39 | 0.27777777777777779 | [0.16666666666666674, 0.33333333333333326] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_c01_eps10 | standardized_mse | 39 | 12.347022190829962 | [7.277807777851452, 84.37994247986882] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_c01_eps10 | bounded_mse | 39 | 0.72147232665357031 | [0.46493933301671536, 0.981720508338917] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_c01_eps1000 | feature_accuracy | 39 | 0.33333333333333337 | [0.2777777777777778, 0.38888888888888884] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_c01_eps1000 | standardized_mse | 39 | 6.4760322343647898 | [2.811892208425047, 306.98642015381097] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_c01_eps1000 | bounded_mse | 39 | 0.7029874258905997 | [0.46493933301671536, 0.981720508338917] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_c01_eps1000000 | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_c01_eps1000000 | standardized_mse | 39 | 0.0070819854820941509 | [0.004201989714846705, 0.02425216066571874] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_c01_eps1000000 | bounded_mse | 39 | 0.0070819854820941509 | [0.004201989714846705, 0.02425216066571874] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_half_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_half_clip_only | standardized_mse | 39 | 0.00040216358288585951 | [0.00016088793444179377, 0.0012896341087546594] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_half_clip_only | bounded_mse | 39 | 0.00040216358288585951 | [0.00016088793444179377, 0.0012896341087546594] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_half_eps10 | feature_accuracy | 39 | 0.27777777777777779 | [0.22222222222222232, 0.33333333333333337] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_half_eps10 | standardized_mse | 39 | 46.535422164214417 | [4.930691828830808, 300.3328747004484] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_half_eps10 | bounded_mse | 39 | 0.72147232665357031 | [0.46493933301671536, 0.981720508338917] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_half_eps1000 | feature_accuracy | 39 | 0.33333333333333337 | [0.33333333333333326, 0.38888888888888884] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_half_eps1000 | standardized_mse | 39 | 4.701871970568309 | [2.3866017099922487, 288.6867879510989] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_half_eps1000 | bounded_mse | 39 | 0.72147232665357031 | [0.46493933301671536, 0.9219957575486094] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_half_eps1000000 | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_half_eps1000000 | standardized_mse | 39 | 0.004407428674809772 | [0.0026452642168739585, 0.01100074888143332] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_half_eps1000000 | bounded_mse | 39 | 0.004407428674809772 | [0.0026452642168739585, 0.01100074888143332] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_clip_only | feature_accuracy | 39 | 1 | [1.0, 1.0] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_clip_only | standardized_mse | 39 | 0.00040216358288585951 | [0.00016088793444179377, 0.0012896341087546594] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_clip_only | bounded_mse | 39 | 0.00040216358288585951 | [0.00016088793444179377, 0.0012896341087546594] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_eps10 | feature_accuracy | 39 | 0.33333333333333326 | [0.2777777777777778, 0.38888888888888884] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_eps10 | standardized_mse | 39 | 24.934889461184543 | [6.322041514212614, 247.10003109552997] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_eps10 | bounded_mse | 39 | 0.72147232665357031 | [0.46493933301671536, 0.981720508338917] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_eps1000 | feature_accuracy | 39 | 0.33333333333333337 | [0.2777777777777778, 0.4444444444444444] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_eps1000 | standardized_mse | 39 | 8.3566904170976972 | [2.9561103425350135, 314.32026705284346] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_eps1000 | bounded_mse | 39 | 0.72147232665357031 | [0.42123441778088405, 0.981720508338917] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_eps1000000 | feature_accuracy | 39 | 1 | [0.9444444444444444, 1.0] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_eps1000000 | standardized_mse | 39 | 0.0068019541053374105 | [0.004469990458104444, 0.021477790573738536] | [13, 27] |
| paysim | tableak_batch2 | b2b_global_cmed_eps1000000 | bounded_mse | 39 | 0.0068019541053374105 | [0.004469990458104444, 0.021477790573738536] | [13, 27] |

### Matched tests and Holm correction

The summary contains median paired differences, ranks 13/27 intervals, lower/higher wins, ties and effective n. For feature accuracy, transform_lower favors lower recovery for the transform; for bounded MSE, transform_higher favors lower recovery. Both directions are retained.

| dataset | cell | transform | clipping | metric | direction | status | summary | raw_p | holm_p | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baf | ratio_batch1 | dna_v1_conservative | global | feature_accuracy | transform_lower | ASSESSABLE | {"effective_n": 39, "higher_wins": 39, "interval": [0.5135135135135135, 0.5945945945945946], "lower_wins": 0, "median": 0.5675675675675675, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.8189894035458565e-12, "raw_p_lower": 1.0, "ties": 0} | 1 | 1 | NA |
| baf | ratio_batch1 | dna_v1_conservative | global | feature_accuracy | transform_higher | ASSESSABLE | {"effective_n": 39, "higher_wins": 39, "interval": [0.5135135135135135, 0.5945945945945946], "lower_wins": 0, "median": 0.5675675675675675, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.8189894035458565e-12, "raw_p_lower": 1.0, "ties": 0} | 1.8189894035458565e-12 | 2.6193447411060333e-10 | NA |
| baf | ratio_batch1 | dna_v1_conservative | global | bounded_mse | transform_lower | ASSESSABLE | {"effective_n": 31, "higher_wins": 0, "interval": [-0.9533232834228269, -0.40772987140766703], "lower_wins": 31, "median": -0.6746543926489342, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 4.656612873077393e-10, "ties": 8} | 4.6566128730773926e-10 | 6.1001628637313843e-08 | NA |
| baf | ratio_batch1 | dna_v1_conservative | global | bounded_mse | transform_higher | ASSESSABLE | {"effective_n": 31, "higher_wins": 0, "interval": [-0.9533232834228269, -0.40772987140766703], "lower_wins": 31, "median": -0.6746543926489342, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 4.656612873077393e-10, "ties": 8} | 1 | 1 | NA |
| baf | ratio_batch1 | dna_v1_conservative | per_tensor | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | ratio_batch1 | dna_v1_conservative | per_tensor | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | ratio_batch1 | dna_v1_conservative | per_tensor | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | ratio_batch1 | dna_v1_conservative | per_tensor | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | ratio_batch1 | dna_v2_0p95 | global | feature_accuracy | transform_lower | ASSESSABLE | {"effective_n": 39, "higher_wins": 38, "interval": [0.45945945945945943, 0.5405405405405406], "lower_wins": 1, "median": 0.5135135135135135, "n": 39, "ranks": [13, 27], "raw_p_higher": 7.275957614183426e-11, "raw_p_lower": 0.999999999998181, "ties": 0} | 0.99999999999818101 | 1 | NA |
| baf | ratio_batch1 | dna_v2_0p95 | global | feature_accuracy | transform_higher | ASSESSABLE | {"effective_n": 39, "higher_wins": 38, "interval": [0.45945945945945943, 0.5405405405405406], "lower_wins": 1, "median": 0.5135135135135135, "n": 39, "ranks": [13, 27], "raw_p_higher": 7.275957614183426e-11, "raw_p_lower": 0.999999999998181, "ties": 0} | 7.2759576141834259e-11 | 9.822542779147625e-09 | NA |
| baf | ratio_batch1 | dna_v2_0p95 | global | bounded_mse | transform_lower | ASSESSABLE | {"effective_n": 27, "higher_wins": 0, "interval": [-0.8046187171772401, -0.113963423882527], "lower_wins": 27, "median": -0.5704743281176454, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 7.450580596923828e-09, "ties": 12} | 7.4505805969238281e-09 | 9.4622373580932617e-07 | NA |
| baf | ratio_batch1 | dna_v2_0p95 | global | bounded_mse | transform_higher | ASSESSABLE | {"effective_n": 27, "higher_wins": 0, "interval": [-0.8046187171772401, -0.113963423882527], "lower_wins": 27, "median": -0.5704743281176454, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 7.450580596923828e-09, "ties": 12} | 1 | 1 | NA |
| baf | ratio_batch1 | dna_v2_0p95 | per_tensor | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | ratio_batch1 | dna_v2_0p95 | per_tensor | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | ratio_batch1 | dna_v2_0p95 | per_tensor | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | ratio_batch1 | dna_v2_0p95 | per_tensor | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | tableak_batch1 | dna_v1_conservative | global | feature_accuracy | transform_lower | ASSESSABLE | {"effective_n": 12, "higher_wins": 10, "interval": [0.0, 0.0], "lower_wins": 2, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.019287109375, "raw_p_lower": 0.996826171875, "ties": 27} | 0.996826171875 | 1 | NA |
| baf | tableak_batch1 | dna_v1_conservative | global | feature_accuracy | transform_higher | ASSESSABLE | {"effective_n": 12, "higher_wins": 10, "interval": [0.0, 0.0], "lower_wins": 2, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.019287109375, "raw_p_lower": 0.996826171875, "ties": 27} | 0.019287109375 | 1 | NA |
| baf | tableak_batch1 | dna_v1_conservative | global | bounded_mse | transform_lower | ASSESSABLE | {"effective_n": 39, "higher_wins": 2, "interval": [-0.009253869435313535, -0.0031830000437910212], "lower_wins": 37, "median": -0.005827998863114685, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.9999999999272404, "raw_p_lower": 1.420630724169314e-09, "ties": 0} | 1.4206307241693139e-09 | 1.8468199414201081e-07 | NA |
| baf | tableak_batch1 | dna_v1_conservative | global | bounded_mse | transform_higher | ASSESSABLE | {"effective_n": 39, "higher_wins": 2, "interval": [-0.009253869435313535, -0.0031830000437910212], "lower_wins": 37, "median": -0.005827998863114685, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.9999999999272404, "raw_p_lower": 1.420630724169314e-09, "ties": 0} | 0.99999999992724042 | 1 | NA |
| baf | tableak_batch1 | dna_v1_conservative | per_tensor | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | tableak_batch1 | dna_v1_conservative | per_tensor | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | tableak_batch1 | dna_v1_conservative | per_tensor | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | tableak_batch1 | dna_v1_conservative | per_tensor | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | tableak_batch1 | dna_v2_0p95 | global | feature_accuracy | transform_lower | ASSESSABLE | {"effective_n": 10, "higher_wins": 10, "interval": [0.0, 0.0], "lower_wins": 0, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.0009765625, "raw_p_lower": 1.0, "ties": 29} | 1 | 1 | NA |
| baf | tableak_batch1 | dna_v2_0p95 | global | feature_accuracy | transform_higher | ASSESSABLE | {"effective_n": 10, "higher_wins": 10, "interval": [0.0, 0.0], "lower_wins": 0, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.0009765625, "raw_p_lower": 1.0, "ties": 29} | 0.0009765625 | 0.1220703125 | NA |
| baf | tableak_batch1 | dna_v2_0p95 | global | bounded_mse | transform_lower | ASSESSABLE | {"effective_n": 39, "higher_wins": 0, "interval": [-0.010361202121795232, -0.005169093620685438], "lower_wins": 39, "median": -0.006868179376079291, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 1.8189894035458565e-12, "ties": 0} | 1.8189894035458565e-12 | 2.6193447411060333e-10 | NA |
| baf | tableak_batch1 | dna_v2_0p95 | global | bounded_mse | transform_higher | ASSESSABLE | {"effective_n": 39, "higher_wins": 0, "interval": [-0.010361202121795232, -0.005169093620685438], "lower_wins": 39, "median": -0.006868179376079291, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 1.8189894035458565e-12, "ties": 0} | 1 | 1 | NA |
| baf | tableak_batch1 | dna_v2_0p95 | per_tensor | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | tableak_batch1 | dna_v2_0p95 | per_tensor | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | tableak_batch1 | dna_v2_0p95 | per_tensor | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | tableak_batch1 | dna_v2_0p95 | per_tensor | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | tableak_batch2 | dna_v1_conservative | global | feature_accuracy | transform_lower | ASSESSABLE | {"effective_n": 37, "higher_wins": 35, "interval": [0.04054054054054057, 0.09459459459459463], "lower_wins": 2, "median": 0.05405405405405406, "n": 39, "ranks": [13, 27], "raw_p_higher": 5.122274160385132e-09, "raw_p_lower": 0.9999999997235136, "ties": 2} | 0.99999999972351361 | 1 | NA |
| baf | tableak_batch2 | dna_v1_conservative | global | feature_accuracy | transform_higher | ASSESSABLE | {"effective_n": 37, "higher_wins": 35, "interval": [0.04054054054054057, 0.09459459459459463], "lower_wins": 2, "median": 0.05405405405405406, "n": 39, "ranks": [13, 27], "raw_p_higher": 5.122274160385132e-09, "raw_p_lower": 0.9999999997235136, "ties": 2} | 5.1222741603851318e-09 | 6.5565109252929688e-07 | NA |
| baf | tableak_batch2 | dna_v1_conservative | global | bounded_mse | transform_lower | ASSESSABLE | {"effective_n": 39, "higher_wins": 2, "interval": [-0.02553310022581502, -0.015068453274222593], "lower_wins": 37, "median": -0.020741598412754893, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.9999999999272404, "raw_p_lower": 1.420630724169314e-09, "ties": 0} | 1.4206307241693139e-09 | 1.8468199414201081e-07 | NA |
| baf | tableak_batch2 | dna_v1_conservative | global | bounded_mse | transform_higher | ASSESSABLE | {"effective_n": 39, "higher_wins": 2, "interval": [-0.02553310022581502, -0.015068453274222593], "lower_wins": 37, "median": -0.020741598412754893, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.9999999999272404, "raw_p_lower": 1.420630724169314e-09, "ties": 0} | 0.99999999992724042 | 1 | NA |
| baf | tableak_batch2 | dna_v1_conservative | per_tensor | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | tableak_batch2 | dna_v1_conservative | per_tensor | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | tableak_batch2 | dna_v1_conservative | per_tensor | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | tableak_batch2 | dna_v1_conservative | per_tensor | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | tableak_batch2 | dna_v2_0p95 | global | feature_accuracy | transform_lower | ASSESSABLE | {"effective_n": 37, "higher_wins": 36, "interval": [0.05405405405405406, 0.1216216216216216], "lower_wins": 1, "median": 0.06756756756756754, "n": 39, "ranks": [13, 27], "raw_p_higher": 2.764863893389702e-10, "raw_p_lower": 0.999999999992724, "ties": 2} | 0.99999999999272404 | 1 | NA |
| baf | tableak_batch2 | dna_v2_0p95 | global | feature_accuracy | transform_higher | ASSESSABLE | {"effective_n": 37, "higher_wins": 36, "interval": [0.05405405405405406, 0.1216216216216216], "lower_wins": 1, "median": 0.06756756756756754, "n": 39, "ranks": [13, 27], "raw_p_higher": 2.764863893389702e-10, "raw_p_lower": 0.999999999992724, "ties": 2} | 2.7648638933897018e-10 | 3.6496203392744064e-08 | NA |
| baf | tableak_batch2 | dna_v2_0p95 | global | bounded_mse | transform_lower | ASSESSABLE | {"effective_n": 39, "higher_wins": 1, "interval": [-0.031427133956862, -0.01898832913394424], "lower_wins": 38, "median": -0.024691026399874413, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.999999999998181, "raw_p_lower": 7.275957614183426e-11, "ties": 0} | 7.2759576141834259e-11 | 9.822542779147625e-09 | NA |
| baf | tableak_batch2 | dna_v2_0p95 | global | bounded_mse | transform_higher | ASSESSABLE | {"effective_n": 39, "higher_wins": 1, "interval": [-0.031427133956862, -0.01898832913394424], "lower_wins": 38, "median": -0.024691026399874413, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.999999999998181, "raw_p_lower": 7.275957614183426e-11, "ties": 0} | 0.99999999999818101 | 1 | NA |
| baf | tableak_batch2 | dna_v2_0p95 | per_tensor | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | tableak_batch2 | dna_v2_0p95 | per_tensor | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | tableak_batch2 | dna_v2_0p95 | per_tensor | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| baf | tableak_batch2 | dna_v2_0p95 | per_tensor | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| ieee_cis | ratio_batch1 | dna_v1_conservative | global | feature_accuracy | transform_lower | ASSESSABLE | {"effective_n": 39, "higher_wins": 0, "interval": [-0.1166666666666667, -0.021428571428571463], "lower_wins": 39, "median": -0.033333333333333326, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 1.8189894035458565e-12, "ties": 0} | 1.8189894035458565e-12 | 2.6193447411060333e-10 | NA |
| ieee_cis | ratio_batch1 | dna_v1_conservative | global | feature_accuracy | transform_higher | ASSESSABLE | {"effective_n": 39, "higher_wins": 0, "interval": [-0.1166666666666667, -0.021428571428571463], "lower_wins": 39, "median": -0.033333333333333326, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 1.8189894035458565e-12, "ties": 0} | 1 | 1 | NA |
| ieee_cis | ratio_batch1 | dna_v1_conservative | global | bounded_mse | transform_lower | ASSESSABLE | {"effective_n": 39, "higher_wins": 39, "interval": [0.2683233801175984, 0.5209870176787749], "lower_wins": 0, "median": 0.3552153664997271, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.8189894035458565e-12, "raw_p_lower": 1.0, "ties": 0} | 1 | 1 | NA |
| ieee_cis | ratio_batch1 | dna_v1_conservative | global | bounded_mse | transform_higher | ASSESSABLE | {"effective_n": 39, "higher_wins": 39, "interval": [0.2683233801175984, 0.5209870176787749], "lower_wins": 0, "median": 0.3552153664997271, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.8189894035458565e-12, "raw_p_lower": 1.0, "ties": 0} | 1.8189894035458565e-12 | 2.6193447411060333e-10 | NA |
| ieee_cis | ratio_batch1 | dna_v1_conservative | per_tensor | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| ieee_cis | ratio_batch1 | dna_v1_conservative | per_tensor | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| ieee_cis | ratio_batch1 | dna_v1_conservative | per_tensor | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| ieee_cis | ratio_batch1 | dna_v1_conservative | per_tensor | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| ieee_cis | ratio_batch1 | dna_v2_0p95 | global | feature_accuracy | transform_lower | ASSESSABLE | {"effective_n": 39, "higher_wins": 0, "interval": [-0.44047619047619047, -0.22142857142857142], "lower_wins": 39, "median": -0.34523809523809523, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 1.8189894035458565e-12, "ties": 0} | 1.8189894035458565e-12 | 2.6193447411060333e-10 | NA |
| ieee_cis | ratio_batch1 | dna_v2_0p95 | global | feature_accuracy | transform_higher | ASSESSABLE | {"effective_n": 39, "higher_wins": 0, "interval": [-0.44047619047619047, -0.22142857142857142], "lower_wins": 39, "median": -0.34523809523809523, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 1.8189894035458565e-12, "ties": 0} | 1 | 1 | NA |
| ieee_cis | ratio_batch1 | dna_v2_0p95 | global | bounded_mse | transform_lower | ASSESSABLE | {"effective_n": 39, "higher_wins": 39, "interval": [0.2625385277659346, 0.567643956999584], "lower_wins": 0, "median": 0.34574607178834715, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.8189894035458565e-12, "raw_p_lower": 1.0, "ties": 0} | 1 | 1 | NA |
| ieee_cis | ratio_batch1 | dna_v2_0p95 | global | bounded_mse | transform_higher | ASSESSABLE | {"effective_n": 39, "higher_wins": 39, "interval": [0.2625385277659346, 0.567643956999584], "lower_wins": 0, "median": 0.34574607178834715, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.8189894035458565e-12, "raw_p_lower": 1.0, "ties": 0} | 1.8189894035458565e-12 | 2.6193447411060333e-10 | NA |
| ieee_cis | ratio_batch1 | dna_v2_0p95 | per_tensor | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| ieee_cis | ratio_batch1 | dna_v2_0p95 | per_tensor | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| ieee_cis | ratio_batch1 | dna_v2_0p95 | per_tensor | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| ieee_cis | ratio_batch1 | dna_v2_0p95 | per_tensor | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| ieee_cis | tableak_batch1 | dna_v1_conservative | global | feature_accuracy | transform_lower | ASSESSABLE | {"effective_n": 3, "higher_wins": 0, "interval": [0.0, 0.0], "lower_wins": 3, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 0.125, "ties": 36} | 0.125 | 1 | NA |
| ieee_cis | tableak_batch1 | dna_v1_conservative | global | feature_accuracy | transform_higher | ASSESSABLE | {"effective_n": 3, "higher_wins": 0, "interval": [0.0, 0.0], "lower_wins": 3, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 0.125, "ties": 36} | 1 | 1 | NA |
| ieee_cis | tableak_batch1 | dna_v1_conservative | global | bounded_mse | transform_lower | ASSESSABLE | {"effective_n": 39, "higher_wins": 39, "interval": [0.0005151162900172424, 0.0008021615217752056], "lower_wins": 0, "median": 0.0006963590488210389, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.8189894035458565e-12, "raw_p_lower": 1.0, "ties": 0} | 1 | 1 | NA |
| ieee_cis | tableak_batch1 | dna_v1_conservative | global | bounded_mse | transform_higher | ASSESSABLE | {"effective_n": 39, "higher_wins": 39, "interval": [0.0005151162900172424, 0.0008021615217752056], "lower_wins": 0, "median": 0.0006963590488210389, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.8189894035458565e-12, "raw_p_lower": 1.0, "ties": 0} | 1.8189894035458565e-12 | 2.6193447411060333e-10 | NA |
| ieee_cis | tableak_batch1 | dna_v1_conservative | per_tensor | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| ieee_cis | tableak_batch1 | dna_v1_conservative | per_tensor | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| ieee_cis | tableak_batch1 | dna_v1_conservative | per_tensor | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| ieee_cis | tableak_batch1 | dna_v1_conservative | per_tensor | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| ieee_cis | tableak_batch1 | dna_v2_0p95 | global | feature_accuracy | transform_lower | ASSESSABLE | {"effective_n": 0, "higher_wins": 0, "interval": [0.0, 0.0], "lower_wins": 0, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 1.0, "ties": 39} | 1 | 1 | NA |
| ieee_cis | tableak_batch1 | dna_v2_0p95 | global | feature_accuracy | transform_higher | ASSESSABLE | {"effective_n": 0, "higher_wins": 0, "interval": [0.0, 0.0], "lower_wins": 0, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 1.0, "ties": 39} | 1 | 1 | NA |
| ieee_cis | tableak_batch1 | dna_v2_0p95 | global | bounded_mse | transform_lower | ASSESSABLE | {"effective_n": 39, "higher_wins": 20, "interval": [-4.210740805615787e-06, 3.6780508112760336e-06], "lower_wins": 19, "median": 2.863754265658967e-07, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.5, "raw_p_lower": 0.6253706876195793, "ties": 0} | 0.62537068761957926 | 1 | NA |
| ieee_cis | tableak_batch1 | dna_v2_0p95 | global | bounded_mse | transform_higher | ASSESSABLE | {"effective_n": 39, "higher_wins": 20, "interval": [-4.210740805615787e-06, 3.6780508112760336e-06], "lower_wins": 19, "median": 2.863754265658967e-07, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.5, "raw_p_lower": 0.6253706876195793, "ties": 0} | 0.5 | 1 | NA |
| ieee_cis | tableak_batch1 | dna_v2_0p95 | per_tensor | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| ieee_cis | tableak_batch1 | dna_v2_0p95 | per_tensor | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| ieee_cis | tableak_batch1 | dna_v2_0p95 | per_tensor | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| ieee_cis | tableak_batch1 | dna_v2_0p95 | per_tensor | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| ieee_cis | tableak_batch2 | dna_v1_conservative | global | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | original P34C estimator not qualified |
| ieee_cis | tableak_batch2 | dna_v1_conservative | global | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | original P34C estimator not qualified |
| ieee_cis | tableak_batch2 | dna_v1_conservative | global | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | original P34C estimator not qualified |
| ieee_cis | tableak_batch2 | dna_v1_conservative | global | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | original P34C estimator not qualified |
| ieee_cis | tableak_batch2 | dna_v1_conservative | per_tensor | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | original P34C estimator not qualified |
| ieee_cis | tableak_batch2 | dna_v1_conservative | per_tensor | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | original P34C estimator not qualified |
| ieee_cis | tableak_batch2 | dna_v1_conservative | per_tensor | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | original P34C estimator not qualified |
| ieee_cis | tableak_batch2 | dna_v1_conservative | per_tensor | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | original P34C estimator not qualified |
| ieee_cis | tableak_batch2 | dna_v2_0p95 | global | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | original P34C estimator not qualified |
| ieee_cis | tableak_batch2 | dna_v2_0p95 | global | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | original P34C estimator not qualified |
| ieee_cis | tableak_batch2 | dna_v2_0p95 | global | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | original P34C estimator not qualified |
| ieee_cis | tableak_batch2 | dna_v2_0p95 | global | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | original P34C estimator not qualified |
| ieee_cis | tableak_batch2 | dna_v2_0p95 | per_tensor | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | original P34C estimator not qualified |
| ieee_cis | tableak_batch2 | dna_v2_0p95 | per_tensor | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | original P34C estimator not qualified |
| ieee_cis | tableak_batch2 | dna_v2_0p95 | per_tensor | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | original P34C estimator not qualified |
| ieee_cis | tableak_batch2 | dna_v2_0p95 | per_tensor | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | original P34C estimator not qualified |
| paysim | ratio_batch1 | dna_v1_conservative | global | feature_accuracy | transform_lower | ASSESSABLE | {"effective_n": 2, "higher_wins": 0, "interval": [0.0, 0.0], "lower_wins": 2, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 0.25, "ties": 37} | 0.25 | 1 | NA |
| paysim | ratio_batch1 | dna_v1_conservative | global | feature_accuracy | transform_higher | ASSESSABLE | {"effective_n": 2, "higher_wins": 0, "interval": [0.0, 0.0], "lower_wins": 2, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 0.25, "ties": 37} | 1 | 1 | NA |
| paysim | ratio_batch1 | dna_v1_conservative | global | bounded_mse | transform_lower | ASSESSABLE | {"effective_n": 39, "higher_wins": 39, "interval": [0.00023156655678346582, 0.004320375559346979], "lower_wins": 0, "median": 0.000865936918141788, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.8189894035458565e-12, "raw_p_lower": 1.0, "ties": 0} | 1 | 1 | NA |
| paysim | ratio_batch1 | dna_v1_conservative | global | bounded_mse | transform_higher | ASSESSABLE | {"effective_n": 39, "higher_wins": 39, "interval": [0.00023156655678346582, 0.004320375559346979], "lower_wins": 0, "median": 0.000865936918141788, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.8189894035458565e-12, "raw_p_lower": 1.0, "ties": 0} | 1.8189894035458565e-12 | 2.6193447411060333e-10 | NA |
| paysim | ratio_batch1 | dna_v1_conservative | per_tensor | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | ratio_batch1 | dna_v1_conservative | per_tensor | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | ratio_batch1 | dna_v1_conservative | per_tensor | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | ratio_batch1 | dna_v1_conservative | per_tensor | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | ratio_batch1 | dna_v2_0p95 | global | feature_accuracy | transform_lower | ASSESSABLE | {"effective_n": 4, "higher_wins": 0, "interval": [0.0, 0.0], "lower_wins": 4, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 0.0625, "ties": 35} | 0.0625 | 1 | NA |
| paysim | ratio_batch1 | dna_v2_0p95 | global | feature_accuracy | transform_higher | ASSESSABLE | {"effective_n": 4, "higher_wins": 0, "interval": [0.0, 0.0], "lower_wins": 4, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 0.0625, "ties": 35} | 1 | 1 | NA |
| paysim | ratio_batch1 | dna_v2_0p95 | global | bounded_mse | transform_lower | ASSESSABLE | {"effective_n": 39, "higher_wins": 39, "interval": [0.003587299852116323, 0.01767797079774048], "lower_wins": 0, "median": 0.008932671166520514, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.8189894035458565e-12, "raw_p_lower": 1.0, "ties": 0} | 1 | 1 | NA |
| paysim | ratio_batch1 | dna_v2_0p95 | global | bounded_mse | transform_higher | ASSESSABLE | {"effective_n": 39, "higher_wins": 39, "interval": [0.003587299852116323, 0.01767797079774048], "lower_wins": 0, "median": 0.008932671166520514, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.8189894035458565e-12, "raw_p_lower": 1.0, "ties": 0} | 1.8189894035458565e-12 | 2.6193447411060333e-10 | NA |
| paysim | ratio_batch1 | dna_v2_0p95 | per_tensor | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | ratio_batch1 | dna_v2_0p95 | per_tensor | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | ratio_batch1 | dna_v2_0p95 | per_tensor | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | ratio_batch1 | dna_v2_0p95 | per_tensor | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | tableak_batch1 | dna_v1_conservative | global | feature_accuracy | transform_lower | ASSESSABLE | {"effective_n": 0, "higher_wins": 0, "interval": [0.0, 0.0], "lower_wins": 0, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 1.0, "ties": 39} | 1 | 1 | NA |
| paysim | tableak_batch1 | dna_v1_conservative | global | feature_accuracy | transform_higher | ASSESSABLE | {"effective_n": 0, "higher_wins": 0, "interval": [0.0, 0.0], "lower_wins": 0, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 1.0, "ties": 39} | 1 | 1 | NA |
| paysim | tableak_batch1 | dna_v1_conservative | global | bounded_mse | transform_lower | ASSESSABLE | {"effective_n": 39, "higher_wins": 38, "interval": [0.0001219869678434178, 0.0010783306521527497], "lower_wins": 1, "median": 0.0003014450283158652, "n": 39, "ranks": [13, 27], "raw_p_higher": 7.275957614183426e-11, "raw_p_lower": 0.999999999998181, "ties": 0} | 0.99999999999818101 | 1 | NA |
| paysim | tableak_batch1 | dna_v1_conservative | global | bounded_mse | transform_higher | ASSESSABLE | {"effective_n": 39, "higher_wins": 38, "interval": [0.0001219869678434178, 0.0010783306521527497], "lower_wins": 1, "median": 0.0003014450283158652, "n": 39, "ranks": [13, 27], "raw_p_higher": 7.275957614183426e-11, "raw_p_lower": 0.999999999998181, "ties": 0} | 7.2759576141834259e-11 | 9.822542779147625e-09 | NA |
| paysim | tableak_batch1 | dna_v1_conservative | per_tensor | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | tableak_batch1 | dna_v1_conservative | per_tensor | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | tableak_batch1 | dna_v1_conservative | per_tensor | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | tableak_batch1 | dna_v1_conservative | per_tensor | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | tableak_batch1 | dna_v2_0p95 | global | feature_accuracy | transform_lower | ASSESSABLE | {"effective_n": 0, "higher_wins": 0, "interval": [0.0, 0.0], "lower_wins": 0, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 1.0, "ties": 39} | 1 | 1 | NA |
| paysim | tableak_batch1 | dna_v2_0p95 | global | feature_accuracy | transform_higher | ASSESSABLE | {"effective_n": 0, "higher_wins": 0, "interval": [0.0, 0.0], "lower_wins": 0, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 1.0, "raw_p_lower": 1.0, "ties": 39} | 1 | 1 | NA |
| paysim | tableak_batch1 | dna_v2_0p95 | global | bounded_mse | transform_lower | ASSESSABLE | {"effective_n": 39, "higher_wins": 19, "interval": [-2.674720442407153e-05, 1.1264341697484554e-05], "lower_wins": 20, "median": -5.412201054627586e-06, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.6253706876195793, "raw_p_lower": 0.5, "ties": 0} | 0.5 | 1 | NA |
| paysim | tableak_batch1 | dna_v2_0p95 | global | bounded_mse | transform_higher | ASSESSABLE | {"effective_n": 39, "higher_wins": 19, "interval": [-2.674720442407153e-05, 1.1264341697484554e-05], "lower_wins": 20, "median": -5.412201054627586e-06, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.6253706876195793, "raw_p_lower": 0.5, "ties": 0} | 0.62537068761957926 | 1 | NA |
| paysim | tableak_batch1 | dna_v2_0p95 | per_tensor | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | tableak_batch1 | dna_v2_0p95 | per_tensor | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | tableak_batch1 | dna_v2_0p95 | per_tensor | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | tableak_batch1 | dna_v2_0p95 | per_tensor | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | tableak_batch2 | dna_v1_conservative | global | feature_accuracy | transform_lower | ASSESSABLE | {"effective_n": 10, "higher_wins": 4, "interval": [0.0, 0.0], "lower_wins": 6, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.828125, "raw_p_lower": 0.376953125, "ties": 29} | 0.376953125 | 1 | NA |
| paysim | tableak_batch2 | dna_v1_conservative | global | feature_accuracy | transform_higher | ASSESSABLE | {"effective_n": 10, "higher_wins": 4, "interval": [0.0, 0.0], "lower_wins": 6, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.828125, "raw_p_lower": 0.376953125, "ties": 29} | 0.828125 | 1 | NA |
| paysim | tableak_batch2 | dna_v1_conservative | global | bounded_mse | transform_lower | ASSESSABLE | {"effective_n": 36, "higher_wins": 29, "interval": [0.00023080192625303693, 0.0015357171185517715], "lower_wins": 7, "median": 0.0007589987725525155, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.00015627557877451181, "raw_p_lower": 0.999965199152939, "ties": 3} | 0.99996519915293902 | 1 | NA |
| paysim | tableak_batch2 | dna_v1_conservative | global | bounded_mse | transform_higher | ASSESSABLE | {"effective_n": 36, "higher_wins": 29, "interval": [0.00023080192625303693, 0.0015357171185517715], "lower_wins": 7, "median": 0.0007589987725525155, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.00015627557877451181, "raw_p_lower": 0.999965199152939, "ties": 3} | 0.00015627557877451181 | 0.019690722925588489 | NA |
| paysim | tableak_batch2 | dna_v1_conservative | per_tensor | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | tableak_batch2 | dna_v1_conservative | per_tensor | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | tableak_batch2 | dna_v1_conservative | per_tensor | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | tableak_batch2 | dna_v1_conservative | per_tensor | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | tableak_batch2 | dna_v2_0p95 | global | feature_accuracy | transform_lower | ASSESSABLE | {"effective_n": 8, "higher_wins": 2, "interval": [0.0, 0.0], "lower_wins": 6, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.96484375, "raw_p_lower": 0.14453125, "ties": 31} | 0.14453125 | 1 | NA |
| paysim | tableak_batch2 | dna_v2_0p95 | global | feature_accuracy | transform_higher | ASSESSABLE | {"effective_n": 8, "higher_wins": 2, "interval": [0.0, 0.0], "lower_wins": 6, "median": 0.0, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.96484375, "raw_p_lower": 0.14453125, "ties": 31} | 0.96484375 | 1 | NA |
| paysim | tableak_batch2 | dna_v2_0p95 | global | bounded_mse | transform_lower | ASSESSABLE | {"effective_n": 35, "higher_wins": 20, "interval": [-3.767207914170102e-05, 0.0001343920016076566], "lower_wins": 15, "median": 9.756574921736987e-07, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.24977991660125554, "raw_p_lower": 0.8447476704604924, "ties": 4} | 0.84474767046049237 | 1 | NA |
| paysim | tableak_batch2 | dna_v2_0p95 | global | bounded_mse | transform_higher | ASSESSABLE | {"effective_n": 35, "higher_wins": 20, "interval": [-3.767207914170102e-05, 0.0001343920016076566], "lower_wins": 15, "median": 9.756574921736987e-07, "n": 39, "ranks": [13, 27], "raw_p_higher": 0.24977991660125554, "raw_p_lower": 0.8447476704604924, "ties": 4} | 0.24977991660125554 | 1 | NA |
| paysim | tableak_batch2 | dna_v2_0p95 | per_tensor | feature_accuracy | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | tableak_batch2 | dna_v2_0p95 | per_tensor | feature_accuracy | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | tableak_batch2 | dna_v2_0p95 | per_tensor | bounded_mse | transform_lower | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |
| paysim | tableak_batch2 | dna_v2_0p95 | per_tensor | bounded_mse | transform_higher | NOT_ASSESSABLE | NA | 1 | 1 | frozen utility match NOT_ASSESSABLE |

### Figure and plain reading

Panel estimator selection uses only all 39 unprotected targets: highest median feature accuracy, fixed cell-order tie break, before protected measurements are submitted.

| cell | dataset | protected_results_inspected | unprotected_medians |
| --- | --- | --- | --- |
| ratio_batch1 | baf | False | {"ratio_batch1": 1.0, "tableak_batch1": 1.0, "tableak_batch2": 1.0} |
| ratio_batch1 | ieee_cis | False | {"ratio_batch1": 1.0, "tableak_batch1": 1.0} |
| ratio_batch1 | paysim | False | {"ratio_batch1": 1.0, "tableak_batch1": 1.0, "tableak_batch2": 1.0} |

![Privacy-utility frontier](/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/artifacts/priority35b/figure/label_layout_v2/frontier.png)

[PDF](/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/artifacts/priority35b/figure/label_layout_v2/frontier.pdf); [exact figure data CSV](/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/artifacts/priority35b/figure/label_layout_v2/frontier_data.csv).

Caption: x is utility loss (negative mean paired Delta-F1) from Config-C training, with 95% t intervals over 11 seeds; y is median feature accuracy from the Priority 34C single-gradient setting, with ranks 13/27 over 39 row sets. These axes come from different configurations, not a single deployable-system experiment. Data-free lines are mean/mode and empirical marginal. DP joint lines are separate by clip norm; per-tensor DP and clip-only points are separate. Lower y means less recovery by the plotted estimator.

Read the transforms relative to both utility error bars and leakage intervals, then consult the complete paired Holm table for inferential claims. Overlapping bars alone neither establish equivalence nor invalidate a paired contrast. A clip-only match cannot support superiority to formal DP at matched utility.

### Limitations and independent verification

This is a server-side analysis with seed access for v2. It is not metadata-free seed identification and does not establish a formal privacy guarantee for the transforms. DP accounting is replace-one client-upload accounting (50 full-participation releases, delta=1e-5), not a claimed record-level DP guarantee. P34C estimator qualification concerns its single-gradient setting, not realistic Config-C Adam uploads. The frontier is limited to the frozen grids, budgets and models; selection of the plotted estimator does not replace reporting every qualified estimator. Utility reuse and post-hoc B2b scope are explicit.

Independent checks recompute scientific payloads, gradients, clipping/noise receipts, metrics, controls, saved objectives, selection/pooling, source-ID firewall, CSV medians/intervals/sign tests and Holm. The categorical decoder and transform/least-squares helper replays remain shared-source dependencies; their source hashes are audited, and they are not described as independent implementations. No optimization or measurement is repeated during verification.

Verification receipt: {"checks": {"compile": true, "csv_recomputation": true, "diff_check": true, "figure_data": true, "figure_visual_qa": true, "no_live_workers": true, "scientific_bundles": true, "source_input_hashes": true, "tests": true, "utility_hashes": true}, "controls": 312, "csv_recomputation": "PASS", "scientific_jobs": 9984}

This rendered section alone is not a COMPLETE seal: the final full manifest, report-prefix assertion and fresh post-publication no-live audit are additionally required.

### Plain reading of the verified frontier

BAF is the only dataset with a DP matched-utility arm: joint C=0.7420758605003357, epsilon=1,000,000. IEEE-CIS and PaySim match only clip-only C_med, with NO DP guarantee. All per-tensor matches remain NOT_ASSESSABLE. The primary ratio estimator indicates greater recovery from both transforms than matched DP on BAF; lower accuracy on IEEE-CIS versus clip-only; and no feature-accuracy difference established on PaySim. These are estimator-specific, different-configuration comparisons, not a universal privacy frontier or deployable Config-C record-recovery claim. All existing 144 paired tests remain authoritative; consult bounded-MSE and all-estimator contrasts too.

### Post-hoc descriptive estimator sensitivity (no new tests)

Selection followed the pre-registered highest-unprotected-median rule with fixed estimator-order tie break. Ratio b=1 was selected from a three-way 100% tie on BAF and PaySim, and a two-way 100% tie on IEEE-CIS; TabLeak b=2 was not qualified on IEEE-CIS and is explicitly NOT_APPLICABLE. No protected result selected the primary estimator.

The closed-form ratio is invariant to positive common rescaling of the gradient: joint clip-only rescales numerator and denominator equally, leaving the ratio and its 100% median accuracy intact. Additive noise does not cancel; the ratio is highly noise-sensitive here. This can favor DP on the ratio axis even when optimization-based TabLeak still recovers records. Under TabLeak b=1 the BAF matched DP arm leaks about as much as the transforms (all median feature accuracies 100%); at b=2 it is lower, 93.24% versus 100%, so this is descriptive similarity/high residual recovery, NOT equivalence. Existing Holm-significant BAF b=2 differences remain reported. IEEE-CIS and PaySim's matched clip-only arms likewise have 100% TabLeak medians, but must not be described as matched DP. Stronger C=0.01 epsilon=10 DP reduces TabLeak recovery much more, at substantial training utility loss.

The following medians and ranks-13/27 intervals use the same 39 row sets per available cell. They were independently recomputed from the already verified per-target CSV; no new statistical test, measurement, family or decision was introduced. Intervals and medians below are in percent.

| Dataset | Arm | Estimator | n | Median accuracy (%) | Ranks 13/27 interval (%) | Status |
| --- | --- | --- | --- | --- | --- | --- |
| baf | unprotected | ratio_batch1 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| baf | unprotected | tableak_batch1 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| baf | unprotected | tableak_batch2 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| baf | dna_v1_conservative | ratio_batch1 | 39 | 97.297297 | [97.297297, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| baf | dna_v1_conservative | tableak_batch1 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| baf | dna_v1_conservative | tableak_batch2 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| baf | dna_v2_0p95 | ratio_batch1 | 39 | 94.594595 | [91.891892, 97.297297] | DESCRIPTIVE_EXISTING_RESULTS |
| baf | dna_v2_0p95 | tableak_batch1 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| baf | dna_v2_0p95 | tableak_batch2 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| baf | local_dp_global_eps10 | ratio_batch1 | 39 | 40.540541 | [35.135135, 43.243243] | DESCRIPTIVE_EXISTING_RESULTS |
| baf | local_dp_global_eps10 | tableak_batch1 | 39 | 32.432432 | [29.729730, 37.837838] | DESCRIPTIVE_EXISTING_RESULTS |
| baf | local_dp_global_eps10 | tableak_batch2 | 39 | 36.486486 | [33.783784, 39.189189] | DESCRIPTIVE_EXISTING_RESULTS |
| baf | local_dp_per_tensor_eps10 | ratio_batch1 | 39 | 40.540541 | [37.837838, 43.243243] | DESCRIPTIVE_EXISTING_RESULTS |
| baf | local_dp_per_tensor_eps10 | tableak_batch1 | 39 | 32.432432 | [29.729730, 35.135135] | DESCRIPTIVE_EXISTING_RESULTS |
| baf | local_dp_per_tensor_eps10 | tableak_batch2 | 39 | 36.486486 | [33.783784, 39.189189] | DESCRIPTIVE_EXISTING_RESULTS |
| baf | b2b_global_cmed_half_eps1000000 | ratio_batch1 | 39 | 40.540541 | [37.837838, 43.243243] | DESCRIPTIVE_EXISTING_RESULTS |
| baf | b2b_global_cmed_half_eps1000000 | tableak_batch1 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| baf | b2b_global_cmed_half_eps1000000 | tableak_batch2 | 39 | 93.243243 | [87.837838, 94.594595] | DESCRIPTIVE_EXISTING_RESULTS |
| ieee_cis | unprotected | ratio_batch1 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| ieee_cis | unprotected | tableak_batch1 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| ieee_cis | unprotected | tableak_batch2 | 0 | NA | NA | NOT_APPLICABLE: not qualified in P34C |
| ieee_cis | dna_v1_conservative | ratio_batch1 | 39 | 96.666667 | [88.333333, 97.857143] | DESCRIPTIVE_EXISTING_RESULTS |
| ieee_cis | dna_v1_conservative | tableak_batch1 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| ieee_cis | dna_v1_conservative | tableak_batch2 | 0 | NA | NA | NOT_APPLICABLE: not qualified in P34C |
| ieee_cis | dna_v2_0p95 | ratio_batch1 | 39 | 65.476190 | [55.952381, 77.857143] | DESCRIPTIVE_EXISTING_RESULTS |
| ieee_cis | dna_v2_0p95 | tableak_batch1 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| ieee_cis | dna_v2_0p95 | tableak_batch2 | 0 | NA | NA | NOT_APPLICABLE: not qualified in P34C |
| ieee_cis | local_dp_global_eps10 | ratio_batch1 | 39 | 60.952381 | [59.761905, 62.380952] | DESCRIPTIVE_EXISTING_RESULTS |
| ieee_cis | local_dp_global_eps10 | tableak_batch1 | 39 | 37.037037 | [31.481481, 40.740741] | DESCRIPTIVE_EXISTING_RESULTS |
| ieee_cis | local_dp_global_eps10 | tableak_batch2 | 0 | NA | NA | NOT_APPLICABLE: not qualified in P34C |
| ieee_cis | local_dp_per_tensor_eps10 | ratio_batch1 | 39 | 60.476190 | [58.809524, 61.190476] | DESCRIPTIVE_EXISTING_RESULTS |
| ieee_cis | local_dp_per_tensor_eps10 | tableak_batch1 | 39 | 33.333333 | [29.629630, 37.037037] | DESCRIPTIVE_EXISTING_RESULTS |
| ieee_cis | local_dp_per_tensor_eps10 | tableak_batch2 | 0 | NA | NA | NOT_APPLICABLE: not qualified in P34C |
| ieee_cis | b2b_global_cmed_clip_only | ratio_batch1 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| ieee_cis | b2b_global_cmed_clip_only | tableak_batch1 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| ieee_cis | b2b_global_cmed_clip_only | tableak_batch2 | 0 | NA | NA | NOT_APPLICABLE: not qualified in P34C |
| paysim | unprotected | ratio_batch1 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| paysim | unprotected | tableak_batch1 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| paysim | unprotected | tableak_batch2 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| paysim | dna_v1_conservative | ratio_batch1 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| paysim | dna_v1_conservative | tableak_batch1 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| paysim | dna_v1_conservative | tableak_batch2 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| paysim | dna_v2_0p95 | ratio_batch1 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| paysim | dna_v2_0p95 | tableak_batch1 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| paysim | dna_v2_0p95 | tableak_batch2 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| paysim | local_dp_global_eps10 | ratio_batch1 | 39 | 77.777778 | [66.666667, 77.777778] | DESCRIPTIVE_EXISTING_RESULTS |
| paysim | local_dp_global_eps10 | tableak_batch1 | 39 | 22.222222 | [22.222222, 33.333333] | DESCRIPTIVE_EXISTING_RESULTS |
| paysim | local_dp_global_eps10 | tableak_batch2 | 39 | 33.333333 | [22.222222, 38.888889] | DESCRIPTIVE_EXISTING_RESULTS |
| paysim | local_dp_per_tensor_eps10 | ratio_batch1 | 39 | 66.666667 | [66.666667, 77.777778] | DESCRIPTIVE_EXISTING_RESULTS |
| paysim | local_dp_per_tensor_eps10 | tableak_batch1 | 39 | 22.222222 | [22.222222, 33.333333] | DESCRIPTIVE_EXISTING_RESULTS |
| paysim | local_dp_per_tensor_eps10 | tableak_batch2 | 39 | 27.777778 | [22.222222, 33.333333] | DESCRIPTIVE_EXISTING_RESULTS |
| paysim | b2b_global_cmed_clip_only | ratio_batch1 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| paysim | b2b_global_cmed_clip_only | tableak_batch1 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |
| paysim | b2b_global_cmed_clip_only | tableak_batch2 | 39 | 100.000000 | [100.000000, 100.000000] | DESCRIPTIVE_EXISTING_RESULTS |

### Preserved numerical audit and presentation provenance

All 9,984 measurements were preserved. Original failed-verifier outputs/logs remain unchanged. Exact integer-binomial p-values and separately recomputed Holm values remain authoritative and bit-for-bit unchanged. SciPy is only a cross-check at relative 1e-12 with absolute floor 1e-15; maximum discrepancy was 2.220446049250313e-16. No raw or Holm value is within that envelope of 0.05 (the only p-value decision threshold): minimum distances 0.012499999999999997 and 0.030309277074411514. Approved float64 MSE reduction-order cross-check allows 1e-12 + 2*gamma_N*max(abs(values)), while requiring stored/native scores exactly; maximum MSE difference 1.4901161193847656e-08, median zero. Neither numerical allowance changed wins/losses/ties, raw/adjusted p-values, thresholds, matching, NA status, gates or figure points.

The post-hoc label-layout version preserves all original assets and failed visual-QA evidence. Only annotation positions and their leader-line endpoints changed. Point/error-bar/curve/style/scale fingerprints are identical to the original renderer; both CSV copies have SHA-256 e815f3ba6261bd70fa7e989e812292398a486aa9e30b012e23389cd90bbf1172. The new PDF/PNG were visually inspected at single-column width before publication. Neither axis selection nor the interpretation was changed by presentation.

