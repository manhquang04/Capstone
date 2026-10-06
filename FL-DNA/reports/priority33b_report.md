# Priority33b: tabular individual-record recovery

Single-gradient FedSGD, batch8, known labels, official TabLeak1500iterations ×30member ensemble. This differs from P32 multi-step local Adam FL. Accuracy is logical mixed-feature accuracy after permutation alignment, not BN batch-mean recovery.

## Qualification

| Dataset | Stage | Control | Attack median (%) | Control median (%) | Paired median difference | Wins/losses/ties | Exact p | Gate |
|---|---|---|---|---|---|---|---|---|
| baf | n8 | empirical_mean | 64.1891891891892 | 58.749999999999986 | 5.444819819819816 | 6/2/0 | 0.14453125 | NOT_ASSESSABLE |
| baf | n8 | mean_mode | 64.1891891891892 | 40.03378378378378 | 24.324324324324333 | 8/0/0 | 0.00390625 | NOT_ASSESSABLE |
| ieee_cis | n8 | empirical_mean | 84.83796296296296 | 87.14506172839506 | -1.7554012345679055 | 3/5/0 | 0.85546875 | NOT_ASSESSABLE |
| ieee_cis | n8 | mean_mode | 84.83796296296296 | 62.96296296296296 | 23.379629629629626 | 6/2/0 | 0.14453125 | NOT_ASSESSABLE |

Qualification failure is NOT evidence of privacy protection. No tuning on confirmatory targets, no scope/budget reduction to force a passing instrument.

## Representation and feature selection

baf: 58 original → 58 retained encoded coordinates, 37 logical features. Train-only additional standardization/bounds; continuous values never integer-rounded. Complete categorical blocks retained; frequency-encoded categories scored as continuous surrogates, not original categories.

Retained coordinates: income, name_email_similarity, prev_address_months_count, current_address_months_count, customer_age, days_since_request, intended_balcon_amount, zip_count_4w, velocity_6h, velocity_24h, velocity_4w, bank_branch_count_8w, date_of_birth_distinct_emails_4w, credit_risk_score, email_is_free, phone_home_valid, phone_mobile_valid, bank_months_count, has_other_cards, proposed_credit_limit, foreign_request, session_length_in_minutes, keep_alive_session, device_distinct_emails_8w, device_fraud_count, month, prev_address_months_count:missing, current_address_months_count:missing, credit_risk_score:missing, bank_months_count:missing, session_length_in_minutes:missing, device_distinct_emails_8w:missing, payment_type=AA, payment_type=AB, payment_type=AC, payment_type=AD, payment_type=AE, employment_status=CA, employment_status=CB, employment_status=CC, employment_status=CD, employment_status=CE, employment_status=CF, employment_status=CG, housing_status=BA, housing_status=BB, housing_status=BC, housing_status=BD, housing_status=BE, housing_status=BF, housing_status=BG, source=INTERNET, source=TELEAPP, device_os=linux, device_os=macintosh, device_os=other, device_os=windows, device_os=x11

ieee_cis: 476 original → 60 retained encoded coordinates, 54 logical features. Train-only additional standardization/bounds; continuous values never integer-rounded. Complete categorical blocks retained; frequency-encoded categories scored as continuous surrogates, not original categories.

Retained coordinates: card3, V15, V16, V17, V18, V33, V34, V37, V38, V39, V40, V42, V43, V44, V45, V47, V51, V52, V58, V74, V78, V79, V80, V81, V85, V86, V87, V92, V93, V94, V123, V170, V171, V177, V188, V189, V199, V200, V201, V217, V219, V228, V230, V231, V232, V233, V242, V244, V246, V257, V258, id_17, ProductCD=C, ProductCD=H, ProductCD=R, ProductCD=S, ProductCD=W, id_35=F, id_35=T, id_35=__MISSING__

## Paired confirmatory tests

Accuracy in percent; lower means stronger protection. Paired effect = DNA−DP accuracy. Intervals use ranks13/27 of39. Fixed16-test Holm family includes p1 placeholders for gated absent cells.

| Dataset | Transform | Comparator | Direction | DNA median | DP median | Paired median [13/27] | W/L/T | Raw p | Holm p | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| ieee_cis | dna_v1_conservative | distortion | DNA_stronger_protection | NA | NA | NA | NA | 1 (reserved) | 1 | NOT_ASSESSABLE |
| ieee_cis | dna_v1_conservative | distortion | DP_stronger_protection | NA | NA | NA | NA | 1 (reserved) | 1 | NOT_ASSESSABLE |
| ieee_cis | dna_v1_conservative | utility | DNA_stronger_protection | NA | NA | NA | NA | 1 (reserved) | 1 | NOT_ASSESSABLE |
| ieee_cis | dna_v1_conservative | utility | DP_stronger_protection | NA | NA | NA | NA | 1 (reserved) | 1 | NOT_ASSESSABLE |
| ieee_cis | dna_v2_0p95 | distortion | DNA_stronger_protection | NA | NA | NA | NA | 1 (reserved) | 1 | NOT_ASSESSABLE |
| ieee_cis | dna_v2_0p95 | distortion | DP_stronger_protection | NA | NA | NA | NA | 1 (reserved) | 1 | NOT_ASSESSABLE |
| ieee_cis | dna_v2_0p95 | utility | DNA_stronger_protection | NA | NA | NA | NA | 1 (reserved) | 1 | NOT_ASSESSABLE |
| ieee_cis | dna_v2_0p95 | utility | DP_stronger_protection | NA | NA | NA | NA | 1 (reserved) | 1 | NOT_ASSESSABLE |
| baf | dna_v1_conservative | distortion | DNA_stronger_protection | NA | NA | NA | NA | 1 (reserved) | 1 | NOT_ASSESSABLE |
| baf | dna_v1_conservative | distortion | DP_stronger_protection | NA | NA | NA | NA | 1 (reserved) | 1 | NOT_ASSESSABLE |
| baf | dna_v1_conservative | utility | DNA_stronger_protection | NA | NA | NA | NA | 1 (reserved) | 1 | NOT_ASSESSABLE |
| baf | dna_v1_conservative | utility | DP_stronger_protection | NA | NA | NA | NA | 1 (reserved) | 1 | NOT_ASSESSABLE |
| baf | dna_v2_0p95 | distortion | DNA_stronger_protection | NA | NA | NA | NA | 1 (reserved) | 1 | NOT_ASSESSABLE |
| baf | dna_v2_0p95 | distortion | DP_stronger_protection | NA | NA | NA | NA | 1 (reserved) | 1 | NOT_ASSESSABLE |
| baf | dna_v2_0p95 | utility | DNA_stronger_protection | NA | NA | NA | NA | 1 (reserved) | 1 | NOT_ASSESSABLE |
| baf | dna_v2_0p95 | utility | DP_stronger_protection | NA | NA | NA | NA | 1 (reserved) | 1 | NOT_ASSESSABLE |

## Calibration and execution status

Both datasets failed n8 qualification. n24, distortion calibration, utility training/grid and n39 reconstruction were NOT RUN under the frozen gate rule. The39source-ID batches were reserved for pairing/firewall assertions only; no confirmatory records/gradients were inspected. Later-stage implementations were not empirically exercised, and no claim is made for those paths.

Corrected qualification:16/16 completed, zero numerical failures, wall time 851.722s (14.20min). First infrastructure-only launch:16errors before any optimization result, retained separately. n24/calibration/confirmatory runtime:0 when gated off.

## Commands, disclosures and integrity

- `.venv-phase1/bin/python -B -m unittest tests.test_priority33b -v` (6 pre-run synthetic tests).
- `.venv-phase1/bin/python -B experiments/priority33b_qualify.py --prepare`.
- Detached `nohup .../.venv-phase1/bin/python -B -u experiments/priority33b_qualify.py --supervise`; exact worker commands in runs.jsonl.
- First preparation failed before any scientific job: trusted historical pickle namespace collision. Reused P33a source-ID extraction after verifying each original bundle SHA.
- Initial16qualification jobs failed at first native closure with callback keyword TypeError, before optimizer updates or metrics. Technical replay amendment fixes only argument names; all original errors kept, root gates superseded as infrastructure-only. attempt2 uses byte-identical target manifests and unchanged seeds/budgets.
- All new workloads CPU, one torch thread per process,4workers; nonfinite loss/logits/candidates/outputs fail closed, never clamp numerical failures. Official feature projection/bounds are frozen attack operations.
- Individual records scored up to permutation; this does not establish per-record DP. DP accounting, when available, is one whole-update add/remove release at delta1e-5.
- PLANNED, NOT RUN: utility uses disclosed FC/selected-representation model variant; no BN-invalid historical utility or repaired raw-BN variant silently substituted. Validation only,50rounds/16pairedseeds/P32 F1 rule and bracketed grid; test not used for selection.
- Qualification and reserved confirmatory source IDs are fresh/disjoint; IEEE TransactionID and BAF original-row domains remain explicit.
- Independent exact integer tails agree with SciPy and two Holm implementations.

SHA256 manifests and final_checks.json are the completion evidence; pending or gate-failed cells never count as privacy successes.

Actual post-gate commands: `.venv-phase1/bin/python -B -u experiments/priority33b_run_cells.py --supervise` (zero jobs by gating), `.venv-phase1/bin/python -B experiments/analyze_priority33b.py`, and `.venv-phase1/bin/python -B experiments/verify_priority33b.py`. Comparator synthetic tests: `tests.test_priority33b_comparators` (3tests); final verification runs9tests total, py_compile and git diff --check. No confirmatory p-values were observed: p1 values are fixed-family reservations only.

## SHA-256

Full output/code checksums: `artifacts/priority33b/sha256_manifest.json`. Original and corrected execution freezes remain separate; the corrected freeze is `artifacts/priority33b/attempt2/execution_freeze.json`.

| Input/code | SHA-256 |
|---|---|
| artifacts/priority33b/attempt2/baf/adapter.json | `3d15d1255975295af024121f7ac2786bddaedcf9794942da58d502454869a030` |
| artifacts/priority33b/attempt2/baf/targets.json | `54e0f14b34429af6556fd2231b84a128db34ddd7a209d5222cc2001ecad3a2c3` |
| artifacts/priority33b/attempt2/ieee_cis/adapter.json | `5fb9df0fe7023a73927ffc4e475bac11c18f53e3fb43687d52835f5c45fdf4f7` |
| artifacts/priority33b/attempt2/ieee_cis/targets.json | `6a818b25b4da1ac2409a7ae5a0f1559a40824f81c5706d0240655f1efc6988a7` |
| datasets/baf/Base.csv | `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809` |
| datasets/ieee-fraud-detection/train_identity.csv | `b63c725d8377be90a995268d97f347c17d456b95db45807adcf9f59cd603c37c` |
| datasets/ieee-fraud-detection/train_transaction.csv | `3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642` |
| experiments/analyze_priority33b.py | `a629342d23ae2c00de4fa21a1f56cf410a45bd971e8a9634a308348190a34a5e` |
| experiments/priority33b_comparators.py | `8120144c43a28b27575af476a79577bdedbaefa139b73f1b188a509d9dd34694` |
| experiments/priority33b_core.py | `6ce6f9578a321b3ade3744a7acff12bab329f3b95ad6a5a7e060e0d5dbd8b004` |
| experiments/priority33b_qualify.py | `2d5e3e31aba7e85dd3f2da8442af40bcc925cf64786c8a80bf824fed0ec594fc` |
| experiments/priority33b_run_cells.py | `c7ac6fd50ec638779f02e2efb0c489e427e9c6945b9c04614626bd813c35ce6e` |
| experiments/verify_priority33b.py | `0185292009bc0479962d33a6a4d92fde10d3638c3feb6e9aeb3f1e7e29933fe7` |
| protocols/amendments/2026-10-02_priority33b_keyword_compatibility_replay.md | `14667a5b33dd115de5d3fbde08b29de34917f7a256534516e72455a3a6332d0c` |
| protocols/amendments/2026-10-02_priority33b_tabular_record_recovery.md | `7d285eca2748f569908ab0958e7ca08987b1dbc5aa76fb9210e35556b37abe05` |
| protocols/amendments/2026-10-02_priority33b_utility_execution_clarification.md | `2a97c9fa36511f7f7343416e57dede311a25fc2a486bfe26e974061bb082dd2c` |
