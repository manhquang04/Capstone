# Priority33a — historical DP BN validity and batch-mean recovery

Status: FINAL; generated 2026-10-02T15:47:07.623981+00:00.

Protocol saved before any run: protocols/amendments/2026-10-02_priority33a_bn_audit_mean_recovery.md. Earlier artifacts, Latex/, external_defenses/ and datasets are unchanged.

## A1 historical DP utility validity

Native replay jobs finished: 21/21. Exact original entry points/seeds/settings on MPS; torch threads=1. CPU/MPS evaluation uses the actual final test inputs and unchanged final checkpoint, not a substitute model.

| Source | Variant | sigma | Seed | First aggregate negative round | First client negative round | Final negative BN coordinates | CPU finite probabilities | MPS finite probabilities | Validity/reproduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P25 | full_state_single_clip | 1e-05 | 2501101 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P25 | full_state_single_clip | 1e-05 | 2501102 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P25 | full_state_single_clip | 1e-05 | 2501103 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P25 | full_state_single_clip | 3e-05 | 2501101 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P25 | full_state_single_clip | 3e-05 | 2501102 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P25 | full_state_single_clip | 3e-05 | 2501103 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P25b | full_state_single_clip | 1e-05 | 2501201 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P25b | full_state_single_clip | 1e-05 | 2501202 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P25b | full_state_single_clip | 1e-05 | 2501203 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P25b | full_state_single_clip | 3e-05 | 2501201 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P25b | full_state_single_clip | 3e-05 | 2501202 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P25b | full_state_single_clip | 3e-05 | 2501203 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P27 | full_state_single_clip | 3e-05 | 270201 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P27 | full_state_single_clip | 3e-05 | 270202 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P27 | full_state_single_clip | 3e-05 | 270203 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P27 | per_tensor_clip | 3e-05 | 270201 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P27 | per_tensor_clip | 3e-05 | 270202 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P27 | per_tensor_clip | 3e-05 | 270203 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P27 | fedbn_trainable_only | 1e-06 | 270201 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P27 | fedbn_trainable_only | 1e-06 | 270202 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |
| P27 | fedbn_trainable_only | 1e-06 | 270203 | none | none | 0 | 100001/100001 | 100001/100001 | metrics match; ordinary BN VALID |

Ordinary BN validity requires no aggregate negative variance, finite CPU logits/probabilities, and reproduction within 1e-8 (counts exact). Client-state violations are separately reported. A reproduced MPS utility number alone is not a valid-domain guarantee. Intermediate failures are not hidden by a finite final endpoint. Only three seeds per selected cell are audited; no extrapolation to unobserved runs.

P25 uses seeds 2501101–2501103. P25b uses the first three NEW seeds 2501201–2501203, not duplicate P25 jobs. P27 fedbn_trainable_only sigma=1e-6 is the historical selected but NOT_BRACKETED fallback control; it does not noise BN buffers and must not be presented as an established utility match.

## A2 qualification

All new training is CPU/thread1; actual forward logits and local/transmitted/aggregate/final states fail closed on negative BN variance or non-finite values. No clamps, excluded seeds, post-gate feature reduction or raw-BN substitution.

P32 did not save checkpoints, so baseline seed321000 is replayed per dataset and checked against its stored validation/test metrics before saving a fixed final checkpoint. IEEE-CIS retains 476 features and a 128-row first linear layer; BAF retains 58 features. P24 plain least-squares recovery and train-only standardized MSE are unchanged; v2 uses key-known sketch-space least squares, not transpose lift.

| Dataset | Gate | Reference | W/L/T | p | Median recovered MSE | Median reference MSE | Gate passes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ieee_cis | n8 | prior | 0/8/0 | 1.0 | 336535.0372200974 | 0.13865086478951727 | False |
| ieee_cis | n8 | decoy | 5/3/0 | 0.36328125 | 336535.0372200974 | 336537.1239023258 | False |
| baf | n8 | prior | 8/0/0 | 0.00390625 | 3.545335743508696e-10 | 0.4689970374633934 | True |
| baf | n8 | decoy | 8/0/0 | 0.00390625 | 3.545335743508696e-10 | 0.3740927154609769 | True |
| baf | n24 | prior | 24/0/0 | 5.960464477539063e-08 | 3.095188252888289e-10 | 0.47123665601360176 | True |
| baf | n24 | decoy | 24/0/0 | 5.960464477539063e-08 | 3.095188252888289e-10 | 0.3761199511770224 | True |

Fresh dataset-qualified source IDs are disjoint across n8/n24/n39 and excluded from P5 manifests and actual P6/P8/P10/P12 IEEE target bundles (eight provenance files,444 unique TransactionIDs). The protected quantity is a four-record batch mean, not individual-record recovery.

## A2 paired confirmatory tests

Fixed Holm family: 16 tests (two datasets × two transforms × two DP comparators × two directions). Gated absent cells reserve p=1 and remain NOT_ASSESSABLE. Higher MSE indicates stronger protection. Intervals are paired DNA−DP differences at ranks13/27 of39, not differences of marginal medians. Qualification tests are outside this family.

| Dataset | DNA | DP arm | Direction | Status | W/L/T | Raw p | Holm p | Median DNA MSE | Median DP MSE | Paired median Δ | Ranks13/27 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ieee_cis | dna_v1_conservative | distortion | DNA_better | NOT_ASSESSABLE | —/—/— | 1.0 | 1.0 | — | — | — | — |
| ieee_cis | dna_v1_conservative | distortion | DP_better | NOT_ASSESSABLE | —/—/— | 1.0 | 1.0 | — | — | — | — |
| ieee_cis | dna_v1_conservative | utility | DNA_better | NOT_ASSESSABLE | —/—/— | 1.0 | 1.0 | — | — | — | — |
| ieee_cis | dna_v1_conservative | utility | DP_better | NOT_ASSESSABLE | —/—/— | 1.0 | 1.0 | — | — | — | — |
| ieee_cis | dna_v2_0p95 | distortion | DNA_better | NOT_ASSESSABLE | —/—/— | 1.0 | 1.0 | — | — | — | — |
| ieee_cis | dna_v2_0p95 | distortion | DP_better | NOT_ASSESSABLE | —/—/— | 1.0 | 1.0 | — | — | — | — |
| ieee_cis | dna_v2_0p95 | utility | DNA_better | NOT_ASSESSABLE | —/—/— | 1.0 | 1.0 | — | — | — | — |
| ieee_cis | dna_v2_0p95 | utility | DP_better | NOT_ASSESSABLE | —/—/— | 1.0 | 1.0 | — | — | — | — |
| baf | dna_v1_conservative | distortion | DNA_better | VALID | 9/30/0 | 0.9998529615622829 | 1.0 | 18.471631957349977 | 36.63756207128195 | -23.85029707665918 | [-33.030458596365996, -13.017699350634839] |
| baf | dna_v1_conservative | distortion | DP_better | VALID | 30/9/0 | 0.00053250981727615 | 0.00798764725914225 | 18.471631957349977 | 36.63756207128195 | -23.85029707665918 | [-33.030458596365996, -13.017699350634839] |
| baf | dna_v1_conservative | utility | DNA_better | NOT_ASSESSABLE | —/—/— | 1.0 | 1.0 | — | — | — | — |
| baf | dna_v1_conservative | utility | DP_better | NOT_ASSESSABLE | —/—/— | 1.0 | 1.0 | — | — | — | — |
| baf | dna_v2_0p95 | distortion | DNA_better | VALID | 0/39/0 | 1.0 | 1.0 | 0.04499028348398336 | 152.96437675762394 | -152.90374438727068 | [-182.61600155519366, -104.01579599771509] |
| baf | dna_v2_0p95 | distortion | DP_better | VALID | 39/0/0 | 1.8189894035458565e-12 | 2.9103830456733704e-11 | 0.04499028348398336 | 152.96437675762394 | -152.90374438727068 | [-182.61600155519366, -104.01579599771509] |
| baf | dna_v2_0p95 | utility | DNA_better | NOT_ASSESSABLE | —/—/— | 1.0 | 1.0 | — | — | — | — |
| baf | dna_v2_0p95 | utility | DP_better | NOT_ASSESSABLE | —/—/— | 1.0 | 1.0 | — | — | — | — |

## Calibration and gates

ieee_cis: NOT_ASSESSABLE; Unprotected batch-mean qualification n8 failed

### baf distortion_calibration.json

```json
{
  "dna_v1_conservative": {
    "clip": 5.664803171157837,
    "sigma": 0.005510361657108481,
    "median_distortion": 0.35188181698322296,
    "delta": 1e-05,
    "accounting": {
      "epsilon": 17344.59109188362,
      "alpha": 1.03,
      "rdp_at_alpha": 16960.826909717947
    }
  },
  "dna_v2_0p95": {
    "clip": 5.664803171157837,
    "sigma": 0.011058386328972115,
    "median_distortion": 0.7016658186912537,
    "delta": 1e-05,
    "accounting": {
      "epsilon": 4523.405785463024,
      "alpha": 1.05,
      "rdp_at_alpha": 4293.14727616362
    }
  }
}
```

### baf utility_calibration.json

```json
{
  "dna_v1_conservative": {
    "status": "NOT_ASSESSABLE",
    "reason": "Required full-state CPU transform violates BN/finite training gate",
    "failed_seeds": [
      330100,
      330101,
      330102,
      330103,
      330104,
      330105,
      330106,
      330107,
      330108,
      330109,
      330110,
      330111,
      330112,
      330113,
      330114,
      330115
    ]
  },
  "dna_v2_0p95": {
    "status": "NOT_ASSESSABLE",
    "reason": "Required full-state CPU transform violates BN/finite training gate",
    "failed_seeds": [
      330100,
      330101,
      330102,
      330103,
      330104,
      330105,
      330106,
      330107,
      330108,
      330109,
      330110,
      330111,
      330112,
      330113,
      330114,
      330115
    ]
  }
}
```

baf: CONFIRMATORY_COMPLETE; 

Utility grid uses validation F1 only, 16 paired seeds, full-state CPU transforms, frozen eight sigma points and at most one extension. A BN-invalid transform cannot supply a utility target: that arm is NOT_ASSESSABLE, not evidence of privacy. Failed required transform replicates prevent unnecessary baseline/DP calibration jobs; the failure records remain intact.

## Commands, disclosures and hashes

Commands actually executed: `.venv-phase1/bin/python -B -u experiments/priority33a_audit.py --supervise`; `.venv-phase1/bin/python -B -u experiments/priority33a_bn_mean.py --supervise`; `.venv-phase1/bin/python -B experiments/analyze_priority33a.py` (initial partial report); `.venv-phase1/bin/python -B -u experiments/analyze_priority33a.py --watch` (periodic partial reports, then final analysis). Detached jobs use nohup/start_new_session; exact per-job argv is in artifacts/priority33a/runs.jsonl.

Infrastructure disclosures: initial shell-background launch did not persist and started zero jobs. The first detached launch resolved the virtualenv symlink to the base interpreter, so all21 jobs failed to import torch before any scientific round. Those artifacts are preserved under A1/. The corrected unchanged scientific configuration runs under A1_attempt2/ using the virtualenv path without symlink resolution. A2's waiting-only supervisor was stopped before any training to complete pre-run harness validation; it is relaunched with the finalized runner. No scientific results were discarded or tuned.

Amendment SHA256: `4becc2ae968f5d5a05d5136e982c76e3c36cbdde769ba3ee51c6b60f86020fff`. Per-source/original-output hashes: A1_attempt2_execution_freeze.json; prepared data/source hashes: A2_execution_freeze.json; final manifest: sha256_manifest.json. Per-target receipts and result.json are stored alongside each confirmatory target. Synthetic pre-run adapter/gate unit tests:6/6 pass.

Independent exact sign tests (integer binomial tails) agree with SciPy; independent NumPy Holm construction agrees within1e-15. Final compilation/diff and unit checks pass; see final_checks.json. Independent completion command: `.venv-phase1/bin/python -B experiments/verify_priority33a_final.py`.

## Completion audit and plain conclusions

Historical DP utility audit: 21/21 audited jobs meet ordinary aggregate-BN/CPU-finiteness/reproduction criteria; 0/21 do not. This statement is limited to the sampled seeds. Negative client buffers, if any, remain separately visible in the main table.

Actual final-checkpoint endpoints (historical validation threshold, no retuning):

| Source | Variant | sigma | Seed | CPU test F1 | MPS test F1 | CPU ROC-AUC | MPS ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P25 | full_state_single_clip | 1e-05 | 2501101 | 0.7725321888412018 | 0.7725321888412018 | 0.995865871105713 | 0.995865871105713 |
| P25 | full_state_single_clip | 1e-05 | 2501102 | 0.7454545454545455 | 0.7454545454545455 | 0.9949250544573023 | 0.9949250544573023 |
| P25 | full_state_single_clip | 1e-05 | 2501103 | 0.7488584474885843 | 0.7488584474885843 | 0.9890996910153524 | 0.9890996910153522 |
| P25 | full_state_single_clip | 3e-05 | 2501101 | 0.7672413793103448 | 0.7672413793103448 | 0.9952928120086734 | 0.9952928120086735 |
| P25 | full_state_single_clip | 3e-05 | 2501102 | 0.7230046948356806 | 0.7230046948356806 | 0.9940159838702066 | 0.9940159838702066 |
| P25 | full_state_single_clip | 3e-05 | 2501103 | 0.7302904564315353 | 0.7302904564315353 | 0.9890056947311162 | 0.9890056947311163 |
| P25b | full_state_single_clip | 1e-05 | 2501201 | 0.7489361702127659 | 0.7489361702127659 | 0.9948594666289129 | 0.9948594666289129 |
| P25b | full_state_single_clip | 1e-05 | 2501202 | 0.7553648068669527 | 0.7553648068669527 | 0.9821069418468041 | 0.9821069418468041 |
| P25b | full_state_single_clip | 1e-05 | 2501203 | 0.7614678899082569 | 0.7614678899082569 | 0.9844527351599193 | 0.9844527351599195 |
| P25b | full_state_single_clip | 3e-05 | 2501201 | 0.7363636363636363 | 0.7363636363636363 | 0.9963115578638332 | 0.9963115578638332 |
| P25b | full_state_single_clip | 3e-05 | 2501202 | 0.7678571428571428 | 0.7678571428571428 | 0.981177612770703 | 0.9811776127707031 |
| P25b | full_state_single_clip | 3e-05 | 2501203 | 0.7258064516129032 | 0.7258064516129032 | 0.9847549048829013 | 0.9847549048829013 |
| P27 | full_state_single_clip | 3e-05 | 270201 | 0.7219917012448133 | 0.7219917012448133 | 0.9922094078870567 | 0.9922094078870567 |
| P27 | full_state_single_clip | 3e-05 | 270202 | 0.748898678414097 | 0.748898678414097 | 0.9955722394432316 | 0.9955722394432316 |
| P27 | full_state_single_clip | 3e-05 | 270203 | 0.7381974248927038 | 0.7381974248927038 | 0.9888435492003409 | 0.9888435492003409 |
| P27 | per_tensor_clip | 3e-05 | 270201 | 0.7321428571428572 | 0.7321428571428572 | 0.9923038698836837 | 0.9923038698836837 |
| P27 | per_tensor_clip | 3e-05 | 270202 | 0.7129629629629628 | 0.7129629629629628 | 0.9947359752265846 | 0.9947359752265845 |
| P27 | per_tensor_clip | 3e-05 | 270203 | 0.7520661157024793 | 0.7520661157024793 | 0.9873191173073628 | 0.9873191173073627 |
| P27 | fedbn_trainable_only | 1e-06 | 270201 | 0.4502617801047121 | 0.4502617801047121 | 0.7939925507750696 | 0.7939958107618061 |
| P27 | fedbn_trainable_only | 1e-06 | 270202 | 0.574468085106383 | 0.574468085106383 | 0.9337108087499286 | 0.9337108475592945 |
| P27 | fedbn_trainable_only | 1e-06 | 270203 | 0.6761904761904761 | 0.6761904761904761 | 0.9297004041141652 | 0.9297040133851949 |

DP accounting shown for A2 is one clipped update-vector release with add/remove sensitivity C and delta=1e-5. It is not a record-level privacy guarantee for the entire 50-round training procedure. Invalid BN training or missing utility brackets are NOT_ASSESSABLE, never privacy superiority.

Additional descriptive Prior reference on the same39 targets (no new tests, attacker selection or verdict-rule changes):

| Dataset | DNA | Median Prior MSE | Median DNA MSE | Median distortion-DP MSE | Practical reference flag |
| --- | --- | --- | --- | --- | --- |
| baf | dna_v1_conservative | 0.4922581452109313 | 18.471631957349977 | 36.63756207128195 | both no better than Prior |
| baf | dna_v2_0p95 | 0.4922581452109313 | 0.04499028348398336 | 152.96437675762394 | not both at Prior reference |

The v1 ranking should therefore not be interpreted as a practically useful recovery difference when both reconstructions are worse than a data-free Prior. Numerical validity of the sampled historical DP jobs also does not rehabilitate BN-invalid DNA utility targets identified in P32b, or prove that every unobserved utility-grid replicate is valid.

The source firewall reads eight earlier IEEE provenance files, including the actual P6/P8/P10/P12 bundle 'targets' containers in addition to P5 manifests, and excludes444 unique TransactionIDs before any A2 qualification. IEEE's n24/confirmatory runs and expensive utility grid are skipped after its n8 failure. BAF's DP utility grid is not run because neither full-state transform supplies a BN-valid utility target.

Final receipt-contract repair: v1's archived transmitted_payload.pt contained unused DNATransformStats (including encoder-only raw/update difference diagnostics). Those diagnostics were never read by recovery. Original diagnostic receipts remain intact but are superseded as server receipts; authoritative packets are receipt_contract_replay/.../server_observable_receipt.pt, with v1 only kind/q and v2 only kind/q/official decoder metadata. All78 recoveries were replayed on exactly the same39 targets per defense, without new training or changes to the attacker. Maximum score difference is 1.177e-13; every paired sign, raw p and Holm p is unchanged. This is a payload-schema repair, not a design change or new replicate set. Command actually executed: `.venv-phase1/bin/python -B experiments/repair_priority33a_receipts.py`.

Pre-registered P27-style v1 debias (descriptive only, never used to select the primary attacker by ground truth):

| Dataset | n | Median plain standardized MSE | Median debiased standardized MSE |
| --- | --- | --- | --- |
| baf | 39 | 18.471631957349977 | 22.321119745681266 |

Requirement-level evidence: completion_audit.json; final_checks.json; sha256_manifest.json. A2 gate outcomes and unavailable utility arms are explicit, not silently omitted.
