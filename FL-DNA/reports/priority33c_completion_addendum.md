# Priority33c — verified completion addendum

Completed 2026-10-03 (Asia/Ho_Chi_Minh). This new administrative addendum
finalizes, but does not overwrite, priority33c_report.md, COMPLETE.json, the
original manifest or any earlier artifact. Read both reports together.

Final independent audit PASS: 160/160 image utility jobs, all brackets valid;
669/669 image reconstruction results and arrays/receipt hashes; exact frozen
jobs/4800 iterations/one restart; fresh source-disjoint targets and pairing;
independently recomputed paired effects, exact sign tests and Holm20/135;
12 unit tests, py_compile and git diff --check PASS; no live scientific workers.
Evidence: artifacts/priority33c/final_verified_receipt.json and the additional
sha256_manifest_final.csv (includes raw CIFAR/PaySim data, reused P31 artifacts,
original and correction sources, both reports, outputs and PROJECT.md).
The original sha256_manifest.csv also passed without rewriting it.

## Interruptions and resolved gates

Original image reconstruction stopped before any result because noncontiguous
NCHW input was incompatible with LeNetZhu view(). A separately saved pre-run
technical amendment and new contiguous-input wrapper fixed storage layout only.
Exact image values/gradients/RNG equality tests passed. Original drivers, freeze,
failure markers and logs remain unchanged. No utility replay, target redraw,
budget reduction or outcome-dependent tuning occurred. Corrected image pool ran
3567.31 seconds (59.46 minutes), then analysis 1.53 seconds. Utility stage runtimes
and exact commands are preserved in runs.jsonl/supervisor.jsonl/launch receipts.

PaySim completed48 jobs:16 baseline PASS,16 v1 and16 v2 full-state CPU jobs
failed BN/numerical guards. Their dependent per-tensor utility-matched BN arm is
NOT_ASSESSABLE. No sigma/epsilon or confirmatory targets exist for this undefined
comparator. Failures were not clamped/excluded or replaced by raw BN, and are not
privacy evidence. Four reserved hypotheses remain p=1 in both frozen families.

## Confirmatory image conclusions

Lower PSNR means poorer recovery/stronger protection. Effects are paired
DNA-minus-comparator PSNR in dB; intervals are ranks13/27 of39. Direction shown
is the stronger protection test; p-adjusted below uses combined135. Both
directions and own-family Holm20 are in the unchanged report/paired_tests.csv.

| DNA setting | Comparator | Median PSNR DNA / comparator | Median SSIM DNA / comparator | Paired effect [13,27] | Direction W/L/T | Raw p | Holm135 p | Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 medium | Unprotected | 23.5035 / 23.7255 | .872359 / .879141 | −.1933 [−.3249,−.0857] | DNA34/5/0 | 1.21495e−6 | .000110561 | Small reduction vs unprotected |
| v1 stronger | Unprotected | 23.3489 / 23.7255 | .868665 / .879141 | −.2645 [−.4376,−.1621] | DNA35/4/0 | 1.67658e−7 | 1.60952e−5 | Small reduction vs unprotected |
| v1 medium | Distortion DP | 23.5035 / 23.5737 | .872359 / .865667 | +.0818 [−.0583,+.1543] | DP22/17/0 | .261199 | 1 | Neither direction significant |
| v1 stronger | Distortion DP | 23.3489 / 23.2294 | .868665 / .862982 | +.1377 [−.0637,+.2520] | DP25/14/0 | .0540645 | 1 | Neither direction significant |
| v1 medium | Single-clip utility DP | 23.5035 / 23.2844 | .872359 / .861493 | +.3083 [+.1891,+.4277] | DP35/4/0 | 1.67658e−7 | 1.60952e−5 | DP stronger |
| v1 stronger | Single-clip utility DP | 23.3489 / 23.3496 | .868665 / .862890 | +.1350 [+.0190,+.2428] | DP27/12/0 | .0118514 | .948108 | Not significant after Holm |
| v1 conservative | Per-tensor utility DP | 23.6335 / 18.5508 | .875396 / .636821 | +4.9379 [+4.3660,+6.6218] | DP39/0/0 | 1.81899e−12 | 2.45564e−10 | DP stronger |
| v2 .95/.01 | Per-tensor utility DP | 23.7320 / 18.5284 | .877713 / .632531 | +5.0978 [+4.5171,+6.4583] | DP39/0/0 | 1.81899e−12 | 2.45564e−10 | DP stronger |

Gray/CIFAR-mean reference medians are12.80395/12.98037dB on C1; no image
confirmatory contrast has both arms at/below reference. No equivalence claim
follows from nonsignificance. The medium/stronger reductions vs unprotected are
small, and none of these settings establishes superiority over DP.

## Calibrated DP releases

All epsilon values are one release, delta1e−5, add/remove update-level RDP,
not record-level privacy and not whole-training accounting. They are extremely
large; these comparators should not be described as strong formal DP.

| Arm | Sigma | C / convention | Epsilon |
| --- | --- | --- | --- |
| Single-clip utility, v1 medium/stronger (also conservative transfer) | .001 | C16.82906017 | 504798.5261 |
| Single-clip utility, v2 C2 transfer | .0003 | C16.82906017 | 5571550.6427 |
| Per-tensor utility, conservative/v2 | .001 | C_l p95; whitened sensitivity sqrt8 | 4013572.3091 |
| Distortion DP, medium | .0006636354460 | C31.46451832 | 1142531.2413 |
| Distortion DP, stronger | .0007987918991 | C31.46451832 | 789622.1587 |

Largest eligible utility sigma is bracketed by the next point below tolerance;
baseline validation mean .4593333 exceeds .40. Distortion development n24
matching error is zero to floating-point precision. All C_l values and complete
grids are in image_clip.json/image_calibration.json/image_distortion.json.
C2 is descriptive: separate batch4/untrained and batch1/trained sensitivities,
24 targets each, calibration transferred rather than refit. The unchanged report
contains median PSNR/SSIM for all10 sensitivity arms, no confirmatory p-values.

## Full Priority33 coverage summary

| Dataset / instrument | Comparator coverage | Result / gap resolution |
| --- | --- | --- |
| CIFAR10 gradient recovery,33c | Medium/stronger vs unprotected/distortion/single utility; conservative/v2 vs per-tensor utility; two sensitivities | Closed as specified; no DNA-over-DP superiority; per-tensor DP stronger |
| BAF four-record BN batch mean,33a | Distortion DP | DP stronger for both; paired DNA−DP MSE−23.8503[−33.0305,−13.0177]v1 and−152.9037[−182.6160,−104.0158]v2; v1/DP both worse than prior reference |
| BAF BN batch mean,33a | Utility DP | NOT_ASSESSABLE: full-state CPU transform BN gate failure |
| IEEE-CIS BN batch mean,33a | Distortion/utility DP | NOT_ASSESSABLE: unprotected n8 qualification failed |
| BAF and IEEE-CIS individual records,33b | Distortion/utility DP | NOT_ASSESSABLE: official TabLeak n8 fails required two-reference qualification; no tuning/n24/n39 |
| PaySim BN batch mean,33c | Per-tensor utility DP | NOT_ASSESSABLE: no BN-valid full-state CPU transform utility target |
| PaySim historical DP utility audit,33a A1 | Selected historical noise levels |21 sampled MPS replays reproduce metrics; variance nonnegative and CPU/MPS finite, sampled scope only |
| Adult | No new Priority33 arm | Earlier evidence unchanged in existing115 family |

BN batch-mean recovery is not individual-record recovery. TabLeak33b is a
single FedSGD gradient, unlike multi-step Adam FL. All earlier scope qualifiers
remain; not-assessable cells are unresolved coverage, not successful protection.

Commands for correction/final audit (cwd FL-DNA):

```text
.venv-phase1/bin/python -B -m unittest tests.test_priority33c_contiguous tests.test_priority33c_image_utility tests.test_priority33c_image_recovery tests.test_priority33c_statistics
nohup .venv-phase1/bin/python -B -u experiments/supervise_priority33c_contiguous.py
.venv-phase1/bin/python -B experiments/verify_priority33c_completion.py --verify-only
.venv-phase1/bin/python -B experiments/verify_priority33c_completion.py
git diff --check
```
