# Priority 31 — CIFAR-10 image utility-matched DP

Status: COMPLETED (unavailable cells remain NOT_ASSESSABLE).

This adds evidence; it does not replace any earlier RQ1 result. No Latex/ or external_defenses/ file was edited.

## Frozen design and provenance

- Amendment: protocols/amendments/2026-10-01_priority31_image_utility_dp.md.
- Official inversefed LeNet-Zhu, initialization seed 42, 15,826 trainable parameters, no BN buffers.
- Utility-development scale: 12,000 CIFAR train images and 3,000 validation images; split source-ID overlap=0. Test images were excluded from all selections.
- Paired seeds 51016–51031 vary minibatch order; initial weights are shared. Frozen P30 transform adapters retain their deterministic seeds (v1 30001, v2 30002); quantization seed per tensor is unchanged.
- This is a development-scale utility match, not a claim of full-data converged CIFAR utility. Baseline quality was checked on validation, not test.

Selected learning rate **0.1**, epochs **100**, from two validation-selection seeds only. All six predeclared 100-epoch selection jobs were completed; their 30/60/100-epoch results are in selection.json.

C = **16.829060173**, p95 of 94 minibatch trainable-gradient norms at the fixed initial model. This is not a p95 estimated throughout training. Per-run actual clipping rates are in utility_per_seed.csv.

DP clips the full trainable gradient vector and adds coordinate noise N(0,(sigma C)^2). RDP uses update-level add/remove adjacency, sensitivity ratio 1, delta=1e-5, one release; no record-level or subsampling claim.

## Utility calibration

| Method | n | Validation mean | Validation SD | Paired validation Δ | SD(Δ) | Test mean (descriptive) |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 16 | 0.459333333 | 0.00920144916 | 0 | 0 | 0.467375 |
| dna_v1_conservative | 16 | 0.453354167 | 0.010063111 | -0.00597916667 | 0.00484342816 | 0.46485625 |
| dna_v2_0p95 | 16 | 0.458520833 | 0.00918127624 | -0.0008125 | 0.00454111872 | 0.46611875 |
| dp_0.0001 | 16 | 0.458979167 | 0.00988112211 | -0.000354166667 | 0.00192630427 | 0.4668625 |
| dp_0.0003 | 16 | 0.458604167 | 0.0109681146 | -0.000729166667 | 0.00514992808 | 0.46655 |
| dp_0.001 | 16 | 0.452416667 | 0.0123699157 | -0.00691666667 | 0.00927481656 | 0.461225 |
| dp_0.003 | 16 | 0.409229167 | 0.00949519859 | -0.0501041667 | 0.0123442707 | 0.4201 |
| dp_0.01 | 16 | 0.247833333 | 0.0172274544 | -0.2115 | 0.0201979097 | 0.25210625 |
| dp_0.03 | 16 | 0.1008125 | 0.00537341708 | -0.358520833 | 0.0106423116 | 0.10123125 |
| dp_0.1 | 16 | 0.100729167 | 0.0026755927 | -0.358604167 | 0.0101435758 | 0.099325 |
| dp_0.3 | 16 | 0.100541667 | 0.00724990421 | -0.358791667 | 0.0129110203 | 0.100925 |

Baseline mean validation accuracy = 0.459333333; quality gate PASS.

| Transform | Own paired Δ | Eligibility threshold | Selected sigma | Next sigma | Bracketed | ε (one release) |
| --- | --- | --- | --- | --- | --- | --- |
| dna_v1_conservative | -0.00597916667 | -0.0109791667 | 0.001 | 0.003 | True | 504798.526 |
| dna_v2_0p95 | -0.0008125 | -0.0058125 | 0.0003 | 0.001 | True | 5571550.64 |

Matching used the largest eligible sigma and required the next-larger grid point to fail the threshold. Test accuracy did not enter this rule.

## Reconstruction and exact paired tests

Exact S1 test-image target IDs were reused, with source-ID equality asserted against S1, S1c and S0u per target. Existing E2 transform outputs were not rerun. Target IDs and image indices are in paired_*.csv.

The official attack used cosine + TV=0.01, signed Adam lr=0.1, 4,800 iterations, one restart, boxed constraints and decay. Its input is only the noisy clipped gradient. Attack seed=30600+target_id; DP draw seed=510000+101*target_id+the frozen defense offset.

A DNA win means DNA has strictly larger raw input-pixel MSE; exact equality is a tie. PSNR difference below is DNA minus DP: negative means less leakage under DNA.

| Transform | DNA wins / DP wins / ties | Raw p(DNA greater error) | Holm p | Raw p(DP greater error) | Holm p |
| --- | --- | --- | --- | --- | --- |
| dna_v1_conservative | 30 / 9 / 0 | 0.000532509817 | 0.0372756872 | 0.999852962 | 1 |
| dna_v2_0p95 | 19 / 20 / 0 | 0.625370688 | 1 | 0.5 | 1 |

| Transform | Median PSNR DNA / DP | Median SSIM DNA / DP | Median paired PSNR Δ | Order interval ranks 13 / 27 |
| --- | --- | --- | --- | --- |
| dna_v1_conservative | 22.6499628 / 23.1362944 | 0.827214241 / 0.835202515 | -0.295981197 | [-0.416788157, -0.124211167] |
| dna_v2_0p95 | 23.4944579 / 23.3520237 | 0.852278411 / 0.850938022 | 0.00955649722 | [-0.015165196, 0.109318116] |

Rank interval binomial coverage (continuous independent differences) = 0.976297298. This is the requested interval, not an automatically relabeled 95% interval.

Holm family size = **115**: the repaired P30 family's 111 finite raw hypotheses plus two directions for each available P31 cell. All old adjusted p-values were recomputed without altering their raw p-values or files. See expanded_holm.csv.

## Direction and limitations

- dna_v1_conservative: DNA significantly larger reconstruction error.
- dna_v2_0p95: no significant direction under the expanded Holm family.

All comparisons reuse fixed checkpoints and the existing targets, so this extension does not solve the checkpoint-clustering concerns of P18–P19. Non-significance is not proof of equivalence.

all_image_ssim.csv exports descriptive per-arm medians from every already-run image E2/E3 stage (also E1), with superseded/NOT_APPLICABLE stages labeled explicitly. Invalid stages do not enter Holm.

## Exact commands and every-run disclosure

```sh
PYTHONDONTWRITEBYTECODE=1 external_defenses/.venv/bin/python -m tests.test_priority31_image_utility_dp
PYTHONDONTWRITEBYTECODE=1 external_defenses/.venv/bin/python -u experiments/priority31_image_utility_dp.py --stage analysis --workers 8
nohup env PYTHONDONTWRITEBYTECODE=1 external_defenses/.venv/bin/python -u experiments/priority31_image_utility_dp.py --stage all --workers 8 > artifacts/priority31_image_utility_dp/execution.log 2>&1 &
nohup env PYTHONDONTWRITEBYTECODE=1 external_defenses/.venv/bin/python -u experiments/priority31_image_utility_dp.py --stage all --workers 8 > artifacts/priority31_image_utility_dp/execution.log 2>&1 < /dev/null & disown
PYTHONDONTWRITEBYTECODE=1 external_defenses/.venv/bin/python -u experiments/priority31_image_utility_dp.py --stage all --workers 8
PYTHONDONTWRITEBYTECODE=1 external_defenses/.venv/bin/python -u experiments/finalize_priority31_image_utility_dp.py --wait
launchctl submit -l org.fl-dna.priority31 -o /Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/artifacts/priority31_image_utility_dp/supervised_stdout.log -e /Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/artifacts/priority31_image_utility_dp/supervised_stderr.log -- /Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/external_defenses/.venv/bin/python -B -u /Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/experiments/supervise_priority31_image_utility_dp.py
```

The first background launch terminated before invoking the runner. The second terminated after creating split/clip, before any training job. The tracked execution session resumed those unchanged outputs. No calibration outcome was used to alter settings. User interruption of the chat did not terminate the tracked training session.

The tracked session subsequently disappeared after all six selection jobs and sixteen baseline replicates completed, while eight v1 jobs had started but had no completed output. Cause was not established. Those unfinished jobs restarted with identical settings; every completed job was skipped. A launchctl submission stalled before Python entered the supervisor (no journal or scientific outputs); that job was removed. The working supervisor instead used subprocess.Popen(start_new_session=True), with logs redirected to detached_stdout.log and detached_stderr.log. See detached_launch.json, runs.jsonl and supervisor.jsonl.

Completed utility/selection jobs: 182. Completed reconstruction targets: 78. Full invocation/job/error/resume journal: runs.jsonl; timing per completed job is stored in JSON.

Final checks: synthetic tests PASS; py_compile PASS; git diff --check PASS. final_checks.json contains exact check output. Experimental workload completed before finalization; thread count is 1 per worker.

## Artifacts and hashes

All new output paths and SHA-256 values are listed in sha256_manifest.csv (including every training result, reconstruction array/JSON, paired CSV, analysis CSV, split, clip, selection, calibration and checks). Report/manifest hashes are independently exported in final_hashes.json to avoid circular self-hashing.

Post-completion verification (2026-10-02): independently checked all 78 paired
target/source IDs, exact-binomial counts/p-values, the 115-hypothesis Holm
adjustment, requested ranks and original source hashes. The initial manifest
had two stale *log* hashes because the supervisor appended its exit/completed
records after finalizer hashing. Hashes were refreshed after all workloads
exited; no scientific artifact or numerical result was changed.

```sh
external_defenses/.venv/bin/python -B experiments/verify_priority31_final.py
external_defenses/.venv/bin/python -m py_compile experiments/verify_priority31_final.py
git diff --check
```

See independent_verification.json. final_hashes.json and the manifest itself
are excluded from the manifest to avoid circular self-hashing.
