# Priority 33c — staged pre-run protocol

Date: 2026-10-03 (Asia/Ho_Chi_Minh). Earlier artifacts are immutable.
Prerequisite P33b is complete with explicit NOT_ASSESSABLE qualification outcomes;
this does not establish privacy. No Latex/, external_defenses/ or dataset edits.

## Scope and execution gates

C1: fresh source-disjoint CIFAR-10 targets, n=39, v1 medium and stronger,
server-knowledge-matched native E2 versus unprotected, distortion-matched DP,
and utility-matched DP. Preserve the P30 official 4800-iteration configuration,
TV 0.01, one restart. Selection must use only the observable objective.
C2: descriptive n=24; explicitly separate batch-size sensitivity (batch4,
untrained checkpoint) and checkpoint sensitivity (batch1, trained checkpoint).
Arms: unprotected, v1 conservative, v2, utility-matched DP for each transform.
D: per-tensor clipping/noise, utility-matched versus conservative/v2 on images
and the PaySim BN-statistics channel. No substitution of raw-BN training for
invalid full-state utility targets. No claim of record-level DP.

CPU training, torch threads=1. Any non-finite model/update/loss/output or negative
BN running_var is a recorded failure; never clamp or replace seeds. Resume only
validated results with matching source/configuration hashes. Detached launches
must check for existing workers. All jobs and errors are disclosed in logs.

The only scientific job authorized by this initial phase freeze is the exact
P31 baseline checkpoint replay below. Before additional calibration or target
creation, append a separately saved pre-run execution annex specifying verified
medium/stronger parameters, source exclusion inventory, target seeds, complete
grids, per-tensor clipping norms, distortion calibration, gates, and full test
IDs. Do not start those workloads while any detail remains unresolved.

## Phase 1 — missing P31 checkpoint reproduction (FROZEN)

P31 saved utility metrics, not trained checkpoints. Reproduce baseline seed
51016, official inversefed LeNetZhu model seed42, CPU SGD lr0.1 momentum0,
100 epochs, batch256, CrossEntropyLoss, P31 split.json (12000 train/3000 val),
same torch.Generator randperm schedule. No learning-rate or length selection.
Read-only parent sources: experiments/priority31_image_utility_dp.py and its
imported P30 native adapters/comparator module. Save a new final_state.pt only
under artifacts/priority33c/checkpoint_replay/.
Quality gates: all floating tensors/outputs finite, BN variance nonnegative;
validation/test accuracy exactly equal the stored P31 seed51016 result, and
validation accuracy >=0.40. Test is replay verification only, never selection.
Fail-closed on disagreement: do not use the checkpoint for C2, preserve evidence.

## Planned statistics (not yet authorized to execute)

Exact one-sided sign tests in both directions, ties excluded, all-tie p=1.
Primary image quality PSNR (lower means stronger protection), SSIM descriptive.
BN channel standardized batch-mean MSE (higher means stronger protection).
Freeze every test ID in execution annex, Holm over entire P33c family including
reserved unassessable hypotheses p=1; also combine the unchanged existing115
raw p-values. n39 paired median differences and one-based order ranks13/27;
n24 sensitivity descriptive only, no incorrectly labeled n39 intervals.
Independently recompute statistics, preserve raw paired CSVs and hashes.

## Deliverables

progress.json/log and checklist.json; per-job configuration/source hashes;
reports/priority33c_report.md with commands, deviations, gates, effect sizes,
and final P33a/b/c evidence table. Finish py_compile and git diff --check;
never mark an unresolved stage complete.
