# Priority35B B2b — baseline telemetry prerequisite, post hoc

Date: 2026-10-10, Asia/Ho_Chi_Minh. Written before replay or inspection of any new baseline upload norm. This addendum follows B2 and does not alter its results, original grid, matching threshold, or prospective Holm family. It authorizes only the approved unprotected baseline telemetry prerequisite. No B2b clip-only/DP arm may start until computed C values, source hashes and the full B2b execution amendment are frozen.

## Prespecified statistic and source population

For each dataset BAF, IEEE-CIS, PaySim, C_med is the median of the L2 norms of the unprotected non-BN trainable client update deltas, pooled over all three clients, rounds1–50 and the eleven paired P34A baseline seeds321000–321010: exactly1650 norms per dataset. Give each client/round/seed equal weight. For the even count, take the arithmetic mean of sorted positions825 and826 (one-based). No removal, replacement or weighting by partition size. No norm from any DP run is eligible. Proposed new C values are0.1, C_med/2 and C_med; existing C=.01 is used only for the extra clip-only control.

The measured delta is the P34A client's uploaded non-BN state minus the shared pre-round non-BN state, before aggregation. This is the same update object clipped in P34B/B2, not the L2 norm of an absolute model state. BN affine tensors and buffers are omitted.

## Read-only telemetry and exact replay

Replay unchanged P34A `train_job` with its original config, local BN, datasets, partition, operation order, dropout, fresh Adam, validation threshold and random seeds. Output only under artifacts/priority35b/b2b/baseline_telemetry/. CPU one torch thread per process and one inter-op thread. Call the original upload function and return its original output unchanged. Telemetry uses detached clones of uploaded tensors and previous shared tensors; subtract in their original float32 dtype/device and compute the concatenated L2 norm with torch.linalg.vector_norm, without casting either input or changing the originals. Record dtype/device, tensor order, and each round/client norm. Telemetry consumes no random numbers and changes no training/aggregation operation order. Norm is a descriptive float32 reduction; Python statistics.median aggregates the stored scalar norms. No float64 cast of a telemetry tensor is introduced.

The process pool uses spawn, max4; every job resets the original public seed exactly as P34A. Synthetic serial/parallel and instrumented/uninstrumented identity tests must pass before replay. All arrays/tensors must be finite; negative BN running variance, missing output, duplicate/missing telemetry or hash drift fail closed. Submitted jobs drain if one fails; do not analyze any incomplete or non-exact dataset.

Every replayed seed must have byte-identical final global non-BN and three local-BN checkpoint tensors, including dtype, shape, order, all buffers and integer counters; all six test/validation probability arrays must have identical bytes; all per-client and mean-client test/validation metrics and validation-selected thresholds must be exactly equal to P34A. Verify historical source/config/result/artifact hashes. No approximate tolerance is allowed for these checks. Archive failed attempts without replacing seeds or replaying them automatically.

Before computing C_med, the supervisor and all workers must have exited. An explicit independent post-exit verifier reloads and checks all33pairs again, complete1650norm coverage per dataset, replay hashes and historical hashes. Only then compute C_med and produce an additive values/source-hash annex. Freeze its digest before any B2b arm. Keep old manifests, logs, P34A files and B2 outputs unchanged; no commit, B3 or leakage run.

Infrastructure: detached33-job supervisor with resume only after explicit authorization, no-live audit and hash-validated skips. No ownership-socket EOF is treated as proof of process exit. Use periodic30-minute monitoring, not a sleep/poll loop. Final B2/B2b COMPLETE status is not granted by baseline telemetry completion alone.
