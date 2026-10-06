# Priority34C — development-only distortion calibration contract

Before any reserved development24 batch access. Only dataset/instrument cells
passing both n8 and n24 controls are eligible. Require qualification supervisor
and workers exited, stage completion, all prior frozen source/input digests
unchanged, and valid hashed outputs. No new targets are drawn; use the separately
reserved development24 IDs, disjoint from every qualification/confirmatory ID.

Capture raw gradients from exactly the same public checkpoint/domain/loss as
qualification. No reconstruction optimization is run on these developmental
records. Apply registered v1/v2 with unchanged keys/settings. For v2, defender
distortion uses the original server R-transpose lift, NOT key-known attack LS.
Clip C=1.01*maximum norm of24 developmental gradients. For each DNA arm, draw
24 independent unit Gaussian vectors, with a separate private OS-entropy seed
per cell/arm/record before freezing. Fit sigma = median DNA L2 distortion /
(C*median unit-Gaussian norm). SD=C*sigma, client-side before upload.
Gate requires sigma>0 and median of per-record absolute relative distortion
mismatches<=.05, exactly the P33B matching rule. No additional grid, retuning,
target replacement or confirmatory-based correction after seeing the result.

A failed distortion gate makes that comparator NOT_ASSESSABLE; it does not
invalidate an independently qualified local-epsilon10 contrast. Primary DNA
arms and epsilon10 remain on the same individual gradient and fixed source IDs.
Local epsilon10 continues globalC=.01/SD.08031273039270019, never the central
aggregate comparator. Calibration seeds/noise are defender-private and never
included in attack receipts or printed. Raw developmental artifacts and scoring
truth are audit-private, not a deployable DP release.

CPU/thread1, detached four-worker stage, validated result resume, original
attempts preserved. Source/input/private-seed file hash freeze before the first
development job. Scientific failures drain submitted stage and require direction;
do not analyze missing gradients, clamp or mark them as privacy evidence.
Stage completion is not P34C completion. Confirmatory attack/report drivers and
their additive execution freeze remain required before n39 batch access.
