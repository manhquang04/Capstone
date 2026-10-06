# Priority 32 BN-domain repair: trainable-only transform / raw BN variant

Date: 2026-10-02. Status: FROZEN / AUTHORIZED.
Execution authorized: true. Human approved this specific repair with "có"
after the explicit trainable-only/raw-BN proposal. The prior draft, original
amendment, 189 attempts, 88 finite outputs and all earlier findings remain
unchanged. This is a separately labeled scientific protocol variant, NOT an
infrastructure replay or a replacement of earlier full-state RQ2 evidence.

## Fixed repair and scope

Same transform formulas and constants. V1 conservative .08/.88/.45 and v2
.95/.01 apply to trainable parameter deltas only. All nontrainable BN buffers
retain untransformed local-client values and use ordinary sample-weighted
FedAvg. They remain transmitted RAW: this is NOT FedBN, and raw BN leakage
is explicitly outside any privacy protection claimed for this repaired
utility variant. Never clamp variance, impute predictions or exclude a seed.
The implementation calls the original state-transform functions, then restores
nontrainable entries. Discarded buffer transforms are computed solely to retain
original tensor indices/randomness; they are never installed in the model.

Rerun all 126 transform jobs, including all 25 finite original transform
jobs, in artifacts/priority32_multidataset/repair_trainable_raw_bn_20261002.
Use byte-identical copied/checksummed original 63 baseline outputs for pairing.
Original prepared train/validation/test caches and quality gates are reused,
not regenerated. Seeds321000–321020, three datasets, paired n21, margins
F1 .02/AUC .005, descriptive PR-AUC, 50 rounds, final checkpoint, Adam .001,
focal .95/gamma2, one local epoch, batch1024, K3, validation threshold unchanged.
Four parallel workers, each torch threads1. Estimated ~1–2 hours for training;
cost stages serial after workers exit, original 90-cell design unchanged.

## Checks and execution

Before repaired training: unit tests for bit-identical local buffers,
bit-identical trainable transform values vs original state functions using
same indices/seeds, identity baseline pairing, finite nonnegative BN aggregation.
Freeze code/amendment/preparation/baseline hashes and all126 configs before
execution. Resume only completed outputs matching those config hashes; collect
and disclose EVERY failure while all submitted jobs finish. No early stopping,
no data/parameter tuning. Any new missing/nonfinite output blocks statistical
claims and needs direction. Progress/checklist/journal after every job.

After complete coverage: paired Student-t95%CI df20 with original margins,
independent recomputation from JSON; original synthetic named-parameter RQ3
benchmark and descriptive DP cost unchanged. Repaired report must explicitly
contrast protocol definitions and disclose original failures/diagnostic replay.
No edits to Latex, external_defenses, datasets, original artifacts/reports,
or original runner. Priority33 waits for P32 final report and checks.
