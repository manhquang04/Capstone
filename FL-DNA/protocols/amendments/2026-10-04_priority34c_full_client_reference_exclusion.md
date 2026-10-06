# Priority34C — conservative full-client reference exclusion

Before metadata audit correction/replay. Early Phase3 full-client evaluator
targets store x/y only. They cannot be silently skipped. Their parent frozen
protocol_lock.json specifies max_rows/data_seed; run_phase3_full_client.py calls
the unchanged PaySim loader, whose _read_dataset samples a stratified population
with precisely those settings. Reconstruct the source-ID SAMPLE, not feature
records, from the raw isFraud column and sklearn train_test_split. Conservatively
exclude that entire sampled population, a superset of each client's targets.
Do not attempt outcome-based nearest-feature identity matching.

Only explicitly identified Phase3 full-client x/y-only bundles may use this
rule, with protocol/raw hashes, sampling seed/max_rows, count and ID-set digest.
Other ID-less candidates remain failures needing review. This is a stronger
pre-outcome exclusion, not seed replacement or scientific outcome tuning.
Freeze supplementary protocol/raw inputs with the final historical receipt.
All earlier files remain unchanged. No new target IDs or recovery are generated
by this metadata audit.
