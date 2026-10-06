# P34D synthetic test process isolation

Before utility preparation freeze/probes/training, the combined unittest process
passed all12 utility tests but failed to import the tabular checkpoint suite:
the vendored image package uses the top-level name `models`, which shadows
the fraud-model package in that interpreter. Preserved preflight receipt in
artifacts/priority34d/image_utility_preflight_failure.json. No image science ran.

Run the two suites in separate Python interpreters. Scientific drivers already
use separate subprocesses. No model, seed, protocol, data, source frozen by the
checkpoint stage or benchmark result is modified. This is test-harness process
isolation, not a scientific repair or reason to exclude an outcome.
