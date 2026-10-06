# Priority34C — trusted bundle import-path preflight correction

Before correction/replay. PaySim historical audit completed832356excluded IDs
and1780candidate paths. IEEE bundle deserialization then failed because the
standalone script did not add the repository root to sys.path, although trusted
project pickles reference experiments classes. Add the root explicitly, matching
the earlier audit scripts; do not change any original module or pickle.
Retain the completed PaySim receipt byte-identical and the failure receipt.
No new target sampling, scientific job or reconstruction outcome occurred.
