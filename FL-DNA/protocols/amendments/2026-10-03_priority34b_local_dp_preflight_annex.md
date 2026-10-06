# Local-DP extension synthetic preflight metadata correction

Recorded before correction and before source freeze or research-data runs.
The synthetic image50round integration fixture caught TypeError: duplicate
client/no_bn_transmitted fields when immutable P34B image runner embeds the
client-side clipping/noise receipt. Preserve
artifacts/priority34b_local_dp/PREFLIGHT_FAILURE_1791034008077214000.json.
No research-data utility, target, calibration or reconstruction result exists.

Wrapper correction: retain full local receipt (including client/round/noBN) in
local_upload_audits, but omit client/no_bn_transmitted from the receipt returned
to the immutable image runner, which inserts the identical fields itself.
No state, clipping, Gaussian sample, private key, seed, query, objective or
training schedule changes. No original source is modified. Rerun all9 synthetic
tests and compile/diff checks before freeze/launch. Disclose the failed test in
the final extension report; do not classify it as a scientific utility failure.
