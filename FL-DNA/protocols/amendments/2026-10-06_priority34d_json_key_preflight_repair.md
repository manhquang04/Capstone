# P34D infrastructure-only JSON-key preflight repair

User approved this amendment before implementation. Scope is solely exact
preflight comparison: normalize JSON object keys to strings on both sides,
then compare without numeric tolerance or value conversion. Identical int/float
and string keys compare equally; a real value mismatch still fails. Ambiguous
key collisions and non-JSON keys fail closed.

The original preparation failure and all existing logs remain byte-for-byte
unchanged and are labelled infrastructure-only by an additive receipt. No
calibration value, sigma, target list, seed, frozen source, comparison tolerance,
training job or recovery algorithm changes. A new administrative preparation
adapter performs the original preparation steps with the normalized comparison,
retains the prior failure and records any new failure under a separate name.
Original scientific files are not edited. Existing frozen launch/worker paths
remain the execution paths. Before launch, all16 calibration cells must match,
reservation IDs must equal the independently audited prespecified reservation,
and all frozen source/input hashes must pass. Preserve all old manifests.

The final P34D report must disclose the original preflight failure, this exact
representation-only repair, its unit tests, audit receipts and absence of
scientific replay or outcome changes. This amendment does not claim completion.
