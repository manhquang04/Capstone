# Priority34C — historical source-ID audit

Before running the provenance audit. Read only trusted project-owned earlier
target manifests/bundles under artifacts, not new P34C records. Enumerate JSON
files named target/source_id and torch bundles named target/bundle whose path
contains the dataset alias (PaySim, IEEE, BAF); exclude the P34C namespace.
Extract source_ids, source_ids_json and transaction_ids recursively; bind each
receipt to the original file hash and retain its exact extracted IDs. Deduplicate
byte-identical bundles for parsing but preserve all paths in provenance. Any
candidate without source IDs fails closed for explicit review; no silent skip.
Use an isolated subprocess before importing official TabLeak, because old
project and official packages share module names. Do not print bundle tensors,
keys or private noise seeds. No raw records enter the attack from this audit.

This first audit is preparatory: final launch still requires checking candidate
coverage against earlier experiment scripts and additional differently-named
source manifests if discovered, and hashing all final exclusion receipts.
It does not allocate real P34C target IDs or authorize scientific jobs.
