# Priority34C — IEEE historical firewall field

Before audit correction/replay. Preserved ID-less candidate failure: IEEE
target_firewall.json uses new_source_rows. Its corresponding project generator
stores both raw source rows and transaction_ids in the target bundle. Add the
explicit new_source_rows alias to conservative exclusions; bundles continue to
provide TransactionID-domain exclusions required by P32. Do not confuse raw
row numbers with TransactionIDs or replace the latter with the former. Both
sets may be excluded conservatively; actual bundle TransactionIDs are mandatory.
Permit metadata audit CLI dataset selection to avoid reparsing the already
completed unchanged PaySim receipt. Final freeze still verifies all three
receipt inputs and candidate coverage. No new target/attack run or tuning.
