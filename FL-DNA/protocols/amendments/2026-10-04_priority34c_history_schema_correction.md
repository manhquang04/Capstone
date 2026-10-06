# Priority34C — historical provenance schema correction

Before correcting/replaying a metadata-only audit. The first audit failed closed
because an earlier PaySim provenance file stores its raw row IDs as source_rows,
not source_ids. Inspection of create_phase4_source_disjoint_targets.py confirms
these are the same row IDs, and priority33c_paysim_bn.py also supports both names.
Preserve failure receipt; add source_rows and the P33C targets.json sources field.
No target sampling, record access, attack outcomes or scientific jobs occurred.

Coverage must also include generically named older PaySim targets.pt files, not
only paths containing PaySim. Enumerate target-named torch files except explicitly
other-dataset paths (IEEE, BAF, CIFAR, MNIST, Adult); their extracted source row IDs
are conservatively excluded from PaySim. Explicit other-dataset files remain
audited within their corresponding dataset scope. Any ambiguous positive ID
extraction is conservative exclusion, never outcome-selected inclusion.
Actual seed/model/target budgets and scoring contracts remain unchanged.
