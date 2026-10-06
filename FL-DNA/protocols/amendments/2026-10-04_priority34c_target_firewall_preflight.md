# Priority34C — source-ID allocation preflight

Written before synthetic allocation tests or real target allocation. Allocation
uses source IDs only, never feature values, labels or reconstruction outcomes.
For each dataset, sort eligible unique training source IDs after excluding a
separately audited union of all historical targets. Apply numpy default_rng
permutation with dataset seeds PaySim340320, IEEE-CIS340321, BAF340322.
Allocate in fixed cell order ratio_batch1, tableak_batch1, tableak_batch2 and
stage order n8, n24, development24, n39, with record batch sizes1,1,2. All
stages/cells are source-disjoint; confirmatory IDs may be reserved but their
records must not be loaded before qualification and development matching pass.
Each dataset needs380 eligible records. Failed gates leave reservations unused;
never recycle, replace or draw more targets after observing outcomes.

The pure allocation helper rejects duplicate source IDs, inadequate pools,
unknown dataset/cell/stage, nonintegral IDs and incomplete manifest schemas.
Replay is bound to the source-ID pool, historical exclusions and fixed seed.
Historical-exclusion discovery/provenance and final execution freeze remain
pending; passing these synthetic tests is not a complete source-disjoint audit
and does not authorize a real target draw or recovery launch.
