# Amendment: RQ2-v2 seed cap increase and single confirmatory execution

**Date:** 2026-09-16  
**Status:** CONFIRMATORY FROZEN AFTER THIS AMENDMENT  
**Supervisor:** Manh Quang

This amendment raises the RQ2 DNA Transform v2 maximum feasible seed count
from 30 to 60. The scientific decision rule is unchanged:

- alpha: 0.05
- target power: 0.80
- F1 non-inferiority margin: 0.02
- AUC-ROC non-inferiority margin: 0.005
- endpoint rule: both F1 and AUC-ROC must pass
- primary contrast: `DNA_TRANSFORM_V2_MINUS_BASELINE`

Rationale: the previous 30-seed cap was tied to the earlier machine-hour
planning ceiling. That machine-hour ceiling has been removed by supervisor
decision. The v2 pilot showed higher paired F1 variance than v1 and therefore
requires a larger confirmatory sample size. The frozen v2 pilot analysis
requires 52 seeds; the new cap of 60 provides safety margin while preserving
the rule that confirmatory execution uses the required n, not every available
seed under the cap.

The confirmatory seed count for this execution is exactly 52. The 52 seeds are
generated as the first 31 bits of
SHA-256(`RQ2-V2|CONFIRMATORY|2026-09-16|i`), i=0.., excluding zero, duplicates,
the eight RQ2-v2 pilot seeds, and prior RQ2 historical seeds found in frozen
RQ2 config files. The resulting list is frozen in
`protocols/config/rq2_v2_confirmatory.json`.

Exactly one confirmatory execution is authorized for these 52 seeds. Baseline
and DNA Transform v2 must share the same seed realization within each pair:
dataset subsampling/split, client partition, model initialization, and loader
order are all controlled by the same `FL_RUN_SEED`.

No seed, margin, endpoint rule, method, model setting, DNA v2 setting, or
stopping rule may be changed after seeing confirmatory outcomes. Reruns are
allowed only for demonstrable infrastructure failure, with failed records
retained. `torch.set_num_threads(1)` and per-process thread limits remain
unchanged.
