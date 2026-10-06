# Priority 32: proposed BN-domain repair (separate protocol variant)

Date: 2026-10-02. Status: DRAFT / AWAITING SPECIFIC BUFFER-HANDLING APPROVAL.
Execution authorized: false. No repaired training may run under this draft.

Cause evidenced in reports/priority32_numerical_failure_diagnosis_20261002.md:
transforming signed running_var deltas produces negative variance states;
the diagnosed BAF/v1 seed has 3 negative coordinates at final evaluation.
This is a scientific protocol-scope issue, not a recoverable infrastructure
failure. Old outputs and original amendment remain preserved.

## Proposed change (requires supervisor choice)

Keep existing DNA math unchanged. Apply v1 conservative or v2 ratio .95 /
eta .01 only to named_parameters trainable deltas, preserving the original
state_dict tensor indices for seed derivation. Restore all BN/nontrainable
buffers to each client's original local values before ordinary FedAvg.
No clamping, numerical imputation, recalibration of variances or FedBN.
BN buffers remain transmitted RAW, and privacy implications must be disclosed.

Rerun all 126 transform jobs (3 datasets x 2 transforms x 21 seeds) into a
new repaired-variant directory; do not mix the 25 finite original transform
jobs with repaired outputs. Reuse all 63 original finite baseline jobs with
SHA/config checks. Same prepared splits/source IDs, seeds321000-321020,
50 rounds, Adam .001, focal alpha .95/gamma2, batch1024, K3, validation
threshold rule, final-round selection and margins .02/.005. One Torch thread
per process, four worker processes, resume only validated completed repaired
outputs. Collect every failure without abandoning future collection; all jobs
must be disclosed and any further numerical failure stops statistical claims.

Separate paired Student-t analysis at n21 using the original margins, plus
independent verification. Label as trainable-only transform / raw BN variant,
not replication of the frozen full-state transform. RQ3 synthetic
named-parameter benchmarks retain their original design. No older report,
conclusion or artifact is replaced; Priority 33 waits for valid P32 completion.

Required tests: buffers bit-identical to untransformed local values;
trainable transform bit-identical to the original function for same update,
tensor index/config/seed; ordinary nonnegative BN aggregation; paired result
coverage exactly21; no RNG/optimizer/model changes. Freeze implementation and
new execution manifest before repaired jobs, after explicit approval.
