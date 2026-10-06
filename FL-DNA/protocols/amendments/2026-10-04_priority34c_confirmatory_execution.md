# Priority34C — confirmatory execution and observable selection

Written before any n39 batch access. Qualification sources and outputs remain
immutable. Require qualification and development stage completion, no live
stage workers, unchanged freezes and the verified local-P34B prerequisite.
Only cells passing both controls at n8 and fresh n24 are eligible. Use the
existing 39 reserved source-disjoint targets, not replacement records.

Each target has unprotected, v1 conservative, key-known v2, and LOCAL epsilon10
arms, plus one client-side distortion DP arm per DNA method whose development
matching gate passed. Global epsilon10 clipping C=.01, Gaussian SD
.08031273039270019, sensitivity2C, per-client update-level RDP over50 rounds,
delta1e-5: not record DP. No central/noisy-aggregate comparator. Independent
OS-entropy seeds for each DP arm/target are created before execution freeze in
a permission0600 private file; never supplied to attacks or printed.

The epsilon10 clipping implementation retains P34B's deterministic inward
float32 rounding multiplier (1−4*machine epsilon) after global clipping and
before adding float64 Gaussian noise then casting to the input dtype. This
preserves the intended L2 sensitivity bound after serialization. The separately
matched distortion comparator retains the P33B developmental mechanism.

All arms use identical public model, labels, target and public attack seed.
Native TabLeak retains P33B official1500iterations/30ensemble and v2 sketch loss.
V1 has plain and P33B structure-debiased candidates; select the smaller minimum
observable ensemble objective, never target accuracy. Ratio uses the same two
visible v1 candidates: decode each first-layer weight/bias ratio, then compute
the cosine mismatch of that candidate's gradient against its visible candidate
payload on the public model and known label. Select smaller mismatch with plain
winning ties. This selection uses no truth or raw protected gradient. V2 ratio
uses the frozen LSMR inverse, not scaled transpose. Ratio candidates are not
clamped; non-finite logits/gradients or zero denominator fail closed. Any candidate
numerical failure is a job failure, not silently discarded.

Scoring is performed only after candidate selection, with unchanged P33B mixed
feature accuracy and matching. Protected attack receipts contain only transmitted
payload, public model/labels, public transform metadata and no BN tensor. Raw
gradient, noise realization and private seed are not attacker inputs. Separate
truth and reconstruction artifacts are audit-private, outside a DP release.

CPU/thread1, four detached subprocess workers, exclusive stage ownership;
validated outputs skipped only on authorized infrastructure resume. Preserve
all attempts. Scientific failures drain submitted jobs and require direction.
Do not analyze missing outputs. Fixed72 directional tests include p1 reservations
for gated cells/comparators. n39 medians and paired effects have ranks13/27
intervals. Batch1 is the easiest recovery setting; single-gradient public-initial
checkpoint measurements differ from multi-step Adam FL utility. Final report
and independent audits, not stage completion, are the completion gate.
